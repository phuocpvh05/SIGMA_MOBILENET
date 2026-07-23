#include "sigma_mobilenet_reference.h"
#include "mobilenet_payload_generated.h"
#include "mobilenet_timing_generated.h"

#include "xil_cache.h"
#include "xil_io.h"
#include "xiltimer.h"

#include <stdint.h>
#include <stdio.h>

#define SIGMA_BASE               0xA0000000u
#define SIGMA_CONTROL            (SIGMA_BASE + 0x0000u)
#define SIGMA_STATUS             (SIGMA_BASE + 0x0004u)
#define SIGMA_CYCLES             (SIGMA_BASE + 0x0008u)
#define SIGMA_ID                 (SIGMA_BASE + 0x000cu)
#define SIGMA_LOAD_CYCLES        (SIGMA_BASE + 0x0010u)
#define SIGMA_CORE_CYCLES        (SIGMA_BASE + 0x0014u)
#define SIGMA_POST_CYCLES        (SIGMA_BASE + 0x0018u)
#define SIGMA_DEPTHWISE_CYCLES   (SIGMA_BASE + 0x001cu)
#define SIGMA_POINTWISE_CYCLES   (SIGMA_BASE + 0x0020u)
#define SIGMA_CLOCK_HZ_REG       (SIGMA_BASE + 0x0024u)
#define SIGMA_PROFILE            (SIGMA_BASE + 0x0028u)
#define SIGMA_IMAGE_COUNT        (SIGMA_BASE + 0x002cu)
#define SIGMA_IMAGE_HASH         (SIGMA_BASE + 0x0030u)
#define SIGMA_RUN_COUNT          (SIGMA_BASE + 0x0034u)
#define SIGMA_IMAGE              (SIGMA_BASE + 0x1000u)
#define SIGMA_SIGNATURE          0x5349474du
#define SIGMA_PROFILE_MOB6       0x4d4f4236u
#define MNIST_TEST_IMAGES        10000u
#define MNIST_IMAGE_WORDS        784u
#define PROGRESS_INTERVAL        500u

extern const uint16_t sigma_mnist10k_images[];
extern const uint16_t sigma_mnist10k_images_broken[];
extern const uint16_t sigma_mnist10k_images_poor[];
extern const uint8_t sigma_mnist10k_labels[];

static uint16_t activation_banks[3u * SIGMA_MOBILE_BANK_WORDS];

typedef struct {
    uint32_t status;
    uint32_t cycles;
    uint32_t load_cycles;
    uint32_t core_cycles;
    uint32_t post_cycles;
    uint32_t depthwise_cycles;
    uint32_t pointwise_cycles;
    uint32_t image_count;
    uint32_t image_hash;
    uint32_t run_count;
} sigma_perf_t;

typedef struct {
    uint32_t cpu_correct;
    uint32_t fpga_correct;
    uint32_t cpu_fpga_matches;
    uint64_t cpu_total_us;
    uint64_t fpga_compute_us;
    uint64_t fpga_load_us;
    uint64_t fpga_e2e_us;
    uint64_t wall_us;
    uint64_t cycles_per_frame;
} condition_result_t;

typedef struct {
    const char *name;
    const char *description;
    const uint16_t *images;
} condition_t;

static uint64_t ticks_to_us(uint64_t ticks) {
    return (ticks * 1000000ull + (uint64_t)COUNTS_PER_SECOND / 2ull) /
           (uint64_t)COUNTS_PER_SECOND;
}

static uint64_t ratio_milli(uint64_t numerator, uint64_t denominator) {
    return denominator ? (numerator * 1000ull) / denominator : 0ull;
}

static uint64_t throughput_milli(uint64_t images, uint64_t elapsed_us) {
    return elapsed_us ? (images * 1000000000ull) / elapsed_us : 0ull;
}

static uint32_t accuracy_basis_points(uint32_t correct, uint32_t total) {
    return total ? (uint32_t)(((uint64_t)correct * 10000ull) / total) : 0u;
}

static uint32_t image_hash_bf16(const uint16_t *image) {
    uint32_t hash = 0u;
    for (uint32_t index = 0; index < MNIST_IMAGE_WORDS; ++index) {
        hash = (hash << 1) | (hash >> 31);
        hash ^= image[index];
    }
    return hash;
}

static void print_decimal_milli(
    const char *name,
    uint64_t value,
    const char *suffix) {
    printf("%-24s: %llu.%03llu%s\r\n",
           name,
           (unsigned long long)(value / 1000ull),
           (unsigned long long)(value % 1000ull),
           suffix);
}

static void print_accuracy(
    const char *name,
    uint32_t correct,
    uint32_t total) {
    const uint32_t basis_points = accuracy_basis_points(correct, total);
    printf("%-24s: %lu / %lu (%lu.%02lu%%)\r\n",
           name,
           (unsigned long)correct,
           (unsigned long)total,
           (unsigned long)(basis_points / 100u),
           (unsigned long)(basis_points % 100u));
}

static int run_fpga_once(
    const uint16_t *image,
    uint32_t *prediction,
    sigma_perf_t *perf,
    uint64_t *load_ticks,
    uint64_t *e2e_ticks) {
    XTime e2e_begin, load_done, done;
    const uint32_t run_count_before = Xil_In32(SIGMA_RUN_COUNT);
    XTime_GetTime(&e2e_begin);
    // Clear the sticky status and the image-transfer diagnostics before the
    // new frame.  Clearing after the load would also erase its checksum.
    Xil_Out32(SIGMA_CONTROL, 2u);
    for (uint32_t index = 0; index < MNIST_IMAGE_WORDS; ++index) {
        Xil_Out32(SIGMA_IMAGE + 4u * index, image[index]);
    }
    XTime_GetTime(&load_done);

    const uint32_t expected_hash = image_hash_bf16(image);
    const uint32_t hardware_words = Xil_In32(SIGMA_IMAGE_COUNT);
    const uint32_t hardware_hash = Xil_In32(SIGMA_IMAGE_HASH);
    if ((hardware_words != MNIST_IMAGE_WORDS) ||
        (hardware_hash != expected_hash)) {
        printf("FAIL: AXI image transfer words=%lu/784 hash=0x%08lx/0x%08lx\r\n",
               (unsigned long)hardware_words,
               (unsigned long)hardware_hash,
               (unsigned long)expected_hash);
        return -3;
    }

    Xil_Out32(SIGMA_CONTROL, 1u);
    const XTime timeout =
        load_done + (XTime)(5ull * (uint64_t)COUNTS_PER_SECOND);
    uint32_t status;
    do {
        status = Xil_In32(SIGMA_STATUS);
        if (status & 4u) {
            return -1;
        }
        XTime_GetTime(&done);
        if (done > timeout) {
            return -2;
        }
    } while (!(status & 2u));

    const uint32_t run_count_after = Xil_In32(SIGMA_RUN_COUNT);
    if (run_count_after != run_count_before + 1u) {
        printf("FAIL: stale completion run_count=%lu expected=%lu\r\n",
               (unsigned long)run_count_after,
               (unsigned long)(run_count_before + 1u));
        return -4;
    }

    *prediction = (status >> 16) & 0xfu;
    perf->status = status;
    perf->cycles = Xil_In32(SIGMA_CYCLES);
    perf->load_cycles = Xil_In32(SIGMA_LOAD_CYCLES);
    perf->core_cycles = Xil_In32(SIGMA_CORE_CYCLES);
    perf->post_cycles = Xil_In32(SIGMA_POST_CYCLES);
    perf->depthwise_cycles = Xil_In32(SIGMA_DEPTHWISE_CYCLES);
    perf->pointwise_cycles = Xil_In32(SIGMA_POINTWISE_CYCLES);
    perf->image_count = hardware_words;
    perf->image_hash = hardware_hash;
    perf->run_count = run_count_after;
    *load_ticks = (uint64_t)(load_done - e2e_begin);
    *e2e_ticks = (uint64_t)(done - e2e_begin);
    return 0;
}

static int benchmark_condition(
    const condition_t *condition,
    uint64_t fpga_clock_hz,
    condition_result_t *result) {
    int cpu_prediction = -1;
    if (sigma_mobilenet_reference(
            sigma_mobilenet_weights,
            condition->images,
            activation_banks,
            &cpu_prediction) != 0) {
        printf("FAIL [%s]: Cortex-A53 warm-up\r\n", condition->name);
        return 2;
    }

    uint32_t fpga_prediction = 0xffffffffu;
    sigma_perf_t warmup_perf = {0};
    uint64_t warmup_load = 0ull;
    uint64_t warmup_e2e = 0ull;
    if (run_fpga_once(
            condition->images,
            &fpga_prediction,
            &warmup_perf,
            &warmup_load,
            &warmup_e2e) != 0) {
        printf("FAIL [%s]: FPGA warm-up\r\n", condition->name);
        return 3;
    }
    if ((int)fpga_prediction != cpu_prediction) {
        printf("FAIL [%s]: FPGA warm-up prediction=%lu CPU=%d; "
               "full 10K run cancelled\r\n",
               condition->name,
               (unsigned long)fpga_prediction,
               cpu_prediction);
        printf("  RTL profile/status   : 0x%08lx / 0x%08lx "
               "(layer=%lu)\r\n",
               (unsigned long)Xil_In32(SIGMA_PROFILE),
               (unsigned long)warmup_perf.status,
               (unsigned long)((warmup_perf.status >> 8) & 0xffu));
        printf("  image words/hash     : %lu / 0x%08lx "
               "(expected 784 / 0x%08lx)\r\n",
               (unsigned long)warmup_perf.image_count,
               (unsigned long)warmup_perf.image_hash,
               (unsigned long)image_hash_bf16(condition->images));
        printf("  cycles total/load/core/depthwise/pointwise/post: "
               "%lu/%lu/%lu/%lu/%lu/%lu\r\n",
               (unsigned long)warmup_perf.cycles,
               (unsigned long)warmup_perf.load_cycles,
               (unsigned long)warmup_perf.core_cycles,
               (unsigned long)warmup_perf.depthwise_cycles,
               (unsigned long)warmup_perf.pointwise_cycles,
               (unsigned long)warmup_perf.post_cycles);
        printf("  completed run count  : %lu\r\n",
               (unsigned long)warmup_perf.run_count);
        return 4;
    }
    printf("\r\n------------------------------------------------------------\r\n");
    printf("Condition: %s - %s\r\n", condition->name, condition->description);
    printf("Warm-up complete; 10,000 measured images start now.\r\n");

    uint32_t cpu_correct = 0u;
    uint32_t fpga_correct = 0u;
    uint32_t cpu_fpga_matches = 0u;
    uint32_t mismatch_examples = 0u;
    uint64_t cpu_total_ticks = 0ull;
    uint64_t fpga_total_cycles = 0ull;
    uint64_t fpga_total_load_cycles = 0ull;
    uint64_t fpga_total_core_cycles = 0ull;
    uint64_t fpga_total_post_cycles = 0ull;
    uint64_t fpga_total_depthwise_cycles = 0ull;
    uint64_t fpga_total_pointwise_cycles = 0ull;
    uint64_t fpga_total_load_ticks = 0ull;
    uint64_t fpga_total_e2e_ticks = 0ull;
    XTime benchmark_begin, benchmark_end;
    XTime_GetTime(&benchmark_begin);

    for (uint32_t sample = 0; sample < MNIST_TEST_IMAGES; ++sample) {
        const uint16_t *image =
            condition->images + (uint64_t)sample * MNIST_IMAGE_WORDS;
        const uint32_t label = sigma_mnist10k_labels[sample];

        XTime cpu_begin, cpu_end;
        cpu_prediction = -1;
        XTime_GetTime(&cpu_begin);
        const int cpu_result = sigma_mobilenet_reference(
            sigma_mobilenet_weights,
            image,
            activation_banks,
            &cpu_prediction);
        XTime_GetTime(&cpu_end);
        if (cpu_result != 0) {
            printf("FAIL [%s]: Cortex-A53 sample %lu, code %d\r\n",
                   condition->name,
                   (unsigned long)sample,
                   cpu_result);
            return 4;
        }
        cpu_total_ticks += (uint64_t)(cpu_end - cpu_begin);

        sigma_perf_t fpga_perf = {0};
        uint64_t fpga_load_ticks = 0ull;
        uint64_t fpga_e2e_ticks = 0ull;
        const int fpga_result = run_fpga_once(
            image,
            &fpga_prediction,
            &fpga_perf,
            &fpga_load_ticks,
            &fpga_e2e_ticks);
        if (fpga_result != 0) {
            printf("FAIL [%s]: FPGA sample %lu, code %d\r\n",
                   condition->name,
                   (unsigned long)sample,
                   fpga_result);
            return 5;
        }
        fpga_total_cycles += fpga_perf.cycles;
        fpga_total_load_cycles += fpga_perf.load_cycles;
        fpga_total_core_cycles += fpga_perf.core_cycles;
        fpga_total_post_cycles += fpga_perf.post_cycles;
        fpga_total_depthwise_cycles += fpga_perf.depthwise_cycles;
        fpga_total_pointwise_cycles += fpga_perf.pointwise_cycles;
        fpga_total_load_ticks += fpga_load_ticks;
        fpga_total_e2e_ticks += fpga_e2e_ticks;

        if ((uint32_t)cpu_prediction == label) {
            ++cpu_correct;
        }
        if (fpga_prediction == label) {
            ++fpga_correct;
        }
        if ((uint32_t)cpu_prediction == fpga_prediction) {
            ++cpu_fpga_matches;
        } else if (mismatch_examples < 10u) {
            printf("MISMATCH [%s] sample=%lu label=%lu cpu=%d fpga=%lu\r\n",
                   condition->name,
                   (unsigned long)sample,
                   (unsigned long)label,
                   cpu_prediction,
                   (unsigned long)fpga_prediction);
            ++mismatch_examples;
        }

        if (((sample + 1u) % PROGRESS_INTERVAL) == 0u) {
            printf("[%s] %5lu / %lu | CPU correct=%lu | "
                   "FPGA correct=%lu | match=%lu\r\n",
                   condition->name,
                   (unsigned long)(sample + 1u),
                   (unsigned long)MNIST_TEST_IMAGES,
                   (unsigned long)cpu_correct,
                   (unsigned long)fpga_correct,
                   (unsigned long)cpu_fpga_matches);
        }
    }
    XTime_GetTime(&benchmark_end);

    const uint64_t cpu_total_us = ticks_to_us(cpu_total_ticks);
    const uint64_t fpga_compute_us =
        (fpga_total_cycles * 1000000ull + fpga_clock_hz / 2ull) /
        fpga_clock_hz;
    const uint64_t fpga_load_us = ticks_to_us(fpga_total_load_ticks);
    const uint64_t fpga_e2e_us = ticks_to_us(fpga_total_e2e_ticks);
    const uint64_t benchmark_wall_us =
        ticks_to_us((uint64_t)(benchmark_end - benchmark_begin));
    const uint64_t compute_speedup =
        ratio_milli(cpu_total_us, fpga_compute_us);
    const uint64_t e2e_speedup =
        ratio_milli(cpu_total_us, fpga_e2e_us);
    const uint64_t cpu_throughput =
        throughput_milli(MNIST_TEST_IMAGES, cpu_total_us);
    const uint64_t fpga_throughput =
        throughput_milli(MNIST_TEST_IMAGES, fpga_e2e_us);

    printf("\r\n%s MNIST-10K measured results\r\n", condition->name);
    print_accuracy("Cortex-A53 accuracy", cpu_correct, MNIST_TEST_IMAGES);
    print_accuracy("FPGA accuracy", fpga_correct, MNIST_TEST_IMAGES);
    print_accuracy("CPU/FPGA agreement", cpu_fpga_matches, MNIST_TEST_IMAGES);
    printf("Cortex-A53 total       : %llu us\r\n",
           (unsigned long long)cpu_total_us);
    printf("FPGA compute total     : %llu us\r\n",
           (unsigned long long)fpga_compute_us);
    printf("PS AXI image load      : %llu us\r\n",
           (unsigned long long)fpga_load_us);
    printf("FPGA end-to-end total  : %llu us\r\n",
           (unsigned long long)fpga_e2e_us);
    printf("Complete test wall time: %llu us\r\n",
           (unsigned long long)benchmark_wall_us);
    const uint64_t classified_cycles =
        fpga_total_load_cycles + fpga_total_core_cycles +
        fpga_total_post_cycles + fpga_total_depthwise_cycles +
        fpga_total_pointwise_cycles;
    const uint64_t control_cycles =
        (fpga_total_cycles >= classified_cycles) ?
        (fpga_total_cycles - classified_cycles) : 0ull;
    printf("FPGA clock configured : %llu.%03llu MHz\r\n",
           (unsigned long long)(fpga_clock_hz / 1000000ull),
           (unsigned long long)((fpga_clock_hz % 1000000ull) / 1000ull));
#if SIGMA_POST_ROUTE_VALID
    printf("Post-route Fmax limit  : %lu.%03lu MHz (WNS %ld ps)\r\n",
           (unsigned long)(SIGMA_POST_ROUTE_FMAX_KHZ / 1000u),
           (unsigned long)(SIGMA_POST_ROUTE_FMAX_KHZ % 1000u),
           (long)SIGMA_POST_ROUTE_WNS_PS);
#else
    printf("Post-route Fmax limit  : not exported for this RTL build\r\n");
#endif
    printf("FPGA cycles/frame      : %llu\r\n",
           (unsigned long long)(fpga_total_cycles / MNIST_TEST_IMAGES));
    printf("  generic load/frame   : %llu\r\n",
           (unsigned long long)(fpga_total_load_cycles / MNIST_TEST_IMAGES));
    printf("  generic core/frame   : %llu\r\n",
           (unsigned long long)(fpga_total_core_cycles / MNIST_TEST_IMAGES));
    printf("  depthwise fast/frame : %llu\r\n",
           (unsigned long long)(fpga_total_depthwise_cycles / MNIST_TEST_IMAGES));
    printf("  pointwise fast/frame : %llu\r\n",
           (unsigned long long)(fpga_total_pointwise_cycles / MNIST_TEST_IMAGES));
    printf("  post-process/frame   : %llu\r\n",
           (unsigned long long)(fpga_total_post_cycles / MNIST_TEST_IMAGES));
    printf("  controller/frame     : %llu\r\n",
           (unsigned long long)(control_cycles / MNIST_TEST_IMAGES));
    print_decimal_milli("Compute speedup", compute_speedup, "x");
    print_decimal_milli("End-to-end speedup", e2e_speedup, "x");
    print_decimal_milli("Cortex-A53 images/s", cpu_throughput, "");
    print_decimal_milli("FPGA images/s", fpga_throughput, "");

    if (cpu_fpga_matches != MNIST_TEST_IMAGES) {
        printf("NOTE: %lu CPU/FPGA boundary-case prediction differences\r\n",
               (unsigned long)(MNIST_TEST_IMAGES - cpu_fpga_matches));
    }

    result->cpu_correct = cpu_correct;
    result->fpga_correct = fpga_correct;
    result->cpu_fpga_matches = cpu_fpga_matches;
    result->cpu_total_us = cpu_total_us;
    result->fpga_compute_us = fpga_compute_us;
    result->fpga_load_us = fpga_load_us;
    result->fpga_e2e_us = fpga_e2e_us;
    result->wall_us = benchmark_wall_us;
    result->cycles_per_frame = fpga_total_cycles / MNIST_TEST_IMAGES;
    return 0;
}

int main(void) {
    Xil_ICacheEnable();
    Xil_DCacheEnable();

    printf("\r\n============================================================\r\n");
    printf("SIGMA MobileNetV2-0.25: MNIST robustness board benchmark\r\n");
    printf("30,000 inferences: 10K clean + 10K broken + 10K poor/noisy\r\n");
    printf("Cortex-A53 and FPGA consume the same BF16 images/weights\r\n");
    printf("64-PE SIGMA mesh + timing-hardened 256-bit fast engine (MOB6)\r\n");
    printf("============================================================\r\n");

    const uint32_t signature = Xil_In32(SIGMA_ID);
    if (signature != SIGMA_SIGNATURE) {
        printf("FAIL: SIGMA ID=0x%08lx, expected 0x%08lx\r\n",
               (unsigned long)signature,
               (unsigned long)SIGMA_SIGNATURE);
        return 1;
    }
    const uint32_t profile = Xil_In32(SIGMA_PROFILE);
    const uint64_t fpga_clock_hz = Xil_In32(SIGMA_CLOCK_HZ_REG);
    if ((profile != SIGMA_PROFILE_MOB6) || (fpga_clock_hz == 0ull)) {
        printf("FAIL: incompatible MobileNet RTL profile=0x%08lx clock=%llu\r\n",
               (unsigned long)profile,
               (unsigned long long)fpga_clock_hz);
        return 1;
    }

    const condition_t conditions[] = {
        {"clean", "original MNIST test images", sigma_mnist10k_images},
        {"broken", "two short stroke segments removed", sigma_mnist10k_images_broken},
        {"poor", "shifted, low-contrast, noisy and locally erased", sigma_mnist10k_images_poor},
    };
    condition_result_t results[3] = {{0}};
    uint32_t total_cpu_correct = 0u;
    uint32_t total_fpga_correct = 0u;
    uint32_t total_matches = 0u;

    for (uint32_t index = 0; index < 3u; ++index) {
        const int status = benchmark_condition(
            &conditions[index], fpga_clock_hz, &results[index]);
        if (status != 0) {
            return status;
        }
        total_cpu_correct += results[index].cpu_correct;
        total_fpga_correct += results[index].fpga_correct;
        total_matches += results[index].cpu_fpga_matches;
    }

    printf("\r\n============================================================\r\n");
    printf("SIGMA MobileNet robustness summary (measured on board)\r\n");
    printf("============================================================\r\n");
    for (uint32_t index = 0; index < 3u; ++index) {
        const uint32_t cpu_bp = accuracy_basis_points(
            results[index].cpu_correct, MNIST_TEST_IMAGES);
        const uint32_t fpga_bp = accuracy_basis_points(
            results[index].fpga_correct, MNIST_TEST_IMAGES);
        const uint32_t agree_bp = accuracy_basis_points(
            results[index].cpu_fpga_matches, MNIST_TEST_IMAGES);
        printf("%-8s CPU=%lu.%02lu%% FPGA=%lu.%02lu%% "
               "agreement=%lu.%02lu%% cycles/frame=%llu\r\n",
               conditions[index].name,
               (unsigned long)(cpu_bp / 100u),
               (unsigned long)(cpu_bp % 100u),
               (unsigned long)(fpga_bp / 100u),
               (unsigned long)(fpga_bp % 100u),
               (unsigned long)(agree_bp / 100u),
               (unsigned long)(agree_bp % 100u),
               (unsigned long long)results[index].cycles_per_frame);
    }
    print_accuracy("Overall CPU accuracy", total_cpu_correct, 30000u);
    print_accuracy("Overall FPGA accuracy", total_fpga_correct, 30000u);
    print_accuracy("Overall agreement", total_matches, 30000u);
    printf("PASS: all 30,000 CPU and FPGA inferences completed.\r\n");
    printf("Accuracy is reported independently; close-logit CPU/FPGA "
           "differences are not transport or execution failures.\r\n");
    return 0;
}
