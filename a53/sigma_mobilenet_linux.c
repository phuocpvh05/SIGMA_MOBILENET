#define _POSIX_C_SOURCE 200809L

#include "sigma_mobilenet_reference.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define SIGMA_BASE_DEFAULT 0xA0000000ull
#define SIGMA_MAP_BYTES 0x10000u
#define SIGMA_CONTROL 0x0000u
#define SIGMA_STATUS 0x0004u
#define SIGMA_CYCLES 0x0008u
#define SIGMA_ID 0x000cu
#define SIGMA_CLOCK_HZ_REG 0x0024u
#define SIGMA_PROFILE 0x0028u
#define SIGMA_IMAGE 0x1000u
#define SIGMA_SIGNATURE 0x5349474du
#define SIGMA_PROFILE_MOB6 0x4d4f4236u

typedef struct {
    const char *weights_path;
    const char *image_path;
    const char *json_path;
    uint64_t base;
    int runs;
    int run_software;
    int run_hardware;
} options_t;

static double monotonic_ms(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC_RAW, &value);
    return value.tv_sec * 1000.0 + value.tv_nsec / 1000000.0;
}

static int compare_double(const void *left, const void *right) {
    const double a = *(const double *)left;
    const double b = *(const double *)right;
    return (a > b) - (a < b);
}

static int compare_u32(const void *left, const void *right) {
    const uint32_t a = *(const uint32_t *)left;
    const uint32_t b = *(const uint32_t *)right;
    return (a > b) - (a < b);
}

static double median_double(const double *values, int count) {
    double *copy = malloc((size_t)count * sizeof(*copy));
    if (!copy) return -1.0;
    memcpy(copy, values, (size_t)count * sizeof(*copy));
    qsort(copy, (size_t)count, sizeof(*copy), compare_double);
    const double result = (count & 1) ? copy[count / 2] :
        (copy[count / 2 - 1] + copy[count / 2]) / 2.0;
    free(copy);
    return result;
}

static uint32_t median_u32(const uint32_t *values, int count) {
    uint32_t *copy = malloc((size_t)count * sizeof(*copy));
    if (!copy) return 0;
    memcpy(copy, values, (size_t)count * sizeof(*copy));
    qsort(copy, (size_t)count, sizeof(*copy), compare_u32);
    const uint32_t result = (count & 1) ? copy[count / 2] :
        (uint32_t)(((uint64_t)copy[count / 2 - 1] + copy[count / 2]) / 2u);
    free(copy);
    return result;
}

static int read_hex_mem(const char *path, uint16_t *words, size_t expected) {
    FILE *file = fopen(path, "r");
    if (!file) {
        fprintf(stderr, "Cannot open %s: %s\n", path, strerror(errno));
        return -1;
    }
    size_t count = 0;
    unsigned value;
    while (fscanf(file, "%x", &value) == 1) {
        if (count >= expected || value > 0xffffu) {
            fprintf(stderr, "Invalid or excessive BF16 data in %s\n", path);
            fclose(file);
            return -1;
        }
        words[count++] = (uint16_t)value;
    }
    fclose(file);
    if (count != expected) {
        fprintf(stderr, "%s: expected %zu words, found %zu\n", path, expected, count);
        return -1;
    }
    return 0;
}

static long read_cpu_khz(void) {
    FILE *file = fopen("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", "r");
    long value = -1;
    if (file) {
        if (fscanf(file, "%ld", &value) != 1) value = -1;
        fclose(file);
    }
    return value;
}

static void mmio_write(volatile uint8_t *base, uint32_t offset, uint32_t value) {
    *(volatile uint32_t *)(base + offset) = value;
    __sync_synchronize();
}

static uint32_t mmio_read(volatile uint8_t *base, uint32_t offset) {
    const uint32_t value = *(volatile uint32_t *)(base + offset);
    __sync_synchronize();
    return value;
}

static int run_hardware(
    const options_t *options,
    const uint16_t *image,
    int *prediction,
    uint32_t *cycles_median,
    uint32_t *clock_hz,
    double *load_median_ms,
    double *e2e_median_ms) {
    const int memory = open("/dev/mem", O_RDWR | O_SYNC);
    if (memory < 0) {
        fprintf(stderr, "Cannot open /dev/mem (run as root): %s\n", strerror(errno));
        return -1;
    }
    volatile uint8_t *registers = mmap(
        NULL, SIGMA_MAP_BYTES, PROT_READ | PROT_WRITE, MAP_SHARED,
        memory, (off_t)options->base);
    if (registers == MAP_FAILED) {
        fprintf(stderr, "mmap(0x%" PRIx64 ") failed: %s\n", options->base, strerror(errno));
        close(memory);
        return -1;
    }
    if (mmio_read(registers, SIGMA_ID) != SIGMA_SIGNATURE) {
        fprintf(stderr, "SIGMA ID mismatch at 0x%" PRIx64 "; program the PS-AXI bitstream first\n",
                options->base);
        munmap((void *)registers, SIGMA_MAP_BYTES);
        close(memory);
        return -1;
    }
    if (mmio_read(registers, SIGMA_PROFILE) != SIGMA_PROFILE_MOB6) {
        fprintf(stderr, "MobileNet RTL profile mismatch; rebuild/program the optimized bitstream\n");
        munmap((void *)registers, SIGMA_MAP_BYTES);
        close(memory);
        return -1;
    }
    *clock_hz = mmio_read(registers, SIGMA_CLOCK_HZ_REG);
    if (*clock_hz == 0u) return -1;

    uint32_t *cycle_samples = calloc((size_t)options->runs, sizeof(*cycle_samples));
    double *load_samples = calloc((size_t)options->runs, sizeof(*load_samples));
    double *e2e_samples = calloc((size_t)options->runs, sizeof(*e2e_samples));
    if (!cycle_samples || !load_samples || !e2e_samples) return -1;
    int expected_prediction = -1;

    for (int run = 0; run < options->runs; ++run) {
        const double e2e_begin = monotonic_ms();
        const double load_begin = e2e_begin;
        for (uint32_t index = 0; index < SIGMA_MOBILE_IMAGE_WORDS; ++index) {
            mmio_write(registers, SIGMA_IMAGE + index * 4u, image[index]);
        }
        load_samples[run] = monotonic_ms() - load_begin;
        mmio_write(registers, SIGMA_CONTROL, 2u);
        mmio_write(registers, SIGMA_CONTROL, 1u);

        uint32_t status = 0;
        const double timeout = monotonic_ms() + 5000.0;
        uint32_t polls = 0;
        do {
            status = mmio_read(registers, SIGMA_STATUS);
            if (status & 4u) {
                fprintf(stderr, "SIGMA hardware error, status=0x%08" PRIx32 "\n", status);
                return -1;
            }
            if ((++polls & 0xffffu) == 0u) sched_yield();
            if (monotonic_ms() > timeout) {
                fprintf(stderr, "SIGMA hardware timeout\n");
                return -1;
            }
        } while (!(status & 2u));

        e2e_samples[run] = monotonic_ms() - e2e_begin;
        cycle_samples[run] = mmio_read(registers, SIGMA_CYCLES);
        const int current_prediction = (int)((status >> 16) & 0xfu);
        if (expected_prediction >= 0 && current_prediction != expected_prediction) {
            fprintf(stderr, "Hardware prediction changed: %d -> %d\n",
                    expected_prediction, current_prediction);
            return -1;
        }
        expected_prediction = current_prediction;
    }

    *prediction = expected_prediction;
    *cycles_median = median_u32(cycle_samples, options->runs);
    *load_median_ms = median_double(load_samples, options->runs);
    *e2e_median_ms = median_double(e2e_samples, options->runs);
    free(cycle_samples);
    free(load_samples);
    free(e2e_samples);
    munmap((void *)registers, SIGMA_MAP_BYTES);
    close(memory);
    return 0;
}

static void usage(const char *program) {
    fprintf(stderr,
        "Usage: %s [--weights FILE] [--image FILE] [--runs N] [--base HEX]\n"
        "          [--software-only|--hardware-only] [--json FILE]\n", program);
}

static int parse_options(int argc, char **argv, options_t *options) {
    *options = (options_t){
        .weights_path = "mobilenet_onchip_bf16.mem",
        .image_path = "mobilenet_board_image.mem",
        .json_path = "sigma_mobilenet_a53_vs_fpga.json",
        .base = SIGMA_BASE_DEFAULT,
        .runs = 5,
        .run_software = 1,
        .run_hardware = 1,
    };
    for (int index = 1; index < argc; ++index) {
        if (!strcmp(argv[index], "--weights") && index + 1 < argc) options->weights_path = argv[++index];
        else if (!strcmp(argv[index], "--image") && index + 1 < argc) options->image_path = argv[++index];
        else if (!strcmp(argv[index], "--json") && index + 1 < argc) options->json_path = argv[++index];
        else if (!strcmp(argv[index], "--runs") && index + 1 < argc) options->runs = atoi(argv[++index]);
        else if (!strcmp(argv[index], "--base") && index + 1 < argc) options->base = strtoull(argv[++index], NULL, 0);
        else if (!strcmp(argv[index], "--software-only")) options->run_hardware = 0;
        else if (!strcmp(argv[index], "--hardware-only")) options->run_software = 0;
        else { usage(argv[0]); return -1; }
    }
    if (options->runs < 1 || options->runs > 1000) return -1;
    return 0;
}

int main(int argc, char **argv) {
    options_t options;
    if (parse_options(argc, argv, &options)) return 2;

    uint16_t *image = malloc(SIGMA_MOBILE_IMAGE_WORDS * sizeof(*image));
    uint16_t *weights = NULL;
    uint16_t *banks = NULL;
    if (!image || read_hex_mem(options.image_path, image, SIGMA_MOBILE_IMAGE_WORDS)) return 2;
    if (options.run_software) {
        weights = malloc(SIGMA_MOBILE_WEIGHT_WORDS * sizeof(*weights));
        banks = malloc(3u * SIGMA_MOBILE_BANK_WORDS * sizeof(*banks));
        if (!weights || !banks || read_hex_mem(options.weights_path, weights, SIGMA_MOBILE_WEIGHT_WORDS)) return 2;
    }

    int software_prediction = -1;
    double software_median_ms = -1.0;
    if (options.run_software) {
        double *samples = calloc((size_t)options.runs, sizeof(*samples));
        // One untimed run warms instruction/data caches, matching standard CPU methodology.
        if (sigma_mobilenet_reference(weights, image, banks, &software_prediction)) return 3;
        for (int run = 0; run < options.runs; ++run) {
            const double begin = monotonic_ms();
            int prediction = -1;
            if (sigma_mobilenet_reference(weights, image, banks, &prediction)) return 3;
            samples[run] = monotonic_ms() - begin;
            if (prediction != software_prediction) return 3;
        }
        software_median_ms = median_double(samples, options.runs);
        free(samples);
    }

    int hardware_prediction = -1;
    uint32_t hardware_cycles = 0;
    uint32_t hardware_clock_hz = 0;
    double hardware_load_ms = -1.0;
    double hardware_e2e_ms = -1.0;
    if (options.run_hardware && run_hardware(
            &options, image, &hardware_prediction, &hardware_cycles,
            &hardware_clock_hz,
            &hardware_load_ms, &hardware_e2e_ms)) return 4;

    const double hardware_compute_ms = options.run_hardware ?
        hardware_cycles * 1000.0 / hardware_clock_hz : -1.0;
    const double compute_speedup = options.run_software && options.run_hardware ?
        software_median_ms / hardware_compute_ms : -1.0;
    const double e2e_speedup = options.run_software && options.run_hardware ?
        software_median_ms / hardware_e2e_ms : -1.0;

    printf("============================================================\n");
    printf("SIGMA MobileNet Cortex-A53 vs FPGA (same Genesys ZU)\n");
    printf("Runs                 : %d\n", options.runs);
    printf("A53 cores online     : %ld\n", sysconf(_SC_NPROCESSORS_ONLN));
    const long cpu_khz = read_cpu_khz();
    if (cpu_khz > 0) printf("A53 current clock     : %.3f MHz\n", cpu_khz / 1000.0);
    if (options.run_software) {
        printf("A53 C prediction      : %d\n", software_prediction);
        printf("A53 C median          : %.6f ms\n", software_median_ms);
    }
    if (options.run_hardware) {
        printf("FPGA prediction       : %d\n", hardware_prediction);
        printf("FPGA clock configured: %.3f MHz\n", hardware_clock_hz / 1.0e6);
        printf("FPGA cycles/frame     : %" PRIu32 "\n", hardware_cycles);
        printf("FPGA compute          : %.6f ms\n", hardware_compute_ms);
        printf("PS AXI image load     : %.6f ms\n", hardware_load_ms);
        printf("PS AXI end-to-end     : %.6f ms\n", hardware_e2e_ms);
    }
    if (options.run_software && options.run_hardware) {
        printf("Compute speedup       : %.3fx\n", compute_speedup);
        printf("End-to-end speedup    : %.3fx\n", e2e_speedup);
        printf("Prediction match      : %s\n",
               software_prediction == hardware_prediction ? "YES" : "NO");
    }
    printf("============================================================\n");

    FILE *json = fopen(options.json_path, "w");
    if (json) {
        fprintf(json,
            "{\n  \"network\": \"MobileNetV2-0.25 MNIST\",\n"
            "  \"runs\": %d,\n  \"cpu_clock_khz\": %ld,\n"
            "  \"a53_prediction\": %d,\n  \"a53_median_ms\": %.9f,\n"
            "  \"fpga_prediction\": %d,\n  \"fpga_clock_mhz\": %.6f,\n"
            "  \"fpga_cycles_per_frame\": %" PRIu32 ",\n  \"fpga_compute_ms\": %.9f,\n"
            "  \"ps_axi_image_load_ms\": %.9f,\n  \"ps_axi_end_to_end_ms\": %.9f,\n"
            "  \"compute_speedup\": %.9f,\n  \"end_to_end_speedup\": %.9f\n}\n",
            options.runs, cpu_khz, software_prediction, software_median_ms,
            hardware_prediction, hardware_clock_hz / 1.0e6,
            hardware_cycles, hardware_compute_ms,
            hardware_load_ms, hardware_e2e_ms, compute_speedup, e2e_speedup);
        fclose(json);
        printf("Report: %s\n", options.json_path);
    }

    free(image);
    free(weights);
    free(banks);
    return (options.run_software && options.run_hardware &&
            software_prediction != hardware_prediction) ? 5 : 0;
}
