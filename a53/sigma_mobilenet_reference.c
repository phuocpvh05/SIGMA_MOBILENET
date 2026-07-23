#include "sigma_mobilenet_reference.h"

#include <stddef.h>
#include <string.h>

enum { SIGMA_KIND_CONV = 0, SIGMA_KIND_DEPTHWISE = 1, SIGMA_KIND_LINEAR = 2 };

float sigma_bf16_to_float(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

uint16_t sigma_float_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static float postprocess(
    float value,
    const sigma_mobilenet_layer_t *layer,
    const uint16_t *skip,
    size_t output_index) {
    if (layer->residual) {
        value += sigma_bf16_to_float(skip[output_index]);
    }
    if (layer->relu6) {
        if (value < 0.0f) value = 0.0f;
        if (value > 6.0f) value = 6.0f;
    }
    return value;
}

static void run_standard_conv(
    const sigma_mobilenet_layer_t *layer,
    const uint16_t *weights,
    const uint16_t *source,
    const uint16_t *skip,
    uint16_t *destination) {
    const uint32_t matrix_k = (uint32_t)layer->kernel * layer->kernel * layer->in_c;
    const uint32_t matrix_n = layer->out_c;
    const uint16_t *matrix = weights + layer->weight_offset;

    for (uint32_t oy = 0; oy < layer->out_h; ++oy) {
        for (uint32_t ox = 0; ox < layer->out_w; ++ox) {
            for (uint32_t n = 0; n < layer->out_c; ++n) {
                float accumulator = sigma_bf16_to_float(matrix[matrix_k * matrix_n + n]);
                uint32_t k = 0;
                for (uint32_t ky = 0; ky < layer->kernel; ++ky) {
                    const int iy = (int)(oy * layer->stride + ky) - (int)layer->padding;
                    for (uint32_t kx = 0; kx < layer->kernel; ++kx) {
                        const int ix = (int)(ox * layer->stride + kx) - (int)layer->padding;
                        if (iy >= 0 && iy < layer->in_h && ix >= 0 && ix < layer->in_w) {
                            const uint32_t input_base =
                                ((uint32_t)iy * layer->in_w + (uint32_t)ix) * layer->in_c;
                            for (uint32_t c = 0; c < layer->in_c; ++c, ++k) {
                                accumulator += sigma_bf16_to_float(source[input_base + c]) *
                                               sigma_bf16_to_float(matrix[k * matrix_n + n]);
                            }
                        } else {
                            k += layer->in_c;
                        }
                    }
                }
                const size_t output_index =
                    ((size_t)oy * layer->out_w + ox) * layer->out_c + n;
                accumulator = postprocess(accumulator, layer, skip, output_index);
                destination[output_index] = sigma_float_to_bf16(accumulator);
            }
        }
    }
}

static void run_depthwise(
    const sigma_mobilenet_layer_t *layer,
    const uint16_t *weights,
    const uint16_t *source,
    const uint16_t *skip,
    uint16_t *destination) {
    const uint32_t channel_words = (uint32_t)layer->kernel * layer->kernel + 1u;
    const uint16_t *matrix = weights + layer->weight_offset;
    for (uint32_t oy = 0; oy < layer->out_h; ++oy) {
        for (uint32_t ox = 0; ox < layer->out_w; ++ox) {
            for (uint32_t c = 0; c < layer->out_c; ++c) {
                const uint16_t *channel_weights = matrix + c * channel_words;
                float accumulator = sigma_bf16_to_float(channel_weights[channel_words - 1u]);
                uint32_t k = 0;
                for (uint32_t ky = 0; ky < layer->kernel; ++ky) {
                    const int iy = (int)(oy * layer->stride + ky) - (int)layer->padding;
                    for (uint32_t kx = 0; kx < layer->kernel; ++kx, ++k) {
                        const int ix = (int)(ox * layer->stride + kx) - (int)layer->padding;
                        if (iy >= 0 && iy < layer->in_h && ix >= 0 && ix < layer->in_w) {
                            const uint32_t input_index =
                                ((uint32_t)iy * layer->in_w + (uint32_t)ix) * layer->in_c + c;
                            accumulator += sigma_bf16_to_float(source[input_index]) *
                                           sigma_bf16_to_float(channel_weights[k]);
                        }
                    }
                }
                const size_t output_index =
                    ((size_t)oy * layer->out_w + ox) * layer->out_c + c;
                accumulator = postprocess(accumulator, layer, skip, output_index);
                destination[output_index] = sigma_float_to_bf16(accumulator);
            }
        }
    }
}

static void run_linear(
    const sigma_mobilenet_layer_t *layer,
    const uint16_t *weights,
    const uint16_t *source,
    const uint16_t *skip,
    uint16_t *destination) {
    const uint16_t *matrix = weights + layer->weight_offset;
    for (uint32_t n = 0; n < layer->out_c; ++n) {
        float accumulator = sigma_bf16_to_float(matrix[layer->in_c * layer->out_c + n]);
        for (uint32_t k = 0; k < layer->in_c; ++k) {
            accumulator += sigma_bf16_to_float(source[k]) *
                           sigma_bf16_to_float(matrix[k * layer->out_c + n]);
        }
        accumulator = postprocess(accumulator, layer, skip, n);
        destination[n] = sigma_float_to_bf16(accumulator);
    }
}

int sigma_mobilenet_reference(
    const uint16_t *weights,
    const uint16_t *image,
    uint16_t *banks,
    int *prediction) {
    if (!weights || !image || !banks || !prediction) return -1;
    memset(banks, 0, 3u * SIGMA_MOBILE_BANK_WORDS * sizeof(uint16_t));
    memcpy(banks, image, SIGMA_MOBILE_IMAGE_WORDS * sizeof(uint16_t));

    for (uint32_t index = 0; index < SIGMA_MOBILE_LAYER_COUNT; ++index) {
        const sigma_mobilenet_layer_t *layer = &sigma_mobilenet_layers[index];
        const uint16_t *source = banks + (size_t)layer->src_bank * SIGMA_MOBILE_BANK_WORDS;
        const uint16_t *skip = banks + (size_t)layer->skip_bank * SIGMA_MOBILE_BANK_WORDS;
        uint16_t *destination = banks + (size_t)layer->dst_bank * SIGMA_MOBILE_BANK_WORDS;
        if (layer->kind == SIGMA_KIND_CONV) {
            run_standard_conv(layer, weights, source, skip, destination);
        } else if (layer->kind == SIGMA_KIND_DEPTHWISE) {
            run_depthwise(layer, weights, source, skip, destination);
        } else if (layer->kind == SIGMA_KIND_LINEAR) {
            run_linear(layer, weights, source, skip, destination);
        } else {
            return -2;
        }
    }

    const sigma_mobilenet_layer_t *last = &sigma_mobilenet_layers[SIGMA_MOBILE_LAYER_COUNT - 1u];
    const uint16_t *logits = banks + (size_t)last->dst_bank * SIGMA_MOBILE_BANK_WORDS;
    int best = 0;
    float best_value = sigma_bf16_to_float(logits[0]);
    for (int index = 1; index < 10; ++index) {
        const float value = sigma_bf16_to_float(logits[index]);
        if (value > best_value) {
            best = index;
            best_value = value;
        }
    }
    *prediction = best;
    return 0;
}

