#ifndef SIGMA_MOBILENET_REFERENCE_H
#define SIGMA_MOBILENET_REFERENCE_H

#include <stdint.h>
#include "mobilenet_layers_generated.h"

#ifdef __cplusplus
extern "C" {
#endif

// banks must contain 3 * SIGMA_MOBILE_BANK_WORDS uint16_t elements.
int sigma_mobilenet_reference(
    const uint16_t *weights,
    const uint16_t *image,
    uint16_t *banks,
    int *prediction);

float sigma_bf16_to_float(uint16_t value);
uint16_t sigma_float_to_bf16(float value);

#ifdef __cplusplus
}
#endif
#endif
