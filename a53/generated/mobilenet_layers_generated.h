// Generated from cnn/mobilenet_onchip_manifest.json; do not edit.
#ifndef SIGMA_MOBILENET_LAYERS_GENERATED_H
#define SIGMA_MOBILENET_LAYERS_GENERATED_H
#include <stdint.h>
#define SIGMA_MOBILE_LAYER_COUNT 53u
#define SIGMA_MOBILE_WEIGHT_WORDS 245450u
#define SIGMA_MOBILE_BANK_WORDS 12288u
#define SIGMA_MOBILE_IMAGE_WORDS 784u
typedef struct {
    uint32_t weight_offset;
    uint16_t in_c, out_c;
    uint8_t in_h, in_w, out_h, out_w;
    uint8_t kernel, stride, padding;
    uint8_t kind, src_bank, dst_bank, skip_bank;
    uint8_t relu6, residual;
} sigma_mobilenet_layer_t;
static const sigma_mobilenet_layer_t sigma_mobilenet_layers[53] = {
    {0u, 1u, 8u, 28u, 28u, 14u, 14u, 3u, 2u, 1u, 0u, 0u, 1u, 0u, 1u, 0u}, // 0: features.0.0
    {80u, 8u, 8u, 14u, 14u, 14u, 14u, 3u, 1u, 1u, 1u, 1u, 0u, 1u, 1u, 0u}, // 1: features.1.conv.0.0
    {160u, 8u, 8u, 14u, 14u, 14u, 14u, 1u, 1u, 0u, 0u, 0u, 2u, 1u, 0u, 1u}, // 2: features.1.conv.1
    {232u, 8u, 48u, 14u, 14u, 14u, 14u, 1u, 1u, 0u, 0u, 2u, 0u, 2u, 1u, 0u}, // 3: features.2.conv.0.0
    {664u, 48u, 48u, 14u, 14u, 7u, 7u, 3u, 2u, 1u, 1u, 0u, 1u, 2u, 1u, 0u}, // 4: features.2.conv.1.0
    {1144u, 48u, 8u, 7u, 7u, 7u, 7u, 1u, 1u, 0u, 0u, 1u, 0u, 2u, 0u, 0u}, // 5: features.2.conv.2
    {1536u, 8u, 48u, 7u, 7u, 7u, 7u, 1u, 1u, 0u, 0u, 0u, 1u, 0u, 1u, 0u}, // 6: features.3.conv.0.0
    {1968u, 48u, 48u, 7u, 7u, 7u, 7u, 3u, 1u, 1u, 1u, 1u, 2u, 0u, 1u, 0u}, // 7: features.3.conv.1.0
    {2448u, 48u, 8u, 7u, 7u, 7u, 7u, 1u, 1u, 0u, 0u, 2u, 1u, 0u, 0u, 1u}, // 8: features.3.conv.2
    {2840u, 8u, 48u, 7u, 7u, 7u, 7u, 1u, 1u, 0u, 0u, 1u, 0u, 1u, 1u, 0u}, // 9: features.4.conv.0.0
    {3272u, 48u, 48u, 7u, 7u, 4u, 4u, 3u, 2u, 1u, 1u, 0u, 1u, 1u, 1u, 0u}, // 10: features.4.conv.1.0
    {3752u, 48u, 8u, 4u, 4u, 4u, 4u, 1u, 1u, 0u, 0u, 1u, 0u, 1u, 0u, 0u}, // 11: features.4.conv.2
    {4144u, 8u, 48u, 4u, 4u, 4u, 4u, 1u, 1u, 0u, 0u, 0u, 1u, 0u, 1u, 0u}, // 12: features.5.conv.0.0
    {4576u, 48u, 48u, 4u, 4u, 4u, 4u, 3u, 1u, 1u, 1u, 1u, 2u, 0u, 1u, 0u}, // 13: features.5.conv.1.0
    {5056u, 48u, 8u, 4u, 4u, 4u, 4u, 1u, 1u, 0u, 0u, 2u, 1u, 0u, 0u, 1u}, // 14: features.5.conv.2
    {5448u, 8u, 48u, 4u, 4u, 4u, 4u, 1u, 1u, 0u, 0u, 1u, 0u, 1u, 1u, 0u}, // 15: features.6.conv.0.0
    {5880u, 48u, 48u, 4u, 4u, 4u, 4u, 3u, 1u, 1u, 1u, 0u, 2u, 1u, 1u, 0u}, // 16: features.6.conv.1.0
    {6360u, 48u, 8u, 4u, 4u, 4u, 4u, 1u, 1u, 0u, 0u, 2u, 0u, 1u, 0u, 1u}, // 17: features.6.conv.2
    {6752u, 8u, 48u, 4u, 4u, 4u, 4u, 1u, 1u, 0u, 0u, 0u, 1u, 0u, 1u, 0u}, // 18: features.7.conv.0.0
    {7184u, 48u, 48u, 4u, 4u, 2u, 2u, 3u, 2u, 1u, 1u, 1u, 0u, 0u, 1u, 0u}, // 19: features.7.conv.1.0
    {7664u, 48u, 16u, 2u, 2u, 2u, 2u, 1u, 1u, 0u, 0u, 0u, 1u, 0u, 0u, 0u}, // 20: features.7.conv.2
    {8448u, 16u, 96u, 2u, 2u, 2u, 2u, 1u, 1u, 0u, 0u, 1u, 0u, 1u, 1u, 0u}, // 21: features.8.conv.0.0
    {10080u, 96u, 96u, 2u, 2u, 2u, 2u, 3u, 1u, 1u, 1u, 0u, 2u, 1u, 1u, 0u}, // 22: features.8.conv.1.0
    {11040u, 96u, 16u, 2u, 2u, 2u, 2u, 1u, 1u, 0u, 0u, 2u, 0u, 1u, 0u, 1u}, // 23: features.8.conv.2
    {12592u, 16u, 96u, 2u, 2u, 2u, 2u, 1u, 1u, 0u, 0u, 0u, 1u, 0u, 1u, 0u}, // 24: features.9.conv.0.0
    {14224u, 96u, 96u, 2u, 2u, 2u, 2u, 3u, 1u, 1u, 1u, 1u, 2u, 0u, 1u, 0u}, // 25: features.9.conv.1.0
    {15184u, 96u, 16u, 2u, 2u, 2u, 2u, 1u, 1u, 0u, 0u, 2u, 1u, 0u, 0u, 1u}, // 26: features.9.conv.2
    {16736u, 16u, 96u, 2u, 2u, 2u, 2u, 1u, 1u, 0u, 0u, 1u, 0u, 1u, 1u, 0u}, // 27: features.10.conv.0.0
    {18368u, 96u, 96u, 2u, 2u, 2u, 2u, 3u, 1u, 1u, 1u, 0u, 2u, 1u, 1u, 0u}, // 28: features.10.conv.1.0
    {19328u, 96u, 16u, 2u, 2u, 2u, 2u, 1u, 1u, 0u, 0u, 2u, 0u, 1u, 0u, 1u}, // 29: features.10.conv.2
    {20880u, 16u, 96u, 2u, 2u, 2u, 2u, 1u, 1u, 0u, 0u, 0u, 1u, 0u, 1u, 0u}, // 30: features.11.conv.0.0
    {22512u, 96u, 96u, 2u, 2u, 2u, 2u, 3u, 1u, 1u, 1u, 1u, 0u, 0u, 1u, 0u}, // 31: features.11.conv.1.0
    {23472u, 96u, 24u, 2u, 2u, 2u, 2u, 1u, 1u, 0u, 0u, 0u, 1u, 0u, 0u, 0u}, // 32: features.11.conv.2
    {25800u, 24u, 144u, 2u, 2u, 2u, 2u, 1u, 1u, 0u, 0u, 1u, 0u, 1u, 1u, 0u}, // 33: features.12.conv.0.0
    {29400u, 144u, 144u, 2u, 2u, 2u, 2u, 3u, 1u, 1u, 1u, 0u, 2u, 1u, 1u, 0u}, // 34: features.12.conv.1.0
    {30840u, 144u, 24u, 2u, 2u, 2u, 2u, 1u, 1u, 0u, 0u, 2u, 0u, 1u, 0u, 1u}, // 35: features.12.conv.2
    {34320u, 24u, 144u, 2u, 2u, 2u, 2u, 1u, 1u, 0u, 0u, 0u, 1u, 0u, 1u, 0u}, // 36: features.13.conv.0.0
    {37920u, 144u, 144u, 2u, 2u, 2u, 2u, 3u, 1u, 1u, 1u, 1u, 2u, 0u, 1u, 0u}, // 37: features.13.conv.1.0
    {39360u, 144u, 24u, 2u, 2u, 2u, 2u, 1u, 1u, 0u, 0u, 2u, 1u, 0u, 0u, 1u}, // 38: features.13.conv.2
    {42840u, 24u, 144u, 2u, 2u, 2u, 2u, 1u, 1u, 0u, 0u, 1u, 0u, 1u, 1u, 0u}, // 39: features.14.conv.0.0
    {46440u, 144u, 144u, 2u, 2u, 1u, 1u, 3u, 2u, 1u, 1u, 0u, 1u, 1u, 1u, 0u}, // 40: features.14.conv.1.0
    {47880u, 144u, 40u, 1u, 1u, 1u, 1u, 1u, 1u, 0u, 0u, 1u, 0u, 1u, 0u, 0u}, // 41: features.14.conv.2
    {53680u, 40u, 240u, 1u, 1u, 1u, 1u, 1u, 1u, 0u, 0u, 0u, 1u, 0u, 1u, 0u}, // 42: features.15.conv.0.0
    {63520u, 240u, 240u, 1u, 1u, 1u, 1u, 3u, 1u, 1u, 1u, 1u, 2u, 0u, 1u, 0u}, // 43: features.15.conv.1.0
    {65920u, 240u, 40u, 1u, 1u, 1u, 1u, 1u, 1u, 0u, 0u, 2u, 1u, 0u, 0u, 1u}, // 44: features.15.conv.2
    {75560u, 40u, 240u, 1u, 1u, 1u, 1u, 1u, 1u, 0u, 0u, 1u, 0u, 1u, 1u, 0u}, // 45: features.16.conv.0.0
    {85400u, 240u, 240u, 1u, 1u, 1u, 1u, 3u, 1u, 1u, 1u, 0u, 2u, 1u, 1u, 0u}, // 46: features.16.conv.1.0
    {87800u, 240u, 40u, 1u, 1u, 1u, 1u, 1u, 1u, 0u, 0u, 2u, 0u, 1u, 0u, 1u}, // 47: features.16.conv.2
    {97440u, 40u, 240u, 1u, 1u, 1u, 1u, 1u, 1u, 0u, 0u, 0u, 1u, 0u, 1u, 0u}, // 48: features.17.conv.0.0
    {107280u, 240u, 240u, 1u, 1u, 1u, 1u, 3u, 1u, 1u, 1u, 1u, 0u, 0u, 1u, 0u}, // 49: features.17.conv.1.0
    {109680u, 240u, 80u, 1u, 1u, 1u, 1u, 1u, 1u, 0u, 0u, 0u, 1u, 0u, 0u, 0u}, // 50: features.17.conv.2
    {128960u, 80u, 1280u, 1u, 1u, 1u, 1u, 1u, 1u, 0u, 0u, 1u, 0u, 0u, 1u, 0u}, // 51: features.18.0
    {232640u, 1280u, 10u, 1u, 1u, 1u, 1u, 1u, 1u, 0u, 2u, 0u, 1u, 0u, 0u, 0u}, // 52: classifier.1
};
#endif
