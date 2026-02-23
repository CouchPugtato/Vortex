#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct YoloTrtDims {
    int32_t nb_dims;
    int32_t dims[8];
} YoloTrtDims;

void* yolo_trt_create(const char* engine_path, YoloTrtDims* input_dims, YoloTrtDims* output_dims);
int32_t yolo_trt_infer(void* handle, const float* input, float* output, size_t output_len);
void yolo_trt_destroy(void* handle);

#ifdef __cplusplus
}
#endif
