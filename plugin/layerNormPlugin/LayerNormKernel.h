#ifndef TRT_LAYERNORM_KERNEL_H
#define TRT_LAYERNORM_KERNEL_H

#include <cuda_runtime_api.h>
#include <stdint.h>
#include <cuda.h>
#include "cuda_fp16.h"

using half = __half;

template <typename T>
int32_t computeLayerNorm(T const *pInput, T const *gamma, T const *beta, T *pOutput, cudaStream_t stream, const int nBlock);

#endif