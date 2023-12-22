#ifndef TRT_LAYERNORM_KERNEL_H
#define TRT_LAYERNORM_KERNEL_H

#include <cuda_runtime_api.h>
#include <stdint.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include <algorithm>

using half = __half;

template <typename T>
int32_t computeLayerNorm(
    const int gridSize, const int nHiddenDimension, T const *pInput, T *pOutput,
    T const *gamma, T const *beta, const float epsilon, cudaStream_t stream);

#endif