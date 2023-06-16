#include "LayerNormKernel.h"
#include <stdio.h>

template <typename T>
__global__ void layerNormKernel(T const *pInput, T const *gamma, T const *beta, T *pOutput)
{
    // nDim=768
    const int tx = threadIdx.x, index = blockIdx.x * 768 + threadIdx.x;

    // __shared__ float temp[256];
    __shared__ T temp[256];

    T value0 = pInput[index];
    T value1 = pInput[index + 256];
    T value2 = pInput[index + 512];

    // temp[tx] = static_cast<float>(value0 + value1 + value2);
    temp[tx] = value0 + value1 + value2;
    __syncthreads();

    for (int stride = 128; stride >= 1; stride /= 2)
    {
        // float tmp = temp[tx];
        // printf(" %f ", tmp);
        if (tx < stride)
        {
            temp[tx] += temp[tx + stride];
        }
        __syncthreads();
    }


    // T mean = static_cast<T>(temp[0] / 768);
    T mean = temp[0] / static_cast<T>(768);
    __syncthreads();

    // float tmp1 = static_cast<float>(value0 - mean);
    // float tmp2 = static_cast<float>(value1 - mean);
    // float tmp3 = static_cast<float>(value2 - mean);
    T tmp1 = value0 - mean;
    T tmp2 = value1 - mean;
    T tmp3 = value2 - mean;

    temp[tx] = tmp1 * tmp1 + tmp2 * tmp2 + tmp3 * tmp3;
    __syncthreads();

    for (int stride = 128; stride >= 1; stride /= 2)
    {
        if (tx < stride)
        {
            temp[tx] += temp[tx + stride];
        }
        __syncthreads();
    }
    // T var = static_cast<T>(temp[0] / 768);
    T var = temp[0] / static_cast<T>(768);
    
    const T epsilon = 6e-6;

    pOutput[index]       = (value0 - mean) * static_cast<T>(rsqrtf(var + epsilon)) * gamma[tx] + beta[tx];
    pOutput[index + 256] = (value1 - mean) * static_cast<T>(rsqrtf(var + epsilon)) * gamma[tx + 256] + beta[tx + 256];
    pOutput[index + 512] = (value2 - mean) * static_cast<T>(rsqrtf(var + epsilon)) * gamma[tx + 512] + beta[tx + 512];

    // if (std::is_same<T, float>::value)
    // {
    //     pOutput[index]       = (value0 - mean) * rsqrtf(var + epsilon) * gamma[tx] + beta[tx];
    //     pOutput[index + 256] = (value1 - mean) * rsqrtf(var + epsilon) * gamma[tx + 256] + beta[tx + 256];
    //     pOutput[index + 512] = (value2 - mean) * rsqrtf(var + epsilon) * gamma[tx + 512] + beta[tx + 512];
    // }
    // else
    // {
    //     pOutput[index]       = (value0 - mean) * hrsqrt(var + epsilon) * gamma[tx] + beta[tx];
    //     pOutput[index + 256] = (value1 - mean) * hrsqrt(var + epsilon) * gamma[tx + 256] + beta[tx + 256];
    //     pOutput[index + 512] = (value2 - mean) * hrsqrt(var + epsilon) * gamma[tx + 512] + beta[tx + 512];
    // }

}

template <typename T>
int32_t computeLayerNorm(T const *pInput, T const *gamma, T const *beta, T *pOutput, cudaStream_t stream, const int nBlock)
{
    layerNormKernel<T><<<nBlock, 256, 0, stream>>>(pInput, gamma, beta, pOutput);
    return 0;
}

template int computeLayerNorm<float>(
    float const *, float const *, float const *, float *, cudaStream_t, const int);

// template int computeLayerNorm<half>(
//     half const *, half const *, half const *, half *, cudaStream_t, const int);
