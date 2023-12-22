#include "LayerNormKernel.h"

// Tool Functions
template<typename T>
using kvp = cub::KeyValuePair<T, T>;

template<typename T>
struct mySum
{
    __host__ __device__ __forceinline__ kvp<T> operator()(const kvp<T> &a, const kvp<T> &b) const
    {
        return kvp<T>(a.key + b.key, a.value + b.value);
    }
};

// For Vectorize Memory Access
template<int VPT>
struct BytesToType;

template<>
struct BytesToType<2>
{
    using type = uint16_t;
};
template<>
struct BytesToType<4>
{
    using type = uint32_t;
};
template<>
struct BytesToType<8>
{
    using type = uint64_t;
};
template<>
struct BytesToType<16>
{
    using type = float4;
};

template<int Bytes>
__device__ inline void copy(const void *local, void *data)
{
    using T = typename BytesToType<Bytes>::type;

    const T *in  = static_cast<const T *>(local);
    T       *out = static_cast<T *>(data);
    *out         = *in;
}

// nHiddenDimension<=32, doing reduce in warp
// Single-pass algorithm
template<typename T, typename OP_T, int TPB>
__global__ void LayerNormSmallKernel(const int nHiddenDimension, const T *input, const T *gamma, const T *beta, T *output, const float epsilon)
{
    const int index       = blockIdx.x * nHiddenDimension + threadIdx.x;
    const T   denominator = T(1) / T(nHiddenDimension);
    OP_T      val         = 0;

    kvp<OP_T> threadData(0, 0);
    // cub::WarpReduce will consider all threads in warp
    // init for valid threads only
    if (threadIdx.x < nHiddenDimension)
    {
        val        = (OP_T)input[index];
        OP_T tmp0  = val * (OP_T)denominator;
        OP_T tmp1  = val * tmp0;
        threadData = mySum<OP_T>()(threadData, kvp<OP_T>(tmp0, tmp1));
    }
    
    using WarpReduce = cub::WarpReduce<kvp<OP_T>, TPB>;
    __shared__ typename WarpReduce::TempStorage temp;
    __shared__ OP_T                             mu, rsigma;
    
    const auto sumKV = WarpReduce(temp).Reduce(threadData, mySum<OP_T>());
    
    if (threadIdx.x == 0)
    {
        mu     = sumKV.key;
        rsigma = rsqrt(sumKV.value - mu * mu + (OP_T)epsilon);
    }
    __syncthreads();

    if (threadIdx.x < nHiddenDimension)
    {
        const OP_T g = gamma[threadIdx.x], b = beta[threadIdx.x];
        output[index] = (val - mu) * rsigma * g + b;
    }
}

template __global__ void LayerNormSmallKernel<float, float, 32>(const int, const float *, const float *, const float *, float *, const float);
template __global__ void LayerNormSmallKernel<__half, float, 32>(const int, const __half *, const __half *, const __half *, __half *, const float);

// Two-Pass Algorithm, using Vectorize Memory Access
template<typename T, typename OP_T, int TPB, int VPT>
__global__ void LayerNormMediumKernel(const int nHiddenDimension, const T *input, const T *gamma, const T *beta, T *output, const float epsilon)
{
    // consider the registers used in a block, such as using data type float and nHiddenDimension is 1024,
    // localX:      256 thread/block * 4 element/thread (i.e. VPT) * 4 Byte/element = 4 KiB
    // localBeta:   1024 element / block * 4 Byte / element = 4 KiB
    // localGamma:  1024 element / block * 4 Byte / element = 4 KiB
    // localBias:   1024 element / block * 4 Byte / element = 4 KiB

    const int  index = blockIdx.x * nHiddenDimension + threadIdx.x * VPT;
    T          localX[VPT], localGamma[VPT], localBeta[VPT];

    OP_T local_sum {0};
    OP_T local_var_sum {0};

    copy<sizeof(T) * VPT>(&input[index], localX);
    copy<sizeof(T) * VPT>(&beta[threadIdx.x * VPT], localBeta);
    copy<sizeof(T) * VPT>(&gamma[threadIdx.x * VPT], localGamma);
    
    // Blockwise shared mean and var
    __shared__ OP_T                              mu, rsigma;
    // Using cub for reduce
    using BlockReduce = cub::BlockReduce<OP_T, TPB>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    // First reduce for mean value 
#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        const OP_T tmp = (OP_T)localX[it];
        local_sum = local_sum + tmp;
    }

    OP_T &ref_0 = local_sum;
    OP_T sum = BlockReduce(temp_storage).Sum(ref_0);
    if (threadIdx.x == 0)
    {
        mu = sum / nHiddenDimension;
    }
    // Wait for the reducing end, real sum will be stored in thread 0 of block
    __syncthreads();
    
    // Second reduce for var value 
#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        const OP_T diff = (OP_T)localX[it] - mu;
        local_var_sum += diff * diff;
    }

    OP_T &ref_1 = local_var_sum;
    OP_T var_sum = BlockReduce(temp_storage).Sum(ref_1);
    if (threadIdx.x == 0)
    {
        rsigma = rsqrt(var_sum / nHiddenDimension + (OP_T)epsilon);
    }
    __syncthreads();

#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        localX[it] = (OP_T)localGamma[it] * ((OP_T)localX[it] - mu) * rsigma + (OP_T)localBeta[it];
    }

    copy<sizeof(T) * VPT>(localX, &output[index]);
}

template __global__ void LayerNormMediumKernel<float, float, 64, 4>(const int, const float *, const float *, const float *, float *, const float);
template __global__ void LayerNormMediumKernel<__half, float, 64, 4>(const int, const __half *, const __half *, const __half *, __half *, const float);
template __global__ void LayerNormMediumKernel<float, float, 256, 4>(const int, const float *, const float *, const float *, float *, const float);
template __global__ void LayerNormMediumKernel<__half, float, 256, 4>(const int, const __half *, const __half *, const __half *, __half *, const float);

// Not using vectorize memory access
template<typename T, typename OP_T, int TPB>
__global__ void LayerNormGeneralKernel(const int nHiddenDimension, const T *input, const T *gamma, const T *beta, T *output, const float epsilon)
{
    const int  offset      = blockIdx.x * nHiddenDimension;
    const OP_T denominator = OP_T(1) / OP_T(nHiddenDimension);
    kvp<OP_T>  threadData(0, 0);

    for (int i = threadIdx.x; i < nHiddenDimension; i += TPB)
    {
        const int  index = offset + i;
        OP_T       val   = input[index];
        const OP_T tmp   = val * denominator;
        threadData       = mySum<OP_T>()(threadData, kvp<OP_T>(tmp, tmp * val));
        output[index]    = val;
    }

    using BlockReduce = cub::BlockReduce<kvp<OP_T>, TPB>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ OP_T                              mu, rsigma;

    const auto sumKV = BlockReduce(temp).Reduce(threadData, mySum<OP_T>());

    if (threadIdx.x == 0)
    {
        mu     = sumKV.key;
        rsigma = rsqrt(sumKV.value - mu * mu + (OP_T)epsilon);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < nHiddenDimension; i += TPB)
    {
        const int index = offset + i;
        output[index]   = ((OP_T)output[index] - mu) * rsigma * (OP_T)gamma[i] + (OP_T)beta[i];
    }
}

template __global__ void LayerNormGeneralKernel<float, float, 256>(const int, const float *, const float *, const float *, float *, const float);
template __global__ void LayerNormGeneralKernel<__half, float, 256>(const int, const __half *, const __half *, const __half *, __half *, const float);


template <typename T>
int32_t computeLayerNorm(
    const int gridSize, const int nHiddenDimension, T const *pInput, T *pOutput,
    T const *gamma, T const *beta, const float epsilon, cudaStream_t stream)
{
    // Handle n Values per Thread 
    constexpr int VPT = 16 / sizeof(T);
    if (nHiddenDimension <= 32)
    {
        // Threads per Block
        constexpr int TPB = 32;
        LayerNormSmallKernel<T, float, TPB><<<gridSize, nHiddenDimension, 0, stream>>>(nHiddenDimension, pInput, gamma, beta, pOutput, epsilon);
    }
    else if (nHiddenDimension == 256)
    {
        constexpr int TPB = 256 / VPT;
        (LayerNormMediumKernel<T, float, TPB, VPT>)<<<gridSize, TPB, 0, stream>>>(nHiddenDimension, pInput, gamma, beta, pOutput, epsilon);
    }
    else if (nHiddenDimension == 1024)
    {
        constexpr int TPB = 1024 / VPT;
        (LayerNormMediumKernel<T, float, TPB, VPT>)<<<gridSize, TPB, 0, stream>>>(nHiddenDimension, pInput, gamma, beta, pOutput, epsilon);
    }
    else
    {
        constexpr int TPB = 256;
        (LayerNormGeneralKernel<T, float, TPB>)<<<gridSize, TPB, 0, stream>>>(nHiddenDimension, pInput, gamma, beta, pOutput, epsilon);
    }
    return 0;
}

template int computeLayerNorm<float>(const int, const int, float const *, float *, float const *, float const *, const float, cudaStream_t);
template int computeLayerNorm<half>(const int, const int, half const *, half *, half const *, half const *, const float, cudaStream_t);
