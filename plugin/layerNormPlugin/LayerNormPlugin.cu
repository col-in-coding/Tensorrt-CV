#include <iostream>
#include <chrono>
#include <thread>

#include "LayerNormPlugin.h"

using namespace nvinfer1;

PluginFieldCollection LayerNormPluginCreator::fc_{};
std::vector<PluginField> LayerNormPluginCreator::attr_;

int32_t LayerNormPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    WHERE_AM_I();
    const int gridSize = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1];
    const int nHiddenDimension = inputDesc[0].dims.d[2];
    const float epsilon = 1e-6;

    int status = -1;
    switch (inputDesc[0].type)
    {
    case DataType::kFLOAT:
    {
        // 当 trt config设置为FP16的时候，即使输入是FP32，内部会尝试转换成FP16推理，如果确定速度更快就使用FP16
        // std::this_thread::sleep_for(std::chrono::milliseconds(40));
        // printf("===> using FP32 kernel\n");
        auto const input = static_cast<float const*>(inputs[0]);
        auto const gamma = static_cast<float const*>(inputs[1]);
        auto const beta = static_cast<float const*>(inputs[2]);
        auto output = static_cast<float *>(outputs[0]);
        status = computeLayerNorm<float>(gridSize, nHiddenDimension, input, output, gamma, beta, epsilon, stream);
        break;
    }
    case DataType::kHALF:
    {
        // printf("===> using FP16 kernel\n");
        auto const input = static_cast<half const*>(inputs[0]);
        auto const gamma = static_cast<half const*>(inputs[1]);
        auto const beta = static_cast<half const*>(inputs[2]);
        auto output = static_cast<half *>(outputs[0]);
        status = computeLayerNorm<half>(gridSize, nHiddenDimension, input, output, gamma, beta, epsilon, stream);
        break;
    }
    default:
    {
        printf("Datatype not implemented yet. %s, %d", __FILE__, __LINE__);
        break;
    }
    }
    return status;
}

REGISTER_TENSORRT_PLUGIN(LayerNormPluginCreator);

