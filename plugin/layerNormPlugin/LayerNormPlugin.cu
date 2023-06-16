/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
 
#include "LayerNormPlugin.h"
#include "LayerNormKernel.h"

#include <iostream>
#include <chrono>
#include <thread>

using namespace nvinfer1;

PluginFieldCollection LayerNormPluginCreator::fc_{};
std::vector<PluginField> LayerNormPluginCreator::attr_;

int32_t LayerNormPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    WHERE_AM_I();
    const int nBlock = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1];

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
        computeLayerNorm<float>(input, gamma, beta, output, stream, nBlock);
        break;
    }
    // case DataType::kHALF:
    // {
    //     printf("===> using FP16 kernel\n");
    //     auto const input = static_cast<half const*>(inputs[0]);
    //     // for 
    //     // float tmp = temp[tx];
    //     // printf(" %f ", tmp);
    //     auto const gamma = static_cast<half const*>(inputs[1]);
    //     auto const beta = static_cast<half const*>(inputs[2]);
    //     auto output = static_cast<half *>(outputs[0]);
    //     computeLayerNorm<half>(input, gamma, beta, output, stream, nBlock);
    //     break;
    // }
    default:
    {
        printf("===> Datatype not implemented yet. %s, %s", __FILE__, __LINE__);
        break;
    }
    }
    return 0;
}

REGISTER_TENSORRT_PLUGIN(LayerNormPluginCreator);

