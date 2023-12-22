#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import ctypes
import numpy as np
from cuda import cudart  # 使用 cuda runtime API
import tensorrt as trt
# import nvtx

soFilePath = '/workspace/Github/Tensorrt-CV/build/out/liblayernorm_plugin.so'
epsilon = 6e-6


np.random.seed(97)


def check(a, b, weak=False, checkEpsilon=1e-5):
    if weak:
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        res = np.all(np.abs(a - b) < checkEpsilon)
    else:
        res = np.all(a == b)
    diff0 = np.max(np.abs(a - b))
    diff1 = np.max(np.abs(a - b) / (np.abs(b) + checkEpsilon))
    print("check:%s, absDiff=%f, relDiff=%f" % (res, diff0, diff1))


def layerNormCPU(bufferH):
    _x, gamma, beta = bufferH
    nHiddenSize = bufferH[0].shape[2]
    _0 = np.mean(_x, 2)[:, :, np.newaxis]
    _1 = _x - _0
    _2 = _1 * _1
    _3 = np.mean(_2, 2)[:, :, np.newaxis]
    _4 = np.array(epsilon, dtype=np.float32)
    _5 = _4.reshape(1, 1, 1)
    _6 = _3 + _5
    _7 = np.sqrt(_6)
    _8 = 1 / _7  # 1/sqrt(...)
    _9 = gamma
    _10 = _9.reshape(1, 1, nHiddenSize)
    _11 = _8 * _10  # gamma/sqrt(...)
    _12 = _0 * _11  # bμ/sqrt(...)
    _13 = beta
    _14 = _13.reshape(1, 1, nHiddenSize)
    _15 = _14 - _12  # beta-bμ/sqrt(...)
    _16 = _x * _11  # bx/sqrt(...)
    _17 = _15 + _16  # gamma(x-μ)/sqrt(...)+beta
    _18 = _17.reshape(bufferH[0].shape[0], bufferH[0].shape[1], bufferH[0].shape[2])
    return _18


def getLayerNormPlugin():
    for c in trt.get_plugin_registry().plugin_creator_list:
        # print(c.name)
        # if c.name == 'MyLayerNorm':
        if c.name == 'LayerNorm':
            return c.create_plugin(c.name, trt.PluginFieldCollection([]))
    return None


def run(input_shape, isFP16=False):
    nBS, nSL, nEmbedding = input_shape
    # logger = trt.Logger(trt.Logger.VERBOSE)
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(soFilePath)

    builder = trt.Builder(logger)
    network = builder.create_network(1 << 0)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 6 << 30)

    inputTensorList = []
    if isFP16:
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.STRICT_TYPES)
        inputTensorList.append(
            network.add_input('inputT', trt.float16, [-1, -1, nEmbedding]))
        inputTensorList.append(
            network.add_input('gemmaT', trt.float16, [1, 1, nEmbedding]))
        inputTensorList.append(
            network.add_input('betaT', trt.float16, [1, 1, nEmbedding]))
    else:
        inputTensorList.append(
            network.add_input('inputT', trt.float32, [-1, -1, nEmbedding]))
        inputTensorList.append(
            network.add_input('gemmaT', trt.float32, [1, 1, nEmbedding]))
        inputTensorList.append(
            network.add_input('betaT', trt.float32, [1, 1, nEmbedding]))

    profile = builder.create_optimization_profile()
    profile.set_shape('inputT', [1, 1, nEmbedding], [nBS, nSL, nEmbedding], [nBS, nSL, nEmbedding])
    config.add_optimization_profile(profile)

    pluginLayer = network.add_plugin_v2(inputTensorList, getLayerNormPlugin())
    # force trt to use fp16 mode
    # pluginLayer.precision = trt.DataType.HALF

    # output_layer = pluginLayer.get_output(0)
    # print("input type: ", pluginLayer.get_input(0).dtype)
    # print("output type: ", output_layer.dtype)
    # exit()

    network.mark_output(pluginLayer.get_output(0))

    engineString = builder.build_serialized_network(network, config)
    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

    context = engine.create_execution_context()
    context.set_binding_shape(0, [nBS, nSL, nEmbedding])
    print("Binding all? %s" %
          (["No", "Yes"][int(context.all_binding_shapes_specified)]))

    nInput = [
        engine.get_tensor_mode(engine.get_tensor_name(i))
        for i in range(engine.num_bindings)
    ].count(trt.TensorIOMode.INPUT)
    nOutput = engine.num_bindings - nInput
    # for i in range(engine.num_bindings):
    #     print(
    #         engine.get_tensor_mode(engine.get_tensor_name(i)).name,
    #         engine.get_binding_dtype(i), engine.get_binding_shape(i),
    #         context.get_binding_shape(i))

    bufferH = []
    shape = nBS, nSL, nEmbedding
    if isFP16:
        bufferH.append(
            np.random.rand(np.prod(shape)).reshape(shape).astype(np.float16) * 2 - 1)
        # gamma
        bufferH.append(np.random.rand(1, 1, nEmbedding).astype(np.float16))
        # beta
        bufferH.append(np.random.rand(1, 1, nEmbedding).astype(np.float16))
    else:
        bufferH.append(
            np.random.rand(np.prod(shape)).reshape(shape).astype(np.float32) * 2 - 1)
        # gamma
        bufferH.append(np.random.rand(1, 1, nEmbedding).astype(np.float32))
        # beta
        bufferH.append(np.random.rand(1, 1, nEmbedding).astype(np.float32))

    bufferH.append(
        np.empty(context.get_binding_shape(3),
                 dtype=trt.nptype(engine.get_binding_dtype(3))))
    # print("===> output type :", engine.get_binding_dtype(3))
    # exit(0)

    bufferD = []
    for i in range(engine.num_bindings):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    for i in range(nInput):
        cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data,
                          bufferH[i].nbytes,
                          cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    # with nvtx.annotate("execute_v2", color="purple"):
    context.execute_v2(bufferD)

    for i in range(nInput, nInput + nOutput):
        cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i],
                          bufferH[i].nbytes,
                          cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)


    temp1 = bufferH[-1]
    temp2 = layerNormCPU(bufferH[:3])
    # print(temp1.sum(), temp2.sum())
    # print(bufferH[0])
    # print(temp1)
    # print(temp2)

    # check(temp1, temp2, weak=False)
    check(temp1, temp2, weak=True)

    for b in bufferD:
        cudart.cudaFree(b)


if __name__ == '__main__':
    os.system("rm -f ./*.trt")
    np.set_printoptions(precision=4, linewidth=200, suppress=True)

    # run((4, 64, 32), isFP16=True)
    # run((4, 64, 256), isFP16=True)
    run((4, 64, 1024), isFP16=False)
    # run((4, 64, 768), isFP16=True)