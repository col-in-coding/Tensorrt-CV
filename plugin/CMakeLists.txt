#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
add_custom_target(plugin)

set(TARGET_NAME nvinfer_plugin)
set(SHARED_TARGET ${TARGET_NAME})
set(STATIC_TARGET ${TARGET_NAME}_static)

set(PLUGIN_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PLUGIN_EXPORT_MAP ${PLUGIN_DIR}/exports.map)

if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()
set(ENABLED_SMS "-DENABLE_SM72 -DENABLE_SM75 -DENABLE_SM80 -DENABLE_SM86 -DENABLE_SM87 -DENABLE_SM89 -DENABLE_SM90")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ENABLED_SMS}")
set(PLUGIN_SOURCES)
set(PLUGIN_CU_SOURCES)

set(PLUGIN_LISTS
    batchedNMSPlugin
    layerNormPlugin
)

include_directories(common common/kernels ${CMAKE_SOURCE_DIR}/third_party)

# Add common
add_subdirectory(common)

foreach(PLUGIN_ITER ${PLUGIN_LISTS})
    include_directories(${PLUGIN_ITER})
    add_subdirectory(${PLUGIN_ITER})
endforeach(PLUGIN_ITER)
