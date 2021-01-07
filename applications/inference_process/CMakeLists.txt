#
# Copyright (c) 2019-2020 Arm Limited. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

set(TR_ARENA_SIZE "200000" CACHE STRING "Arena size.")
set(TR_PRINT_OUTPUT_BYTES "" CACHE STRING "Print output data.")

add_library(inference_process STATIC)
target_include_directories(inference_process PUBLIC include
                           PRIVATE ${TENSORFLOW_PATH} ${TENSORFLOW_PATH}/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include)
target_link_libraries(inference_process PUBLIC tflu cmsis_core cmsis_device)
target_sources(inference_process PRIVATE src/inference_process.cpp)

# Set arena size
target_compile_definitions(inference_process PRIVATE TENSOR_ARENA_SIZE=${TR_ARENA_SIZE})
