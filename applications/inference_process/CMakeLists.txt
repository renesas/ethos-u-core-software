#
# SPDX-FileCopyrightText: Copyright 2019-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

set(TR_PRINT_OUTPUT_BYTES "" CACHE STRING "Print output data.")

add_library(inference_process INTERFACE)

target_include_directories(inference_process INTERFACE include)

target_link_libraries(inference_process INTERFACE tflu cmsis_core cmsis_device ethosu_crc)

if (TARGET arm_profiler)
    target_link_libraries(inference_process INTERFACE arm_profiler)
endif()

if (TARGET ethosu_log)
    target_link_libraries(inference_process INTERFACE ethosu_log)
endif()

target_sources(inference_process INTERFACE src/inference_process.cpp)

if (DEFINED INFERENCE_PROCESS_OPS_RESOLVER)
    target_compile_definitions(inference_process INTERFACE INFERENCE_PROCESS_OPS_RESOLVER=${INFERENCE_PROCESS_OPS_RESOLVER})
endif()
