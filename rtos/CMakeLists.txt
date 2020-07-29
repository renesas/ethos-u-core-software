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

add_library(ethosu_core_rtos INTERFACE)

# Include FreeRTOS kernel & link with ethosu_rtos target
if(CORE_SOFTWARE_RTOS_LOWER STREQUAL "freertos")
    set(FREERTOS_KERNEL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/freertos/freertos_kernel" CACHE PATH "Path to FreeRTOS kernel.")
    add_subdirectory(freertos_config)
    target_link_libraries(ethosu_core_rtos INTERFACE freertos_kernel)
endif()