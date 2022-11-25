#
# Copyright (c) 2019-2022 Arm Limited. All rights reserved.
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

# Include FreeRTOS kernel
if(CORE_SOFTWARE_RTOS_LOWER STREQUAL "freertos" OR CORE_SOFTWARE_RTOS_LOWER STREQUAL "all")
    set(FREERTOS_KERNEL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/freertos/freertos_kernel" CACHE PATH "Path to FreeRTOS kernel.")
    add_subdirectory(freertos_config)
endif()

# Include ThreadX
# Only enable cortex-m55 until cortex-m33 is sorted out upstream
if((CMAKE_SYSTEM_PROCESSOR MATCHES "cortex-m55(\\+|$)") AND
    (CORE_SOFTWARE_RTOS_LOWER STREQUAL "threadx" OR CORE_SOFTWARE_RTOS_LOWER STREQUAL "all"))
    #threadx build requires 2 defines: THREADX_ARCH (format: cortex_m4) and THREADX_TOOLCHAIN (ac5/ac6/gnu).
    #set them according to existing defines CMAKE_SYSTEM_PROCESSOR and CMAKE_CXX_COMPILER_ID:
    string(REGEX MATCH "^cortex-m([0-9]+[a-z]*)" THREADX_ARCH ${CMAKE_SYSTEM_PROCESSOR})
    string(REPLACE "-" "_" THREADX_ARCH ${THREADX_ARCH})
    string(TOLOWER ${CMAKE_CXX_COMPILER_ID} THREADX_TOOLCHAIN)

    #if armclang, change the format to "ac6".
    #if(THREADX_TOOLCHAIN STREQUAL "armclang")
    #    set(THREADX_TOOLCHAIN "ac6")
    #endif()

    #if ThreadX supports this configuration, build ThreadX.
    if (EXISTS ${CMAKE_CURRENT_LIST_DIR}/threadx/ports/${THREADX_ARCH}/${THREADX_TOOLCHAIN}/CMakeLists.txt)
        add_subdirectory(threadx)

        if(THREADX_ARCH STREQUAL "cortex_m33" OR THREADX_ARCH STREQUAL "cortex_m55")
            target_compile_definitions(threadx PUBLIC TX_SINGLE_MODE_SECURE)
        endif()

        target_compile_options(threadx PRIVATE
            -Wno-unused-function
            -Wno-unused-parameter)
    else()
        message(STATUS "Skipping ThreadX build - doesn't support toolchain=${THREADX_TOOLCHAIN} and cpu=${THREADX_ARCH}")
    endif()
else()
    message(STATUS "Disabling ThreadX for ${CMAKE_SYSTEM_PROCESSOR}, due to upstream issues")
    # Disable cpu until upstream issue with missing files is sorted out
endif()
