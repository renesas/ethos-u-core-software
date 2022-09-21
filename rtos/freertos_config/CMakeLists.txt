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

if(CMAKE_SYSTEM_PROCESSOR MATCHES "cortex-m3(\\+|$)")
    add_compile_definitions(CPU_CORTEX_M3)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "cortex-m33(\\+|$)")
    add_compile_definitions(CPU_CORTEX_M33)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "cortex-m4(\\+|$)")
    add_compile_definitions(CPU_CORTEX_M4)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "cortex-m55(\\+|$)")
    add_compile_definitions(CPU_CORTEX_M55)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "cortex-m7(\\+|$)")
    add_compile_definitions(CPU_CORTEX_M7)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "cortex-m85(\\+|$)")
    add_compile_definitions(CPU_CORTEX_M85)
else()
    message(FATAL_ERROR "Unsupported compiler ${CMAKE_SYSTEM_PROCESSOR}.")
endif()

if((${TARGET} STREQUAL "mps2") OR (${CMAKE_SYSTEM_PROCESSOR} MATCHES "cortex-m55(\\+|$)"))
    set("SYSTEM_CORE_CLOCK" "25000000")
else()
    set("SYSTEM_CORE_CLOCK" "32000000")
endif()


configure_file(FreeRTOSConfig.h.in FreeRTOSConfig.h)
set(FREERTOS_CONFIG_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR})

# FreeRTOS kernel library
add_library(freertos_kernel STATIC)

target_compile_options(freertos_kernel PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${TOOLCHAIN_CXX_OPTIONS}> "-Wno-redundant-decls")

# Set include directories
target_include_directories(freertos_kernel PUBLIC
    ${FREERTOS_KERNEL_PATH}/include
    ${FREERTOS_CONFIG_INCLUDE_DIR}
)

# Add kernel source files
target_sources(freertos_kernel PRIVATE
    ${FREERTOS_KERNEL_PATH}/tasks.c
    ${FREERTOS_KERNEL_PATH}/list.c
    ${FREERTOS_KERNEL_PATH}/queue.c
)

# Add the heap implementation
# heap_3 uses the compiler supplied malloc & free for the heap.
target_sources(freertos_kernel PRIVATE
    ${FREERTOS_KERNEL_PATH}/portable/MemMang/heap_3.c)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "cortex-m3(\\+|$)")
target_sources(freertos_kernel PRIVATE
    ${FREERTOS_KERNEL_PATH}/portable/GCC/ARM_CM3/port.c
)
target_include_directories(freertos_kernel PUBLIC
    ${FREERTOS_KERNEL_PATH}/portable/GCC/ARM_CM3
)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "cortex-m4(\\+|$)")
    target_sources(freertos_kernel PRIVATE
        ${FREERTOS_KERNEL_PATH}/portable/GCC/ARM_CM4F/port.c
    )
    target_include_directories(freertos_kernel PUBLIC
        ${FREERTOS_KERNEL_PATH}/portable/GCC/ARM_CM4F
    )
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "cortex-m7(\\+|$)")
    target_sources(freertos_kernel PRIVATE
        ${FREERTOS_KERNEL_PATH}/portable/GCC/ARM_CM7/r0p1/port.c
    )
    target_include_directories(freertos_kernel PUBLIC
        ${FREERTOS_KERNEL_PATH}/portable/GCC/ARM_CM7/r0p1
    )
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "cortex-m33(\\+|$)")
    target_sources(freertos_kernel PRIVATE
        ${FREERTOS_KERNEL_PATH}/portable/GCC/ARM_CM33_NTZ/non_secure/port.c
        ${FREERTOS_KERNEL_PATH}/portable/GCC/ARM_CM33_NTZ/non_secure/portasm.c
    )
    target_include_directories(freertos_kernel PUBLIC
        ${FREERTOS_KERNEL_PATH}/portable/GCC/ARM_CM33_NTZ/non_secure
    )
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "cortex-m55(\\+|$)")
# TODO: using the same port files as Cortex-M33. Update for Cortex-M55.
    target_sources(freertos_kernel PRIVATE
        ${FREERTOS_KERNEL_PATH}/portable/GCC/ARM_CM33_NTZ/non_secure/port.c
        ${FREERTOS_KERNEL_PATH}/portable/GCC/ARM_CM33_NTZ/non_secure/portasm.c
    )
    target_include_directories(freertos_kernel PUBLIC
        ${FREERTOS_KERNEL_PATH}/portable/GCC/ARM_CM33_NTZ/non_secure
    )
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "cortex-m85(\\+|$)")
# TODO: using the same port files as Cortex-M33. Update for Cortex-M85.
    target_sources(freertos_kernel PRIVATE
        ${FREERTOS_KERNEL_PATH}/portable/GCC/ARM_CM33_NTZ/non_secure/port.c
        ${FREERTOS_KERNEL_PATH}/portable/GCC/ARM_CM33_NTZ/non_secure/portasm.c
    )
    target_include_directories(freertos_kernel PUBLIC
        ${FREERTOS_KERNEL_PATH}/portable/GCC/ARM_CM33_NTZ/non_secure
    )
else()
    message(FATAL_ERROR "Invalid processor ${CMAKE_SYSTEM_PROCESSOR} when building FreeRTOS kernel")
endif()
