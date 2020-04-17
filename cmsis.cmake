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

# Extract the CPU number from the system processor
string(REGEX MATCH "^cortex-m([0-9]+)$" CPU_NUMBER ${CMAKE_SYSTEM_PROCESSOR})
if(NOT CPU_NUMBER)
    message(FATAL_ERROR "System processor '${CMAKE_SYSTEM_PROCESSOR}' not supported. Should be cortex-m<nr>.")
endif()
string(REGEX REPLACE "^cortex-m([0-9]+)$" "\\1" CPU_NUMBER ${CMAKE_SYSTEM_PROCESSOR})

set(ARM_CPU "ARMCM${CPU_NUMBER}")

# CMSIS core library
add_library(cmsis_core INTERFACE)
target_include_directories(cmsis_core INTERFACE ${CMSIS_PATH}/CMSIS/Core/Include)

# CMSIS device library
add_library(cmsis_device OBJECT)
target_sources(cmsis_device PRIVATE
    ${CMSIS_PATH}/Device/ARM/${ARM_CPU}/Source/startup_${ARM_CPU}.c
    ${CMSIS_PATH}/Device/ARM/${ARM_CPU}/Source/system_${ARM_CPU}.c)
target_compile_definitions(cmsis_device PRIVATE ${ARM_CPU})
target_include_directories(cmsis_device PRIVATE ${CMSIS_PATH}/Device/ARM/${ARM_CPU}/Include)
target_link_libraries(cmsis_device PRIVATE cmsis_core)
