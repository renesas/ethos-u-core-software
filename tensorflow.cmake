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

include(ProcessorCount)
ProcessorCount(J)

set(TFLU_CC "${CMAKE_C_COMPILER} --target=${CMAKE_C_COMPILER_TARGET} -mcpu=${CMAKE_SYSTEM_PROCESSOR}${CPU_FEATURES}")
set(TFLU_CXX "${CMAKE_CXX_COMPILER} --target=${CMAKE_C_COMPILER_TARGET} -mcpu=${CMAKE_SYSTEM_PROCESSOR}${CPU_FEATURES}")
set(TFLU_AR ${CMAKE_AR})

set(TFLU_PATH "${TENSORFLOW_PATH}/tensorflow/lite/micro")
set(TFLU_GENDIR ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/)
set(TFLU_TARGET "lib")
set(TFLU_TARGET_ARCH ${CMAKE_SYSTEM_PROCESSOR}${CPU_FEATURES})
set(TFLU_ETHOSU_LIBS $<TARGET_FILE:ethosu_core_driver>)

if(CORE_SOFTWARE_BACKEND STREQUAL NPU)
    list(APPEND TFLU_TAGS "ARM_NPU")
endif()

string(JOIN TFLU_TAGS " " TFLU_TAGS)

# Command and target
add_custom_target(tflu_gen ALL
                  COMMAND make -j${J} -f ${TFLU_PATH}/tools/make/Makefile microlite TARGET=${TFLU_TARGET} TARGET_ARCH=${TFLU_TARGET_ARCH} CC_TOOL=${TFLU_CC} CXX_TOOL=${TFLU_CXX} AR_TOOL=${TFLU_AR} GENDIR=${TFLU_GENDIR} CMSIS_PATH=${CMSIS_PATH} ARM_NPU_PATH=${CORE_DRIVER_PATH} ETHOSU_LIBS=${TFLU_ETHOSU_LIBS} TAGS="${TFLU_TAGS}"
                  WORKING_DIRECTORY ${TENSORFLOW_PATH})

# Create library and link library to custom target
add_library(tflu STATIC IMPORTED)
set_property(TARGET tflu PROPERTY IMPORTED_LOCATION ${TFLU_GENDIR}/lib/libtensorflow-microlite.a)
add_dependencies(tflu tflu_gen)
target_include_directories(tflu INTERFACE ${TENSORFLOW_PATH})
