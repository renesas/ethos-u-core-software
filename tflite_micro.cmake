#
# Copyright (c) 2021 Arm Limited. All rights reserved.
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

if (CMAKE_CXX_COMPILER_ID STREQUAL "ARMClang")
    set(TFLU_TOOLCHAIN "armclang")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(TFLU_TOOLCHAIN "gcc")
else ()
    message(FATAL_ERROR "No compiler ID is set")
endif()

get_filename_component(TFLU_TARGET_TOOLCHAIN_ROOT ${CMAKE_C_COMPILER} DIRECTORY)

set(TFLU_TARGET_TOOLCHAIN_ROOT "'${TFLU_TARGET_TOOLCHAIN_ROOT}'/")
set(TFLU_PATH "${TENSORFLOW_PATH}/tensorflow/lite/micro")
set(TFLU_GENDIR "${CMAKE_CURRENT_BINARY_DIR}/tflite_micro/")
set(TFLU_TARGET "cortex_m_generic")
set(TFLU_TARGET_ARCH ${CMAKE_SYSTEM_PROCESSOR}${CPU_FEATURES}
    CACHE STRING "Tensorflow Lite for Microcontrollers target architecture")
set(TFLU_BUILD_TYPE "release" CACHE STRING "Tensorflow Lite Mirco build type, can be release or debug")
set(TFLU_OPTIMIZATION_LEVEL CACHE STRING "Tensorflow Lite Micro kernel optimization level")


if (TFLU_PREBUILT_LIBRARY_PATH)
    set(TFLU_IMPORTED_LIB_PATH "${TFLU_PREBUILT_LIBRARY_PATH}")
    message(STATUS "Using a prebuilt TensorFlow Lite for Microcontrollers library: ${TFLU_IMPORTED_LIB_PATH}")
else()
    if(CORE_SOFTWARE_ACCELERATOR STREQUAL NPU)
        set(TFLU_ETHOSU_LIBS $<TARGET_FILE:ethosu_core_driver>)
        # Set preference for ethos-u over cmsis-nn
        set(TFLU_OPTIMIZED_KERNEL_DIR "cmsis_nn")
        set(TFLU_CO_PROCESSOR "ethos_u")
    elseif(CORE_SOFTWARE_ACCELERATOR STREQUAL CMSIS-NN)
        set(TFLU_OPTIMIZED_KERNEL_DIR "cmsis_nn")
    endif()

    # Windows: change to relative paths.
    if (CMAKE_HOST_SYSTEM_NAME STREQUAL Windows)
        file(RELATIVE_PATH CMSIS_PATH ${TENSORFLOW_PATH} ${CMSIS_PATH})
        file(RELATIVE_PATH CORE_DRIVER_PATH ${TENSORFLOW_PATH} ${CORE_DRIVER_PATH})
    endif()

    # Command and target
    add_custom_target(tflu_gen ALL
                  COMMAND make -f ${TFLU_PATH}/tools/make/Makefile third_party_downloads
                  COMMAND make -j${J} -f ${TFLU_PATH}/tools/make/Makefile microlite
                          TARGET_TOOLCHAIN_ROOT=${TFLU_TARGET_TOOLCHAIN_ROOT}
                          TOOLCHAIN=${TFLU_TOOLCHAIN}
                          GENDIR=${TFLU_GENDIR}
                          TARGET=${TFLU_TARGET}
                          TARGET_ARCH=${TFLU_TARGET_ARCH}
                          OPTIMIZED_KERNEL_DIR="${TFLU_OPTIMIZED_KERNEL_DIR}"
                          CO_PROCESSOR="${TFLU_CO_PROCESSOR}"
                          $<$<BOOL:${FLOAT}>:FLOAT=${FLOAT}>
                          BUILD_TYPE=${TFLU_BUILD_TYPE}
                          $<$<BOOL:${TFLU_OPTIMIZATION_LEVEL}>:KERNEL_OPTIMIZATION_LEVEL=${TFLU_OPTIMIZATION_LEVEL}>
                          $<$<BOOL:${TFLU_OPTIMIZATION_LEVEL}>:CORE_OPTIMIZATION_LEVEL=${TFLU_OPTIMIZATION_LEVEL}>
                          CMSIS_PATH=${CMSIS_PATH}
                          ETHOSU_DRIVER_PATH=${CORE_DRIVER_PATH}
                          ETHOSU_DRIVER_LIBS=${TFLU_ETHOSU_LIBS}
                  BYPRODUCTS ${CMAKE_CURRENT_SOURCE_DIR}/tensorflow/tensorflow/lite/micro/tools/make/downloads
                  WORKING_DIRECTORY ${TENSORFLOW_PATH})

    set(TFLU_IMPORTED_LIB_PATH "${TFLU_GENDIR}/lib/libtensorflow-microlite.a")
endif()

# Create library and link library to custom target
add_library(tflu STATIC IMPORTED)
set_property(TARGET tflu PROPERTY IMPORTED_LOCATION "${TFLU_IMPORTED_LIB_PATH}")
add_dependencies(tflu tflu_gen)
target_include_directories(tflu INTERFACE
    ${TENSORFLOW_PATH})
target_compile_options(tflu INTERFACE
    -I${TENSORFLOW_PATH}/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include)
target_compile_definitions(tflu INTERFACE TF_LITE_MICRO TF_LITE_STATIC_MEMORY)

if(${TFLU_BUILD_TYPE} STREQUAL "release")
    target_compile_definitions(tflu INTERFACE TF_LITE_STRIP_ERROR_STRINGS)
endif()

if(CORE_SOFTWARE_ACCELERATOR STREQUAL NPU)
    target_link_libraries(tflu INTERFACE ethosu_core_driver)
endif()

# Install libraries and header files
get_target_property(TFLU_IMPORTED_LOCATION tflu IMPORTED_LOCATION)
install(FILES ${TFLU_IMPORTED_LOCATION} DESTINATION "lib")
