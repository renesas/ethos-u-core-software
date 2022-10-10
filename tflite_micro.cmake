#
# SPDX-FileCopyrightText: Copyright 2021-2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

add_library(tflu STATIC)

set(TFLU_PATH "${TENSORFLOW_PATH}/tensorflow/lite/micro")
set(TFLU_BUILD_TYPE "release" CACHE STRING "Tensorflow Lite Mirco build type, can be release or debug")
set(TFLU_OPTIMIZATION_LEVEL "-O2" CACHE STRING "Tensorflow Lite Micro kernel optimization level")

#############################################################################
# Helpers
#############################################################################

include(FetchContent)

# Download third party
macro(download_third_party target)
    cmake_parse_arguments(DOWNLOAD "" "URL;URL_MD5;SOURCE_DIR" "" ${ARGN})

    message("Downloading ${DOWNLOAD_URL}")

    FetchContent_Declare(${target}
                         URL ${DOWNLOAD_URL}
                         URL_MD5 ${DOWNLOAD_MD5}
                         SOURCE_DIR ${DOWNLOAD_SOURCE_DIR}
                         ${PATCH_COMMAND})

    FetchContent_GetProperties(${target})
    if (NOT ${target}_POPULATED)
        FetchContent_Populate(${target})
    endif()
endmacro()

function(tensorflow_source_exists RESULT TARGET SOURCE)
    get_target_property(SOURCES ${TARGET} SOURCES)

    # Loop over source files already added to this target
    foreach(TMP ${SOURCES})
        get_filename_component(SOURCE_NAME ${SOURCE} NAME)
        get_filename_component(TMP_NAME ${TMP} NAME)

        # Check if file already exists
        if (${SOURCE_NAME} STREQUAL ${TMP_NAME})
            set(${RESULT} TRUE PARENT_SCOPE)
            return()
        endif()
    endforeach()

    set(${RESULT} FALSE PARENT_SCOPE)
endfunction()

function(tensorflow_target_sources_glob TARGET GLOB UNIQUE)
    foreach (EXPR ${ARGN})
        # Find files matching globbing expression
        file(${GLOB} SOURCES ${EXPR})

        # Remove tests
        list(FILTER SOURCES EXCLUDE REGEX ".*_test\.cc")

        # Add files to target
        foreach(SOURCE ${SOURCES})
            tensorflow_source_exists(SOURCE_EXISTS ${TARGET} ${SOURCE})
            if (NOT ${UNIQUE} OR NOT ${SOURCE_EXISTS})
                target_sources(${TARGET} PRIVATE ${SOURCE})
            endif()
        endforeach()
    endforeach()
endfunction()

#############################################################################
# Download thirdparty
#############################################################################

# Flatbuffers
# Synch revision with 'tensorflow/lite/micro/tools/make/flatbuffers_download.sh'
download_third_party(tensorflow-flatbuffers
    URL "https://github.com/google/flatbuffers/archive/a66de58af9565586832c276fbb4251fc416bf07f.zip"
    URL_MD5 51a7a96747e1c33eb4aac6d52513a02f)

target_include_directories(tflu PUBLIC
    ${tensorflow-flatbuffers_SOURCE_DIR}/include)

target_compile_definitions(tflu PUBLIC
    FLATBUFFERS_LOCALE_INDEPENDENT=0)

# Gemlowp
# Synch revision with 'tensorflow/lite/micro/tools/make/third_party_downloads.inc'
download_third_party(tensorflow-gemlowp
    URL "https://github.com/google/gemmlowp/archive/719139ce755a0f31cbf1c37f7f98adcc7fc9f425.zip"
    URL_MD5 7e8191b24853d75de2af87622ad293ba)

target_include_directories(tflu PUBLIC
    ${tensorflow-gemlowp_SOURCE_DIR})

# Ruy
# Synch revision with 'tensorflow/lite/micro/tools/make/third_party_downloads.inc'
download_third_party(tensorflow-ruy
    URL "https://github.com/google/ruy/archive/d37128311b445e758136b8602d1bbd2a755e115d.zip"
    URL_MD5 abf7a91eb90d195f016ebe0be885bb6e)

target_include_directories(tflu PUBLIC
    ${tensorflow-ruy_SOURCE_DIR})

#############################################################################
# CMSIS-NN
#############################################################################

if (NOT ${CORE_SOFTWARE_ACCELERATOR} STREQUAL "CPU")
    add_subdirectory(${CMSIS_NN_PATH} cmsis_nn)

    target_compile_options(cmsis-nn PRIVATE
        ${TFLU_OPTIMIZATION_LEVEL})

    tensorflow_target_sources_glob(tflu GLOB TRUE
        ${TFLU_PATH}/kernels/cmsis_nn/*.cc)

    target_include_directories(tflu PUBLIC
        ${CMSIS_NN_PATH})

    target_compile_definitions(tflu PUBLIC
        CMSIS_NN)

    target_link_libraries(tflu PUBLIC
        cmsis-nn)
endif()

#############################################################################
# Ethos-U
#############################################################################

if(TARGET ethosu_core_driver)
    tensorflow_target_sources_glob(tflu GLOB TRUE
        ${TFLU_PATH}/kernels/ethos_u/*.cc)

    target_link_libraries(tflu PUBLIC
        ethosu_core_driver)
endif()

#############################################################################
# Cortex-M generic
#############################################################################

tensorflow_target_sources_glob(tflu GLOB TRUE
    ${TFLU_PATH}/cortex_m_generic/*.cc)

target_include_directories(tflu PRIVATE
    ${TFLU_PATH}/cortex_m_generic)

# For DWT/PMU counters
target_link_libraries(tflu PRIVATE cmsis_device)
target_compile_definitions(tflu PRIVATE ${ARM_CPU})

if(("${ARM_CPU}" STREQUAL "ARMCM55") OR ("${ARM_CPU}" STREQUAL "ARMCM85"))
    target_compile_definitions(tflu PRIVATE
        ARM_MODEL_USE_PMU_COUNTERS)
endif()

#############################################################################
# Tensorflow micro lite
#############################################################################

tensorflow_target_sources_glob(tflu GLOB TRUE
    ${TFLU_PATH}/*.cc
    ${TFLU_PATH}/arena_allocator/*.cc
    ${TFLU_PATH}/memory_planner/*.cc
    ${TFLU_PATH}/kernels/*.cc)

tensorflow_target_sources_glob(tflu GLOB_RECURSE FALSE
    ${TFLU_PATH}/../c/*.cc
    ${TFLU_PATH}/../core/*.cc
    ${TFLU_PATH}/../kernels/*.cc
    ${TFLU_PATH}/../schema/*.cc)

target_include_directories(tflu PUBLIC
    ${TENSORFLOW_PATH})

target_compile_definitions(tflu PUBLIC
    TF_LITE_STATIC_MEMORY
    $<$<STREQUAL:${TFLU_BUILD_TYPE},"release">:"NDEBUG;TF_LITE_STRIP_ERROR_STRINGS">
    $<$<STREQUAL:${TFLU_BUILD_TYPE},"release_with_logs">:"NDEBUG">)

target_compile_options(tflu
    PRIVATE
        ${TFLU_OPTIMIZATION_LEVEL}
        -fno-unwind-tables
        -ffunction-sections
        -fdata-sections
        -fmessage-length=0
        -funsigned-char
        "$<$<COMPILE_LANGUAGE:CXX>:-fno-rtti;-fno-exceptions;-fno-threadsafe-statics>"

        -Wall
        -Wextra

        -Wdouble-promotion
        -Wmissing-field-initializers
        -Wshadow
        -Wstrict-aliasing
        -Wswitch
        -Wunused-variable
        -Wunused-function
        -Wvla

    PUBLIC
        -Wno-cast-align
        -Wno-null-dereference
        -Wno-unused-parameter
        -Wno-switch-default
)

# Install libraries and header files
install(TARGETS tflu DESTINATION "lib")
