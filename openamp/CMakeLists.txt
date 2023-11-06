#
# SPDX-FileCopyrightText: Copyright 2022-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

function(build_openamp PROJECT_SYSTEM)
    set(PROJECT_PROCESSOR "arm")
    set(PROJECT_MACHINE "cortexm")

    file(GLOB SRCS
        # libmetal
        libmetal/lib/*.c
        libmetal/lib/system/${PROJECT_SYSTEM}/*.c
        libmetal/lib/compiler/gcc/*.c

        # Extra sources
        src/system/${PROJECT_SYSTEM}/${PROJECT_MACHINE}/*.c

        # OpenAMP
        openamp/lib/remoteproc/*.c
        openamp/lib/rpmsg/*.c
        openamp/lib/virtio/*.c)

    add_library(openamp-${PROJECT_SYSTEM} STATIC
        ${SRCS})

    target_include_directories(openamp-${PROJECT_SYSTEM}
        PUBLIC
            ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_SYSTEM}
            openamp/lib/include
            src/system/${PROJECT_SYSTEM})

    target_compile_definitions(openamp-${PROJECT_SYSTEM} PRIVATE
        OPENAMP_VERSION_MAJOR=0
        OPENAMP_VERSION_MINOR=0
        OPENAMP_VERSION_PATCH=0
        OPENAMP_VERSION=0
        METAL_INTERNAL)

    target_link_libraries(openamp-${PROJECT_SYSTEM} PRIVATE
        cmsis_device
        $<$<STREQUAL:${PROJECT_SYSTEM},freertos>:freertos_kernel>)

    target_compile_options(openamp-${PROJECT_SYSTEM} PRIVATE
        -Wno-cast-align
        -Wno-unknown-pragmas
        -Wno-unused-but-set-variable
    )

    # Generate libmetal headers
    file(GLOB_RECURSE HDRS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/libmetal/lib" "libmetal/lib/*.h")

    foreach(HDR ${HDRS})
        configure_file("libmetal/lib/${HDR}" "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_SYSTEM}/metal/${HDR}")
    endforeach()

endfunction()

build_openamp(generic)

if (TARGET freertos_kernel)
    build_openamp(freertos)
endif()
