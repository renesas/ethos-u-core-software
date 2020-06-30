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

cmake_minimum_required(VERSION 3.15.6)

project(core_software VERSION 0.0.1)

#
# Define build options
#

# Setup paths
set(CMSIS_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmsis" CACHE PATH "Path to CMSIS.")
set(CORE_DRIVER_PATH "${CMAKE_CURRENT_SOURCE_DIR}/core_driver" CACHE PATH "Path to core driver.")
set(TENSORFLOW_PATH "${CMAKE_CURRENT_SOURCE_DIR}/tensorflow" CACHE PATH "Path to Tensorflow.")

# Select accelerator for tensorflow
set(CORE_SOFTWARE_ACCELERATOR "NPU" CACHE STRING "Set NPU backend for TFLU")
set_property(CACHE CORE_SOFTWARE_ACCELERATOR PROPERTY STRINGS CPU CMSIS-NN NPU)

# Define build options
set(CORE_SOFTWARE_RTOS "None" CACHE STRING "Select RTOS to include. (None, MbedOS, FreeRTOS, Zephyr)")

#
# Build
#
add_library(ethosu_core INTERFACE)

# Build CMSIS
include(cmsis.cmake)

# Build core driver
if (CORE_SOFTWARE_ACCELERATOR STREQUAL "NPU")
    set(ETHOSU_PMU_INTERACTIVE OFF)
    add_subdirectory(${CORE_DRIVER_PATH} core_driver)
    target_link_libraries(ethosu_core INTERFACE ethosu_core_driver)
endif()

# Build Tensorflow library
include(tensorflow.cmake)

# Merge libraries into static library
target_link_libraries(ethosu_core INTERFACE tflu cmsis_device)
