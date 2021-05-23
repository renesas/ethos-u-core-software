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

add_library(rte_component INTERFACE)
target_include_directories(rte_component INTERFACE lib/rte_component/include)
target_link_libraries(rte_component INTERFACE cmsis_device)
target_compile_definitions(rte_component INTERFACE RTE_Components_CMSIS_device_header=\"${ARM_CPU}${ARM_FEATURES}.h\")

add_library(event_recorder INTERFACE)

target_include_directories(event_recorder INTERFACE
    ${CMSIS_VIEW_PATH}/EventRecorder/Include
    ${CMSIS_VIEW_PATH}/EventRecorder/Config)

target_link_libraries(event_recorder INTERFACE rte_component)
target_sources(event_recorder INTERFACE ${CMSIS_VIEW_PATH}/EventRecorder/Source/EventRecorder.c)


