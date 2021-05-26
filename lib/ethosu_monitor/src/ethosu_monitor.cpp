/*
 * Copyright (c) 2021 Arm Limited. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ethosu_monitor.hpp"
#include <stdio.h>

EthosUMonitor::EthosUMonitor(std::vector<int32_t> __eventRecordIds, Backend __backend) :
    eventRecordIds(__eventRecordIds), backend(__backend) {}

void EthosUMonitor::monitorSample(ethosu_driver *drv) {
    // Fetch events
    uint32_t eventCount[ETHOSU_PMU_NCOUNTERS] = {0};
    for (size_t i = 0; i < numEvents; i++) {
        eventCount[i] = ETHOSU_PMU_Get_EVCNTR(drv, i);
        switch (backend) {
        case EVENT_RECORDER:
            EventRecord2(eventRecordIds[i], ethosuEventIds[i], eventCount[i]);
            break;
        case PRINTF:
        default:
            printf("ethosu_pmu_cntr%d : %u\n", i, eventCount[i]);
        }
    }
}

void EthosUMonitor::release(ethosu_driver *drv) {
    ETHOSU_PMU_Disable(drv);
}
