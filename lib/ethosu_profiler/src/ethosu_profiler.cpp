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

#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/micro_time.h"

#include <string.h>

#include "ethosu_profiler.hpp"
#include <ethosu_driver.h>
#include <inttypes.h>
#include <stdio.h>

namespace {

uint64_t GetCurrentEthosuTicks(struct ethosu_driver *drv) {
    return ETHOSU_PMU_Get_CCNTR_v2(drv);
}

void InitEthosuPMUCounters(struct ethosu_driver *drv, ethosu_pmu_event_type *ethosu_pmu_cntrs) {
    ETHOSU_PMU_Enable_v2(drv);

    ETHOSU_PMU_CNTR_Enable_v2(drv,
                              ETHOSU_PMU_CNT1_Msk | ETHOSU_PMU_CNT2_Msk | ETHOSU_PMU_CNT3_Msk | ETHOSU_PMU_CNT4_Msk |
                                  ETHOSU_PMU_CCNT_Msk);

    for (int i = 0; i < ETHOSU_PMU_NCOUNTERS; i++) {
        ETHOSU_PMU_Set_EVTYPER_v2(drv, i, ethosu_pmu_cntrs[i]);
    }

    ETHOSU_PMU_EVCNTR_ALL_Reset_v2(drv);
}

uint32_t GetEthosuPMUEventCounter(struct ethosu_driver *drv, int counter) {
    return ETHOSU_PMU_Get_EVCNTR_v2(drv, counter);
}
} // namespace

namespace tflite {

EthosUProfiler::EthosUProfiler(ethosu_pmu_event_type event0,
                               ethosu_pmu_event_type event1,
                               ethosu_pmu_event_type event2,
                               ethosu_pmu_event_type event3,
                               size_t max_events) :
    max_events_(max_events) {
    tags_        = std::make_unique<const char *[]>(max_events_);
    start_ticks_ = std::make_unique<uint64_t[]>(max_events_);
    end_ticks_   = std::make_unique<uint64_t[]>(max_events_);

    for (size_t i = 0; i < ETHOSU_PMU_NCOUNTERS; i++) {
        event_counters[i] = 0;
    }

    MonitorEthosuPMUEvents(event0, event1, event2, event3);
}

// NOTE: THIS PROFILER ONLY WORKS ON SYSTEMS WITH 1 NPU
uint32_t EthosUProfiler::BeginEvent(const char *tag) {
    if (num_events_ == max_events_) {
        tflite::GetMicroErrorReporter()->Report("Profiling event overflow, max: %u events", max_events_);
        num_events_ = 0;
    }

    tags_[num_events_] = tag;

    if (strcmp("ethos-u", tag) == 0) {
        struct ethosu_driver *ethosu_drv = ethosu_reserve_driver();
        ETHOSU_PMU_CYCCNT_Reset_v2(ethosu_drv);
        ETHOSU_PMU_PMCCNTR_CFG_Set_Start_Event_v2(ethosu_drv, ETHOSU_PMU_NPU_ACTIVE);
        ETHOSU_PMU_PMCCNTR_CFG_Set_Stop_Event_v2(ethosu_drv, ETHOSU_PMU_NPU_IDLE);
        start_ticks_[num_events_] = GetCurrentEthosuTicks(ethosu_drv);
        InitEthosuPMUCounters(ethosu_drv, ethosu_pmu_cntrs);
        ethosu_release_driver(ethosu_drv);
    } else {
        start_ticks_[num_events_] = GetCurrentTimeTicks();
    }

    end_ticks_[num_events_] = start_ticks_[num_events_] - 1;
    return num_events_++;
}

// NOTE: THIS PROFILER ONLY WORKS ON SYSTEMS WITH 1 NPU
void EthosUProfiler::EndEvent(uint32_t event_handle) {
    TFLITE_DCHECK(event_handle < max_events_);

    if (strcmp("ethos-u", tags_[event_handle]) == 0) {
        struct ethosu_driver *ethosu_drv = ethosu_reserve_driver();
        end_ticks_[event_handle]         = GetCurrentEthosuTicks(ethosu_drv);
        uint32_t ethosu_pmu_counter_end[ETHOSU_PMU_NCOUNTERS];
        ETHOSU_PMU_Disable_v2(ethosu_drv);
        for (size_t i = 0; i < ETHOSU_PMU_NCOUNTERS; i++) {
            ethosu_pmu_counter_end[i] = GetEthosuPMUEventCounter(ethosu_drv, i);
            tflite::GetMicroErrorReporter()->Report(
                "%s : ethosu_pmu_cntr%d : %u", tags_[event_handle], i, ethosu_pmu_counter_end[i]);

            event_counters[i] += ethosu_pmu_counter_end[i];
        }
        ethosu_release_driver(ethosu_drv);
        printf("%s : cycle_cnt : %" PRIu64 " cycles\n",
               tags_[event_handle],
               end_ticks_[event_handle] - start_ticks_[event_handle]);

    } else {
        end_ticks_[event_handle] = GetCurrentTimeTicks();
        printf("%s : cycle_cnt : %" PRIu64 " cycles\n",
               tags_[event_handle],
               end_ticks_[event_handle] - start_ticks_[event_handle]);
    }
}

uint64_t EthosUProfiler::GetTotalTicks() const {
    uint64_t ticks = 0;
    for (int i = 0; i < num_events_; ++i) {
        ticks += end_ticks_[i] - start_ticks_[i];
    }

    return ticks;
}

void EthosUProfiler::Log() const {
#if !defined(TF_LITE_STRIP_ERROR_STRINGS)
    for (int i = 0; i < num_events_; ++i) {
        uint64_t ticks = end_ticks_[i] - start_ticks_[i];
        printf("%s took %" PRIu64 " cycles\n", tags_[i], ticks);
    }
#endif
}

void EthosUProfiler::MonitorEthosuPMUEvents(ethosu_pmu_event_type event0,
                                            ethosu_pmu_event_type event1,
                                            ethosu_pmu_event_type event2,
                                            ethosu_pmu_event_type event3) {
    ethosu_pmu_cntrs[0] = event0;
    ethosu_pmu_cntrs[1] = event1;
    ethosu_pmu_cntrs[2] = event2;
    ethosu_pmu_cntrs[3] = event3;
}

uint32_t EthosUProfiler::GetEthosuPMUCounter(int counter) {
    return event_counters[counter];
}

} // namespace tflite
