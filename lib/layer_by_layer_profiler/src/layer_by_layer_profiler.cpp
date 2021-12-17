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

#include "ethosu_log.h"
#include "layer_by_layer_profiler.hpp"
#include <ethosu_driver.h>
#include <inttypes.h>
#include <stdio.h>

namespace {

uint64_t GetCurrentEthosuTicks(struct ethosu_driver *drv) {
    return ETHOSU_PMU_Get_CCNTR(drv);
}

} // namespace

namespace tflite {

LayerByLayerProfiler::LayerByLayerProfiler(const std::vector<uint8_t> &event_config,
                                           bool pmu_cycle_counter_enable,
                                           size_t max_events,
                                           Backend backend,
                                           int32_t event_id) :
    pmu_event_config(event_config),
    pmu_event_count(), pmu_cycle_counter_enable(pmu_cycle_counter_enable), pmu_cycle_counter_count(0),
    max_events_(max_events), backend(backend), event_id(event_id), num_events_(0) {

    tags_        = std::make_unique<const char *[]>(max_events);
    start_ticks_ = std::make_unique<uint64_t[]>(max_events);
    end_ticks_   = std::make_unique<uint64_t[]>(max_events);
}

// NOTE: THIS PROFILER ONLY WORKS ON SYSTEMS WITH 1 NPU
uint32_t LayerByLayerProfiler::BeginEvent(const char *tag) {
    if (num_events_ == max_events_) {
        tflite::GetMicroErrorReporter()->Report("Profiling event overflow, max: %u events", max_events_);
        num_events_ = 0;
    }

    tags_[num_events_] = tag;

    if (strcmp("ethos-u", tag) == 0) {
        struct ethosu_driver *drv = ethosu_reserve_driver();
        size_t numEventCounters   = ETHOSU_PMU_Get_NumEventCounters();

        if (pmu_event_config.size() > numEventCounters) {
            LOG_WARN("PMU event config list is bigger (%lu) than available PMU event counters (%lu)",
                     pmu_event_config.size(),
                     numEventCounters);
            LOG_WARN("PMU event config list will be truncated");
            pmu_event_config.resize(numEventCounters);
        }
        // Enable PMU
        ETHOSU_PMU_Enable(drv);

        for (size_t i = 0; i < pmu_event_config.size(); i++) {
            ETHOSU_PMU_Set_EVTYPER(drv, i, static_cast<ethosu_pmu_event_type>(pmu_event_config[i]));
        }

        ETHOSU_PMU_CNTR_Enable(drv, (1 << pmu_event_config.size()) - 1);
        ETHOSU_PMU_EVCNTR_ALL_Reset(drv);

        // Configure the cycle counter
        if (pmu_cycle_counter_enable) {
            ETHOSU_PMU_CNTR_Disable(drv, ETHOSU_PMU_CCNT_Msk);
            ETHOSU_PMU_CYCCNT_Reset(drv);

            ETHOSU_PMU_PMCCNTR_CFG_Set_Stop_Event(drv, ETHOSU_PMU_NPU_IDLE);
            ETHOSU_PMU_PMCCNTR_CFG_Set_Start_Event(drv, ETHOSU_PMU_NPU_ACTIVE);

            ETHOSU_PMU_CNTR_Enable(drv, ETHOSU_PMU_CCNT_Msk);
        }
        start_ticks_[num_events_] = 0; // Hardware cycle counter has been reset above, thus starts at 0
        ethosu_release_driver(drv);
    } else {
        start_ticks_[num_events_] = GetCurrentTimeTicks();
    }

    end_ticks_[num_events_] =
        start_ticks_[num_events_]; // NOTE: In case an EndEvent() doesn't trigger, cycles reports as 0
    return num_events_++;
}

// NOTE: THIS PROFILER ONLY WORKS ON SYSTEMS WITH 1 NPU
void LayerByLayerProfiler::EndEvent(uint32_t event_handle) {
    TFLITE_DCHECK(event_handle < max_events_);

    if (strcmp("ethos-u", tags_[event_handle]) == 0) {
        struct ethosu_driver *drv = ethosu_reserve_driver();

        end_ticks_[event_handle] = GetCurrentEthosuTicks(drv);
        // Get the cycle count
        if (pmu_cycle_counter_enable) {
            pmu_cycle_counter_count = end_ticks_[event_handle];
        }

        // Save the PMU counter values
        // NOTE: If multiple ethos-u layers, only the latest will be saved
        pmu_event_count.resize(pmu_event_config.size());
        for (size_t i = 0; i < pmu_event_config.size(); i++) {
            pmu_event_count[i] = ETHOSU_PMU_Get_EVCNTR(drv, i);
        }

        // Shut down the PMU
        ETHOSU_PMU_Disable(drv);

        ethosu_release_driver(drv);
    } else {
        end_ticks_[event_handle] = GetCurrentTimeTicks();
    }

    if (backend == PRINTF) {
        if (strcmp("ethos-u", tags_[event_handle]) == 0) {
            for (size_t i = 0; i < pmu_event_count.size(); i++) {
                LOG("ethos-u : ethosu_pmu_cntr%lu : %u\n", i, pmu_event_count[i]);
            }
            LOG("ethos-u : cycle_cnt : %" PRIu64 " cycles\n", pmu_cycle_counter_count);
        } else {
            LOG("%s : cycle_cnt : %" PRIu64 " cycles\n",
                tags_[event_handle],
                end_ticks_[event_handle] - start_ticks_[event_handle]);
        }
    } else {
        EventRecord2(event_id, (int32_t)event_handle, end_ticks_[event_handle] - start_ticks_[event_handle]);
    }
}

uint64_t LayerByLayerProfiler::GetTotalTicks() const {
    uint64_t ticks = 0;

    for (size_t i = 0; i < num_events_; ++i) {
        ticks += end_ticks_[i] - start_ticks_[i];
    }

    return ticks;
}

uint64_t LayerByLayerProfiler::GetPmuCycleCounterCount() const {
    return pmu_cycle_counter_count;
}

const std::vector<uint32_t> &LayerByLayerProfiler::GetPmuEventCount() const {
    return pmu_event_count;
}

void LayerByLayerProfiler::Log() const {

#if !defined(TF_LITE_STRIP_ERROR_STRINGS)
    if (backend == PRINTF) {
        for (size_t i = 0; i < num_events_; ++i) {
            uint64_t ticks = end_ticks_[i] - start_ticks_[i];
            LOG("%s took %" PRIu64 " cycles", tags_[i], ticks);
        }
    }
#endif
}

} // namespace tflite
