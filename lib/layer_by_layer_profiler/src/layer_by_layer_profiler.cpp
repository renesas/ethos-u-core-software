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

LayerByLayerProfiler::LayerByLayerProfiler(size_t max_events, Backend backend, int32_t event_id) :
    max_events_(max_events), backend_(backend), event_id_(event_id), num_events_(0) {

    tags_        = std::make_unique<const char *[]>(max_events_);
    start_ticks_ = std::make_unique<uint64_t[]>(max_events_);
    end_ticks_   = std::make_unique<uint64_t[]>(max_events_);

    struct ethosu_driver *drv = ethosu_reserve_driver();
    ETHOSU_PMU_CNTR_Enable(drv, ETHOSU_PMU_CCNT_Msk);
    ETHOSU_PMU_CYCCNT_Reset(drv);
    ethosu_release_driver(drv);
}

// NOTE: THIS PROFILER ONLY WORKS ON SYSTEMS WITH 1 NPU
uint32_t LayerByLayerProfiler::BeginEvent(const char *tag) {
    if (num_events_ == max_events_) {
        tflite::GetMicroErrorReporter()->Report("Profiling event overflow, max: %u events", max_events_);
        num_events_ = 0;
    }

    tags_[num_events_] = tag;

    if (strcmp("ethos-u", tag) == 0) {
        struct ethosu_driver *ethosu_drv = ethosu_reserve_driver();
        ETHOSU_PMU_CYCCNT_Reset(ethosu_drv);
        ETHOSU_PMU_PMCCNTR_CFG_Set_Start_Event(ethosu_drv, ETHOSU_PMU_NPU_ACTIVE);
        ETHOSU_PMU_PMCCNTR_CFG_Set_Stop_Event(ethosu_drv, ETHOSU_PMU_NPU_IDLE);
        start_ticks_[num_events_] = GetCurrentEthosuTicks(ethosu_drv);
        ethosu_release_driver(ethosu_drv);
    } else {
        start_ticks_[num_events_] = GetCurrentTimeTicks();
    }

    end_ticks_[num_events_] = start_ticks_[num_events_] - 1;
    return num_events_++;
}

// NOTE: THIS PROFILER ONLY WORKS ON SYSTEMS WITH 1 NPU
void LayerByLayerProfiler::EndEvent(uint32_t event_handle) {
    TFLITE_DCHECK(event_handle < max_events_);

    if (strcmp("ethos-u", tags_[event_handle]) == 0) {
        struct ethosu_driver *ethosu_drv = ethosu_reserve_driver();
        end_ticks_[event_handle]         = GetCurrentEthosuTicks(ethosu_drv);
        ethosu_release_driver(ethosu_drv);
    } else {
        end_ticks_[event_handle] = GetCurrentTimeTicks();
    }

    if (backend_ == PRINTF) {
        printf("%s : cycle_cnt : %" PRIu64 " cycles\n",
               tags_[event_handle],
               end_ticks_[event_handle] - start_ticks_[event_handle]);
    } else {
        EventRecord2(event_id_, (int32_t)event_handle, end_ticks_[event_handle] - start_ticks_[event_handle]);
    }
}

uint64_t LayerByLayerProfiler::GetTotalTicks() const {
    uint64_t ticks = 0;

    for (size_t i = 0; i < num_events_; ++i) {
        ticks += end_ticks_[i] - start_ticks_[i];
    }

    return ticks;
}

void LayerByLayerProfiler::Log() const {

#if !defined(TF_LITE_STRIP_ERROR_STRINGS)
    if (backend_ == PRINTF) {
        for (size_t i = 0; i < num_events_; ++i) {
            uint64_t ticks = end_ticks_[i] - start_ticks_[i];
            printf("%s took %" PRIu64 " cycles\n", tags_[i], ticks);
        }
    }
#endif
}

} // namespace tflite
