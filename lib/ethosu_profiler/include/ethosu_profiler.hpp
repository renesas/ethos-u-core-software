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

#ifndef ETHOSU_PROFILER_H
#define ETHOSU_PROFILER_H

#include "tensorflow/lite/kernels/internal/compatibility.h"
#include <memory>
#include <pmu_ethosu.h>

// NOTE: This profiler only works on systems with 1 NPU due to the use of
// ethosu_reserve_driver().
namespace tflite {
class EthosUProfiler : public MicroProfiler {
public:
    EthosUProfiler(ethosu_pmu_event_type event0 = ETHOSU_PMU_NO_EVENT,
                   ethosu_pmu_event_type event1 = ETHOSU_PMU_NO_EVENT,
                   ethosu_pmu_event_type event2 = ETHOSU_PMU_NO_EVENT,
                   ethosu_pmu_event_type event3 = ETHOSU_PMU_NO_EVENT,
                   size_t max_events            = 200);
    uint32_t BeginEvent(const char *tag);
    void EndEvent(uint32_t event_handle);
    uint64_t GetTotalTicks() const;
    void Log() const;
    uint32_t GetEthosuPMUCounter(int counter);

private:
    void MonitorEthosuPMUEvents(ethosu_pmu_event_type event0,
                                ethosu_pmu_event_type event1,
                                ethosu_pmu_event_type event2,
                                ethosu_pmu_event_type event3);

    size_t max_events_;
    std::unique_ptr<const char *[]> tags_;
    std::unique_ptr<uint64_t[]> start_ticks_;
    std::unique_ptr<uint64_t[]> end_ticks_;

    int num_events_ = 0;

    ethosu_pmu_event_type ethosu_pmu_cntrs[ETHOSU_PMU_NCOUNTERS];

    uint32_t event_counters[ETHOSU_PMU_NCOUNTERS];

    TF_LITE_REMOVE_VIRTUAL_DELETE;
};

} // namespace tflite

#endif
