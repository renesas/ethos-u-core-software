/*
 * Copyright (c) 2019-2020 Arm Limited. All rights reserved.
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

#pragma once

#include <queue>
#include <stdlib.h>
#include <string>
#include <vector>

namespace InferenceProcess {
struct DataPtr {
    void *data;
    size_t size;

    DataPtr(void *data = nullptr, size_t size = 0);

    void invalidate();
    void clean();
};

struct InferenceJob {
    std::string name;
    DataPtr networkModel;
    std::vector<DataPtr> input;
    std::vector<DataPtr> output;
    std::vector<DataPtr> expectedOutput;
    size_t numBytesToPrint;

    InferenceJob();
    InferenceJob(const std::string &name,
                 const DataPtr &networkModel,
                 const std::vector<DataPtr> &input,
                 const std::vector<DataPtr> &output,
                 const std::vector<DataPtr> &expectedOutput,
                 size_t numBytesToPrint);

    void invalidate();
    void clean();
};

class InferenceProcess {
public:
    InferenceProcess();

    bool push(const InferenceJob &job);
    bool runJob(InferenceJob &job);
    bool run(bool exitOnEmpty = true);

private:
    volatile uint32_t lock;
    std::queue<InferenceJob> inferenceJobQueue;

    void getLock();
    void freeLock();
};
} // namespace InferenceProcess
