/*
 * Copyright (c) 2020 Arm Limited. All rights reserved.
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

#ifndef MESSAGE_PROCESS_H
#define MESSAGE_PROCESS_H

#include <ethosu_core_interface.h>
#include <inference_process.hpp>

#include <cstddef>
#include <cstdio>
#include <vector>

namespace MessageProcess {

template <uint32_t SIZE>
struct Queue {
    ethosu_core_queue_header header;
    uint8_t data[SIZE];

    constexpr Queue() : header({SIZE, 0, 0}) {}

    constexpr ethosu_core_queue *toQueue() {
        return reinterpret_cast<ethosu_core_queue *>(&header);
    }
};

class QueueImpl {
public:
    struct Vec {
        const void *base;
        size_t length;
    };

    QueueImpl(ethosu_core_queue &queue);

    bool empty() const;
    size_t available() const;
    size_t capacity() const;
    bool read(uint8_t *dst, uint32_t length);
    bool write(const Vec *vec, size_t length);
    bool write(const uint32_t type, const void *src = nullptr, uint32_t length = 0);
    bool skip(uint32_t length);

    template <typename T>
    bool read(T &dst) {
        return read(reinterpret_cast<uint8_t *>(&dst), sizeof(dst));
    }

    template <typename T>
    bool readOrSkip(T &dst, uint32_t expectedLength) {
        if (expectedLength == sizeof(dst)) {
            return read(reinterpret_cast<uint8_t *>(&dst), sizeof(dst));
        } else {
            return skip(expectedLength);
        }
    }

    template <typename T>
    bool write(const uint32_t type, const T &src) {
        return write(type, reinterpret_cast<const void *>(&src), sizeof(src));
    }

private:
    ethosu_core_queue &queue;
};

class MessageProcess {
public:
    MessageProcess(ethosu_core_queue &in, ethosu_core_queue &out, InferenceProcess::InferenceProcess &inferenceProcess);

    void run();
    void handleIrq();
    bool handleMessage();
    void sendPong();
    void sendInferenceRsp(uint64_t userArg, std::vector<InferenceProcess::DataPtr> &ofm, bool failed);

private:
    QueueImpl queueIn;
    QueueImpl queueOut;
    InferenceProcess::InferenceProcess &inferenceProcess;
};

} // namespace MessageProcess

#endif
