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

#include <message_process.hpp>

#include "cmsis_compiler.h"

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <inttypes.h>

using namespace std;
using namespace InferenceProcess;

namespace MessageProcess {

QueueImpl::QueueImpl(ethosu_core_queue &_queue) : queue(_queue) {}

bool QueueImpl::empty() const {
    return queue.header.read == queue.header.write;
}

size_t QueueImpl::available() const {
    size_t avail = queue.header.write - queue.header.read;

    if (queue.header.read > queue.header.write) {
        avail += queue.header.size;
    }

    return avail;
}

size_t QueueImpl::capacity() const {
    return queue.header.size - available();
}

bool QueueImpl::read(uint8_t *dst, uint32_t length) {
    const uint8_t *end = dst + length;
    uint32_t rpos      = queue.header.read;

    invalidateHeaderData();

    if (length > available()) {
        return false;
    }

    while (dst < end) {
        *dst++ = queue.data[rpos];
        rpos   = (rpos + 1) % queue.header.size;
    }

    queue.header.read = rpos;

    cleanHeader();

    return true;
}

bool QueueImpl::write(const Vec *vec, size_t length) {
    size_t total = 0;

    for (size_t i = 0; i < length; i++) {
        total += vec[i].length;
    }

    invalidateHeader();

    if (total > capacity()) {
        return false;
    }

    uint32_t wpos = queue.header.write;

    for (size_t i = 0; i < length; i++) {
        const uint8_t *src = reinterpret_cast<const uint8_t *>(vec[i].base);
        const uint8_t *end = src + vec[i].length;

        while (src < end) {
            queue.data[wpos] = *src++;
            wpos             = (wpos + 1) % queue.header.size;
        }
    }

    // Update the write position last
    queue.header.write = wpos;

    cleanHeaderData();

    return true;
}

bool QueueImpl::write(const uint32_t type, const void *src, uint32_t length) {
    ethosu_core_msg msg = {type, length};
    Vec vec[2]          = {{&msg, sizeof(msg)}, {src, length}};

    return write(vec, 2);
}

bool QueueImpl::skip(uint32_t length) {
    uint32_t rpos = queue.header.read;

    invalidateHeader();

    if (length > available()) {
        return false;
    }

    queue.header.read = (rpos + length) % queue.header.size;

    cleanHeader();

    return true;
}

void QueueImpl::cleanHeader() const {
#if defined(__DCACHE_PRESENT) && (__DCACHE_PRESENT == 1U)
    SCB_CleanDCache_by_Addr(reinterpret_cast<uint32_t *>(&queue.header), sizeof(queue.header));
#endif
}

void QueueImpl::cleanHeaderData() const {
#if defined(__DCACHE_PRESENT) && (__DCACHE_PRESENT == 1U)
    SCB_CleanDCache_by_Addr(reinterpret_cast<uint32_t *>(&queue.header), sizeof(queue.header));
    SCB_CleanDCache_by_Addr(reinterpret_cast<uint32_t *>(queue.data), queue.header.size);
#endif
}

void QueueImpl::invalidateHeader() const {
#if defined(__DCACHE_PRESENT) && (__DCACHE_PRESENT == 1U)
    SCB_InvalidateDCache_by_Addr(reinterpret_cast<uint32_t *>(&queue.header), sizeof(queue.header));
#endif
}

void QueueImpl::invalidateHeaderData() const {
#if defined(__DCACHE_PRESENT) && (__DCACHE_PRESENT == 1U)
    SCB_InvalidateDCache_by_Addr(reinterpret_cast<uint32_t *>(&queue.header), sizeof(queue.header));
    SCB_InvalidateDCache_by_Addr(reinterpret_cast<uint32_t *>(queue.data), queue.header.size);
#endif
}

MessageProcess::MessageProcess(ethosu_core_queue &in,
                               ethosu_core_queue &out,
                               Mailbox::Mailbox &mbox,
                               ::InferenceProcess::InferenceProcess &_inferenceProcess) :
    queueIn(in),
    queueOut(out), mailbox(mbox), inferenceProcess(_inferenceProcess) {
    mailbox.registerCallback(mailboxCallback, reinterpret_cast<void *>(this));
}

void MessageProcess::run() {
    while (true) {
        // Handle all messages in queue
        while (handleMessage())
            ;

        // Wait for event
        __WFE();
    }
}

void MessageProcess::handleIrq() {
    __SEV();
}

bool MessageProcess::handleMessage() {
    ethosu_core_msg msg;

    // Read msg header
    if (!queueIn.read(msg)) {
        return false;
    }

    printf("Message. type=%" PRIu32 ", length=%" PRIu32 "\n", msg.type, msg.length);

    switch (msg.type) {
    case ETHOSU_CORE_MSG_PING:
        printf("Ping\n");
        sendPong();
        break;
    case ETHOSU_CORE_MSG_INFERENCE_REQ: {
        ethosu_core_inference_req req;

        if (!queueIn.readOrSkip(req, msg.length)) {
            printf("Failed to read payload.\n");
            return false;
        }

        printf("InferenceReq. user_arg=0x%" PRIx64 ", network={0x%" PRIx32 ", %" PRIu32 "}",
               req.user_arg,
               req.network.ptr,
               req.network.size);

        printf(", ifm_count=%" PRIu32 ", ifm=[", req.ifm_count);
        for (uint32_t i = 0; i < req.ifm_count; ++i) {
            if (i > 0) {
                printf(", ");
            }

            printf("{0x%" PRIx32 ", %" PRIu32 "}", req.ifm[i].ptr, req.ifm[i].size);
        }
        printf("]");

        printf(", ofm_count=%" PRIu32 ", ofm=[", req.ofm_count);
        for (uint32_t i = 0; i < req.ofm_count; ++i) {
            if (i > 0) {
                printf(", ");
            }

            printf("{0x%" PRIx32 ", %" PRIu32 "}", req.ofm[i].ptr, req.ofm[i].size);
        }
        printf("]\n");

        DataPtr networkModel(reinterpret_cast<void *>(req.network.ptr), req.network.size);

        vector<DataPtr> ifm;
        for (uint32_t i = 0; i < req.ifm_count; ++i) {
            ifm.push_back(DataPtr(reinterpret_cast<void *>(req.ifm[i].ptr), req.ifm[i].size));
        }

        vector<DataPtr> ofm;
        for (uint32_t i = 0; i < req.ofm_count; ++i) {
            ofm.push_back(DataPtr(reinterpret_cast<void *>(req.ofm[i].ptr), req.ofm[i].size));
        }

        vector<DataPtr> expectedOutput;

        InferenceJob job("job", networkModel, ifm, ofm, expectedOutput, -1);
        job.invalidate();

        bool failed = inferenceProcess.runJob(job);
        job.clean();

        sendInferenceRsp(req.user_arg, job.output, failed);
        break;
    }
    default: {
        printf("Unexpected message type: %" PRIu32 ", skipping %" PRIu32 " bytes\n", msg.type, msg.length);

        queueIn.skip(msg.length);
    } break;
    }

    return true;
}

void MessageProcess::sendPong() {
    if (!queueOut.write(ETHOSU_CORE_MSG_PONG)) {
        printf("Failed to write pong.\n");
    }
    mailbox.sendMessage();
}

void MessageProcess::sendInferenceRsp(uint64_t userArg, vector<DataPtr> &ofm, bool failed) {
    ethosu_core_inference_rsp rsp;

    rsp.user_arg  = userArg;
    rsp.ofm_count = ofm.size();
    rsp.status    = failed ? ETHOSU_CORE_STATUS_ERROR : ETHOSU_CORE_STATUS_OK;

    for (size_t i = 0; i < ofm.size(); ++i) {
        rsp.ofm_size[i] = ofm[i].size;
    }

    printf("Sending inference response. userArg=0x%" PRIx64 ", ofm_count=%" PRIu32 ", status=%" PRIu32 "\n",
           rsp.user_arg,
           rsp.ofm_count,
           rsp.status);

    if (!queueOut.write(ETHOSU_CORE_MSG_INFERENCE_RSP, rsp)) {
        printf("Failed to write inference.\n");
    }
    mailbox.sendMessage();
}

void MessageProcess::mailboxCallback(void *userArg) {
    MessageProcess *_this = reinterpret_cast<MessageProcess *>(userArg);
    _this->handleIrq();
}

} // namespace MessageProcess
