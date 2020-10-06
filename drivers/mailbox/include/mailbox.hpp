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

#ifndef MAILBOX_HPP
#define MAILBOX_HPP

#include <cstddef>
#include <list>

namespace Mailbox {

class Mailbox {
public:
    Mailbox();
    virtual ~Mailbox();
    virtual bool sendMessage() = 0;
    virtual void handleMessage() = 0;
    virtual bool verifyHardware();
    typedef void (*CallbackFptr)(void *userArg);
    void registerCallback(CallbackFptr callback, void *userArg);
    void deregisterCallback(CallbackFptr callback, void *userArg);

protected:
    void notify();
    uint32_t read32(volatile uint32_t *baseAddr, const uint32_t offset);
    void write32(volatile uint32_t *baseAddr, const uint32_t offset, const uint32_t value);

private:
    struct Callback {
        bool operator==(const Callback &b) const;
        CallbackFptr callback;
        void *userArg;
    };

    std::list<Callback> callbacks;
};

} // namespace Mailbox

#endif
