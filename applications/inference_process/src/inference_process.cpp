/*
 * SPDX-FileCopyrightText: Copyright 2019-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#ifndef INFERENCE_PROCESS_OPS_RESOLVER
#include "micro_mutable_all_ops_resolver.h"
#else
#define _STRINGIFY(a) #a
#define STRINGIFY(a)  _STRINGIFY(a)
#include STRINGIFY(INFERENCE_PROCESS_OPS_RESOLVER)
#endif
#include "tensorflow/lite/micro/cortex_m_generic/debug_log_callback.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_time.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "arm_profiler.hpp"
#include "cmsis_compiler.h"
#include "crc.hpp"
#include "ethosu_log.h"
#include "inference_process.hpp"

#include <inttypes.h>

using namespace std;

namespace {
const char *BASE64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

void printBase64(const uint8_t *data, size_t len) {
    size_t count     = len / 3;
    size_t remainder = len % 3;
    char buf[]       = "====";
    while (count--) {
        buf[0] = BASE64[data[0] >> 2];
        buf[1] = BASE64[(data[0] & 3) << 4 | data[1] >> 4];
        buf[2] = BASE64[(data[1] & 0xf) << 2 | data[2] >> 6];
        buf[3] = BASE64[data[2] & 0x3f];
        LOG("%s", buf);
        data += 3;
    }
    if (remainder) {
        uint8_t b2 = remainder > 1 ? data[1] : 0;
        buf[0]     = BASE64[data[0] >> 2];
        buf[1]     = BASE64[(data[0] & 3) << 4 | b2 >> 4];
        buf[2]     = remainder > 1 ? BASE64[(b2 & 0xf) << 2] : '=';
        buf[3]     = '=';
        LOG("%s", buf);
    }
}

} // namespace

namespace InferenceProcess {
DataPtr::DataPtr(void *_data, size_t _size) : data(_data), size(_size) {}

void DataPtr::invalidate() {
#if defined(__DCACHE_PRESENT) && (__DCACHE_PRESENT == 1U)
    SCB_InvalidateDCache_by_Addr(reinterpret_cast<uint32_t *>(data), size);
#endif
}

void DataPtr::clean() {
#if defined(__DCACHE_PRESENT) && (__DCACHE_PRESENT == 1U)
    SCB_CleanDCache_by_Addr(reinterpret_cast<uint32_t *>(data), size);
#endif
}

char *DataPtr::begin() const {
    return static_cast<char *>(data);
}

char *DataPtr::end() const {
    return static_cast<char *>(data) + size;
}

InferenceJob::InferenceJob() : numBytesToPrint(0), externalContext(nullptr) {}

InferenceJob::InferenceJob(const string &_name,
                           const DataPtr &_networkModel,
                           const vector<DataPtr> &_input,
                           const vector<DataPtr> &_output,
                           const vector<DataPtr> &_expectedOutput,
                           const size_t _numBytesToPrint,
                           void *_externalContext) :
    name(_name),
    networkModel(_networkModel), input(_input), output(_output), expectedOutput(_expectedOutput),
    numBytesToPrint(_numBytesToPrint), externalContext(_externalContext) {}

void InferenceJob::invalidate() {
    networkModel.invalidate();

    for (auto &it : input) {
        it.invalidate();
    }

    for (auto &it : output) {
        it.invalidate();
    }

    for (auto &it : expectedOutput) {
        it.invalidate();
    }
}

void InferenceJob::clean() {
    networkModel.clean();

    for (auto &it : input) {
        it.clean();
    }

    for (auto &it : output) {
        it.clean();
    }

    for (auto &it : expectedOutput) {
        it.clean();
    }
}

InferenceProcess::InferenceProcess(uint8_t *_tensorArena, size_t _tensorArenaSize) :
    tensorArena(_tensorArena), tensorArenaSize(_tensorArenaSize) {}

bool InferenceProcess::runJob(InferenceJob &job) {
    LOG_INFO("Running inference job: %s", job.name.c_str());

    // Register debug log callback for profiling
    RegisterDebugLogCallback(tfluDebugLog);

    // Get model handle and verify that the version is correct
    const tflite::Model *model = parser.getModel(job.networkModel.data, job.networkModel.size);
    if (model == nullptr) {
        LOG_ERR("Invalid model");
        return true;
    }

    // Create the TFL micro interpreter
    tflite::MicroMutableOpResolver<kNumberOperators> resolver = get_resolver();
    tflite::ArmProfiler profiler;
    tflite::MicroInterpreter interpreter(model, resolver, tensorArena, tensorArenaSize, nullptr, &profiler);

    // Allocate tensors
    TfLiteStatus status = interpreter.AllocateTensors();
    if (status != kTfLiteOk) {
        LOG_ERR("Failed to allocate tensors for inference: job=%s", job.name.c_str());
        return true;
    }

    // Set external context
    if (job.externalContext != nullptr) {
        interpreter.SetMicroExternalContext(job.externalContext);
    }

    // Copy IFM data from job descriptor to TFLu arena
    if (copyIfm(job, interpreter)) {
        return true;
    }

    // Get the current cycle counter value
    uint32_t cpuCyclesBegin = tflite::GetCurrentTimeTicks();

    // Run the inference
    status = interpreter.Invoke();

    // Calculate nbr of CPU cycles for the Invoke call
    job.cpuCycles = tflite::GetCurrentTimeTicks() - cpuCyclesBegin;

    if (status != kTfLiteOk) {
        LOG_ERR("Invoke failed for inference: job=%s", job.name.c_str());
        return true;
    }

    // Copy output data from TFLu arena to job descriptor
    if (copyOfm(job, interpreter)) {
        return true;
    }

    printJob(job, interpreter);

    // Compare the OFM with the expected reference data
    if (compareOfm(job, interpreter)) {
        return true;
    }

    LOG_INFO("\n");
    LOG_INFO("Finished running job: %s", job.name.c_str());

    profiler.ReportResults();

    LOG("\n");
    LOG("Operator(s) total: %" PRIu64 " CPU cycles\n\n", profiler.GetTotalTicks());

    LOG("Inference runtime: %" PRIu64 " CPU cycles total\n\n", job.cpuCycles);

    return false;
}

bool InferenceProcess::copyIfm(InferenceJob &job, tflite::MicroInterpreter &interpreter) {
    // Create a filtered list of non empty input tensors
    vector<TfLiteTensor *> inputTensors;
    for (size_t i = 0; i < interpreter.inputs_size(); ++i) {
        TfLiteTensor *tensor = interpreter.input(i);

        if (tensor != nullptr && tensor->bytes > 0) {
            inputTensors.push_back(tensor);
        }
    }

    if (job.input.size() != inputTensors.size()) {
        LOG_ERR("Number of input buffers does not match number of non empty network tensors: input=%zu, network=%zu",
                job.input.size(),
                inputTensors.size());
        return true;
    }

    // Copy input data from job to TFLu arena
    for (size_t i = 0; i < inputTensors.size(); ++i) {
        DataPtr &input       = job.input[i];
        TfLiteTensor *tensor = inputTensors[i];

        if (input.size != tensor->bytes) {
            LOG_ERR("Job input size does not match network input size: job=%s, index=%zu, input=%zu, network=%u",
                    job.name.c_str(),
                    i,
                    input.size,
                    tensor->bytes);
            return true;
        }

        copy(input.begin(), input.end(), tensor->data.uint8);
    }

    return false;
}

bool InferenceProcess::copyOfm(InferenceJob &job, tflite::MicroInterpreter &interpreter) {
    // Skip copy if output is empty
    if (job.output.empty()) {
        return false;
    }

    if (interpreter.outputs_size() != job.output.size()) {
        LOG_ERR("Output size mismatch: job=%zu, network=%u", job.output.size(), interpreter.outputs_size());
        return true;
    }

    for (unsigned i = 0; i < interpreter.outputs_size(); ++i) {
        DataPtr &output      = job.output[i];
        TfLiteTensor *tensor = interpreter.output(i);

        if (tensor == nullptr) {
            return true;
        }
        if (tensor->bytes > output.size) {
            LOG_ERR("Tensor size mismatch: tensor=%u, expected=%u", tensor->bytes, output.size);
            return true;
        }

        copy(tensor->data.uint8, tensor->data.uint8 + tensor->bytes, output.begin());
    }

    return false;
}

bool InferenceProcess::compareOfm(InferenceJob &job, tflite::MicroInterpreter &interpreter) {
    // Skip verification if expected output is empty
    if (job.expectedOutput.empty()) {
        return false;
    }

    if (job.expectedOutput.size() != interpreter.outputs_size()) {
        LOG_ERR("Expected number of output tensors mismatch: job=%s, expected=%zu, network=%zu",
                job.name.c_str(),
                job.expectedOutput.size(),
                interpreter.outputs_size());
        return true;
    }

    for (unsigned int i = 0; i < interpreter.outputs_size(); i++) {
        const DataPtr &expected    = job.expectedOutput[i];
        const TfLiteTensor *output = interpreter.output(i);

        if (output == nullptr) {
            return true;
        }
        if (expected.size != output->bytes) {
            LOG_ERR("Expected output tensor size mismatch: job=%s, index=%u, expected=%zu, network=%zu",
                    job.name.c_str(),
                    i,
                    expected.size,
                    output->bytes);
            return true;
        }

        const char *exp = expected.begin();
        for (unsigned int j = 0; j < output->bytes; ++j) {
            if (output->data.uint8[j] != exp[j]) {
                LOG_ERR("Expected output tensor data mismatch: job=%s, index=%u, offset=%u, "
                        "expected=%02x, network=%02x\n",
                        job.name.c_str(),
                        i,
                        j,
                        exp[j],
                        output->data.uint8[j]);
                return true;
            }
        }
    }

    return false;
}

void InferenceProcess::printJob(InferenceJob &job, tflite::MicroInterpreter &interpreter) {
    LOG("arena_used_bytes : %zu\n", interpreter.arena_used_bytes());

    // Print all of the output data, or the first NUM_BYTES_TO_PRINT bytes,
    // whichever comes first as well as the output shape.
    LOG("num_of_outputs: %u\n", interpreter.outputs_size());
    LOG("output_begin\n");
    LOG("[\n");

    for (unsigned int i = 0; i < interpreter.outputs_size(); i++) {
        printOutputTensor(interpreter.output(i), job.numBytesToPrint);

        if (i != interpreter.outputs_size() - 1) {
            LOG(",\n");
        }
    }

    LOG("]\n");
    LOG("output_end\n");
}

void InferenceProcess::printOutputTensor(TfLiteTensor *output, size_t bytesToPrint) {
    constexpr auto crc           = Crc();
    const uint32_t crc32         = crc.crc32(output->data.data, output->bytes);
    const size_t numBytesToPrint = min(output->bytes, bytesToPrint);
    int dims_size                = output->dims->size;

    LOG("{\n");
    LOG("\"dims\": [%d,", dims_size);

    for (int i = 0; i < output->dims->size - 1; ++i) {
        LOG("%d,", output->dims->data[i]);
    }

    LOG("%d],\n", output->dims->data[dims_size - 1]);
    LOG("\"data_address\": \"%08" PRIx32 "\",\n", (uint32_t)output->data.data);
    LOG("\"data_bytes\": %d,\n", output->bytes);

    if (numBytesToPrint) {
        LOG("\"crc32\": \"%08" PRIx32 "\",\n", crc32);
        LOG("\"data\":\"");
        printBase64(output->data.uint8, numBytesToPrint);
        LOG("\"\n");
    } else {
        LOG("\"crc32\": \"%08" PRIx32 "\"\n", crc32);
    }

    LOG("}");
}

void InferenceProcess::tfluDebugLog(const char *s) {
    LOG("%s", s);
}

} // namespace InferenceProcess
