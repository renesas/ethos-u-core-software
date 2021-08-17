/*
 * Copyright (c) 2019-2021 Arm Limited. All rights reserved.
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

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/cortex_m_generic/debug_log_callback.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "arm_profiler.hpp"
#ifdef ETHOSU
#include "layer_by_layer_profiler.hpp"
#endif
#include "ethosu_log.h"

#include "inference_process.hpp"

#include "cmsis_compiler.h"

#include <inttypes.h>

using namespace std;

namespace {

void tflu_debug_log(const char *s) {
    LOG("%s", s);
}

void print_output_data(TfLiteTensor *output, size_t bytesToPrint) {
    const int numBytesToPrint = min(output->bytes, bytesToPrint);
    int dims_size             = output->dims->size;
    LOG("{\n");
    LOG("\"dims\": [%d,", dims_size);
    for (int i = 0; i < output->dims->size - 1; ++i) {
        LOG("%d,", output->dims->data[i]);
    }
    LOG("%d],\n", output->dims->data[dims_size - 1]);
    LOG("\"data_address\": \"%08" PRIx32 "\",\n", (uint32_t)output->data.data);
    LOG("\"data\":\"");
    for (int i = 0; i < numBytesToPrint - 1; ++i) {
        /*
         * Workaround an issue when compiling with GCC where by
         * printing only a '\n' the produced global output is wrong.
         */
        if (i % 15 == 0 && i != 0) {
            LOG("0x%02x,\n", output->data.uint8[i]);
        } else {
            LOG("0x%02x,", output->data.uint8[i]);
        }
    }
    LOG("0x%02x\"\n", output->data.uint8[numBytesToPrint - 1]);
    LOG("}");
}

bool copyOutput(const TfLiteTensor &src, InferenceProcess::DataPtr &dst) {
    if (dst.data == nullptr) {
        return false;
    }

    if (src.bytes > dst.size) {
        LOG_ERR("Tensor size mismatch (bytes): actual=%d, expected%d.\n", src.bytes, dst.size);
        return true;
    }

    copy(src.data.uint8, src.data.uint8 + src.bytes, static_cast<uint8_t *>(dst.data));
    dst.size = src.bytes;

    return false;
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

InferenceJob::InferenceJob() : numBytesToPrint(0) {}

InferenceJob::InferenceJob(const string &_name,
                           const DataPtr &_networkModel,
                           const vector<DataPtr> &_input,
                           const vector<DataPtr> &_output,
                           const vector<DataPtr> &_expectedOutput,
                           size_t _numBytesToPrint,
                           const vector<uint8_t> &_pmuEventConfig,
                           const uint32_t _pmuCycleCounterEnable) :
    name(_name),
    networkModel(_networkModel), input(_input), output(_output), expectedOutput(_expectedOutput),
    numBytesToPrint(_numBytesToPrint), pmuEventConfig(_pmuEventConfig), pmuCycleCounterEnable(_pmuCycleCounterEnable),
    pmuEventCount(), pmuCycleCounterCount(0) {}

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

// NOTE: Adding code for get_lock & free_lock with some corrections from
// http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.dai0321a/BIHEJCHB.html
// TODO: check correctness?
void InferenceProcess::getLock() {
    int status = 0;

    do {
        // Wait until lock_var is free
        while (__LDREXW(&lock) != 0)
            ;

        // Try to set lock_var
        status = __STREXW(1, &lock);
    } while (status != 0);

    // Do not start any other memory access until memory barrier is completed
    __DMB();
}

// TODO: check correctness?
void InferenceProcess::freeLock() {
    // Ensure memory operations completed before releasing lock
    __DMB();

    lock = 0;
}

bool InferenceProcess::push(const InferenceJob &job) {
    getLock();
    inferenceJobQueue.push(job);
    freeLock();

    return true;
}

bool InferenceProcess::runJob(InferenceJob &job) {
    LOG_INFO("Running inference job: %s\n", job.name.c_str());

    // Register debug log callback for profiling
    RegisterDebugLogCallback(tflu_debug_log);

    tflite::MicroErrorReporter microErrorReporter;
    tflite::ErrorReporter *reporter = &microErrorReporter;

    // Get model handle and verify that the version is correct
    const tflite::Model *model = ::tflite::GetModel(job.networkModel.data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        LOG_ERR("Model schema version unsupported: version=%" PRIu32 ", supported=%d.\n",
                model->version(),
                TFLITE_SCHEMA_VERSION);
        return true;
    }

    // Create the TFL micro interpreter
    tflite::AllOpsResolver resolver;
#ifdef ETHOSU
    tflite::LayerByLayerProfiler profiler;
#else
    tflite::ArmProfiler profiler;
#endif

    tflite::MicroInterpreter interpreter(model, resolver, tensorArena, tensorArenaSize, reporter, &profiler);

    // Allocate tensors
    TfLiteStatus allocate_status = interpreter.AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        LOG_ERR("Failed to allocate tensors for inference: job=%s\n", job.name.c_str());
        return true;
    }

    // Create a filtered list of non empty input tensors
    vector<TfLiteTensor *> inputTensors;
    for (size_t i = 0; i < interpreter.inputs_size(); ++i) {
        TfLiteTensor *tensor = interpreter.input(i);

        if (tensor->bytes > 0) {
            inputTensors.push_back(tensor);
        }
    }
    if (job.input.size() != inputTensors.size()) {
        LOG_ERR("Number of input buffers does not match number of non empty network tensors: input=%zu, network=%zu\n",
                job.input.size(),
                inputTensors.size());
        return true;
    }

    // Copy input data
    for (size_t i = 0; i < inputTensors.size(); ++i) {
        const DataPtr &input       = job.input[i];
        const TfLiteTensor *tensor = inputTensors[i];

        if (input.size != tensor->bytes) {
            LOG_ERR("Job input size does not match network input size: job=%s, index=%zu, input=%zu, network=%u\n",
                    job.name.c_str(),
                    i,
                    input.size,
                    tensor->bytes);
            return true;
        }

        copy(static_cast<char *>(input.data), static_cast<char *>(input.data) + input.size, tensor->data.uint8);
    }

    // Run the inference
    TfLiteStatus invoke_status = interpreter.Invoke();
    if (invoke_status != kTfLiteOk) {
        LOG_ERR("Invoke failed for inference: job=%s\n", job.name.c_str());
        return true;
    }

    LOG("arena_used_bytes : %zu\n", interpreter.arena_used_bytes());

    LOG("Inference runtime: %u cycles\n", (unsigned int)profiler.GetTotalTicks());

    if (job.pmuCycleCounterEnable != 0) {
        job.pmuCycleCounterCount = profiler.GetTotalTicks();
    }

    // Copy output data
    if (job.output.size() > 0) {
        if (interpreter.outputs_size() != job.output.size()) {
            LOG_ERR("Output size mismatch: job=%zu, network=%u\n", job.output.size(), interpreter.outputs_size());
            return true;
        }

        for (unsigned i = 0; i < interpreter.outputs_size(); ++i) {
            if (copyOutput(*interpreter.output(i), job.output[i])) {
                return true;
            }
        }
    }

    if (job.numBytesToPrint > 0) {
        // Print all of the output data, or the first NUM_BYTES_TO_PRINT bytes,
        // whichever comes first as well as the output shape.
        LOG("num_of_outputs: %d\n", interpreter.outputs_size());
        LOG("output_begin\n");
        LOG("[\n");
        for (unsigned int i = 0; i < interpreter.outputs_size(); i++) {
            TfLiteTensor *output = interpreter.output(i);
            print_output_data(output, job.numBytesToPrint);
            if (i != interpreter.outputs_size() - 1) {
                LOG(",\n");
            }
        }
        LOG("]\n");
        LOG("output_end\n");
    }

    if (job.expectedOutput.size() > 0) {
        if (job.expectedOutput.size() != interpreter.outputs_size()) {
            LOG_ERR("Expected number of output tensors mismatch: job=%s, expected=%zu, network=%zu\n",
                    job.name.c_str(),
                    job.expectedOutput.size(),
                    interpreter.outputs_size());
            return true;
        }

        for (unsigned int i = 0; i < interpreter.outputs_size(); i++) {
            const DataPtr &expected    = job.expectedOutput[i];
            const TfLiteTensor *output = interpreter.output(i);

            if (expected.size != output->bytes) {
                LOG_ERR("Expected output tensor size mismatch: job=%s, index=%u, expected=%zu, network=%zu\n",
                        job.name.c_str(),
                        i,
                        expected.size,
                        output->bytes);
                return true;
            }

            for (unsigned int j = 0; j < output->bytes; ++j) {
                if (output->data.uint8[j] != static_cast<uint8_t *>(expected.data)[j]) {
                    LOG_ERR("Expected output tensor data mismatch: job=%s, index=%u, offset=%u, "
                            "expected=%02x, network=%02x\n",
                            job.name.c_str(),
                            i,
                            j,
                            static_cast<uint8_t *>(expected.data)[j],
                            output->data.uint8[j]);
                    return true;
                }
            }
        }
    }

    LOG_INFO("Finished running job: %s\n", job.name.c_str());

    return false;
} // namespace InferenceProcess

bool InferenceProcess::run(bool exitOnEmpty) {
    bool anyJobFailed = false;

    while (true) {
        getLock();
        bool empty = inferenceJobQueue.empty();
        freeLock();

        if (empty) {
            if (exitOnEmpty) {
                LOG_INFO("Exit from InferenceProcess::run() due to empty job queue\n");
                break;
            }

            continue;
        }

        getLock();
        InferenceJob job = inferenceJobQueue.front();
        inferenceJobQueue.pop();
        freeLock();

        if (runJob(job)) {
            anyJobFailed = true;
            continue;
        }
    }

    return anyJobFailed;
}

} // namespace InferenceProcess
