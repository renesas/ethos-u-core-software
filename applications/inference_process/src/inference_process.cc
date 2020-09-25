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

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "inference_process.hpp"

#include <inttypes.h>

#ifndef TENSOR_ARENA_SIZE
#define TENSOR_ARENA_SIZE (1024)
#endif

using namespace std;

__attribute__((section(".bss.NoInit"), aligned(16))) uint8_t inferenceProcessTensorArena[TENSOR_ARENA_SIZE];

namespace {
void print_output_data(TfLiteTensor *output, size_t bytesToPrint) {
    const int numBytesToPrint = min(output->bytes, bytesToPrint);

    int dims_size = output->dims->size;
    printf("{\n");
    printf("\"dims\": [%d,", dims_size);
    for (int i = 0; i < output->dims->size - 1; ++i) {
        printf("%d,", output->dims->data[i]);
    }
    printf("%d],\n", output->dims->data[dims_size - 1]);

    printf("\"data_address\": \"%08" PRIx32 "\",\n", (uint32_t)output->data.data);
    printf("\"data\":\"");
    for (int i = 0; i < numBytesToPrint - 1; ++i) {
        if (i % 16 == 0 && i != 0) {
            printf("\n");
        }
        printf("0x%02x,", output->data.uint8[i]);
    }
    printf("0x%02x\"\n", output->data.uint8[numBytesToPrint - 1]);
    printf("}");
}

bool copyOutput(const TfLiteTensor &src, InferenceProcess::DataPtr &dst) {
    if (dst.data == nullptr) {
        return false;
    }

    if (src.bytes > dst.size) {
        printf("Tensor size %d does not match output size %d.\n", src.bytes, dst.size);
        return true;
    }

    copy(src.data.uint8, src.data.uint8 + src.bytes, static_cast<uint8_t *>(dst.data));
    dst.size = src.bytes;

    return false;
}

} // namespace

namespace InferenceProcess {
DataPtr::DataPtr(void *data, size_t size) : data(data), size(size) {}

InferenceJob::InferenceJob() : numBytesToPrint(0) {}

InferenceJob::InferenceJob(const string &name,
                           const DataPtr &networkModel,
                           const vector<DataPtr> &input,
                           const vector<DataPtr> &output,
                           const vector<DataPtr> &expectedOutput,
                           size_t numBytesToPrint) :
    name(name),
    networkModel(networkModel), input(input), output(output), expectedOutput(expectedOutput),
    numBytesToPrint(numBytesToPrint) {}

InferenceProcess::InferenceProcess() : lock(0) {}

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
    printf("Running inference job: %s\n", job.name.c_str());

    tflite::MicroErrorReporter microErrorReporter;
    tflite::ErrorReporter *reporter = &microErrorReporter;

    // Get model handle and verify that the version is correct
    const tflite::Model *model = ::tflite::GetModel(job.networkModel.data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Model provided is schema version %" PRIu32 " not equal to supported version %d.\n",
               model->version(),
               TFLITE_SCHEMA_VERSION);
        return true;
    }

    // Create the TFL micro interpreter
    tflite::AllOpsResolver resolver;
    tflite::MicroInterpreter interpreter(model, resolver, inferenceProcessTensorArena, TENSOR_ARENA_SIZE, reporter);

    // Allocate tensors
    TfLiteStatus allocate_status = interpreter.AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        printf("AllocateTensors failed for inference job: %s\n", job.name.c_str());
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
        printf("Number of input buffers does not match number of non empty network tensors. input=%zu, network=%zu\n",
               job.input.size(),
               inputTensors.size());
        return true;
    }

    // Copy input data
    for (size_t i = 0; i < inputTensors.size(); ++i) {
        const DataPtr &input       = job.input[i];
        const TfLiteTensor *tensor = inputTensors[i];

        if (input.size != tensor->bytes) {
            printf("Input size does not match network size. job=%s, index=%zu, input=%zu, network=%u\n",
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
        printf("Invoke failed for inference job: %s\n", job.name.c_str());
        return true;
    }

    // Copy output data
    if (job.output.size() > 0) {
        if (interpreter.outputs_size() != job.output.size()) {
            printf("Number of outputs mismatch. job=%zu, network=%u\n", job.output.size(), interpreter.outputs_size());
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
        printf("num_of_outputs: %d\n", interpreter.outputs_size());
        printf("output_begin\n");
        printf("[\n");

        for (unsigned int i = 0; i < interpreter.outputs_size(); i++) {
            TfLiteTensor *output = interpreter.output(i);
            print_output_data(output, job.numBytesToPrint);
            if (i != interpreter.outputs_size() - 1) {
                printf(",\n");
            }
        }

        printf("]\n");
        printf("output_end\n");
    }

    if (job.expectedOutput.size() > 0) {
        if (job.expectedOutput.size() != interpreter.outputs_size()) {
            printf("Expeded number of output tensors does not match network. job=%s, expected=%zu, network=%zu\n",
                   job.name.c_str(),
                   job.expectedOutput.size(),
                   interpreter.outputs_size());
            return true;
        }

        for (unsigned int i = 0; i < interpreter.outputs_size(); i++) {
            const DataPtr &expected    = job.expectedOutput[i];
            const TfLiteTensor *output = interpreter.output(i);

            if (expected.size != output->bytes) {
                printf(
                    "Expected tensor size does not match network size. job=%s, index=%u, expected=%zu, network=%zu\n",
                    job.name.c_str(),
                    i,
                    expected.size,
                    output->bytes);
                return true;
            }

            for (unsigned int j = 0; j < output->bytes; ++j) {
                if (output->data.uint8[j] != static_cast<uint8_t *>(expected.data)[j]) {
                    printf("Expected tensor size does not match network size. job=%s, index=%u, offset=%u, "
                           "expected=%02x, network=%02x\n",
                           job.name.c_str(),
                           i,
                           j,
                           static_cast<uint8_t *>(expected.data)[j],
                           output->data.uint8[j]);
                }
            }
        }
    }

    printf("Finished running job: %s\n", job.name.c_str());

    return false;
}

bool InferenceProcess::run(bool exitOnEmpty) {
    bool anyJobFailed = false;

    while (true) {
        getLock();
        bool empty = inferenceJobQueue.empty();
        freeLock();

        if (empty) {
            if (exitOnEmpty) {
                printf("Exit from InferenceProcess::run() on empty job queue!\n");
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
