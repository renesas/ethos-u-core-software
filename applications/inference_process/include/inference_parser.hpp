/*
 * SPDX-FileCopyrightText: Copyright 2022-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "tensorflow/lite/schema/schema_generated.h"

#include <stdlib.h>
#include <string>

namespace InferenceProcess {

template <typename T, typename U>
class Array {
public:
    Array() = delete;
    Array(T *const data, U &size, size_t capacity) : _data{data}, _size{size}, _capacity{capacity} {}

    auto size() const {
        return _size;
    }

    auto capacity() const {
        return _capacity;
    }

    void push_back(const T &data) {
        _data[_size++] = data;
    }

private:
    T *const _data;
    U &_size;
    const size_t _capacity{};
};

template <typename T, typename U>
Array<T, U> makeArray(T *const data, U &size, size_t capacity) {
    return Array<T, U>{data, size, capacity};
}

class InferenceParser {
public:
    const tflite::Model *getModel(const void *buffer, size_t size) {
        // Verify buffer
        flatbuffers::Verifier base_verifier(reinterpret_cast<const uint8_t *>(buffer), size);
        if (!tflite::VerifyModelBuffer(base_verifier)) {
            printf("Warning: the model is not valid\n");
            return nullptr;
        }

        // Create model handle
        const tflite::Model *model = tflite::GetModel(buffer);
        if (model->subgraphs() == nullptr) {
            printf("Warning: nullptr subgraph\n");
            return nullptr;
        }

        return model;
    }

    template <typename T, typename U, size_t S>
    bool parseModel(const void *buffer, size_t size, char (&description)[S], T &&ifmDims, U &&ofmDims) {
        const tflite::Model *model = getModel(buffer, size);
        if (model == nullptr) {
            return true;
        }

        // Depending on the src string, strncpy may not add a null-terminator
        // so one is manually added at the end.
        strncpy(description, model->description()->c_str(), S - 1);
        description[S - 1] = '\0';

        // Get input dimensions for first subgraph
        auto *subgraph = *model->subgraphs()->begin();
        bool failed    = getSubGraphDims(subgraph, subgraph->inputs(), ifmDims);
        if (failed) {
            return true;
        }

        // Get output dimensions for last subgraph
        subgraph = *model->subgraphs()->rbegin();
        failed   = getSubGraphDims(subgraph, subgraph->outputs(), ofmDims);
        if (failed) {
            return true;
        }

        return false;
    }

private:
    bool getShapeSize(const flatbuffers::Vector<int32_t> *shape, size_t &size) {
        size = 1;

        if (shape == nullptr) {
            printf("Warning: nullptr shape size.\n");
            return true;
        }

        if (shape->size() == 0) {
            printf("Warning: shape zero size.\n");
            return true;
        }

        for (auto it = shape->begin(); it != shape->end(); ++it) {
            size *= *it;
        }

        return false;
    }

    bool getTensorTypeSize(const enum tflite::TensorType type, size_t &size) {
        switch (type) {
        case tflite::TensorType::TensorType_UINT8:
        case tflite::TensorType::TensorType_INT8:
            size = 1;
            break;
        case tflite::TensorType::TensorType_INT16:
            size = 2;
            break;
        case tflite::TensorType::TensorType_INT32:
        case tflite::TensorType::TensorType_FLOAT32:
            size = 4;
            break;
        default:
            printf("Warning: Unsupported tensor type\n");
            return true;
        }

        return false;
    }

    template <typename T>
    bool getSubGraphDims(const tflite::SubGraph *subgraph, const flatbuffers::Vector<int32_t> *tensorMap, T &dims) {
        if (subgraph == nullptr || tensorMap == nullptr) {
            printf("Warning: nullptr subgraph or tensormap.\n");
            return true;
        }

        if ((dims.capacity() - dims.size()) < tensorMap->size()) {
            printf("Warning: tensormap size is larger than dimension capacity.\n");
            return true;
        }

        for (auto index = tensorMap->begin(); index != tensorMap->end(); ++index) {
            auto tensor = subgraph->tensors()->Get(*index);
            size_t size;
            size_t tensorSize;

            bool failed = getShapeSize(tensor->shape(), size);
            if (failed) {
                return true;
            }

            failed = getTensorTypeSize(tensor->type(), tensorSize);
            if (failed) {
                return true;
            }

            size *= tensorSize;

            if (size > 0) {
                dims.push_back(size);
            }
        }

        return false;
    }
};

} // namespace InferenceProcess
