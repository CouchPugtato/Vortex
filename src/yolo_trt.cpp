#include "yolo_trt.h"

#include <NvInfer.h>
#include <NvInferVersion.h>
#include <cuda_runtime_api.h>

#include <fstream>
#include <memory>
#include <vector>
#include <string>
#include <iostream>

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
};

static TrtLogger gLogger;

template <typename T>
struct TrtDeleter {
    void operator()(T* ptr) const {
        if (!ptr) return;
#if NV_TENSORRT_MAJOR >= 10
        delete ptr;
#else
        ptr->destroy();
#endif
    }
};

struct TrtHandle {
    std::unique_ptr<nvinfer1::IRuntime, TrtDeleter<nvinfer1::IRuntime>> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter<nvinfer1::ICudaEngine>> engine;
    std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter<nvinfer1::IExecutionContext>> context;
    cudaStream_t stream{nullptr};

#if NV_TENSORRT_MAJOR >= 10
    std::string inputName;
    std::string outputName;
#else
    int inputIndex{-1};
    int outputIndex{-1};
#endif
    nvinfer1::Dims inputDims{};
    nvinfer1::Dims outputDims{};

    void* deviceInput{nullptr};
    void* deviceOutput{nullptr};
    size_t inputBytes{0};
    size_t outputBytes{0};
};

static bool readFile(const std::string& path, std::vector<char>& out) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;
    file.seekg(0, std::ios::end);
    size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);
    out.resize(size);
    file.read(out.data(), size);
    return true;
}

static size_t volume(const nvinfer1::Dims& d) {
    size_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) {
        v *= static_cast<size_t>(d.d[i]);
    }
    return v;
}

void* yolo_trt_create(const char* engine_path, YoloTrtDims* input_dims, YoloTrtDims* output_dims) {
    if (!engine_path || !input_dims || !output_dims) {
        return nullptr;
    }

    std::vector<char> engineData;
    if (!readFile(engine_path, engineData)) {
        std::cout << "[TRT] Failed to read engine file: " << engine_path << std::endl;
        return nullptr;
    }

    auto handle = std::make_unique<TrtHandle>();
    handle->runtime.reset(nvinfer1::createInferRuntime(gLogger));
    if (!handle->runtime) return nullptr;

    handle->engine.reset(handle->runtime->deserializeCudaEngine(engineData.data(), engineData.size()));
    if (!handle->engine) return nullptr;

    handle->context.reset(handle->engine->createExecutionContext());
    if (!handle->context) return nullptr;

#if NV_TENSORRT_MAJOR >= 10
    const int nbTensors = handle->engine->getNbIOTensors();
    for (int i = 0; i < nbTensors; ++i) {
        const char* name = handle->engine->getIOTensorName(i);
        if (!name) continue;
        const auto mode = handle->engine->getTensorIOMode(name);
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            handle->inputName = name;
            handle->inputDims = handle->engine->getTensorShape(name);
        } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
            handle->outputName = name;
            handle->outputDims = handle->engine->getTensorShape(name);
        }
    }

    if (handle->inputName.empty() || handle->outputName.empty()) {
        return nullptr;
    }

    for (int i = 0; i < handle->inputDims.nbDims; ++i) {
        if (handle->inputDims.d[i] < 0) {
            return nullptr;
        }
    }
    for (int i = 0; i < handle->outputDims.nbDims; ++i) {
        if (handle->outputDims.d[i] < 0) {
            return nullptr;
        }
    }
#else
    const int nbBindings = handle->engine->getNbBindings();
    for (int i = 0; i < nbBindings; ++i) {
        if (handle->engine->bindingIsInput(i)) {
            handle->inputIndex = i;
            handle->inputDims = handle->engine->getBindingDimensions(i);
        } else {
            handle->outputIndex = i;
            handle->outputDims = handle->engine->getBindingDimensions(i);
        }
    }

    if (handle->inputIndex < 0 || handle->outputIndex < 0) {
        return nullptr;
    }

    if (handle->inputDims.d[0] == -1) {
        auto profileDims = handle->engine->getProfileDimensions(
            handle->inputIndex,
            0,
            nvinfer1::OptProfileSelector::kOPT
        );
        handle->inputDims = profileDims;
        handle->context->setBindingDimensions(handle->inputIndex, handle->inputDims);
    }

    if (!handle->context->allInputDimensionsSpecified()) {
        return nullptr;
    }

    handle->outputDims = handle->context->getBindingDimensions(handle->outputIndex);
#endif
    handle->inputBytes = volume(handle->inputDims) * sizeof(float);
    handle->outputBytes = volume(handle->outputDims) * sizeof(float);

    if (cudaStreamCreate(&handle->stream) != cudaSuccess) {
        return nullptr;
    }

    if (cudaMalloc(&handle->deviceInput, handle->inputBytes) != cudaSuccess) {
        return nullptr;
    }
    if (cudaMalloc(&handle->deviceOutput, handle->outputBytes) != cudaSuccess) {
        return nullptr;
    }

    input_dims->nb_dims = handle->inputDims.nbDims;
    for (int i = 0; i < handle->inputDims.nbDims && i < 8; ++i) {
        input_dims->dims[i] = handle->inputDims.d[i];
    }

    output_dims->nb_dims = handle->outputDims.nbDims;
    for (int i = 0; i < handle->outputDims.nbDims && i < 8; ++i) {
        output_dims->dims[i] = handle->outputDims.d[i];
    }

    return handle.release();
}

int32_t yolo_trt_infer(void* handle_ptr, const float* input, float* output, size_t output_len) {
    if (!handle_ptr || !input || !output) return -1;
    auto* handle = reinterpret_cast<TrtHandle*>(handle_ptr);

    if (output_len * sizeof(float) < handle->outputBytes) {
        return -2;
    }

    if (cudaMemcpyAsync(handle->deviceInput, input, handle->inputBytes, cudaMemcpyHostToDevice, handle->stream) != cudaSuccess) {
        return -3;
    }

#if NV_TENSORRT_MAJOR >= 10
    if (!handle->context->setTensorAddress(handle->inputName.c_str(), handle->deviceInput)) {
        return -4;
    }
    if (!handle->context->setTensorAddress(handle->outputName.c_str(), handle->deviceOutput)) {
        return -4;
    }
    if (!handle->context->enqueueV3(handle->stream)) {
        return -4;
    }
#else
    void* bindings[2] = { nullptr, nullptr };
    bindings[handle->inputIndex] = handle->deviceInput;
    bindings[handle->outputIndex] = handle->deviceOutput;

    if (!handle->context->enqueueV2(bindings, handle->stream, nullptr)) {
        return -4;
    }
#endif

    if (cudaMemcpyAsync(output, handle->deviceOutput, handle->outputBytes, cudaMemcpyDeviceToHost, handle->stream) != cudaSuccess) {
        return -5;
    }

    if (cudaStreamSynchronize(handle->stream) != cudaSuccess) {
        return -6;
    }

    return 0;
}

void yolo_trt_destroy(void* handle_ptr) {
    if (!handle_ptr) return;
    auto* handle = reinterpret_cast<TrtHandle*>(handle_ptr);

    if (handle->deviceInput) cudaFree(handle->deviceInput);
    if (handle->deviceOutput) cudaFree(handle->deviceOutput);
    if (handle->stream) cudaStreamDestroy(handle->stream);

    delete handle;
}
