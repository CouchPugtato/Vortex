// minimal C++ tool to convert an ONNX model to a TensorRT engine
// Usage: ./build_engine <model.onnx> <output.engine>

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <NvInfer.h>
#include <NvOnnxParser.h>

// logger for TensorRT
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TRT] " << msg << std::endl;
    }
} gLogger;

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <model.onnx> <output.engine>" << std::endl;
        return 1;
    }

    std::string onnxPath = argv[1];
    std::string enginePath = argv[2];

    std::cout << "Building Engine..." << std::endl;
    std::cout << "Input ONNX: " << onnxPath << std::endl;
    std::cout << "Output Engine: " << enginePath << std::endl;

    // 1. create Builder
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    if (!builder) return 1;

    // 2. create Network Definition (Explicit Batch)
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network) return 1;

    // 3. parse ONNX Model
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
    if (!parser->parseFromFile(onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "Failed to parse ONNX file" << std::endl;
        return 1;
    }

    // 4. create Config
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) return 1;

    // allow FP16 for speedup on Orin Nano
    if (builder->platformHasFastFp16()) {
        std::cout << "Enabling FP16 Mode" << std::endl;
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 30);

    // 5. build Serialized Engine
    auto serializedModel = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!serializedModel) {
        std::cerr << "Engine serialization failed" << std::endl;
        return 1;
    }

    // 6. save to File
    std::ofstream outfile(enginePath, std::ios::binary);
    outfile.write(reinterpret_cast<const char*>(serializedModel->data()), serializedModel->size());

    std::cout << "Success! Engine saved to " << enginePath << std::endl;
    return 0;
}
