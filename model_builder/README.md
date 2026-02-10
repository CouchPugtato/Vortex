# How to Build the TensorRT Engine Tool

This tool converts an `.onnx` model into a `.engine` file optimized for the Jetson Orin Nano.

## Prerequisites
- **JetPack** installed (includes CUDA, TensorRT, and C++ compilers)
- `cmake` installed (`sudo apt install cmake`)

## 1. Build the Tool
Run these commands in this directory:

```bash
mkdir build
cd build
cmake ..
make
```

## 2. Usage
Once built, you can convert your ONNX model:

```bash
./build_engine /path/to/your_model.onnx /path/to/output.engine
```

## 3. Workflow for Custom Dataset
1.  **Train** a model on PC/Colab (YOLOv8).
2.  **Export** to ONNX: `yolo export model=best.pt format=onnx opset=12`.
3.  **Transfer** the `best.onnx` file to the Jetson.
4.  **Run** this tool on the Jetson to generate `best.engine`.
