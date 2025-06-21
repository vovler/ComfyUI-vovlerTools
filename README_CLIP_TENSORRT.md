# CLIP to TensorRT Converter

This project provides two versions of dual CLIP to TensorRT converters for ComfyUI:

## Version 1: DualCLIPToTensorRT
- **Method**: Direct PyTorch → ONNX → TensorRT conversion
- **Dependencies**: Basic TensorRT and ONNX
- **Pros**: Fewer dependencies, simpler setup
- **Cons**: May have compatibility issues with complex CLIP models, especially CLIP-G

## Version 2: DualCLIPToTensorRTV2 (Recommended)
- **Method**: Hugging Face Optimum-based conversion
- **Dependencies**: Optimum, Transformers, ONNXRuntime
- **Pros**: 
  - Better ONNX export compatibility
  - Advanced optimization options (O1, O2, O3, O4)
  - More robust handling of complex models
  - Better quantization and calibration
- **Cons**: More dependencies required

## Installation

### Basic Installation (V1 only)
```bash
pip install tensorrt onnx
```

### Full Installation (V1 + V2)
```bash
pip install tensorrt onnx
pip install optimum[onnxruntime-gpu] transformers onnxruntime-gpu
```

## Usage

1. **Place CLIP models**: Put your `.safetensors` CLIP model files in `ComfyUI/models/clip/`

2. **Convert to TensorRT**:
   - Use "Dual CLIP to TensorRT Converter" for V1
   - Use "Dual CLIP to TensorRT V2 (Optimum)" for V2 (recommended)

3. **Load and Use**:
   - Use "Load CLIP (TensorRT)" to load the converted engines
   - Use "CLIP Text Encode (TensorRT)" for text encoding

## Troubleshooting

### V1 Issues
- **Black/blank images**: Usually indicates corrupted TensorRT engines
- **NaN values**: Engine conversion problems, try V2
- **CLIP-G failures**: Known issue with complex attention patterns

### V2 Solutions
- **Better engine quality**: Optimum provides more robust conversion
- **Multiple optimization levels**: Try different O1-O4 levels
- **Improved quantization**: Better FP16 handling

## Recommendations

1. **Start with V2**: It's more robust and handles edge cases better
2. **Use FP16**: Faster inference with minimal quality loss
3. **Optimization Level O2**: Good balance of speed and stability
4. **Batch sizes**: Start with 1/1/4 for min/opt/max

## Engine Files

Engines are saved in `ComfyUI/models/clip_tensorrt/` with naming:
- V1: `{name}_sdxl_{min}_{opt}_{max}_fp16.engine`
- V2: `{name}_sdxl_{min}_{opt}_{max}_fp16_v2.engine`

Each conversion creates two files:
- `*_clip_l.engine` - CLIP-L text encoder
- `*_clip_g.engine` - CLIP-G text encoder 