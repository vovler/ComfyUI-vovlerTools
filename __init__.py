from .wd14tagger import WD14TaggerAndImageFilterer, WD14TensorRTModelLoader, WDTaggerONNXtoTENSORRT, WD14BlackListLoader
from .clip2tensor import DualCLIPToTensorRT, DualCLIPToTensorRTV2, CLIPTensorRTLoader, CLIPTensorRTTextEncode
from .onnx_converter import SDXLDirectoryToOnnx

NODE_CLASS_MAPPINGS = {
    "WD14TaggerAndImageFilterer": WD14TaggerAndImageFilterer,
    "WD14TensorRTModelLoader": WD14TensorRTModelLoader,
    "WDTaggerONNXtoTENSORRT": WDTaggerONNXtoTENSORRT,
    "WD14BlackListLoader": WD14BlackListLoader,
    "DualCLIPToTensorRT": DualCLIPToTensorRT,
    "DualCLIPToTensorRTV2": DualCLIPToTensorRTV2,
    "CLIPTensorRTLoader": CLIPTensorRTLoader,
    "CLIPTensorRTTextEncode": CLIPTensorRTTextEncode,
    "SDXLDirectoryToOnnx": SDXLDirectoryToOnnx,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WD14TaggerAndImageFilterer": "WD14 Tagger & Image Filterer 🏷️🔍",
    "WD14TensorRTModelLoader": "WD14 TensorRT Model Loader ⚡",
    "WDTaggerONNXtoTENSORRT": "WD14 ONNX to TensorRT Converter 🚀",
    "WD14BlackListLoader": "WD14 BlackList Loader 🚫",
    "DualCLIPToTensorRT": "Dual CLIP to TensorRT Converter 🔄⚡",
    "DualCLIPToTensorRTV2": "Dual CLIP to TensorRT V2 (Optimum) 🔄⚡🎯",
    "CLIPTensorRTLoader": "Load CLIP (TensorRT) ⚡",
    "CLIPTensorRTTextEncode": "CLIP Text Encode (TensorRT) ⚡🏷️",
    "SDXLDirectoryToOnnx": "SDXL Directory to ONNX Converter 🔄⚡🎯",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
