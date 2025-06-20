from .wd14tagger import WD14TaggerAndImageFilterer, WD14TensorRTModelLoader, WDTaggerONNXtoTENSORRT, WD14BlackListLoader

NODE_CLASS_MAPPINGS = {
    "WD14TaggerAndImageFilterer": WD14TaggerAndImageFilterer,
    "WD14TensorRTModelLoader": WD14TensorRTModelLoader,
    "WDTaggerONNXtoTENSORRT": WDTaggerONNXtoTENSORRT,
    "WD14BlackListLoader": WD14BlackListLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WD14TaggerAndImageFilterer": "WD14 Tagger & Image Filterer 🏷️🔍",
    "WD14TensorRTModelLoader": "WD14 TensorRT Model Loader ⚡",
    "WDTaggerONNXtoTENSORRT": "WD14 ONNX to TensorRT Converter 🚀",
    "WD14BlackListLoader": "WD14 BlackList Loader 🚫",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
