import os
import torch
import folder_paths
import tempfile
import shutil
import traceback

# Direct imports based on the provided __init__.py structure.
# If these fail, the libraries are not installed correctly.
import onnxruntime
from optimum.onnxruntime import ORTOptimizer, OptimizationConfig
from optimum.exporters.onnx import export


class SDXLClipToOnnx:
    """
    A ComfyUI node to export SDXL CLIP-L and CLIP-G models to ONNX format.
    This version uses the modern Optimum API and is optimized to detect/use a GPU.
    """
    OUTPUT_NODE = True
    CATEGORY = "Export"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_l": ("CLIP",),
                "clip_g": ("CLIP",),
                "optimization_level": ("INT", {"default": 4, "min": 0, "max": 4, "step": 1, "display": "slider"}),
                "use_fp16": ("BOOLEAN", {"default": True}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    # This string must match the method name below.
    FUNCTION = "export_clips_to_onnx"
    
    def __init__(self):
        self.gpu_available = False
        providers = onnxruntime.get_available_providers()
        if 'CUDAExecutionProvider' in providers:
            self.gpu_available = True
            print("\033[92m[INFO] comfy_onnx_exporter: CUDAExecutionProvider found. GPU will be used for optimization.\033[0m")
        else:
            print("\033[93m[WARNING] comfy_onnx_exporter: CUDAExecutionProvider not found. Optimization will run on CPU.\033[0m")

    def export_single_clip(self, clip_model, model_name_hint, optimization_level, use_fp16):
        if not hasattr(clip_model, 'current_filename') or not clip_model.current_filename:
            print(f"[ERROR] comfy_onnx_exporter: Could not find source filename for {model_name_hint}. Cannot determine save location.")
            return

        source_filename = clip_model.current_filename
        clips_dir = os.path.dirname(folder_paths.get_full_path("clips", source_filename))
        onnx_model_name = os.path.splitext(source_filename)[0] + "_onnx"
        output_path = os.path.join(clips_dir, onnx_model_name)

        if os.path.exists(output_path):
            print(f"[INFO] comfy_onnx_exporter: ONNX model for {model_name_hint} already exists at {output_path}. Skipping.")
            return

        print(f"\n[INFO] comfy_onnx_exporter: Starting export for {model_name_hint} ({source_filename})")
        print(f"[INFO] comfy_onnx_exporter: Target ONNX path: {output_path}")

        # Optimum's tools work best with a directory containing the model.
        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"[INFO] comfy_onnx_exporter: Saving temporary HuggingFace model to {tmpdir}")
            
            pytorch_model = clip_model.model
            tokenizer = clip_model.tokenizer
            
            pytorch_model.save_pretrained(tmpdir)
            tokenizer.save_pretrained(tmpdir)

            try:
                # 1. Export the PyTorch model from the temp directory to ONNX format,
                # also saving it within the same temp directory. The `export` function
                # will add the necessary files like model.onnx.
                print(f"[INFO] comfy_onnx_exporter: Exporting {model_name_hint} to ONNX format...")
                export(model_name_or_path=tmpdir, output=tmpdir, task="feature-extraction")

                # 2. Instantiate the optimizer from the directory containing the new ONNX model.
                optimizer = ORTOptimizer.from_pretrained(tmpdir)

                # 3. Define the optimization configuration.
                optimization_config = OptimizationConfig(
                    optimization_level=optimization_level, 
                    fp16=use_fp16
                )

                # 4. Apply the optimization and save to the final destination.
                device_used = "GPU" if self.gpu_available else "CPU"
                print(f"[INFO] comfy_onnx_exporter: Optimizing model on {device_used} (Level: {optimization_level}, FP16: {use_fp16})...")
                optimizer.optimize(save_dir=output_path, optimization_config=optimization_config)

                print(f"\033[92m[SUCCESS] comfy_onnx_exporter: Successfully exported and optimized {model_name_hint} to:\n{output_path}\033[0m")

            except Exception as e:
                print(f"\033[91m[ERROR] comfy_onnx_exporter: An error occurred during ONNX export for {model_name_hint}: {e}\033[0m")
                traceback.print_exc()
                if os.path.exists(output_path):
                    shutil.rmtree(output_path)

    # CORRECTED a typo in the method name here.
    def export_clips_to_onnx(self, clip_l, clip_g, optimization_level, use_fp16, prompt=None, extra_pnginfo=None):
        self.export_single_clip(clip_l, "CLIP-L", optimization_level, use_fp16)
        self.export_single_clip(clip_g, "CLIP-G", optimization_level, use_fp16)
        
        return { "ui": { "text": ["ONNX export process finished. Check console for details."] } }