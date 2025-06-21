import os
import torch
import folder_paths
import tempfile
import shutil
import traceback

# Direct imports based on the provided __init__.py structure.
import onnxruntime
from optimum.onnxruntime import ORTOptimizer, OptimizationConfig
from optimum.exporters.onnx import export


class SDXLClipToOnnx:
    """
    A ComfyUI node to export SDXL CLIP-L and CLIP-G models to ONNX format.
    This version saves the models as single .onnx files in the /models/clip/ directory
    and correctly handles nested CLIP model structures.
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
    FUNCTION = "export_clips_to_onnx"

    def __init__(self):
        self.gpu_available = False
        providers = onnxruntime.get_available_providers()
        if 'CUDAExecutionProvider' in providers:
            self.gpu_available = True
            print("\033[92m[INFO] comfy_onnx_exporter: CUDAExecutionProvider found. GPU will be used for optimization.\033[0m")
        else:
            print("\033[93m[WARNING] comfy_onnx_exporter: CUDAExecutionProvider not found. Optimization will run on CPU.\033[0m")

    def export_single_clip(self, clip_object, model_key, clip_base_dir, optimization_level, use_fp16):
        """
        Handles the conversion of a single CLIP model and saves it as a single .onnx file.
        Navigates the nested structure of custom CLIP wrappers.
        """
        final_onnx_path = os.path.join(clip_base_dir, f"{model_key}.onnx")

        if os.path.exists(final_onnx_path):
            print(f"[INFO] comfy_onnx_exporter: ONNX model already exists at {final_onnx_path}. Skipping.")
            return

        print(f"\n[INFO] comfy_onnx_exporter: Starting export for {model_key}")
        print(f"[INFO] comfy_onnx_exporter: Target ONNX path: {final_onnx_path}")

        try:
            # CORRECTED: Navigate the nested object structure to find the actual model and tokenizer.
            # 1. Get the main model wrapper (e.g., SD1ClipModel) from the CLIP object.
            model_wrapper = clip_object.cond_stage_model
            # 2. Get the inner model object (e.g., SD1CheckpointClipModel) using its key ('clip_l' or 'clip_g').
            # The name of this key is stored in the wrapper's 'clip' attribute.
            inner_model_wrapper = getattr(model_wrapper, model_wrapper.clip)
            # 3. The actual HuggingFace model is the 'transformer' attribute of the inner wrapper.
            pytorch_model = inner_model_wrapper.transformer

            # Do the same for the tokenizer
            tokenizer_wrapper = clip_object.tokenizer
            inner_tokenizer_wrapper = getattr(tokenizer_wrapper, tokenizer_wrapper.clip)
            tokenizer = inner_tokenizer_wrapper.tokenizer

        except AttributeError as e:
            print(f"\033[91m[ERROR] comfy_onnx_exporter: Could not find the model or tokenizer in the expected object structure for {model_key}. Please check your CLIP loader. Details: {e}\033[0m")
            traceback.print_exc()
            return

        # Temporary directory for initial PyTorch -> ONNX export
        with tempfile.TemporaryDirectory() as tmpdir_export:
            print(f"[INFO] comfy_onnx_exporter: Saving temporary HuggingFace model to {tmpdir_export}")
            
            pytorch_model.save_pretrained(tmpdir_export)
            tokenizer.save_pretrained(tmpdir_export)

            try:
                # 1. Export the model to a standard ONNX model inside the temp directory
                print(f"[INFO] comfy_onnx_exporter: Exporting {model_key} to ONNX format...")
                export(model_name_or_path=tmpdir_export, output=tmpdir_export, task="feature-extraction")

                # 2. Create an optimizer for the exported model
                optimizer = ORTOptimizer.from_pretrained(tmpdir_export)
                
                # 3. Define the optimization configuration
                optimization_config = OptimizationConfig(
                    optimization_level=optimization_level,
                    fp16=use_fp16
                )
                
                # 4. Optimize the model and save the result to a *second* temporary directory
                with tempfile.TemporaryDirectory() as tmpdir_optimized:
                    device_used = "GPU" if self.gpu_available else "CPU"
                    print(f"[INFO] comfy_onnx_exporter: Optimizing model on {device_used} (Level: {optimization_level}, FP16: {use_fp16})...")
                    
                    optimizer.optimize(save_dir=tmpdir_optimized, optimization_config=optimization_config)

                    # 5. The optimized ONNX file is located inside the second temp directory
                    optimized_model_file = os.path.join(tmpdir_optimized, 'model.onnx')

                    # 6. Move just the single .onnx file to the final destination
                    shutil.move(optimized_model_file, final_onnx_path)

                    print(f"\033[92m[SUCCESS] comfy_onnx_exporter: Successfully exported and optimized {model_key} to:\n{final_onnx_path}\033[0m")

            except Exception as e:
                print(f"\033[91m[ERROR] comfy_onnx_exporter: An error occurred during ONNX export for {model_key}: {e}\033[0m")
                traceback.print_exc()

    def export_clips_to_onnx(self, clip_l, clip_g, optimization_level, use_fp16, prompt=None, extra_pnginfo=None):
        # CORRECTED: The folder name is "clip", not "clips"
        clip_dir = folder_paths.get_folder_paths("clip")[0]
        os.makedirs(clip_dir, exist_ok=True)
        
        # Run the export process for each clip
        self.export_single_clip(clip_l, "clip_l", clip_dir, optimization_level, use_fp16)
        self.export_single_clip(clip_g, "clip_g", clip_dir, optimization_level, use_fp16)
        
        return {"ui": {"text": [f"ONNX export finished. Models saved in:\n{clip_dir}"]}}