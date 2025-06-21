import os
import folder_paths
import tempfile
import shutil
import traceback

# Required imports
import onnxruntime
from optimum.onnxruntime import ORTOptimizer, OptimizationConfig
from optimum.exporters.onnx import export


class SDXLClipToOnnx:
    """
    A ComfyUI node to export SDXL CLIP-L and CLIP-G models to ONNX format.
    This version loads the models directly from .safetensors files provided by the user
    and saves them as single .onnx files in the /models/clip/ directory.
    """
    OUTPUT_NODE = True
    CATEGORY = "Export"

    @classmethod
    def INPUT_TYPES(cls):
        # Create dropdowns for all files in the 'clip' directory
        clip_files = folder_paths.get_filename_list("clip")
        
        return {
            "required": {
                "clip_l_name": (clip_files, ),
                "clip_g_name": (clip_files, ),
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

    def export_single_clip(self, clip_filename, model_key, clip_base_dir, optimization_level, use_fp16):
        """
        Handles the conversion of a single CLIP model file and saves it as a single .onnx file.
        """
        # The final destination for the single .onnx file
        final_onnx_path = os.path.join(clip_base_dir, f"{model_key}.onnx")
        
        # The source directory containing the .safetensors and config.json
        source_model_path = folder_paths.get_full_path("clip", clip_filename)
        
        # Optimum expects a directory, not a file path.
        source_model_dir = os.path.dirname(source_model_path)

        if os.path.exists(final_onnx_path):
            print(f"[INFO] comfy_onnx_exporter: ONNX model already exists at {final_onnx_path}. Skipping.")
            return

        print(f"\n[INFO] comfy_onnx_exporter: Starting export for {model_key} from {clip_filename}")
        print(f"[INFO] comfy_onnx_exporter: Target ONNX path: {final_onnx_path}")

        try:
            # Temporary directory for the intermediate ONNX model files
            with tempfile.TemporaryDirectory() as tmpdir:
                # 1. Export the model directly from its source directory to the temp directory
                print(f"[INFO] comfy_onnx_exporter: Exporting {model_key} to ONNX format...")
                export(model_name_or_path=source_model_dir, output=tmpdir, task="feature-extraction")

                # 2. Create an optimizer for the exported model in the temp directory
                optimizer = ORTOptimizer.from_pretrained(tmpdir)
                
                # 3. Define the optimization configuration
                optimization_config = OptimizationConfig(
                    optimization_level=optimization_level,
                    fp16=use_fp16
                )
                
                # 4. Optimize the model and save to a second temp directory
                with tempfile.TemporaryDirectory() as tmpdir_optimized:
                    device_used = "GPU" if self.gpu_available else "CPU"
                    print(f"[INFO] comfy_onnx_exporter: Optimizing model on {device_used} (Level: {optimization_level}, FP16: {use_fp16})...")
                    
                    optimizer.optimize(save_dir=tmpdir_optimized, optimization_config=optimization_config)

                    # 5. The final optimized ONNX file is located inside the second temp directory
                    optimized_model_file = os.path.join(tmpdir_optimized, 'model.onnx')

                    # 6. Move the single .onnx file to its final destination
                    shutil.move(optimized_model_file, final_onnx_path)

                    print(f"\033[92m[SUCCESS] comfy_onnx_exporter: Successfully exported and optimized {model_key} to:\n{final_onnx_path}\033[0m")

        except Exception as e:
            print(f"\033[91m[ERROR] comfy_onnx_exporter: An error occurred during ONNX export for {model_key}: {e}\033[0m")
            traceback.print_exc()

    def export_clips_to_onnx(self, clip_l_name, clip_g_name, optimization_level, use_fp16, prompt=None, extra_pnginfo=None):
        # Get the main ComfyUI clip directory for saving the final files
        clip_dir = folder_paths.get_folder_paths("clip")[0]
        os.makedirs(clip_dir, exist_ok=True)
        
        # Run the export process for each selected clip file
        self.export_single_clip(clip_l_name, "clip_l", clip_dir, optimization_level, use_fp16)
        self.export_single_clip(clip_g_name, "clip_g", clip_dir, optimization_level, use_fp16)
        
        return {"ui": {"text": [f"ONNX export finished. Models saved in:\n{clip_dir}"]}}