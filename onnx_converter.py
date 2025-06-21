import os
import folder_paths
import tempfile
import shutil
import traceback

# Required imports
import onnxruntime
from optimum.onnxruntime import ORTOptimizer, OptimizationConfig
from optimum.exporters.onnx import main_export


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
        self.device = "cpu"
        providers = onnxruntime.get_available_providers()
        if 'CUDAExecutionProvider' in providers:
            self.gpu_available = True
            self.device = "cuda"
            print("\033[92m[INFO] comfy_onnx_exporter: CUDAExecutionProvider found. GPU will be used for optimization.\033[0m")
        else:
            print("\033[93m[WARNING] comfy_onnx_exporter: CUDAExecutionProvider not found. Optimization will run on CPU.\033[0m")

    def export_single_clip(self, clip_filename, model_key, output_dir, optimization_level, use_fp16):
        """
        Handles the conversion of a single CLIP model file and saves it as a single .onnx file.
        """
        final_onnx_path = os.path.join(output_dir, f"{model_key}.onnx")
        
        source_model_path = folder_paths.get_full_path("clip", clip_filename)
        if not source_model_path:
             print(f"\033[91m[ERROR] comfy_onnx_exporter: Could not find model {clip_filename}. Please ensure it's in a ComfyUI 'clip' model directory.\033[0m")
             return
        
        source_model_dir = os.path.dirname(source_model_path)
        
        # CORRECTED: Derive the config filename from the model's filename.
        base_name, _ = os.path.splitext(clip_filename)
        config_filename = f"{base_name}.json"
        source_config_path = os.path.join(source_model_dir, config_filename)

        if not os.path.exists(source_config_path):
            print(f"\033[91m[ERROR] comfy_onnx_exporter: Could not find config file '{config_filename}' for {clip_filename} in {source_model_dir}. A corresponding .json config file is required.\033[0m")
            return
            
        if os.path.exists(final_onnx_path):
            print(f"[INFO] comfy_onnx_exporter: ONNX model already exists at {final_onnx_path}. Skipping.")
            return

        print(f"\n[INFO] comfy_onnx_exporter: Starting export for {model_key} from {clip_filename}")
        print(f"[INFO] comfy_onnx_exporter: Target ONNX path: {final_onnx_path}")

        try:
            with tempfile.TemporaryDirectory() as temp_model_dir:
                # Copy source files to the temporary directory with the standard names that `optimum` expects
                shutil.copyfile(source_model_path, os.path.join(temp_model_dir, "model.safetensors"))
                shutil.copyfile(source_config_path, os.path.join(temp_model_dir, "config.json"))

                print(f"[INFO] comfy_onnx_exporter: Created temporary model structure at {temp_model_dir}")

                with tempfile.TemporaryDirectory() as tmpdir_export:
                    print(f"[INFO] comfy_onnx_exporter: Exporting {model_key} to ONNX format...")
                    
                    main_export(
                        model_name_or_path=temp_model_dir,
                        output=tmpdir_export,
                        task="feature-extraction",
                        framework="pt",
                        library_name="transformers",
                        device=self.device,
                        no_post_process=True
                    )

                    optimizer = ORTOptimizer.from_pretrained(tmpdir_export)
                    optimization_config = OptimizationConfig(
                        optimization_level=optimization_level,
                        fp16=use_fp16
                    )
                    
                    with tempfile.TemporaryDirectory() as tmpdir_optimized:
                        device_used = "GPU" if self.gpu_available else "CPU"
                        print(f"[INFO] comfy_onnx_exporter: Optimizing model on {device_used} (Level: {optimization_level}, FP16: {use_fp16})...")
                        
                        optimizer.optimize(save_dir=tmpdir_optimized, optimization_config=optimization_config)

                        optimized_model_file = os.path.join(tmpdir_optimized, 'model.onnx')
                        shutil.move(optimized_model_file, final_onnx_path)

                        print(f"\033[92m[SUCCESS] comfy_onnx_exporter: Successfully exported and optimized {model_key} to:\n{final_onnx_path}\033[0m")

        except Exception as e:
            print(f"\033[91m[ERROR] comfy_onnx_exporter: An error occurred during ONNX export for {model_key}: {e}\033[0m")
            traceback.print_exc()

    def export_clips_to_onnx(self, clip_l_name, clip_g_name, optimization_level, use_fp16, prompt=None, extra_pnginfo=None):
        output_clip_dir = os.path.join(folder_paths.base_path, "models", "clip")
        os.makedirs(output_clip_dir, exist_ok=True)
        
        self.export_single_clip(clip_l_name, "clip_l", output_clip_dir, optimization_level, use_fp16)
        self.export_single_clip(clip_g_name, "clip_g", output_clip_dir, optimization_level, use_fp16)
        
        return {"ui": {"text": [f"ONNX export finished. Models saved in:\n{output_clip_dir}"]}}