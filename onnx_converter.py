import os
import torch
import folder_paths
import tempfile
import shutil
import traceback

# Required imports for model reconstruction
from transformers import CLIPTextModelWithProjection, CLIPTextConfig

# Optimum and ONNX Runtime imports
import onnxruntime
from optimum.onnxruntime import ORTOptimizer, OptimizationConfig
from optimum.exporters.onnx import export


class SDXLClipToOnnx:
    """
    A ComfyUI node to export SDXL CLIP-L and CLIP-G models to ONNX format.
    This version correctly reconstructs a HuggingFace model from ComfyUI's
    custom CLIP objects before exporting.
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
        Handles the conversion of a single CLIP model by rebuilding it in a
        HuggingFace-compatible format and then exporting to a single .onnx file.
        """
        final_onnx_path = os.path.join(clip_base_dir, f"{model_key}.onnx")

        if os.path.exists(final_onnx_path):
            print(f"[INFO] comfy_onnx_exporter: ONNX model already exists at {final_onnx_path}. Skipping.")
            return

        print(f"\n[INFO] comfy_onnx_exporter: Starting export for {model_key}")
        print(f"[INFO] comfy_onnx_exporter: Target ONNX path: {final_onnx_path}")

        try:
            print(f"[INFO] comfy_onnx_exporter: Reconstructing HuggingFace model for {model_key}...")

            # --- MODEL RECONSTRUCTION ---
            # Create the correct HuggingFace config for the target model.
            if model_key == "clip_l":
                config = CLIPTextConfig(hidden_size=768, intermediate_size=3072, num_attention_heads=12, num_hidden_layers=12, projection_dim=768, vocab_size=49408, max_position_embeddings=77, hidden_act="quick_gelu")
            elif model_key == "clip_g":
                config = CLIPTextConfig(hidden_size=1280, intermediate_size=5120, num_attention_heads=20, num_hidden_layers=32, projection_dim=1280, vocab_size=49408, max_position_embeddings=77, hidden_act="gelu")
            else:
                raise ValueError(f"Unknown model_key: {model_key}")

            # ** CORRECTED ACCESS LOGIC **
            # 1. Get the top-level wrapper model (e.g., SD1ClipModel)
            model_wrapper = clip_object.cond_stage_model
            # 2. Use the '.clip' attribute (e.g., "clip_l") to get the inner model (e.g., SD1CheckpointClipModel)
            inner_model = getattr(model_wrapper, model_wrapper.clip)
            # 3. Get the transformer, which is the actual torch module with weights
            comfy_transformer = inner_model.transformer
            state_dict = comfy_transformer.state_dict()
            
            # 4. Create a new, standard HuggingFace model and load the weights.
            pytorch_model = CLIPTextModelWithProjection(config)
            pytorch_model.load_state_dict(state_dict)
            
            # 5. Get the tokenizer using the same logic.
            tokenizer_wrapper = clip_object.tokenizer
            inner_tokenizer = getattr(tokenizer_wrapper, tokenizer_wrapper.clip)
            tokenizer = inner_tokenizer.tokenizer
            
            print(f"[INFO] comfy_onnx_exporter: Model reconstruction successful.")

        except Exception as e:
            print(f"\033[91m[ERROR] comfy_onnx_exporter: Failed to reconstruct the HuggingFace model from the ComfyUI CLIP object. Details: {e}\033[0m")
            traceback.print_exc()
            return

        # Temporary directory for the export process
        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"[INFO] comfy_onnx_exporter: Saving temporary HuggingFace model to {tmpdir}")
            
            # This now works because `pytorch_model` is a genuine transformers.PreTrainedModel
            pytorch_model.save_pretrained(tmpdir)
            tokenizer.save_pretrained(tmpdir)

            try:
                # Export and Optimize
                print(f"[INFO] comfy_onnx_exporter: Exporting {model_key} to ONNX format...")
                export(model_name_or_path=tmpdir, output=tmpdir, task="feature-extraction")

                optimizer = ORTOptimizer.from_pretrained(tmpdir)
                optimization_config = OptimizationConfig(optimization_level=optimization_level, fp16=use_fp16)
                
                with tempfile.TemporaryDirectory() as tmpdir_optimized:
                    device_used = "GPU" if self.gpu_available else "CPU"
                    print(f"[INFO] comfy_onnx_exporter: Optimizing model on {device_used} (Level: {optimization_level}, FP16: {use_fp16})...")
                    optimizer.optimize(save_dir=tmpdir_optimized, optimization_config=optimization_config)

                    optimized_model_file = os.path.join(tmpdir_optimized, 'model.onnx')
                    shutil.move(optimized_model_file, final_onnx_path)

                    print(f"\033[92m[SUCCESS] comfy_onnx_exporter: Successfully exported and optimized {model_key} to:\n{final_onnx_path}\033[0m")

            except Exception as e:
                print(f"\033[91m[ERROR] comfy_onnx_exporter: An error occurred during the ONNX export/optimization for {model_key}: {e}\033[0m")
                traceback.print_exc()

    def export_clips_to_onnx(self, clip_l, clip_g, optimization_level, use_fp16, prompt=None, extra_pnginfo=None):
        clip_dir = folder_paths.get_folder_paths("clip")[0]
        os.makedirs(clip_dir, exist_ok=True)
        
        self.export_single_clip(clip_l, "clip_l", clip_dir, optimization_level, use_fp16)
        self.export_single_clip(clip_g, "clip_g", clip_dir, optimization_level, use_fp16)
        
        return {"ui": {"text": [f"ONNX export finished. Models saved in:\n{clip_dir}"]}}