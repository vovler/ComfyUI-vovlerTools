import os
import folder_paths
import tempfile
import shutil
import traceback
from pathlib import Path

# Required imports
import torch
import onnxruntime
from optimum.onnxruntime import ORTOptimizer, OptimizationConfig
# The 'convert' module is inside the 'exporters' subdirectory
from optimum.exporters.onnx.convert import export
from diffusers import StableDiffusionXLPipeline
# Import the specific OnnxConfig classes needed for each component
from optimum.exporters.onnx.model_configs import (
    UNetOnnxConfig, 
    CLIPTextOnnxConfig, 
    CLIPTextWithProjectionOnnxConfig, 
    VaeEncoderOnnxConfig, 
    VaeDecoderOnnxConfig
)

# --- The Main Node Class ---

class SDXLDirectoryToOnnx:
    """
    A ComfyUI node to export a model from a diffusers directory structure into
    separate, optimized ONNX files. This is the robust version that correctly

    instantiates OnnxConfig objects and handles missing config values.
    """
    OUTPUT_NODE = True
    CATEGORY = "Export"

    @classmethod
    def INPUT_TYPES(cls):
        diffusers_path = os.path.join(folder_paths.base_path, "models", "diffusers")
        os.makedirs(diffusers_path, exist_ok=True)
        model_dirs = [d for d in os.listdir(diffusers_path) if os.path.isdir(os.path.join(diffusers_path, d))]
        
        return {
            "required": {
                "model_directory": (model_dirs,),
                "output_subfolder_name": ("STRING", {"default": "onnx_from_dir"}),
                "optimization_level": ("INT", {"default": 2, "min": 0, "max": 4, "step": 1, "display": "slider"}),
                "use_fp16": ("BOOLEAN", {"default": True}),
                "export_unet": ("BOOLEAN", {"default": True}),
                "export_clip": ("BOOLEAN", {"default": True}),
                "export_vae": ("BOOLEAN", {"default": True}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "export_from_directory"

    def __init__(self):
        self.gpu_available = 'CUDAExecutionProvider' in onnxruntime.get_available_providers()
        self.device = "cuda" if self.gpu_available else "cpu"
        if self.gpu_available:
            print("\033[92m[INFO] ONNX Exporter: CUDAExecutionProvider found. GPU will be used.\033[0m")
        else:
            print("\033[93m[WARNING] ONNX Exporter: CUDAExecutionProvider not found. CPU will be used.\033[0m")

    def optimize_model(self, model_path: Path, component_name: str, optimization_level: int):
        """Helper to optimize a saved ONNX model."""
        try:
            print(f"[INFO] ONNX Exporter: Optimizing {component_name}...")
            optimizer = ORTOptimizer.from_pretrained(model_path.parent)
            optimization_config = OptimizationConfig(optimization_level=optimization_level)
            
            optimizer.optimize(save_dir=model_path.parent, optimization_config=optimization_config)
            print(f"\033[92m[SUCCESS] ONNX Exporter: Optimization complete for {component_name}.\033[0m")
        except Exception as e:
            print(f"\033[91m[ERROR] ONNX Exporter: Could not optimize {component_name}: {e}\033[0m")
            traceback.print_exc()

    def export_from_directory(self, model_directory, output_subfolder_name, optimization_level, use_fp16, 
                              export_unet, export_clip, export_vae, prompt=None, extra_pnginfo=None):
        
        source_model_dir = Path(folder_paths.base_path) / "models" / "diffusers" / model_directory
        if not source_model_dir.is_dir():
            return {"ui": {"text": [f"ERROR: Source directory not found at '{source_model_dir}'."]}}
            
        final_output_dir = Path(folder_paths.get_output_directory()) / output_subfolder_name
        final_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[INFO] ONNX Exporter: Source model directory: {source_model_dir}")
        print(f"[INFO] ONNX Exporter: Final models will be saved in: {final_output_dir}")

        torch_dtype = torch.float16 if use_fp16 else torch.float32
        onnx_dtype = "fp16" if use_fp16 else "fp32"
        pipeline = None

        try:
            print("\n[INFO] ONNX Exporter: Loading entire pipeline to resolve component configurations...")
            pipeline = StableDiffusionXLPipeline.from_pretrained(source_model_dir, torch_dtype=torch_dtype).to(self.device)
            print("[INFO] ONNX Exporter: Pipeline loaded successfully.")

            # --- UNet Export ---
            if export_unet:
                print("\n[INFO] ONNX Exporter: Exporting UNet...")
                unet_path = final_output_dir / "unet"
                
                # THE KEY FIX: Manually provide the missing projection_dim to the UNet's ONNX config
                unet_config = UNetOnnxConfig(pipeline.unet.config)
                unet_config.values_override = {
                    "text_encoder_projection_dim": pipeline.text_encoder_2.config.projection_dim,
                }
                
                export(model=pipeline.unet, config=unet_config, output=unet_path / "model.onnx", device=self.device, opset=14, dtype=onnx_dtype)
                pipeline.unet.config.save_pretrained(unet_path)
                self.optimize_model(unet_path / "model.onnx", "UNet", optimization_level)

            # --- Text Encoders Export ---
            if export_clip:
                print("\n[INFO] ONNX Exporter: Exporting Text Encoder 1...")
                text_encoder_path = final_output_dir / "text_encoder"
                text_encoder_config = CLIPTextOnnxConfig(pipeline.text_encoder.config, library_name="diffusers")
                export(model=pipeline.text_encoder, config=text_encoder_config, output=text_encoder_path / "model.onnx", device=self.device, opset=14, dtype=onnx_dtype)
                pipeline.text_encoder.config.save_pretrained(text_encoder_path)
                self.optimize_model(text_encoder_path / "model.onnx", "Text Encoder 1", optimization_level)

                print("\n[INFO] ONNX Exporter: Exporting Text Encoder 2...")
                text_encoder_2_path = final_output_dir / "text_encoder_2"
                text_encoder_2_config = CLIPTextWithProjectionOnnxConfig(pipeline.text_encoder_2.config, library_name="diffusers")
                export(model=pipeline.text_encoder_2, config=text_encoder_2_config, output=text_encoder_2_path / "model.onnx", device=self.device, opset=14, dtype=onnx_dtype)
                pipeline.text_encoder_2.config.save_pretrained(text_encoder_2_path)
                self.optimize_model(text_encoder_2_path / "model.onnx", "Text Encoder 2", optimization_level)
                
                print("[INFO] ONNX Exporter: Copying tokenizer files...")
                shutil.copytree(source_model_dir / "tokenizer", final_output_dir / "tokenizer", dirs_exist_ok=True)
                shutil.copytree(source_model_dir / "tokenizer_2", final_output_dir / "tokenizer_2", dirs_exist_ok=True)
                print(f"\033[92m[SUCCESS] ONNX Exporter: Copied tokenizers.\033[0m")

            # --- VAE Export (Split) ---
            if export_vae:
                print("\n[INFO] ONNX Exporter: Starting split export for VAE...")
                # Export VAE Encoder
                vae_encoder_path = final_output_dir / "vae_encoder"
                vae_encoder_config = VaeEncoderOnnxConfig(pipeline.vae.config)
                export(model=pipeline.vae, config=vae_encoder_config, output=vae_encoder_path / "model.onnx", device=self.device, opset=14, dtype=onnx_dtype)
                pipeline.vae.config.save_pretrained(vae_encoder_path)
                self.optimize_model(vae_encoder_path / "model.onnx", "VAE Encoder", optimization_level)

                # Export VAE Decoder
                vae_decoder_path = final_output_dir / "vae_decoder"
                vae_decoder_config = VaeDecoderOnnxConfig(pipeline.vae.config)
                export(model=pipeline.vae, config=vae_decoder_config, output=vae_decoder_path / "model.onnx", device=self.device, opset=14, dtype=onnx_dtype)
                pipeline.vae.config.save_pretrained(vae_decoder_path)
                self.optimize_model(vae_decoder_path / "model.onnx", "VAE Decoder", optimization_level)

        except Exception as e:
            traceback.print_exc()
            return {"ui": {"text": [f"An unexpected error occurred: {e}"]}}
        finally:
            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return {"ui": {"text": [f"ONNX export process finished.\nModels saved in:\n{final_output_dir}"]}}