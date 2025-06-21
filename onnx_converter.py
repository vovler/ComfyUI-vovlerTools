import os
import folder_paths
import tempfile
import shutil
import traceback

# Required imports
import torch
import onnxruntime
from optimum.onnxruntime import ORTOptimizer, OptimizationConfig
from optimum.exporters.onnx import main_export
# CORRECTED: Import AutoencoderKL from its specific submodule
from diffusers.models.autoencoder_kl import AutoencoderKL


# --- Helper function for exporting components ---
def export_component(component_name, source_path, final_output_dir, optimization_level, use_fp16, device="cpu", gpu_available=False):
    """
    Handles the export and optimization of a single, well-defined model component.
    """
    print(f"\n[INFO] ONNX Exporter: Starting export for component: {component_name.upper()}")
    
    # The final location for this specific component (e.g., /output/onnx_models/unet)
    final_component_path = os.path.join(final_output_dir, component_name)
    if os.path.exists(final_component_path):
        print(f"[INFO] ONNX Exporter: Output directory '{final_component_path}' already exists. Skipping component.")
        return True

    # Use the specific tasks registered in optimum's model_configs.py
    task_map = {
        "unet": "semantic-segmentation",
        "text_encoder": "feature-extraction",
        "text_encoder_2": "feature-extraction",
        "vae_encoder": "semantic-segmentation",
        "vae_decoder": "semantic-segmentation",
    }
    task = task_map.get(component_name)
    
    if not task:
        print(f"\033[91m[ERROR] ONNX Exporter: Unknown component '{component_name}'. Cannot determine export task.\033[0m")
        return False

    try:
        with tempfile.TemporaryDirectory() as tmpdir_export:
            print(f"[INFO] ONNX Exporter: Exporting '{component_name}' with task '{task}' to temporary ONNX format...")
            
            main_export(
                model_name_or_path=source_path,
                output=tmpdir_export,
                task=task,
                framework="pt",
                device=device,
                fp16=use_fp16,
                no_post_process=True
            )
            
            optimizer = ORTOptimizer.from_pretrained(tmpdir_export)
            optimization_config = OptimizationConfig(
                optimization_level=optimization_level,
                fp16=False # FP16 conversion is already done during export.
            )
            
            device_used = "GPU" if gpu_available else "CPU"
            print(f"[INFO] ONNX Exporter: Optimizing {component_name} on {device_used} (Level: {optimization_level}, FP16: {use_fp16})...")
            
            optimizer.optimize(save_dir=final_component_path, optimization_config=optimization_config)

            print(f"\033[92m[SUCCESS] ONNX Exporter: Successfully exported and optimized {component_name} to:\n{final_component_path}\033[0m")
            return True

    except Exception as e:
        print(f"\033[91m[ERROR] ONNX Exporter: An error occurred during ONNX export for {component_name}: {e}\033[0m")
        traceback.print_exc()
        return False


# --- The Main Node Class ---

class SDXLDirectoryToOnnx:
    """
    A ComfyUI node to load a model from a diffusers directory structure
    and export its components (UNet, Text Encoders, VAE) into separate ONNX files.
    """
    OUTPUT_NODE = True
    CATEGORY = "Export"

    @classmethod
    def INPUT_TYPES(cls):
        diffusers_path = os.path.join(folder_paths.get_base_path(), "models", "diffusers")
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
            print("\033[92m[INFO] ONNX Exporter: CUDAExecutionProvider found. GPU will be used for export/optimization.\033[0m")
        else:
            print("\033[93m[WARNING] ONNX Exporter: CUDAExecutionProvider not found. Export/optimization will run on CPU.\033[0m")

    def export_from_directory(self, model_directory, output_subfolder_name, optimization_level, use_fp16, 
                              export_unet, export_clip, export_vae, prompt=None, extra_pnginfo=None):
        
        source_model_dir = os.path.join(folder_paths.get_base_path(), "models", "diffusers", model_directory)
        if not os.path.isdir(source_model_dir):
            return {"ui": {"text": [f"ERROR: Source directory not found at '{source_model_dir}'."]}}
            
        final_output_dir = os.path.join(folder_paths.get_output_directory(), output_subfolder_name)
        os.makedirs(final_output_dir, exist_ok=True)
        print(f"[INFO] ONNX Exporter: Source model directory: {source_model_dir}")
        print(f"[INFO] ONNX Exporter: Final models will be saved in: {final_output_dir}")

        try:
            # --- UNet and CLIP Export (Standard) ---
            if export_unet:
                export_component("unet", os.path.join(source_model_dir, "unet"), final_output_dir, optimization_level, use_fp16, self.device, self.gpu_available)
            
            if export_clip:
                export_component("text_encoder", os.path.join(source_model_dir, "text_encoder"), final_output_dir, optimization_level, use_fp16, self.device, self.gpu_available)
                export_component("text_encoder_2", os.path.join(source_model_dir, "text_encoder_2"), final_output_dir, optimization_level, use_fp16, self.device, self.gpu_available)
                
                print("[INFO] ONNX Exporter: Copying tokenizer files...")
                shutil.copytree(os.path.join(source_model_dir, "tokenizer"), os.path.join(final_output_dir, "tokenizer"), dirs_exist_ok=True)
                shutil.copytree(os.path.join(source_model_dir, "tokenizer_2"), os.path.join(final_output_dir, "tokenizer_2"), dirs_exist_ok=True)
                print(f"\033[92m[SUCCESS] ONNX Exporter: Copied tokenizers to {final_output_dir}\033[0m")

            # --- VAE Export (Special Handling) ---
            if export_vae:
                print("\n[INFO] ONNX Exporter: Starting special handling for VAE export.")
                vae_source_path = os.path.join(source_model_dir, "vae")
                if not os.path.isdir(vae_source_path):
                    print(f"\033[93m[WARNING] ONNX Exporter: VAE directory '{vae_source_path}' not found. Skipping VAE export.\033[0m")
                else:
                    with tempfile.TemporaryDirectory() as tmp_vae_dir:
                        print("[INFO] ONNX Exporter: Loading VAE and splitting into encoder/decoder for export...")
                        
                        dtype = torch.float16 if use_fp16 else torch.float32
                        # CORRECTED: Use the correct class name to load the VAE
                        vae = AutoencoderKL.from_pretrained(vae_source_path, torch_dtype=dtype)
                        
                        # Save the encoder and decoder parts to separate temporary directories
                        encoder_path = os.path.join(tmp_vae_dir, 'encoder')
                        decoder_path = os.path.join(tmp_vae_dir, 'decoder')
                        vae.encoder.save_pretrained(encoder_path)
                        vae.decoder.save_pretrained(decoder_path)
                        
                        del vae
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        print("[INFO] ONNX Exporter: VAE split complete. Exporting sub-models...")
                        export_component("vae_encoder", encoder_path, final_output_dir, optimization_level, use_fp16, self.device, self.gpu_available)
                        export_component("vae_decoder", decoder_path, final_output_dir, optimization_level, use_fp16, self.device, self.gpu_available)

        except Exception as e:
            traceback.print_exc()
            return {"ui": {"text": [f"An unexpected error occurred: {e}"]}}

        return {"ui": {"text": [f"ONNX export process finished.\nModels saved in:\n{final_output_dir}"]}}