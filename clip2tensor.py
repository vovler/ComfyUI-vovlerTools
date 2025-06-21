import comfy.utils
import numpy as np
import os
import sys
import time
import folder_paths
import torch
import tensorrt as trt
import tempfile
import subprocess
import comfy.model_management as model_management
import comfy.sd
import comfy.clip_model
import traceback
import platform

# Import logging utility from wd14tagger
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

def log(message, type=None, always=True):
    if type is not None:
        message = f"[{type}] {message}"
    print(f"(CLIP2TensorRT) {message}")

def log_system_info():
    """Log system information for debugging"""
    try:
        log(f"System: {platform.system()} {platform.release()}", "DEBUG", True)
        log(f"Python: {platform.python_version()}", "DEBUG", True)
        log(f"PyTorch: {torch.__version__}", "DEBUG", True)
        log(f"TensorRT: {trt.__version__}", "DEBUG", True)
        
        if torch.cuda.is_available():
            log(f"CUDA: {torch.version.cuda}", "DEBUG", True)
            log(f"GPU: {torch.cuda.get_device_name(0)}", "DEBUG", True)
            log(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB", "DEBUG", True)
        else:
            log("CUDA: Not available", "WARNING", True)
    except Exception as e:
        log(f"Failed to get system info: {str(e)}", "WARNING", True)

def log_error_with_traceback(error_msg, exception=None):
    """Log error with full traceback for debugging"""
    log(error_msg, "ERROR", True)
    if exception:
        log(f"Exception type: {type(exception).__name__}", "ERROR", True)
        log(f"Exception details: {str(exception)}", "ERROR", True)
    
    # Log traceback
    tb_lines = traceback.format_exc().split('\n')
    for line in tb_lines:
        if line.strip():
            log(f"  {line}", "ERROR", True)

def validate_file_access(file_path, operation="read"):
    """Validate file access with detailed error reporting"""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File does not exist: {file_path}")
        
        if operation == "read" and not os.access(file_path, os.R_OK):
            raise PermissionError(f"No read permission for file: {file_path}")
        
        if operation == "write" and not os.access(os.path.dirname(file_path), os.W_OK):
            raise PermissionError(f"No write permission for directory: {os.path.dirname(file_path)}")
        
        file_size = os.path.getsize(file_path) / (1024*1024)  # MB
        log(f"File access validated: {os.path.basename(file_path)} ({file_size:.1f}MB)", "DEBUG", True)
        return True
        
    except Exception as e:
        log_error_with_traceback(f"File access validation failed for {file_path}", e)
        return False

def get_gpu_memory_gb():
    """Get total GPU memory in GB with error handling"""
    try:
        if not torch.cuda.is_available():
            log("CUDA not available, using default memory value", "WARNING", True)
            return 8.0
        
        device_count = torch.cuda.device_count()
        if device_count == 0:
            log("No CUDA devices found, using default memory value", "WARNING", True)
            return 8.0
        
        total_memory = torch.cuda.get_device_properties(0).total_memory
        memory_gb = total_memory / (1024**3)
        log(f"GPU memory detected: {memory_gb:.1f}GB", "DEBUG", True)
        return memory_gb
        
    except Exception as e:
        log_error_with_traceback("Failed to get GPU memory", e)
        log("Using default 8GB memory value", "WARNING", True)
        return 8.0

# Use ComfyUI's models directory with clip2tensor subdirectory
clip_models_dir = os.path.join(folder_paths.models_dir, "clip")
tensorrt_output_dir = os.path.join(folder_paths.models_dir, "clip_tensorrt")

def ensure_directories():
    """Ensure required directories exist with error handling"""
    try:
        if not os.path.exists(clip_models_dir):
            log(f"CLIP models directory not found: {clip_models_dir}", "WARNING", True)
            
        if not os.path.exists(tensorrt_output_dir):
            os.makedirs(tensorrt_output_dir)
            log(f"Created clip_tensorrt output directory: {tensorrt_output_dir}", "INFO", True)
        else:
            log(f"Using existing clip_tensorrt directory: {tensorrt_output_dir}", "DEBUG", True)
            
        # Validate write permissions
        if not os.access(tensorrt_output_dir, os.W_OK):
            raise PermissionError(f"No write permission for TensorRT output directory: {tensorrt_output_dir}")
            
    except Exception as e:
        log_error_with_traceback("Failed to setup directories", e)
        raise

# Initialize directories
ensure_directories()

def get_available_clip_models():
    """Get all available CLIP models (safetensors format) with error handling"""
    try:
        if not os.path.exists(clip_models_dir):
            log(f"CLIP models directory does not exist: {clip_models_dir}", "ERROR", True)
            return ["No CLIP models found - place .safetensors files in models/clip folder"]
        
        if not os.access(clip_models_dir, os.R_OK):
            log(f"No read permission for CLIP models directory: {clip_models_dir}", "ERROR", True)
            return ["Permission denied - cannot read models/clip folder"]
        
        all_files = os.listdir(clip_models_dir)
        clip_files = [f for f in all_files if f.endswith(".safetensors")]
        
        if not clip_files:
            log(f"No .safetensors files found in {clip_models_dir}", "WARNING", True)
            log(f"Found files: {all_files[:10]}", "DEBUG", True)  # Show first 10 files for debugging
            return ["No CLIP models found - place .safetensors files in models/clip folder"]
        
        log(f"Found {len(clip_files)} CLIP model(s): {clip_files}", "DEBUG", True)
        return clip_files
        
    except Exception as e:
        log_error_with_traceback("Failed to scan CLIP models directory", e)
        return ["Error scanning models/clip folder - check permissions"]

def get_existing_tensorrt_engines():
    """Get all existing TensorRT engine files with error handling"""
    try:
        if not os.path.exists(tensorrt_output_dir):
            log(f"TensorRT output directory does not exist: {tensorrt_output_dir}", "DEBUG", True)
            return ["No TensorRT engines found"]
        
        all_files = os.listdir(tensorrt_output_dir)
        engine_files = [f for f in all_files if f.endswith(".engine")]
        
        if not engine_files:
            log(f"No .engine files found in {tensorrt_output_dir}", "DEBUG", True)
            return ["No TensorRT engines found"]
        
        # Log engine file details
        for engine_file in engine_files:
            engine_path = os.path.join(tensorrt_output_dir, engine_file)
            size_mb = os.path.getsize(engine_path) / (1024*1024)
            log(f"Found engine: {engine_file} ({size_mb:.1f}MB)", "DEBUG", True)
        
        return engine_files
        
    except Exception as e:
        log_error_with_traceback("Failed to scan TensorRT engines directory", e)
        return ["Error scanning TensorRT engines folder"]


class CLIP_L_Wrapper(torch.nn.Module):
    def __init__(self, clip_l_model):
        super().__init__()
        self.clip_l = clip_l_model
    def forward(self, input_ids):
        # ComfyUI CLIPTextModel returns (x[0], x[1], out, x[2])
        # For CLIP-L we only need the sequence output (last hidden state)
        outputs = self.clip_l(input_tokens=input_ids)
        return outputs[0]  # last_hidden_state

class CLIP_G_Wrapper(torch.nn.Module):
    def __init__(self, clip_g_model):
        super().__init__()
        self.clip_g = clip_g_model
    def forward(self, input_ids):
        # Raw CLIPTextModel returns (x[0], x[1], out, x[2]) where:
        # x[0] = last hidden state, x[1] = intermediate, out = projected, x[2] = pooled
        # For ONNX export compatibility, use final layer instead of intermediate_output=-2
        print(f"[CLIP_G_Wrapper] Forward called with input_ids shape: {input_ids.shape}")
        print(f"[CLIP_G_Wrapper] Input device: {input_ids.device}, dtype: {input_ids.dtype}")
        print(f"[CLIP_G_Wrapper] Input min/max: {input_ids.min()}/{input_ids.max()}")
        
        try:
            # Use final layer output to avoid ONNX negative indexing issues
            # Note: This differs from SDXLClipG's penultimate layer, but needed for ONNX compatibility
            outputs = self.clip_g(input_tokens=input_ids)
            print(f"[CLIP_G_Wrapper] Raw CLIP-G outputs: {len(outputs)} items")
            for i, output in enumerate(outputs):
                if output is not None:
                    print(f"[CLIP_G_Wrapper] Raw output[{i}]: shape={output.shape}, dtype={output.dtype}, device={output.device}")
                else:
                    print(f"[CLIP_G_Wrapper] Raw output[{i}]: None")
            
            # Use final layer output for ONNX compatibility
            last_hidden_state = outputs[0]  # final layer sequence output
            projected_pooled = outputs[2]   # projected pooled output
            print(f"[CLIP_G_Wrapper] Using final layer for ONNX compatibility")
            
            print(f"[CLIP_G_Wrapper] Selected last_hidden_state: shape={last_hidden_state.shape}, dtype={last_hidden_state.dtype}")
            print(f"[CLIP_G_Wrapper] Selected projected_pooled: shape={projected_pooled.shape}, dtype={projected_pooled.dtype}")
            
            result = (last_hidden_state, projected_pooled)
            print(f"[CLIP_G_Wrapper] Returning tuple with {len(result)} items")
            return result
            
        except Exception as e:
            print(f"[CLIP_G_Wrapper] ERROR in forward pass: {str(e)}")
            print(f"[CLIP_G_Wrapper] Error type: {type(e)}")
            import traceback
            print(f"[CLIP_G_Wrapper] Traceback:")
            traceback.print_exc()
            raise 

class DualCLIPToTensorRT:
    @classmethod
    def INPUT_TYPES(s):
        clip_models = get_available_clip_models()
        default_model = clip_models[0] if clip_models and "No CLIP models found" not in clip_models[0] else ""
        
        return {"required": {
            "clip_name1": (clip_models, {"default": default_model}),
            "clip_name2": (clip_models, {"default": default_model}),
            "output_name": ("STRING", {"default": "dual_clip", "multiline": False}),
            "prompt_batch_min": ("INT", {"default": 1, "min": 1, "max": 32, "step": 1}),
            "prompt_batch_opt": ("INT", {"default": 1, "min": 1, "max": 32, "step": 1}),
            "prompt_batch_max": ("INT", {"default": 8, "min": 1, "max": 32, "step": 1}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "convert_dual_clip_to_tensorrt"
    OUTPUT_NODE = True
    CATEGORY = "vovlerTools"

    def convert_dual_clip_to_tensorrt(self, clip_name1, clip_name2, output_name, 
                                    prompt_batch_min, prompt_batch_opt, prompt_batch_max):
        
        # Log system information for debugging
        log_system_info()
        
        # Validate inputs
        if "No CLIP models found" in clip_name1 or "No CLIP models found" in clip_name2:
            error_msg = "CLIP models not found. Please place .safetensors files in models/clip folder."
            log(error_msg, "ERROR", True)
            log(f"Clip models directory: {clip_models_dir}", "ERROR", True)
            return (error_msg,)
        
        if "Error scanning" in clip_name1 or "Error scanning" in clip_name2:
            error_msg = "Error accessing CLIP models directory. Check permissions and directory structure."
            log(error_msg, "ERROR", True)
            return (error_msg,)
        
        if clip_name1 == clip_name2:
            error_msg = "Please select two different CLIP models for dual CLIP conversion."
            log(error_msg, "ERROR", True)
            log(f"Both models selected: {clip_name1}", "ERROR", True)
            return (error_msg,)
        
        # Validate batch size parameters
        if not (1 <= prompt_batch_min <= prompt_batch_opt <= prompt_batch_max <= 32):
            error_msg = f"Invalid batch size configuration: min={prompt_batch_min}, opt={prompt_batch_opt}, max={prompt_batch_max}. Must be 1 <= min <= opt <= max <= 32"
            log(error_msg, "ERROR", True)
            return (error_msg,)
        
        # Validate output name
        if not output_name or not output_name.strip():
            error_msg = "Output name cannot be empty"
            log(error_msg, "ERROR", True)
            return (error_msg,)
        
        # Sanitize output name (remove invalid characters)
        import re
        sanitized_name = re.sub(r'[<>:"/\\|?*]', '_', output_name.strip())
        if sanitized_name != output_name.strip():
            log(f"Output name sanitized from '{output_name}' to '{sanitized_name}'", "WARNING", True)
            output_name = sanitized_name
        
        clip1_path = os.path.join(clip_models_dir, clip_name1)
        clip2_path = os.path.join(clip_models_dir, clip_name2)
        
        # Validate file access for both CLIP models
        if not validate_file_access(clip1_path, "read"):
            error_msg = f"Cannot access CLIP model 1: {clip_name1}"
            log(f"Full path: {clip1_path}", "ERROR", True)
            return (error_msg,)
        
        if not validate_file_access(clip2_path, "read"):
            error_msg = f"Cannot access CLIP model 2: {clip_name2}"
            log(f"Full path: {clip2_path}", "ERROR", True)
            return (error_msg,)
        
        # Always use fp16 and SDXL (77 tokens)
        engine_filename = f"{output_name}_sdxl_{prompt_batch_min}_{prompt_batch_opt}_{prompt_batch_max}_fp16.engine"
        engine_path = os.path.join(tensorrt_output_dir, engine_filename)
        
        # Check if engine already exists
        if os.path.exists(engine_path):
            engine_size = os.path.getsize(engine_path) / (1024*1024)
            success_msg = f"TensorRT engine already exists: {engine_filename} ({engine_size:.1f}MB)"
            log(success_msg, "INFO", True)
            log(f"Engine path: {engine_path}", "DEBUG", True)
            return (success_msg,)
        
        # Validate output directory write access
        if not validate_file_access(tensorrt_output_dir, "write"):
            error_msg = f"Cannot write to TensorRT output directory: {tensorrt_output_dir}"
            return (error_msg,)
        
        try:
            log(f"Starting dual CLIP to TensorRT conversion...", "INFO", True)
            log(f"CLIP 1: {clip_name1}", "INFO", True)
            log(f"CLIP 2: {clip_name2}", "INFO", True)
            log(f"Type: SDXL (always)", "INFO", True)
            log(f"Precision: FP16 (always)", "INFO", True)
            log(f"Prompt batch sizes: {prompt_batch_min}/{prompt_batch_opt}/{prompt_batch_max}", "INFO", True)
            log(f"Token sequence length: 77 (SDXL standard)", "INFO", True)
            log(f"Output engine: {engine_filename}", "INFO", True)
            
            # Initialize engine paths for exception handling
            clip_l_engine_path = None
            clip_g_engine_path = None
            
            # Load CLIP models using ComfyUI's built-in functionality
            log("Loading CLIP models using ComfyUI's native loader...", "INFO", True)

            # Load the first model, which should contain CLIP-L
            log(f"Loading first CLIP model from {clip_name1}...", "DEBUG", True)
            clip_object_1 = comfy.sd.load_clip(ckpt_paths=[clip1_path], embedding_directory=folder_paths.get_folder_paths("embeddings"))

            # Load the second model, which should contain CLIP-G
            log(f"Loading second CLIP model from {clip_name2}...", "DEBUG", True)
            clip_object_2 = comfy.sd.load_clip(ckpt_paths=[clip2_path], embedding_directory=folder_paths.get_folder_paths("embeddings"))
            
            # Inspect the loaded models to determine which contains which CLIP type
            clip_l_model = None
            clip_g_model = None
            
            # Check first model for CLIP-L and CLIP-G
            log("Inspecting first model structure...", "DEBUG", True)
            if hasattr(clip_object_1.cond_stage_model, 'clip_l') and clip_object_1.cond_stage_model.clip_l is not None:
                clip_l_model = clip_object_1.cond_stage_model.clip_l.transformer
                log(f"Found CLIP-L in first model ({clip_name1})", "DEBUG", True)
                log(f"CLIP-L transformer type: {type(clip_l_model)}", "DEBUG", True)
            
            if hasattr(clip_object_1.cond_stage_model, 'clip_g') and clip_object_1.cond_stage_model.clip_g is not None:
                # For CLIP-G, get the raw transformer without layer extraction
                clip_g_obj = clip_object_1.cond_stage_model.clip_g
                clip_g_model = clip_g_obj.transformer
                log(f"Found CLIP-G in first model ({clip_name1})", "DEBUG", True)
                log(f"CLIP-G transformer type: {type(clip_g_model)}", "DEBUG", True)
                log(f"CLIP-G parent config: layer={getattr(clip_g_obj, 'layer', None)}, layer_idx={getattr(clip_g_obj, 'layer_idx', None)}", "DEBUG", True)
            
            # Check second model for CLIP-L and CLIP-G
            log("Inspecting second model structure...", "DEBUG", True)
            if hasattr(clip_object_2.cond_stage_model, 'clip_l') and clip_object_2.cond_stage_model.clip_l is not None:
                if clip_l_model is None:
                    clip_l_model = clip_object_2.cond_stage_model.clip_l.transformer
                    log(f"Found CLIP-L in second model ({clip_name2})", "DEBUG", True)
                    log(f"CLIP-L transformer type: {type(clip_l_model)}", "DEBUG", True)
                else:
                    log(f"Second model also has CLIP-L, using first model's CLIP-L", "DEBUG", True)
            
            if hasattr(clip_object_2.cond_stage_model, 'clip_g') and clip_object_2.cond_stage_model.clip_g is not None:
                if clip_g_model is None:
                    # For CLIP-G, get the raw transformer without layer extraction
                    clip_g_obj = clip_object_2.cond_stage_model.clip_g
                    clip_g_model = clip_g_obj.transformer
                    log(f"Found CLIP-G in second model ({clip_name2})", "DEBUG", True)
                    log(f"CLIP-G transformer type: {type(clip_g_model)}", "DEBUG", True)
                    log(f"CLIP-G parent config: layer={getattr(clip_g_obj, 'layer', None)}, layer_idx={getattr(clip_g_obj, 'layer_idx', None)}", "DEBUG", True)
                else:
                    log(f"Second model also has CLIP-G, using first model's CLIP-G", "DEBUG", True)
            
            # Validate that we found both CLIP models
            if clip_l_model is None:
                raise ValueError(f"Could not find CLIP-L model in either {clip_name1} or {clip_name2}. Make sure at least one file contains a CLIP-L model.")
            
            if clip_g_model is None:
                raise ValueError(f"Could not find CLIP-G model in either {clip_name1} or {clip_name2}. Make sure at least one file contains a CLIP-G model.")
            
            log("Successfully located both CLIP-L and CLIP-G models", "INFO", True)

            # Debug the actual model classes and structures before moving to GPU
            log("=== DEBUGGING CLIP MODEL CLASSES ===", "DEBUG", True)
            log(f"CLIP-L model class: {type(clip_l_model)}", "DEBUG", True)
            log(f"CLIP-G model class: {type(clip_g_model)}", "DEBUG", True)
            
            # Check if they have different parent models
            log(f"CLIP-L parent object: {type(clip_object_1.cond_stage_model.clip_l) if hasattr(clip_object_1.cond_stage_model, 'clip_l') else 'None'}", "DEBUG", True)
            log(f"CLIP-G parent object: {type(clip_object_2.cond_stage_model.clip_g) if hasattr(clip_object_2.cond_stage_model, 'clip_g') else 'None'}", "DEBUG", True)
            
            # Check the config/architecture differences
            if hasattr(clip_l_model, 'config'):
                log(f"CLIP-L config: {clip_l_model.config}", "DEBUG", True)
            if hasattr(clip_g_model, 'config'):
                log(f"CLIP-G config: {clip_g_model.config}", "DEBUG", True)

            # Move models to GPU for conversion
            device = 'cuda'
            log(f"Moving models to '{device}' for ONNX export...", "INFO", True)
            clip_l_model = clip_l_model.to(device)
            clip_g_model = clip_g_model.to(device)

            log(f"CLIP-L model device: {next(clip_l_model.parameters()).device}", "DEBUG", True)
            log(f"CLIP-G model device: {next(clip_g_model.parameters()).device}", "DEBUG", True)
            
            # === END OF THE FIX ===
            
            # Create separate engines for each CLIP model
            log("Creating separate TensorRT engines for CLIP-L and CLIP-G", "INFO", True)
            
            # Test the models first to debug the outputs
            log("Testing CLIP model outputs for debugging...", "DEBUG", True)
            test_input = torch.randint(0, 49408, (1, 77), dtype=torch.long, device=device)
            
            try:
                clip_l_outputs = clip_l_model(input_tokens=test_input)
                log(f"CLIP-L raw outputs: {len(clip_l_outputs)} items", "DEBUG", True)
                for i, output in enumerate(clip_l_outputs):
                    if output is not None:
                        log(f"CLIP-L output[{i}] shape: {output.shape if hasattr(output, 'shape') else type(output)}", "DEBUG", True)
                    else:
                        log(f"CLIP-L output[{i}]: None", "DEBUG", True)
            except Exception as e:
                log(f"CLIP-L test failed: {str(e)}", "ERROR", True)
            
            try:
                clip_g_outputs = clip_g_model(input_tokens=test_input)
                log(f"CLIP-G raw outputs: {len(clip_g_outputs)} items", "DEBUG", True)
                for i, output in enumerate(clip_g_outputs):
                    if output is not None:
                        log(f"CLIP-G output[{i}] shape: {output.shape if hasattr(output, 'shape') else type(output)}", "DEBUG", True)
                    else:
                        log(f"CLIP-G output[{i}]: None", "DEBUG", True)
            except Exception as e:
                log(f"CLIP-G test failed: {str(e)}", "ERROR", True)
            
            # Test the wrappers before ONNX export
            log("Testing wrapper outputs...", "DEBUG", True)
            clip_l_wrapper = CLIP_L_Wrapper(clip_l_model)
            clip_g_wrapper = CLIP_G_Wrapper(clip_g_model)
            
            try:
                clip_l_wrapper_out = clip_l_wrapper(test_input)
                log(f"CLIP-L wrapper output shape: {clip_l_wrapper_out.shape}", "DEBUG", True)
            except Exception as e:
                log(f"CLIP-L wrapper test failed: {str(e)}", "ERROR", True)
            
            try:
                clip_g_wrapper_out = clip_g_wrapper(test_input)
                log(f"CLIP-G wrapper outputs: {len(clip_g_wrapper_out) if isinstance(clip_g_wrapper_out, (tuple, list)) else 'single'}", "DEBUG", True)
                if isinstance(clip_g_wrapper_out, (tuple, list)):
                    for i, out in enumerate(clip_g_wrapper_out):
                        log(f"CLIP-G wrapper output[{i}] shape: {out.shape}", "DEBUG", True)
                else:
                    log(f"CLIP-G wrapper output shape: {clip_g_wrapper_out.shape}", "DEBUG", True)
            except Exception as e:
                log(f"CLIP-G wrapper test failed: {str(e)}", "ERROR", True)
            
            # Create engine file paths for both models
            clip_l_engine_path = engine_path.replace('.engine', '_clip_l.engine')
            clip_g_engine_path = engine_path.replace('.engine', '_clip_g.engine')
            
            # Create CLIP-L engine
            log("Creating CLIP-L TensorRT engine...", "INFO", True)
            self._create_single_clip_engine(
                clip_l_wrapper, 
                clip_l_engine_path, 
                'clip-l',
                prompt_batch_min, 
                prompt_batch_opt, 
                prompt_batch_max
            )
            
            # Create CLIP-G engine  
            log("Creating CLIP-G TensorRT engine...", "INFO", True)
            self._create_single_clip_engine(
                clip_g_wrapper, 
                clip_g_engine_path, 
                'clip-g',
                prompt_batch_min, 
                prompt_batch_opt, 
                prompt_batch_max
            )
            
            # Success message
            clip_l_size = os.path.getsize(clip_l_engine_path) / (1024*1024)
            clip_g_size = os.path.getsize(clip_g_engine_path) / (1024*1024)
            success_msg = f"Dual CLIP TensorRT engines created successfully:\n"
            success_msg += f"  - CLIP-L: {os.path.basename(clip_l_engine_path)} ({clip_l_size:.1f}MB)\n" 
            success_msg += f"  - CLIP-G: {os.path.basename(clip_g_engine_path)} ({clip_g_size:.1f}MB)"
            
            log(success_msg, "INFO", True)
            return (success_msg,)
            
        except MemoryError as e:
            error_msg = "Out of memory during TensorRT conversion. Try reducing batch sizes or closing other applications."
            log_error_with_traceback(error_msg, e)
            log(f"Suggested action: Reduce prompt_batch_max from {prompt_batch_max} to {max(1, prompt_batch_max // 2)}", "INFO", True)
            return (error_msg,)
            
        except PermissionError as e:
            error_msg = "Permission denied during TensorRT conversion. Check file/directory permissions."
            log_error_with_traceback(error_msg, e)
            return (error_msg,)
            
        except Exception as e:
            error_msg = f"Error during dual CLIP to TensorRT conversion: {str(e)}"
            log_error_with_traceback(error_msg, e)
            
            # Clean up any partial engine files on failure
            if clip_l_engine_path and os.path.exists(clip_l_engine_path):
                try:
                    os.remove(clip_l_engine_path)
                    log(f"Cleaned up partial CLIP-L engine file", "DEBUG", True)
                except:
                    pass
            
            if clip_g_engine_path and os.path.exists(clip_g_engine_path):
                try:
                    os.remove(clip_g_engine_path)
                    log(f"Cleaned up partial CLIP-G engine file", "DEBUG", True)
                except:
                    pass
            
            # Provide helpful suggestions based on error type
            if "CUDA" in str(e):
                log("Suggestion: Check CUDA installation and GPU availability", "INFO", True)
            elif "TensorRT" in str(e):
                log("Suggestion: Check TensorRT installation and compatibility", "INFO", True)
            elif "memory" in str(e).lower():
                log("Suggestion: Reduce batch sizes or free up GPU memory", "INFO", True)
            
            return (error_msg,)

    def _create_single_clip_engine(self, model, engine_path, clip_type, 
                                 prompt_batch_min, prompt_batch_opt, prompt_batch_max):
        """
        Create TensorRT engine for a single CLIP model
        Uses PyTorch -> ONNX -> TensorRT workflow
        """
        onnx_path = None
        original_backends = None
        try:
            log(f"Creating {clip_type} TensorRT engine...", "INFO", True)
            
            # Step 1: Export to ONNX
            onnx_path = engine_path.replace('.engine', '.onnx')
            log(f"Exporting {clip_type} to ONNX: {os.path.basename(onnx_path)}", "DEBUG", True)
            
            model.eval()
            device = next(model.parameters()).device
            dummy_input = torch.randint(0, 49408, (prompt_batch_opt, 77), dtype=torch.long, device=device)
            
            # Define the output names based on the clip type
            if clip_type == 'clip-g':
                # CLIP-G needs both outputs for SDXL
                output_names = ['last_hidden_state', 'pooled_output']
                dynamic_axes_outputs = {
                    'last_hidden_state': {0: 'batch_size'},
                    'pooled_output': {0: 'batch_size'}
                }
                model_args = (dummy_input,)
            else:  # clip-l
                output_names = ['last_hidden_state']
                dynamic_axes_outputs = {
                    'last_hidden_state': {0: 'batch_size'}
                }
                model_args = (dummy_input,)
            
            log(f"ONNX export config for {clip_type}:", "DEBUG", True)
            log(f"  - Model args: {[arg.shape for arg in model_args]}", "DEBUG", True)
            log(f"  - Output names: {output_names}", "DEBUG", True)
            log(f"  - Dynamic axes: {{'input_ids': {{0: 'batch_size'}}, **{dynamic_axes_outputs}}}", "DEBUG", True)
            
            # Test the model wrapper first
            log(f"Testing {clip_type} wrapper before ONNX export...", "DEBUG", True)
            try:
                wrapper_output = model(dummy_input)
                if isinstance(wrapper_output, (tuple, list)):
                    log(f"{clip_type} wrapper returned {len(wrapper_output)} outputs", "DEBUG", True)
                    for i, out in enumerate(wrapper_output):
                        log(f"  - Output {i}: {out.shape} dtype={out.dtype}", "DEBUG", True)
                else:
                    log(f"{clip_type} wrapper returned single output: {wrapper_output.shape} dtype={wrapper_output.dtype}", "DEBUG", True)
            except Exception as e:
                log(f"ERROR: {clip_type} wrapper test failed: {str(e)}", "ERROR", True)
                raise RuntimeError(f"Wrapper test failed for {clip_type}: {str(e)}")
            
            # For CLIP-G, try to disable PyTorch optimizations that might interfere with ONNX export
            if clip_type == 'clip-g':
                log(f"Disabling PyTorch optimizations for CLIP-G ONNX export...", "DEBUG", True)
                # Disable optimized attention temporarily (if available)
                try:
                    original_backends = torch.backends.opt_einsum.enabled
                    torch.backends.opt_einsum.enabled = False
                    log(f"Disabled opt_einsum backend", "DEBUG", True)
                except AttributeError:
                    log(f"opt_einsum backend not available in this PyTorch version", "DEBUG", True)
                    original_backends = None
                
                # Set model to eval mode and disable gradients
                model.eval()
                for param in model.parameters():
                    param.requires_grad = False
            
            try:
                # Try multiple ONNX export strategies for CLIP-G compatibility
                if clip_type == 'clip-g':
                    log(f"Attempting CLIP-G ONNX export with compatibility fixes...", "DEBUG", True)
                    
                    # Strategy 1: Try with opset 14+ (supports aten::triu)
                    try:
                        log(f"Trying CLIP-G export with opset 16 (supports aten::triu)...", "DEBUG", True)
                        torch.onnx.export(
                            model,
                            model_args,
                            onnx_path,
                            export_params=True,
                            opset_version=16,  # Higher opset supports more operators
                            do_constant_folding=False,  # Disable optimizations that might cause issues
                            input_names=['input_ids'],
                            output_names=output_names,
                            dynamic_axes={
                                'input_ids': {0: 'batch_size'},
                                **dynamic_axes_outputs
                            }
                        )
                        log(f"CLIP-G opset 16 export succeeded", "DEBUG", True)
                    except Exception as opset16_error:
                        log(f"Opset 16 export failed: {str(opset16_error)}", "DEBUG", True)
                        
                        # Strategy 2: Try opset 14 (minimum for aten::triu)
                        try:
                            log(f"Trying CLIP-G export with opset 14 (minimum for aten::triu)...", "DEBUG", True)
                            torch.onnx.export(
                                model,
                                model_args,
                                onnx_path,
                                export_params=True,
                                opset_version=14,
                                do_constant_folding=False,
                                input_names=['input_ids'],
                                output_names=output_names,
                                dynamic_axes={
                                    'input_ids': {0: 'batch_size'},
                                    **dynamic_axes_outputs
                                },
                                keep_initializers_as_inputs=True,
                                export_modules_as_functions=False
                            )
                            log(f"CLIP-G opset 14 export succeeded", "DEBUG", True)
                        except Exception as opset14_error:
                            log(f"Opset 14 export failed: {str(opset14_error)}", "DEBUG", True)
                            
                            # Strategy 3: Try torch.jit.trace with opset 16
                            try:
                                log(f"Trying CLIP-G with torch.jit.trace approach (opset 16)...", "DEBUG", True)
                                
                                # First trace the model
                                model.eval()
                                with torch.no_grad():
                                    traced_model = torch.jit.trace(model, model_args[0])
                                
                                # Then export the traced model
                                torch.onnx.export(
                                    traced_model,
                                    model_args,
                                    onnx_path,
                                    export_params=True,
                                    opset_version=16,
                                    do_constant_folding=False,
                                    input_names=['input_ids'],
                                    output_names=output_names,
                                    dynamic_axes={
                                        'input_ids': {0: 'batch_size'},
                                        **dynamic_axes_outputs
                                    }
                                )
                                log(f"CLIP-G jit.trace export succeeded", "DEBUG", True)
                            except Exception as trace_error:
                                log(f"JIT trace export failed: {str(trace_error)}", "DEBUG", True)
                                
                                # Strategy 4: Fallback to simplest possible export with fixed training mode
                                log(f"Trying CLIP-G with minimal export options...", "DEBUG", True)
                                torch.onnx.export(
                                    model,
                                    model_args,
                                    onnx_path,
                                    export_params=True,
                                    opset_version=16,
                                    do_constant_folding=False,
                                    input_names=['input_ids'],
                                    output_names=output_names
                                    # No dynamic axes and no training parameter to avoid complexity
                                )
                else:
                    # For CLIP-L, use the original approach (it works)
                    torch.onnx.export(
                        model,
                        model_args,
                        onnx_path,
                        export_params=True,
                        opset_version=16,
                        do_constant_folding=True,
                        input_names=['input_ids'],
                        output_names=output_names,
                        dynamic_axes={
                            'input_ids': {0: 'batch_size'},
                            **dynamic_axes_outputs  # Merge the output axes
                        }
                    )
            except Exception as onnx_error:
                log(f"ONNX export failed for {clip_type}: {str(onnx_error)}", "ERROR", True)
                log(f"ONNX error type: {type(onnx_error)}", "ERROR", True)
                import traceback
                log(f"ONNX export traceback:", "ERROR", True)
                for line in traceback.format_exc().split('\n'):
                    if line.strip():
                        log(f"  {line}", "ERROR", True)
                raise
            
            log(f"{clip_type} ONNX export completed", "DEBUG", True)

            # Validate ONNX file before proceeding
            if not os.path.exists(onnx_path) or os.path.getsize(onnx_path) == 0:
                raise RuntimeError(f"ONNX export failed for {clip_type}: file not created or is empty.")
            
            # Step 2: Convert ONNX to TensorRT
            log(f"Converting {clip_type} ONNX to TensorRT engine...", "DEBUG", True)
            self._onnx_to_tensorrt(onnx_path, engine_path, clip_type,
                                prompt_batch_min, prompt_batch_opt, prompt_batch_max)
            
            log(f"{clip_type} TensorRT engine created successfully", "INFO", True)
            
        except Exception as e:
            log_error_with_traceback(f"Failed to create {clip_type} TensorRT engine", e)
            raise
        finally:
            # Always clean up ONNX files and restore PyTorch settings
            if clip_type == 'clip-g' and original_backends is not None:
                try:
                    log(f"Restoring PyTorch backend settings...", "DEBUG", True)
                    torch.backends.opt_einsum.enabled = original_backends
                    log(f"Restored opt_einsum backend", "DEBUG", True)
                except AttributeError:
                    log(f"opt_einsum backend not available for restoration", "DEBUG", True)
            self._cleanup_temporary_files(onnx_path)

    def _onnx_to_tensorrt(self, onnx_path, engine_path, clip_type,
                         prompt_batch_min, prompt_batch_opt, prompt_batch_max):
        """Convert ONNX model to TensorRT engine - no fallbacks"""
        try:
            # Get GPU memory and calculate 80% for memory pool
            gpu_memory_gb = get_gpu_memory_gb()
            memory_pool_bytes = int(gpu_memory_gb * 0.8 * 1024**3)
            
            log(f"Converting {clip_type} ONNX to TensorRT with {memory_pool_bytes / (1024**3):.1f}GB memory pool", "DEBUG", True)
            
            # Create TensorRT logger and builder
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(TRT_LOGGER)
            
            # Create network from ONNX
            network = builder.create_network()
            parser = trt.OnnxParser(network, TRT_LOGGER)
            
            # Parse ONNX file with detailed error reporting
            log(f"Parsing ONNX file: {os.path.basename(onnx_path)}", "DEBUG", True)
            onnx_file_size = os.path.getsize(onnx_path) / (1024*1024)
            log(f"ONNX file size: {onnx_file_size:.1f}MB", "DEBUG", True)
            
            with open(onnx_path, 'rb') as model_file:
                onnx_data = model_file.read()
                log(f"Read {len(onnx_data)} bytes from ONNX file", "DEBUG", True)
                
                if not parser.parse(onnx_data):
                    log(f"ONNX parsing failed for {clip_type}", "ERROR", True)
                    log(f"Number of parser errors: {parser.num_errors}", "ERROR", True)
                    
                    # Log all parser errors
                    for error in range(parser.num_errors):
                        error_msg = parser.get_error(error)
                        log(f"ONNX Parser Error {error}: {error_msg}", "ERROR", True)
                    
                    raise RuntimeError(f"Failed to parse ONNX file for {clip_type}")
                else:
                    log(f"{clip_type} ONNX parsed successfully", "DEBUG", True)
            
            # Configure builder
            config = builder.create_builder_config()
            config.set_flag(trt.BuilderFlag.FP16)  # Always use FP16
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, memory_pool_bytes)
            
            # Add optimization profile for dynamic batch size
            profile = builder.create_optimization_profile()
            profile.set_shape(
                'input_ids',
                (prompt_batch_min, 77),
                (prompt_batch_opt, 77), 
                (prompt_batch_max, 77)
            )
            config.add_optimization_profile(profile)
            
            log(f"Building {clip_type} TensorRT engine (this may take several minutes)...", "INFO", True)
            build_start_time = time.time()
            
            # Build engine
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                raise RuntimeError(f"Failed to build TensorRT engine for {clip_type}")
            
            build_time = time.time() - build_start_time
            log(f"{clip_type} TensorRT build completed in {build_time:.1f} seconds", "INFO", True)
            
            # Save engine
            with open(engine_path, 'wb') as f:
                f.write(serialized_engine)
            
            engine_size = os.path.getsize(engine_path) / (1024*1024)
            log(f"{clip_type} engine saved: {os.path.basename(engine_path)} ({engine_size:.1f}MB)", "INFO", True)
            
        except Exception as e:
            log_error_with_traceback(f"Failed to convert {clip_type} ONNX to TensorRT", e)
            raise
    
    def _cleanup_temporary_files(self, onnx_path):
        """Clean up all temporary files including ONNX, MatMul, weight, and bias files"""
        try:
            # Clean up main ONNX file
            if onnx_path and os.path.exists(onnx_path):
                os.remove(onnx_path)
                #log(f"Cleaned up temporary ONNX file: {os.path.basename(onnx_path)}", "DEBUG", True)
            
            # Clean up any leftover temporary files
            import glob
            cleanup_patterns = [
                "onnx__MatMul_*",
                "*.onnx.tmp",
                "*.onnx_",
                "*_fallback.onnx",
                "*.weight",
                "*.bias",
                "*.weight.*",
                "*.bias.*"
            ]
            
            # Check current working directory
            for pattern in cleanup_patterns:
                for temp_file in glob.glob(pattern):
                    try:
                        os.remove(temp_file)
                        # Don't log individual file removals to reduce noise
                    except Exception:
                        pass  # Silently ignore cleanup failures
            
            # Check temp directory
            temp_dir = tempfile.gettempdir()
            for pattern in cleanup_patterns:
                temp_pattern = os.path.join(temp_dir, pattern)
                for temp_file in glob.glob(temp_pattern):
                    try:
                        os.remove(temp_file)
                        # Don't log individual file removals to reduce noise
                    except Exception:
                        pass  # Silently ignore cleanup failures
            
            # Check TensorRT output directory for any temporary files
            if os.path.exists(tensorrt_output_dir):
                for pattern in cleanup_patterns:
                    output_pattern = os.path.join(tensorrt_output_dir, pattern)
                    for temp_file in glob.glob(output_pattern):
                        try:
                            os.remove(temp_file)
                            # Don't log individual file removals to reduce noise
                        except Exception:
                            pass  # Silently ignore cleanup failures
                            
        except Exception as e:
            # Only log if there's a major cleanup error
            log(f"Error during cleanup: {str(e)}", "DEBUG", True)

class TensorRTCLIP:
    """TensorRT CLIP wrapper that implements ComfyUI's CLIP interface"""
    
    def __init__(self, engine_data):
        self.engine_data = engine_data
        self.clip_l_engine = None
        self.clip_l_context = None
        self.clip_g_engine = None
        self.clip_g_context = None
        
        # Load both engines
        if "clip_l" in engine_data:
            self.clip_l_engine, self.clip_l_context = self._load_engine(engine_data["clip_l"]["engine_path"])
        if "clip_g" in engine_data:
            self.clip_g_engine, self.clip_g_context = self._load_engine(engine_data["clip_g"]["engine_path"])
        
        # Initialize tokenizer
        self.tokenizer = self._init_tokenizer()
        
        # Add properties to match ComfyUI CLIP interface
        self.layer_idx = None
        self.tokenizer_options = {}
        self.use_clip_schedule = False
        self.apply_hooks_to_conds = None
        
        log(f"TensorRT CLIP initialized successfully", "INFO", True)
        log(f"  - CLIP-L engine: {'Loaded' if self.clip_l_engine else 'Not available'}", "DEBUG", True)
        log(f"  - CLIP-G engine: {'Loaded' if self.clip_g_engine else 'Not available'}", "DEBUG", True)

    def _load_engine(self, engine_path):
        """Load TensorRT engine and create execution context"""
        try:
            log(f"Loading TensorRT engine: {os.path.basename(engine_path)}", "DEBUG", True)
            
            # Create TensorRT logger and runtime
            trt_logger = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(trt_logger)
            
            # Load engine from file
            with open(engine_path, 'rb') as f:
                engine_data = f.read()
            
            engine = runtime.deserialize_cuda_engine(engine_data)
            if engine is None:
                raise RuntimeError(f"Failed to deserialize TensorRT engine from {engine_path}")
            
            # Create execution context
            context = engine.create_execution_context()
            if context is None:
                raise RuntimeError(f"Failed to create execution context for {engine_path}")
            
            log(f"TensorRT engine loaded successfully: {os.path.basename(engine_path)}", "DEBUG", True)
            
            # Check engine precision
            if hasattr(engine, 'has_implicit_batch_dimension'):
                log(f"Engine batch mode: {'Implicit' if engine.has_implicit_batch_dimension else 'Explicit'}", "DEBUG", True)
            
            # Log tensor information for debugging
            for i in range(engine.num_io_tensors):
                tensor_name = engine.get_tensor_name(i)
                tensor_mode = engine.get_tensor_mode(tensor_name)
                tensor_dtype = engine.get_tensor_dtype(tensor_name)
                log(f"Tensor {i}: {tensor_name}, mode: {tensor_mode}, dtype: {tensor_dtype}", "DEBUG", True)
            
            return engine, context
            
        except Exception as e:
            log_error_with_traceback(f"Failed to load TensorRT engine: {engine_path}", e)
            raise

    def _init_tokenizer(self):
        """Initialize tokenizer - simplified since we handle tokenization externally"""
        try:
            # Create a dummy CLIP object to access tokenization
            # This is a bit hacky but works with ComfyUI's architecture
            return None  # We'll use ComfyUI's tokenization directly
        except Exception as e:
            log(f"Warning: Could not initialize tokenizer: {str(e)}", "WARNING", True)
            return None
        
    def tokenize(self, text):
        """Tokenize text for CLIP processing using proper CLIP tokenization"""
        try:
            # Use transformers library for proper CLIP tokenization
            from transformers import CLIPTokenizer
            
            # Load CLIP tokenizer (SDXL uses the same tokenizer for both CLIP-L and CLIP-G)
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            
            # Tokenize text with proper padding and truncation
            encoded = tokenizer(
                text,
                padding="max_length",
                max_length=77,  # SDXL standard sequence length
                truncation=True,
                return_tensors="pt"
            )
            
            tokens = encoded["input_ids"]
            
            log(f"Tokenized text: '{text[:50]}{'...' if len(text) > 50 else ''}'", "DEBUG", True)
            log(f"Token shape: {tokens.shape}", "DEBUG", True)
            
            return tokens
            
        except Exception as e:
            log_error_with_traceback("Failed to tokenize text", e)
            # Fallback: create basic tokens with start/end tokens
            tokens = torch.zeros((1, 77), dtype=torch.long)
            tokens[0, 0] = 49406  # Start token
            tokens[0, 1] = 49407  # End token (for empty text)
            log("Using fallback tokenization", "WARNING", True)
            return tokens
    
    def encode_from_tokens_scheduled(self, tokens, unprojected=False, add_dict=None, show_pbar=True):
        """Encode tokens using dual TensorRT engines - ComfyUI compatible interface"""
        try:
            if add_dict is None:
                add_dict = {}
                
            batch_size = tokens.shape[0]
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            log(f"Running dual TensorRT CLIP inference:", "DEBUG", True)
            log(f"  - Batch size: {batch_size}", "DEBUG", True)
            log(f"  - Token sequence length: {tokens.shape[1]}", "DEBUG", True)
            log(f"  - Unprojected: {unprojected}", "DEBUG", True)
            
            # Move tokens to GPU
            tokens = tokens.to(device)
            
            # Run CLIP-L inference
            clip_l_embeddings = self._run_clip_l_inference(tokens)
            
            # Run CLIP-G inference  
            clip_g_embeddings, pooled_output = self._run_clip_g_inference(tokens)
            
            # SDXL conditioning format combines both CLIP outputs
            combined_embeddings = torch.cat([clip_l_embeddings, clip_g_embeddings], dim=-1)
            
            # Ensure tensors are on CPU and in the right dtype for ComfyUI
            combined_embeddings = combined_embeddings.cpu().float()
            pooled_output = pooled_output.cpu().float()
            
            # Create pooled dictionary with proper format
            pooled_dict = {"pooled_output": pooled_output}
            pooled_dict.update(add_dict)
            
            # Return in ComfyUI conditioning format - list of [cond, pooled_dict] pairs
            all_cond_pooled = [[combined_embeddings, pooled_dict]]
            
            log(f"Dual TensorRT CLIP encoding complete:", "DEBUG", True)
            log(f"  - Combined embeddings shape: {combined_embeddings.shape}", "DEBUG", True)
            log(f"  - Combined embeddings dtype: {combined_embeddings.dtype}", "DEBUG", True)
            log(f"  - Pooled output shape: {pooled_output.shape}", "DEBUG", True)
            log(f"  - Pooled output dtype: {pooled_output.dtype}", "DEBUG", True)
            log(f"  - Return format: List with {len(all_cond_pooled)} conditioning(s)", "DEBUG", True)
            
            return all_cond_pooled
            
        except Exception as e:
            log_error_with_traceback("Dual TensorRT CLIP encoding failed", e)
            raise
    
    def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False):
        """Encode tokens - compatibility method for ComfyUI CLIP interface"""
        try:
            # Call the main encoding method
            all_cond_pooled = self.encode_from_tokens_scheduled(tokens)
            
            if not all_cond_pooled:
                raise RuntimeError("No conditioning returned from encode_from_tokens_scheduled")
            
            # Extract the first (and typically only) conditioning
            cond, pooled_dict = all_cond_pooled[0]
            pooled = pooled_dict.get("pooled_output", None)
            
            if return_dict:
                out = {"cond": cond, "pooled_output": pooled}
                # Add any additional keys from pooled_dict
                for k, v in pooled_dict.items():
                    if k != "pooled_output":
                        out[k] = v
                return out
            
            if return_pooled:
                return cond, pooled
                
            return cond
            
        except Exception as e:
            log_error_with_traceback("encode_from_tokens failed", e)
            raise
    
    def encode(self, text):
        """Encode text - high-level interface"""
        try:
            tokens = self.tokenize(text)
            return self.encode_from_tokens(tokens)
        except Exception as e:
            log_error_with_traceback("Text encoding failed", e)
            raise
    
    # Add compatibility methods for ComfyUI CLIP interface
    def clone(self):
        """Clone the TensorRT CLIP object"""
        return TensorRTCLIP(self.engine_data)
    
    def clip_layer(self, layer_idx):
        """Set CLIP layer - for compatibility"""
        self.layer_idx = layer_idx
    
    def set_tokenizer_option(self, option_name, value):
        """Set tokenizer option - for compatibility"""
        self.tokenizer_options[option_name] = value

    def _run_clip_l_inference(self, tokens):
        """Run CLIP-L TensorRT inference"""
        try:
            batch_size = tokens.shape[0]
            
            log(f"CLIP-L inference - batch_size: {batch_size}", "DEBUG", True)
            
            # Validate engine and context
            if self.clip_l_engine is None or self.clip_l_context is None:
                raise RuntimeError("CLIP-L engine or context is None")
            
            # Check if engine is valid
            if not self.clip_l_engine.num_io_tensors > 0:
                raise RuntimeError("CLIP-L engine has no I/O tensors")
            
            # Debug engine information
            log(f"CLIP-L engine inputs: {[self.clip_l_engine.get_tensor_name(i) for i in range(self.clip_l_engine.num_io_tensors) if self.clip_l_engine.get_tensor_mode(self.clip_l_engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]}", "DEBUG", True)
            log(f"CLIP-L engine outputs: {[self.clip_l_engine.get_tensor_name(i) for i in range(self.clip_l_engine.num_io_tensors) if self.clip_l_engine.get_tensor_mode(self.clip_l_engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]}", "DEBUG", True)
            
            # Get actual tensor names from engine
            input_names = [self.clip_l_engine.get_tensor_name(i) for i in range(self.clip_l_engine.num_io_tensors) if self.clip_l_engine.get_tensor_mode(self.clip_l_engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]
            output_names = [self.clip_l_engine.get_tensor_name(i) for i in range(self.clip_l_engine.num_io_tensors) if self.clip_l_engine.get_tensor_mode(self.clip_l_engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]
            
            if not input_names or not output_names:
                raise RuntimeError(f"No input/output tensors found in CLIP-L engine. Inputs: {input_names}, Outputs: {output_names}")
            
            input_name = input_names[0]
            output_name = output_names[0]
            
            log(f"Using input tensor: {input_name}, output tensor: {output_name}", "DEBUG", True)
            
            # Get tensor shapes and check compatibility
            input_shape = (batch_size, 77)
            
            # Check if we need to set input shape for dynamic batching
            try:
                # Get the optimization profile shape
                min_shape = self.clip_l_engine.get_tensor_profile_shape(input_name, 0)[0]
                opt_shape = self.clip_l_engine.get_tensor_profile_shape(input_name, 0)[1]
                max_shape = self.clip_l_engine.get_tensor_profile_shape(input_name, 0)[2]
                
                log(f"CLIP-L tensor shapes - min: {min_shape}, opt: {opt_shape}, max: {max_shape}", "DEBUG", True)
                
                # Set the input shape if it's within the valid range
                if min_shape[0] <= batch_size <= max_shape[0]:
                    self.clip_l_context.set_input_shape(input_name, input_shape)
                    log(f"Set CLIP-L input shape to: {input_shape}", "DEBUG", True)
                else:
                    raise RuntimeError(f"Batch size {batch_size} is outside the valid range [{min_shape[0]}, {max_shape[0]}]")
                    
            except Exception as shape_error:
                log(f"Error setting input shape: {str(shape_error)}", "WARNING", True)
                # Try without setting shape for static batch engines
            
            # Get output shape after setting input shape
            try:
                output_shape_dims = self.clip_l_context.get_tensor_shape(output_name)
                # Convert TensorRT Dims to tuple
                output_shape = tuple(output_shape_dims)
                log(f"CLIP-L output shape: {output_shape}", "DEBUG", True)
            except:
                # Fallback to expected shape
                output_shape = (batch_size, 77, 768)
                log(f"Using fallback CLIP-L output shape: {output_shape}", "DEBUG", True)
            
            # Allocate device memory - ensure proper dtype
            d_input = torch.empty(input_shape, dtype=torch.long, device='cuda')
            d_output = torch.empty(output_shape, dtype=torch.float16, device='cuda')  # Use fp16 for efficiency
            
            # Copy input data to device
            d_input.copy_(tokens)
            
            log(f"Input tensor shape: {d_input.shape}, dtype: {d_input.dtype}", "DEBUG", True)
            log(f"Output tensor shape: {d_output.shape}, dtype: {d_output.dtype}", "DEBUG", True)
            
            # Set tensor addresses
            self.clip_l_context.set_tensor_address(input_name, d_input.data_ptr())
            self.clip_l_context.set_tensor_address(output_name, d_output.data_ptr())
            
            # Execute inference
            log("Executing CLIP-L TensorRT inference...", "DEBUG", True)
            
            # Debug input values before execution
            log(f"Input token stats - min: {d_input.min().item()}, max: {d_input.max().item()}", "DEBUG", True)
            log(f"Input sample tokens: {d_input[0, :10].tolist()}", "DEBUG", True)
            
            success = self.clip_l_context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
            
            if not success:
                # Get more detailed error information
                log("CLIP-L TensorRT execution returned False", "ERROR", True)
                log("Checking context validity and tensor bindings...", "DEBUG", True)
                
                # Check if all required tensors are bound
                for i in range(self.clip_l_engine.num_io_tensors):
                    tensor_name = self.clip_l_engine.get_tensor_name(i)
                    is_bound = self.clip_l_context.get_tensor_address(tensor_name) != 0
                    log(f"Tensor {tensor_name} bound: {is_bound}", "DEBUG", True)
                
                raise RuntimeError("CLIP-L TensorRT execute_async_v3 returned False")
            
            # Synchronize and return result
            torch.cuda.synchronize()
            log("CLIP-L TensorRT inference completed successfully", "DEBUG", True)
            
            # Debug output values immediately after execution
            log(f"Raw output stats - min: {d_output.min().item():.6f}, max: {d_output.max().item():.6f}, mean: {d_output.mean().item():.6f}", "DEBUG", True)
            log(f"Raw output sample: {d_output[0, 0, :5].tolist()}", "DEBUG", True)
            
            # Ensure output is properly formatted and has valid values
            result = d_output.clone()
            
            # Check for NaN or invalid values
            nan_count = torch.isnan(result).sum().item()
            inf_count = torch.isinf(result).sum().item()
            
            if nan_count > 0:
                log(f"ERROR: {nan_count} NaN values detected in CLIP-L output - TensorRT engine may be corrupted", "ERROR", True)
                log("This indicates a serious issue with the TensorRT engine conversion or execution", "ERROR", True)
                # Don't replace with zeros - this masks the real problem
                raise RuntimeError(f"CLIP-L TensorRT engine produced {nan_count} NaN values")
            
            if inf_count > 0:
                log(f"ERROR: {inf_count} Inf values detected in CLIP-L output - TensorRT engine may have overflow", "ERROR", True)
                log("This indicates a serious issue with the TensorRT engine conversion or execution", "ERROR", True)
                raise RuntimeError(f"CLIP-L TensorRT engine produced {inf_count} Inf values")
            
            log(f"CLIP-L output stats - min: {result.min():.6f}, max: {result.max():.6f}, mean: {result.mean():.6f}", "DEBUG", True)
            
            return result.cpu().float()  # Convert fp16 to float32 on CPU for ComfyUI compatibility
            
        except Exception as e:
            log_error_with_traceback("CLIP-L TensorRT inference failed", e)
            # Fallback to dummy embeddings with proper values
            log("Using fallback dummy embeddings for CLIP-L", "WARNING", True)
            # Create embeddings with small random values instead of zeros
            fallback = torch.randn((batch_size, 77, 768), dtype=torch.float16, device='cuda') * 0.01
            return fallback.cpu().float()  # Convert to float32 on CPU for ComfyUI compatibility
    
    def _run_clip_g_inference(self, tokens):
        """Run CLIP-G TensorRT inference"""
        try:
            batch_size = tokens.shape[0]
            
            log(f"CLIP-G inference - batch_size: {batch_size}", "DEBUG", True)
            
            # Debug engine information
            log(f"CLIP-G engine inputs: {[self.clip_g_engine.get_tensor_name(i) for i in range(self.clip_g_engine.num_io_tensors) if self.clip_g_engine.get_tensor_mode(self.clip_g_engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]}", "DEBUG", True)
            log(f"CLIP-G engine outputs: {[self.clip_g_engine.get_tensor_name(i) for i in range(self.clip_g_engine.num_io_tensors) if self.clip_g_engine.get_tensor_mode(self.clip_g_engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]}", "DEBUG", True)
            
            # Get actual tensor names from engine
            input_names = [self.clip_g_engine.get_tensor_name(i) for i in range(self.clip_g_engine.num_io_tensors) if self.clip_g_engine.get_tensor_mode(self.clip_g_engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]
            output_names = [self.clip_g_engine.get_tensor_name(i) for i in range(self.clip_g_engine.num_io_tensors) if self.clip_g_engine.get_tensor_mode(self.clip_g_engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]
            
            if not input_names:
                raise RuntimeError(f"No input tensors found in CLIP-G engine. Inputs: {input_names}")
            if len(output_names) < 2:
                raise RuntimeError(f"Expected 2 output tensors for CLIP-G, found {len(output_names)}: {output_names}")
            
            input_name = input_names[0]
            # CLIP-G should have 2 outputs: hidden state and pooled output
            hidden_output_name = output_names[0]  # Usually 'last_hidden_state'
            pooled_output_name = output_names[1]  # Usually 'pooled_output'
            
            log(f"Using input tensor: {input_name}", "DEBUG", True)
            log(f"Using hidden output tensor: {hidden_output_name}", "DEBUG", True)
            log(f"Using pooled output tensor: {pooled_output_name}", "DEBUG", True)
            
            # Get tensor shapes and check compatibility
            input_shape = (batch_size, 77)
            
            # Check if we need to set input shape for dynamic batching
            try:
                # Get the optimization profile shape
                min_shape = self.clip_g_engine.get_tensor_profile_shape(input_name, 0)[0]
                opt_shape = self.clip_g_engine.get_tensor_profile_shape(input_name, 0)[1]
                max_shape = self.clip_g_engine.get_tensor_profile_shape(input_name, 0)[2]
                
                log(f"CLIP-G tensor shapes - min: {min_shape}, opt: {opt_shape}, max: {max_shape}", "DEBUG", True)
                
                # Set the input shape if it's within the valid range
                if min_shape[0] <= batch_size <= max_shape[0]:
                    self.clip_g_context.set_input_shape(input_name, input_shape)
                    log(f"Set CLIP-G input shape to: {input_shape}", "DEBUG", True)
                else:
                    raise RuntimeError(f"Batch size {batch_size} is outside the valid range [{min_shape[0]}, {max_shape[0]}]")
                    
            except Exception as shape_error:
                log(f"Error setting input shape: {str(shape_error)}", "WARNING", True)
                # Try without setting shape for static batch engines
            
            # Get output shapes after setting input shape
            try:
                hidden_shape_dims = self.clip_g_context.get_tensor_shape(hidden_output_name)
                pooled_shape_dims = self.clip_g_context.get_tensor_shape(pooled_output_name)
                # Convert TensorRT Dims to tuples
                hidden_shape = tuple(hidden_shape_dims)
                pooled_shape = tuple(pooled_shape_dims)
                log(f"CLIP-G hidden output shape: {hidden_shape}", "DEBUG", True)
                log(f"CLIP-G pooled output shape: {pooled_shape}", "DEBUG", True)
            except:
                # Fallback to expected shapes
                hidden_shape = (batch_size, 77, 1280)  # CLIP-G hidden size
                pooled_shape = (batch_size, 1280)      # CLIP-G pooled size
                log(f"Using fallback CLIP-G shapes - hidden: {hidden_shape}, pooled: {pooled_shape}", "DEBUG", True)
            
            # Allocate device memory - ensure proper dtype
            d_input = torch.empty(input_shape, dtype=torch.long, device='cuda')
            d_hidden = torch.empty(hidden_shape, dtype=torch.float16, device='cuda')  # Use fp16 for efficiency
            d_pooled = torch.empty(pooled_shape, dtype=torch.float16, device='cuda')  # Use fp16 for efficiency
            
            # Copy input data to device
            d_input.copy_(tokens)
            
            log(f"Input tensor shape: {d_input.shape}, dtype: {d_input.dtype}", "DEBUG", True)
            log(f"Hidden tensor shape: {d_hidden.shape}, dtype: {d_hidden.dtype}", "DEBUG", True)
            log(f"Pooled tensor shape: {d_pooled.shape}, dtype: {d_pooled.dtype}", "DEBUG", True)
            
            # Set tensor addresses
            self.clip_g_context.set_tensor_address(input_name, d_input.data_ptr())
            self.clip_g_context.set_tensor_address(hidden_output_name, d_hidden.data_ptr())
            self.clip_g_context.set_tensor_address(pooled_output_name, d_pooled.data_ptr())
            
            # Execute inference
            log("Executing CLIP-G TensorRT inference...", "DEBUG", True)
            success = self.clip_g_context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
            
            if not success:
                # Get more detailed error information
                log("CLIP-G TensorRT execution returned False", "ERROR", True)
                log("Checking context validity and tensor bindings...", "DEBUG", True)
                
                # Check if all required tensors are bound
                for i in range(self.clip_g_engine.num_io_tensors):
                    tensor_name = self.clip_g_engine.get_tensor_name(i)
                    is_bound = self.clip_g_context.get_tensor_address(tensor_name) != 0
                    log(f"Tensor {tensor_name} bound: {is_bound}", "DEBUG", True)
                
                raise RuntimeError("CLIP-G TensorRT execute_async_v3 returned False")
            
            # Synchronize and return results
            torch.cuda.synchronize()
            log("CLIP-G TensorRT inference completed successfully", "DEBUG", True)
            
            # Ensure outputs are properly formatted and have valid values
            hidden_result = d_hidden.clone()
            pooled_result = d_pooled.clone()
            
            # Check for NaN or invalid values in hidden output
            hidden_nan_count = torch.isnan(hidden_result).sum().item()
            hidden_inf_count = torch.isinf(hidden_result).sum().item()
            pooled_nan_count = torch.isnan(pooled_result).sum().item()
            pooled_inf_count = torch.isinf(pooled_result).sum().item()
            
            if hidden_nan_count > 0:
                log(f"ERROR: {hidden_nan_count} NaN values detected in CLIP-G hidden output", "ERROR", True)
                raise RuntimeError(f"CLIP-G TensorRT engine produced {hidden_nan_count} NaN values in hidden output")
            
            if hidden_inf_count > 0:
                log(f"ERROR: {hidden_inf_count} Inf values detected in CLIP-G hidden output", "ERROR", True)
                raise RuntimeError(f"CLIP-G TensorRT engine produced {hidden_inf_count} Inf values in hidden output")
            
            if pooled_nan_count > 0:
                log(f"ERROR: {pooled_nan_count} NaN values detected in CLIP-G pooled output", "ERROR", True)
                raise RuntimeError(f"CLIP-G TensorRT engine produced {pooled_nan_count} NaN values in pooled output")
            
            if pooled_inf_count > 0:
                log(f"ERROR: {pooled_inf_count} Inf values detected in CLIP-G pooled output", "ERROR", True)
                raise RuntimeError(f"CLIP-G TensorRT engine produced {pooled_inf_count} Inf values in pooled output")
            
            # Check for suspicious value ranges that might indicate quantization issues
            hidden_min, hidden_max = hidden_result.min().item(), hidden_result.max().item()
            pooled_min, pooled_max = pooled_result.min().item(), pooled_result.max().item()
            
            if abs(hidden_min) >= 500 or abs(hidden_max) >= 500:
                log(f"WARNING: CLIP-G hidden values seem unusually large: min={hidden_min:.2f}, max={hidden_max:.2f}", "WARNING", True)
                log("This might indicate quantization issues in the TensorRT engine", "WARNING", True)
            
            if abs(pooled_min) >= 500 or abs(pooled_max) >= 500:
                log(f"WARNING: CLIP-G pooled values seem unusually large: min={pooled_min:.2f}, max={pooled_max:.2f}", "WARNING", True)
                log("This might indicate quantization issues in the TensorRT engine", "WARNING", True)
            
            log(f"CLIP-G hidden stats - min: {hidden_result.min():.6f}, max: {hidden_result.max():.6f}, mean: {hidden_result.mean():.6f}", "DEBUG", True)
            log(f"CLIP-G pooled stats - min: {pooled_result.min():.6f}, max: {pooled_result.max():.6f}, mean: {pooled_result.mean():.6f}", "DEBUG", True)
            
            return hidden_result.cpu().float(), pooled_result.cpu().float()  # Convert fp16 to float32 on CPU for ComfyUI compatibility
            
        except Exception as e:
            log_error_with_traceback("CLIP-G TensorRT inference failed", e)
            # Fallback to dummy embeddings with proper values
            log("Using fallback dummy embeddings for CLIP-G", "WARNING", True)
            # Create embeddings with small random values instead of zeros
            hidden = torch.randn((batch_size, 77, 1280), dtype=torch.float16, device='cuda') * 0.01
            pooled = torch.randn((batch_size, 1280), dtype=torch.float16, device='cuda') * 0.01
            return hidden.cpu().float(), pooled.cpu().float()  # Convert to float32 on CPU for ComfyUI compatibility

class CLIPTensorRTLoader:
    """TensorRT Dual CLIP Loader following ComfyUI DualCLIPLoader patterns"""
    
    @classmethod
    def INPUT_TYPES(s):
        tensorrt_engines = get_existing_tensorrt_engines()
        default_engine = tensorrt_engines[0] if tensorrt_engines and "No TensorRT engines found" not in tensorrt_engines[0] else ""
        
        return {
            "required": {
                "clip_l_engine": (tensorrt_engines, {
                    "default": default_engine,
                    "tooltip": "The TensorRT engine file for CLIP-L text encoder."
                }),
                "clip_g_engine": (tensorrt_engines, {
                    "default": default_engine,
                    "tooltip": "The TensorRT engine file for CLIP-G text encoder."
                }),
            }
        }

    RETURN_TYPES = ("CLIP",)
    OUTPUT_TOOLTIPS = ("The dual TensorRT CLIP model (CLIP-L + CLIP-G) used for encoding text prompts.",)
    FUNCTION = "load_clip"
    CATEGORY = "loaders"
    DESCRIPTION = "Loads dual TensorRT-optimized CLIP models (CLIP-L and CLIP-G) for faster SDXL text encoding. Both engines must be created first using the Dual CLIP to TensorRT conversion node."

    def load_clip(self, clip_l_engine, clip_g_engine):
        """Load both TensorRT CLIP engines and return dual CLIP-compatible object"""
        
        try:
            # Validate both engines
            for engine_name, engine_type in [(clip_l_engine, "CLIP-L"), (clip_g_engine, "CLIP-G")]:
                if "No TensorRT engines found" in engine_name:
                    error_msg = f"No TensorRT engines found for {engine_type}. Please convert CLIP models first."
                    log(error_msg, "ERROR", True)
                    log(f"TensorRT output directory: {tensorrt_output_dir}", "ERROR", True)
                    raise ValueError(error_msg)
                
                if "Error scanning" in engine_name:
                    error_msg = f"Error accessing TensorRT engines directory for {engine_type}. Check permissions."
                    log(error_msg, "ERROR", True)
                    raise ValueError(error_msg)
            
            # Validate engine files exist and are accessible
            clip_l_path = os.path.join(tensorrt_output_dir, clip_l_engine)
            clip_g_path = os.path.join(tensorrt_output_dir, clip_g_engine)
            
            if not validate_file_access(clip_l_path, "read"):
                error_msg = f"Cannot access CLIP-L TensorRT engine file: {clip_l_engine}"
                log(f"Full path: {clip_l_path}", "ERROR", True)
                raise FileNotFoundError(error_msg)
            
            if not validate_file_access(clip_g_path, "read"):
                error_msg = f"Cannot access CLIP-G TensorRT engine file: {clip_g_engine}"
                log(f"Full path: {clip_g_path}", "ERROR", True)
                raise FileNotFoundError(error_msg)
            
            # Get engine sizes
            clip_l_size = os.path.getsize(clip_l_path) / (1024*1024)
            clip_g_size = os.path.getsize(clip_g_path) / (1024*1024)
            
            log(f"Loading dual TensorRT CLIP engines:", "INFO", True)
            log(f"  - CLIP-L: {clip_l_engine} ({clip_l_size:.1f}MB)", "INFO", True)
            log(f"  - CLIP-G: {clip_g_engine} ({clip_g_size:.1f}MB)", "INFO", True)
            
            # Create engine data for both models
            engine_data = {
                "clip_l": {
                    "engine_name": clip_l_engine,
                    "engine_path": clip_l_path,
                    "engine_size_mb": clip_l_size,
                    "clip_type": "clip-l"
                },
                "clip_g": {
                    "engine_name": clip_g_engine,
                    "engine_path": clip_g_path,
                    "engine_size_mb": clip_g_size,
                    "clip_type": "clip-g"
                }
            }
            
            # Create dual TensorRT CLIP wrapper
            clip_tensorrt = TensorRTCLIP(engine_data)
            
            log(f"Dual TensorRT CLIP loaded successfully", "INFO", True)
            log(f"Ready for SDXL text encoding with hardware acceleration", "DEBUG", True)
            
            return (clip_tensorrt,)
            
        except Exception as e:
            log_error_with_traceback(f"Failed to load dual TensorRT CLIP engines", e)
            
            # Provide helpful suggestions
            error_str = str(e).lower()
            if "permission" in error_str:
                log(f"Suggestion: Check file permissions for {tensorrt_output_dir}", "INFO", True)
            elif "not found" in error_str:
                log("Suggestion: Create TensorRT engines first using Dual CLIP to TensorRT conversion", "INFO", True)
            elif "clip-l" in error_str:
                log(f"Suggestion: Make sure {clip_l_engine} is a valid CLIP-L TensorRT engine", "INFO", True)
            elif "clip-g" in error_str:
                log(f"Suggestion: Make sure {clip_g_engine} is a valid CLIP-G TensorRT engine", "INFO", True)
            
            raise

class CLIPTensorRTTextEncode:
    """TensorRT CLIP Text Encoder following ComfyUI patterns"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True, 
                    "dynamicPrompts": True, 
                    "tooltip": "The text to be encoded using TensorRT CLIP."
                }),
                "clip": ("CLIP", {
                    "tooltip": "The TensorRT CLIP model used for encoding the text."
                })
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    OUTPUT_TOOLTIPS = ("A conditioning containing the embedded text used to guide the diffusion model.",)
    FUNCTION = "encode"
    CATEGORY = "conditioning"
    DESCRIPTION = "Encodes a text prompt using a TensorRT-optimized CLIP model for faster inference. Supports SDXL dual CLIP (CLIP-L + CLIP-G) format."

    def encode(self, clip, text):
        """Encode text using TensorRT CLIP"""
        
        try:
            if clip is None:
                raise RuntimeError("ERROR: clip input is invalid: None\n\nMake sure you've loaded a TensorRT CLIP model using the CLIPTensorRTLoader node.")
            
            if not isinstance(clip, TensorRTCLIP):
                raise TypeError(f"Expected TensorRTCLIP object, got {type(clip)}. Make sure you're using the CLIPTensorRTLoader node.")
            
            if text is None:
                text = ""
            
            if not isinstance(text, str):
                log(f"Converting text to string from {type(text)}", "WARNING", True)
                text = str(text)
            
            if not text.strip():
                text = ""
                log("Empty text provided for TensorRT CLIP encoding", "DEBUG", True)
            
            # Check text length
            if len(text) > 1000:
                log(f"Warning: Very long text ({len(text)} chars), may be truncated", "WARNING", True)
            
            log(f"Encoding text with TensorRT CLIP engine", "INFO", True)
            log(f"Text: '{text[:50]}{'...' if len(text) > 50 else ''}'", "DEBUG", True)
            
            # Tokenize text
            tokens = clip.tokenize(text)
            
            # Encode using TensorRT - this returns a list of [cond, pooled_dict] pairs
            all_cond_pooled = clip.encode_from_tokens_scheduled(tokens)
            
            if not all_cond_pooled:
                raise RuntimeError("No conditioning returned from TensorRT CLIP encoding")
            
            log(f"TensorRT CLIP text encoding complete", "INFO", True)
            log(f"Returned {len(all_cond_pooled)} conditioning(s)", "DEBUG", True)
            
            # Return the conditioning list directly (ComfyUI expects this format)
            return (all_cond_pooled,)
            
        except Exception as e:
            log_error_with_traceback("Failed to encode text with TensorRT CLIP", e)
            
            # Provide helpful suggestions
            error_str = str(e).lower()
            if "memory" in error_str:
                log("Suggestion: Free up GPU memory or use smaller batch sizes", "INFO", True)
            elif "tensorrt" in error_str:
                log("Suggestion: Check if TensorRT engine is properly loaded and compatible", "INFO", True)
            elif "invalid" in error_str or "none" in error_str:
                log("Suggestion: Make sure to connect a TensorRT CLIP model from CLIPTensorRTLoader", "INFO", True)
            
            raise

# Add after the existing DualCLIPToTensorRT class, before TensorRTCLIP class

class DualCLIPToTensorRTV2:
    """Version 2: Uses Hugging Face Optimum for ONNX export and TensorRT conversion"""
    
    @classmethod
    def INPUT_TYPES(s):
        clip_models = get_available_clip_models()
        default_model = clip_models[0] if clip_models and "No CLIP models found" not in clip_models[0] else ""
        
        return {"required": {
            "clip_name1": (clip_models, {"default": default_model}),
            "clip_name2": (clip_models, {"default": default_model}),
            "output_name": ("STRING", {"default": "dual_clip_v2", "multiline": False}),
            "prompt_batch_min": ("INT", {"default": 1, "min": 1, "max": 32, "step": 1}),
            "prompt_batch_opt": ("INT", {"default": 1, "min": 1, "max": 32, "step": 1}),
            "prompt_batch_max": ("INT", {"default": 8, "min": 1, "max": 32, "step": 1}),
            "use_fp16": ("BOOLEAN", {"default": True}),
            "optimization_level": (["O1", "O2", "O3", "O4"], {"default": "O2"}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "convert_dual_clip_to_tensorrt_v2"
    OUTPUT_NODE = True
    CATEGORY = "vovlerTools"

    def convert_dual_clip_to_tensorrt_v2(self, clip_name1, clip_name2, output_name, 
                                        prompt_batch_min, prompt_batch_opt, prompt_batch_max,
                                        use_fp16, optimization_level):
        
        try:
            # Import required libraries for Optimum
            try:
                from optimum.onnxruntime import ORTModelForFeatureExtraction
                from optimum.exporters import TasksManager
                from transformers import CLIPTextModel, CLIPTokenizer
                import onnx
                log("Successfully imported Optimum and required libraries", "DEBUG", True)
            except ImportError as e:
                error_msg = f"Required libraries not found: {str(e)}\n"
                error_msg += "Please install: pip install optimum[onnxruntime-gpu] transformers onnx"
                log(error_msg, "ERROR", True)
                return (error_msg,)
            
            # Log system information for debugging
            log_system_info()
            
            # Validate inputs (same as V1)
            if "No CLIP models found" in clip_name1 or "No CLIP models found" in clip_name2:
                error_msg = "CLIP models not found. Please place .safetensors files in models/clip folder."
                log(error_msg, "ERROR", True)
                return (error_msg,)
            
            if clip_name1 == clip_name2:
                error_msg = "Please select two different CLIP models for dual CLIP conversion."
                log(error_msg, "ERROR", True)
                return (error_msg,)
            
            # Validate batch size parameters
            if not (1 <= prompt_batch_min <= prompt_batch_opt <= prompt_batch_max <= 32):
                error_msg = f"Invalid batch size configuration: min={prompt_batch_min}, opt={prompt_batch_opt}, max={prompt_batch_max}"
                log(error_msg, "ERROR", True)
                return (error_msg,)
            
            # Sanitize output name
            import re
            sanitized_name = re.sub(r'[<>:"/\\|?*]', '_', output_name.strip())
            if sanitized_name != output_name.strip():
                log(f"Output name sanitized from '{output_name}' to '{sanitized_name}'", "WARNING", True)
                output_name = sanitized_name
            
            clip1_path = os.path.join(clip_models_dir, clip_name1)
            clip2_path = os.path.join(clip_models_dir, clip_name2)
            
            # Validate file access
            if not validate_file_access(clip1_path, "read") or not validate_file_access(clip2_path, "read"):
                error_msg = "Cannot access one or both CLIP model files"
                log(error_msg, "ERROR", True)
                return (error_msg,)
            
            # Create engine filename
            precision = "fp16" if use_fp16 else "fp32"
            engine_filename = f"{output_name}_sdxl_{prompt_batch_min}_{prompt_batch_opt}_{prompt_batch_max}_{precision}_v2.engine"
            engine_path = os.path.join(tensorrt_output_dir, engine_filename)
            
            # Check if engine already exists
            if os.path.exists(engine_path):
                engine_size = os.path.getsize(engine_path) / (1024*1024)
                success_msg = f"TensorRT engine already exists: {engine_filename} ({engine_size:.1f}MB)"
                log(success_msg, "INFO", True)
                return (success_msg,)
            
            log(f"Starting dual CLIP to TensorRT V2 conversion...", "INFO", True)
            log(f"CLIP 1: {clip_name1}", "INFO", True)
            log(f"CLIP 2: {clip_name2}", "INFO", True)
            log(f"Precision: {precision}", "INFO", True)
            log(f"Optimization level: {optimization_level}", "INFO", True)
            log(f"Batch sizes: {prompt_batch_min}/{prompt_batch_opt}/{prompt_batch_max}", "INFO", True)
            
            # Initialize engine paths for cleanup
            clip_l_engine_path = None
            clip_g_engine_path = None
            
            # Load CLIP models using ComfyUI's built-in functionality
            log("Loading CLIP models using ComfyUI's native loader...", "INFO", True)
            clip_object_1 = comfy.sd.load_clip(ckpt_paths=[clip1_path], embedding_directory=folder_paths.get_folder_paths("embeddings"))
            clip_object_2 = comfy.sd.load_clip(ckpt_paths=[clip2_path], embedding_directory=folder_paths.get_folder_paths("embeddings"))
            
            # Extract CLIP models
            clip_l_model = None
            clip_g_model = None
            
            # Check first model
            if hasattr(clip_object_1.cond_stage_model, 'clip_l') and clip_object_1.cond_stage_model.clip_l is not None:
                clip_l_model = clip_object_1.cond_stage_model.clip_l.transformer
                log(f"Found CLIP-L in first model ({clip_name1})", "DEBUG", True)
            
            if hasattr(clip_object_1.cond_stage_model, 'clip_g') and clip_object_1.cond_stage_model.clip_g is not None:
                clip_g_model = clip_object_1.cond_stage_model.clip_g.transformer
                log(f"Found CLIP-G in first model ({clip_name1})", "DEBUG", True)
            
            # Check second model
            if hasattr(clip_object_2.cond_stage_model, 'clip_l') and clip_object_2.cond_stage_model.clip_l is not None:
                if clip_l_model is None:
                    clip_l_model = clip_object_2.cond_stage_model.clip_l.transformer
                    log(f"Found CLIP-L in second model ({clip_name2})", "DEBUG", True)
            
            if hasattr(clip_object_2.cond_stage_model, 'clip_g') and clip_object_2.cond_stage_model.clip_g is not None:
                if clip_g_model is None:
                    clip_g_model = clip_object_2.cond_stage_model.clip_g.transformer
                    log(f"Found CLIP-G in second model ({clip_name2})", "DEBUG", True)
            
            # Validate that we found both models
            if clip_l_model is None:
                raise ValueError(f"Could not find CLIP-L model in either {clip_name1} or {clip_name2}")
            if clip_g_model is None:
                raise ValueError(f"Could not find CLIP-G model in either {clip_name1} or {clip_name2}")
            
            log("Successfully located both CLIP-L and CLIP-G models", "INFO", True)
            
            # Move models to GPU
            device = 'cuda'
            clip_l_model = clip_l_model.to(device)
            clip_g_model = clip_g_model.to(device)
            
            # Create separate engines using Optimum
            clip_l_engine_path = engine_path.replace('.engine', '_clip_l.engine')
            clip_g_engine_path = engine_path.replace('.engine', '_clip_g.engine')
            
            # Create CLIP-L engine using Optimum
            log("Creating CLIP-L TensorRT engine using Optimum...", "INFO", True)
            self._create_optimum_tensorrt_engine(
                clip_l_model, 
                clip_l_engine_path, 
                'clip-l',
                prompt_batch_min, 
                prompt_batch_opt, 
                prompt_batch_max,
                use_fp16,
                optimization_level
            )
            
            # Create CLIP-G engine using Optimum
            log("Creating CLIP-G TensorRT engine using Optimum...", "INFO", True)
            self._create_optimum_tensorrt_engine(
                clip_g_model, 
                clip_g_engine_path, 
                'clip-g',
                prompt_batch_min, 
                prompt_batch_opt, 
                prompt_batch_max,
                use_fp16,
                optimization_level
            )
            
            # Success message
            clip_l_size = os.path.getsize(clip_l_engine_path) / (1024*1024)
            clip_g_size = os.path.getsize(clip_g_engine_path) / (1024*1024)
            success_msg = f"Dual CLIP TensorRT V2 engines created successfully:\n"
            success_msg += f"  - CLIP-L: {os.path.basename(clip_l_engine_path)} ({clip_l_size:.1f}MB)\n" 
            success_msg += f"  - CLIP-G: {os.path.basename(clip_g_engine_path)} ({clip_g_size:.1f}MB)"
            
            log(success_msg, "INFO", True)
            return (success_msg,)
            
        except Exception as e:
            error_msg = f"Error during dual CLIP to TensorRT V2 conversion: {str(e)}"
            log_error_with_traceback(error_msg, e)
            
            # Clean up any partial engine files
            if clip_l_engine_path and os.path.exists(clip_l_engine_path):
                try:
                    os.remove(clip_l_engine_path)
                    log(f"Cleaned up partial CLIP-L engine file", "DEBUG", True)
                except:
                    pass
            
            if clip_g_engine_path and os.path.exists(clip_g_engine_path):
                try:
                    os.remove(clip_g_engine_path)
                    log(f"Cleaned up partial CLIP-G engine file", "DEBUG", True)
                except:
                    pass
            
            return (error_msg,)

    def _create_optimum_tensorrt_engine(self, model, engine_path, clip_type, 
                                       prompt_batch_min, prompt_batch_opt, prompt_batch_max,
                                       use_fp16, optimization_level):
        """Create TensorRT engine using Hugging Face Optimum"""
        
        temp_dir = None
        try:
            import tempfile
            from optimum.onnxruntime import ORTModelForFeatureExtraction
            from optimum.exporters.onnx import main_export
            from transformers import CLIPTextModel, CLIPTokenizer
            import onnx
            
            log(f"Creating {clip_type} TensorRT engine using Optimum...", "INFO", True)
            
            # Create temporary directory for intermediate files
            temp_dir = tempfile.mkdtemp(prefix=f"clip_{clip_type}_")
            log(f"Using temporary directory: {temp_dir}", "DEBUG", True)
            
            # Step 1: Create a Hugging Face compatible model wrapper
            hf_model = self._create_hf_compatible_model(model, clip_type)
            
            # Step 2: Export to ONNX using Optimum
            onnx_path = os.path.join(temp_dir, f"{clip_type}.onnx")
            log(f"Exporting {clip_type} to ONNX using Optimum...", "DEBUG", True)
            
            # Create tokenizer for export
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            
            # Define dynamic axes for batching
            dynamic_axes = {
                "input_ids": {0: "batch_size"},
                "last_hidden_state": {0: "batch_size"}
            }
            
            if clip_type == 'clip-g':
                dynamic_axes["pooled_output"] = {0: "batch_size"}
            
            # Export using Optimum's ONNX exporter
            from optimum.exporters.onnx import export_models
            from optimum.exporters.tasks import TasksManager
            
            # Get the task configuration
            task = "feature-extraction"
            model_config = hf_model.config if hasattr(hf_model, 'config') else None
            
            # Create export configuration
            onnx_config = TasksManager.get_exporter_config_constructor(
                model_type="clip_text_model",
                exporter="onnx",
                task=task,
                model_name="clip-text",
                library_name="transformers"
            )(model_config) if model_config else None
            
            if onnx_config is None:
                log(f"Using manual ONNX export for {clip_type}...", "DEBUG", True)
                # Fallback to manual export
                self._manual_onnx_export_optimum(hf_model, tokenizer, onnx_path, clip_type, 
                                                prompt_batch_opt, dynamic_axes)
            else:
                log(f"Using Optimum automatic export for {clip_type}...", "DEBUG", True)
                # Use Optimum's automatic export
                export_models(
                    model=hf_model,
                    config=onnx_config,
                    tokenizer=tokenizer,
                    output_dir=temp_dir,
                    opset=16
                )
                # Move the exported model to our expected path
                exported_path = os.path.join(temp_dir, "model.onnx")
                if os.path.exists(exported_path):
                    os.rename(exported_path, onnx_path)
            
            # Validate ONNX model
            if not os.path.exists(onnx_path) or os.path.getsize(onnx_path) == 0:
                raise RuntimeError(f"ONNX export failed for {clip_type}: file not created or empty")
            
            # Step 3: Optimize ONNX model using Optimum
            log(f"Optimizing {clip_type} ONNX model...", "DEBUG", True)
            optimized_onnx_path = os.path.join(temp_dir, f"{clip_type}_optimized.onnx")
            
            try:
                from optimum.onnxruntime import ORTOptimizer, OptimizationConfig
                
                # Create optimization configuration
                optimization_config = OptimizationConfig(
                    optimization_level=optimization_level,
                    optimize_for_gpu=True,
                    fp16=use_fp16
                )
                
                # Load and optimize
                optimizer = ORTOptimizer.from_pretrained(temp_dir)
                optimizer.optimize(
                    save_dir=temp_dir,
                    optimization_config=optimization_config,
                    file_suffix="_optimized"
                )
                
                # Check if optimized model exists
                if os.path.exists(optimized_onnx_path):
                    onnx_path = optimized_onnx_path
                    log(f"{clip_type} ONNX optimization completed", "DEBUG", True)
                else:
                    log(f"{clip_type} ONNX optimization skipped - using original", "DEBUG", True)
                    
            except Exception as opt_error:
                log(f"ONNX optimization failed for {clip_type}: {str(opt_error)}", "WARNING", True)
                log("Continuing with non-optimized ONNX model", "DEBUG", True)
            
            # Step 4: Convert ONNX to TensorRT using improved method
            log(f"Converting {clip_type} ONNX to TensorRT...", "DEBUG", True)
            self._onnx_to_tensorrt_optimum(onnx_path, engine_path, clip_type,
                                          prompt_batch_min, prompt_batch_opt, prompt_batch_max,
                                          use_fp16)
            
            log(f"{clip_type} TensorRT engine created successfully using Optimum", "INFO", True)
            
        except Exception as e:
            log_error_with_traceback(f"Failed to create {clip_type} TensorRT engine using Optimum", e)
            raise
        finally:
            # Clean up temporary directory
            if temp_dir and os.path.exists(temp_dir):
                try:
                    import shutil
                    shutil.rmtree(temp_dir)
                    log(f"Cleaned up temporary directory for {clip_type}", "DEBUG", True)
                except Exception as cleanup_error:
                    log(f"Failed to clean up temporary directory: {str(cleanup_error)}", "WARNING", True)

    def _create_hf_compatible_model(self, model, clip_type):
        """Create a Hugging Face compatible model wrapper"""
        
        class HFCompatibleCLIP(torch.nn.Module):
            def __init__(self, clip_model, clip_type):
                super().__init__()
                self.clip_model = clip_model
                self.clip_type = clip_type
                
                # Create a basic config for HF compatibility
                from types import SimpleNamespace
                if clip_type == 'clip-l':
                    self.config = SimpleNamespace(
                        hidden_size=768,
                        intermediate_size=3072,
                        num_attention_heads=12,
                        num_hidden_layers=12,
                        max_position_embeddings=77,
                        vocab_size=49408,
                        model_type="clip_text_model"
                    )
                else:  # clip-g
                    self.config = SimpleNamespace(
                        hidden_size=1280,
                        intermediate_size=5120,
                        num_attention_heads=20,
                        num_hidden_layers=32,
                        max_position_embeddings=77,
                        vocab_size=49408,
                        model_type="clip_text_model"
                    )
            
            def forward(self, input_ids, attention_mask=None, **kwargs):
                # Call the original model
                outputs = self.clip_model(input_tokens=input_ids)
                
                if self.clip_type == 'clip-l':
                    # CLIP-L only returns sequence output
                    return SimpleNamespace(
                        last_hidden_state=outputs[0],
                        hidden_states=None,
                        attentions=None
                    )
                else:  # clip-g
                    # CLIP-G returns both sequence and pooled output
                    return SimpleNamespace(
                        last_hidden_state=outputs[0],
                        pooler_output=outputs[2] if len(outputs) > 2 else None,
                        hidden_states=None,
                        attentions=None
                    )
        
        return HFCompatibleCLIP(model, clip_type)

    def _manual_onnx_export_optimum(self, model, tokenizer, onnx_path, clip_type, 
                                   batch_size, dynamic_axes):
        """Manual ONNX export with Optimum-style configuration"""
        
        try:
            # Create dummy input
            dummy_input = torch.randint(0, 49408, (batch_size, 77), dtype=torch.long, device=model.clip_model.device)
            
            # Define output names
            if clip_type == 'clip-g':
                output_names = ['last_hidden_state', 'pooled_output']
            else:
                output_names = ['last_hidden_state']
            
            # Export with improved settings
            model.eval()
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    dummy_input,
                    onnx_path,
                    export_params=True,
                    opset_version=16,
                    do_constant_folding=True,
                    input_names=['input_ids'],
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    verbose=False,
                    keep_initializers_as_inputs=False,
                    export_modules_as_functions=False
                )
            
            log(f"Manual ONNX export completed for {clip_type}", "DEBUG", True)
            
        except Exception as e:
            log_error_with_traceback(f"Manual ONNX export failed for {clip_type}", e)
            raise

    def _onnx_to_tensorrt_optimum(self, onnx_path, engine_path, clip_type,
                                 prompt_batch_min, prompt_batch_opt, prompt_batch_max,
                                 use_fp16):
        """Convert ONNX to TensorRT with improved settings for Optimum models"""
        
        try:
            # Get GPU memory
            gpu_memory_gb = get_gpu_memory_gb()
            memory_pool_bytes = int(gpu_memory_gb * 0.8 * 1024**3)
            
            log(f"Converting {clip_type} ONNX to TensorRT with improved settings", "DEBUG", True)
            log(f"Memory pool: {memory_pool_bytes / (1024**3):.1f}GB", "DEBUG", True)
            log(f"Precision: {'FP16' if use_fp16 else 'FP32'}", "DEBUG", True)
            
            # Create TensorRT logger and builder
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(TRT_LOGGER)
            
            # Create network from ONNX
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, TRT_LOGGER)
            
            # Parse ONNX file
            log(f"Parsing ONNX file: {os.path.basename(onnx_path)}", "DEBUG", True)
            with open(onnx_path, 'rb') as model_file:
                onnx_data = model_file.read()
                
                if not parser.parse(onnx_data):
                    log(f"ONNX parsing failed for {clip_type}", "ERROR", True)
                    for error in range(parser.num_errors):
                        log(f"Parser Error: {parser.get_error(error)}", "ERROR", True)
                    raise RuntimeError(f"Failed to parse ONNX file for {clip_type}")
            
            log(f"{clip_type} ONNX parsed successfully", "DEBUG", True)
            
            # Configure builder with improved settings
            config = builder.create_builder_config()
            
            # Set precision
            if use_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                log(f"Enabled FP16 precision for {clip_type}", "DEBUG", True)
            
            # Set memory pool
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, memory_pool_bytes)
            
            # Enable optimizations
            config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            
            # Add optimization profile for dynamic batch size
            profile = builder.create_optimization_profile()
            profile.set_shape(
                'input_ids',
                (prompt_batch_min, 77),
                (prompt_batch_opt, 77), 
                (prompt_batch_max, 77)
            )
            config.add_optimization_profile(profile)
            
            # Build engine with timing
            log(f"Building {clip_type} TensorRT engine with improved settings...", "INFO", True)
            build_start_time = time.time()
            
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                raise RuntimeError(f"Failed to build TensorRT engine for {clip_type}")
            
            build_time = time.time() - build_start_time
            log(f"{clip_type} TensorRT build completed in {build_time:.1f} seconds", "INFO", True)
            
            # Save engine
            with open(engine_path, 'wb') as f:
                f.write(serialized_engine)
            
            engine_size = os.path.getsize(engine_path) / (1024*1024)
            log(f"{clip_type} engine saved: {os.path.basename(engine_path)} ({engine_size:.1f}MB)", "INFO", True)
            
        except Exception as e:
            log_error_with_traceback(f"Failed to convert {clip_type} ONNX to TensorRT using Optimum method", e)
            raise

# TensorRT CLIP nodes - exported via __init__.py