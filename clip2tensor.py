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
            
            #try:
             #   clip_l_outputs = clip_l_model(input_tokens=test_input)
             #   log(f"CLIP-L raw outputs: {len(clip_l_outputs)} items", "DEBUG", True)
             #   for i, output in enumerate(clip_l_outputs):
             #       if output is not None:
             #           log(f"CLIP-L output[{i}] shape: {output.shape if hasattr(output, 'shape') else type(output)}", "DEBUG", True)
             #       else:
             #           log(f"CLIP-L output[{i}]: None", "DEBUG", True)
            #except Exception as e:
            #    log(f"CLIP-L test failed: {str(e)}", "ERROR", True)
            
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
            #clip_l_wrapper = CLIP_L_Wrapper(clip_l_model)
            clip_g_wrapper = CLIP_G_Wrapper(clip_g_model)
            
            #try:
            #    clip_l_wrapper_out = clip_l_wrapper(test_input)
            #    log(f"CLIP-L wrapper output shape: {clip_l_wrapper_out.shape}", "DEBUG", True)
            #except Exception as e:
            #    log(f"CLIP-L wrapper test failed: {str(e)}", "ERROR", True)
            
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
            
            clip_type = 'clip-g'
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
            original_backends = None
            if clip_type == 'clip-g':
                log(f"Disabling PyTorch optimizations for CLIP-G ONNX export...", "DEBUG", True)
                # Disable optimized attention temporarily
                original_backends = torch.backends.opt_einsum.enabled
                torch.backends.opt_einsum.enabled = False
                
                # Set model to eval mode and disable gradients
                model.eval()
                for param in model.parameters():
                    param.requires_grad = False
            
            try:
                # Try multiple ONNX export strategies for CLIP-G compatibility
                if clip_type == 'clip-g':
                    log(f"Attempting CLIP-G ONNX export with compatibility fixes...", "DEBUG", True)
                    
                    # Strategy 1: Try with older opset version (more compatible)
                    try:
                        log(f"Trying CLIP-G export with opset 11 (more compatible)...", "DEBUG", True)
                        torch.onnx.export(
                            model,
                            model_args,
                            onnx_path,
                            export_params=True,
                            opset_version=11,  # Use older, more stable opset
                            do_constant_folding=False,  # Disable optimizations that might cause issues
                            input_names=['input_ids'],
                            output_names=output_names,
                            dynamic_axes={
                                'input_ids': {0: 'batch_size'},
                                **dynamic_axes_outputs
                            },
                            # Additional options for problematic models
                            keep_initializers_as_inputs=True,
                            export_modules_as_functions=False
                        )
                        log(f"CLIP-G opset 11 export succeeded", "DEBUG", True)
                    except Exception as opset11_error:
                        log(f"Opset 11 export failed: {str(opset11_error)}", "DEBUG", True)
                        
                        # Strategy 2: Try torch.jit.trace instead of onnx.export
                        try:
                            log(f"Trying CLIP-G with torch.jit.trace approach...", "DEBUG", True)
                            
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
                                opset_version=11,
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
                            
                            # Strategy 3: Fallback to simplest possible export
                            log(f"Trying CLIP-G with minimal export options...", "DEBUG", True)
                            torch.onnx.export(
                                model,
                                model_args,
                                onnx_path,
                                export_params=True,
                                opset_version=9,  # Very old, very compatible
                                do_constant_folding=False,
                                training=False,
                                input_names=['input_ids'],
                                output_names=output_names
                                # No dynamic axes to avoid complexity
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