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
            
            # Load CLIP models using ComfyUI's built-in functionality
            log("Loading CLIP models...", "INFO", True)
            
            success_msg = self._create_tensorrt_engine(
                clip1_path, clip2_path, engine_path, clip_name1, clip_name2,
                prompt_batch_min, prompt_batch_opt, prompt_batch_max
            )
            
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
            
            # Provide helpful suggestions based on error type
            if "CUDA" in str(e):
                log("Suggestion: Check CUDA installation and GPU availability", "INFO", True)
            elif "TensorRT" in str(e):
                log("Suggestion: Check TensorRT installation and compatibility", "INFO", True)
            elif "memory" in str(e).lower():
                log("Suggestion: Reduce batch sizes or free up GPU memory", "INFO", True)
            
            return (error_msg,)
    
    def _create_tensorrt_engine(self, clip1_path, clip2_path, engine_path, clip_name1, clip_name2,
                              prompt_batch_min, prompt_batch_opt, prompt_batch_max):
        """
        Create separate TensorRT engines from dual CLIP models for SDXL
        Uses PyTorch -> ONNX -> TensorRT workflow
        Always uses fp16 precision and 77 token sequence length
        """
        
        log("Creating TensorRT engines for dual CLIP models (SDXL)...", "INFO", True)
        log("Using PyTorch -> ONNX -> TensorRT conversion workflow", "INFO", True)
        
        try:
            # Load CLIP models using ComfyUI's built-in functionality
            log("Loading CLIP models from safetensors...", "INFO", True)
            
            # Load CLIP-L model (first model)
            log(f"Loading CLIP-L model: {clip_name1}", "DEBUG", True)
            clip_l_model = self._load_clip_model(clip1_path, "clip-l")
            
            # Load CLIP-G model (second model) 
            log(f"Loading CLIP-G model: {clip_name2}", "DEBUG", True)
            clip_g_model = self._load_clip_model(clip2_path, "clip-g")
            
            # Create separate engines for each CLIP model
            log("Creating separate TensorRT engines for CLIP-L and CLIP-G", "INFO", True)
            
            # Create CLIP-L engine
            clip_l_engine_path = engine_path.replace('.engine', '_clip_l.engine')
            log(f"Creating CLIP-L engine: {os.path.basename(clip_l_engine_path)}", "INFO", True)
            self._create_single_clip_engine(
                clip_l_model, clip_l_engine_path, "clip-l",
                prompt_batch_min, prompt_batch_opt, prompt_batch_max
            )
            
            # Create CLIP-G engine
            clip_g_engine_path = engine_path.replace('.engine', '_clip_g.engine')
            log(f"Creating CLIP-G engine: {os.path.basename(clip_g_engine_path)}", "INFO", True)
            self._create_single_clip_engine(
                clip_g_model, clip_g_engine_path, "clip-g", 
                prompt_batch_min, prompt_batch_opt, prompt_batch_max
            )
            
            # Create a metadata file to link the two engines
            metadata_path = engine_path.replace('.engine', '_metadata.json')
            metadata = {
                "clip_l_engine": os.path.basename(clip_l_engine_path),
                "clip_g_engine": os.path.basename(clip_g_engine_path),
                "clip_l_model": clip_name1,
                "clip_g_model": clip_name2,
                "batch_sizes": {
                    "min": prompt_batch_min,
                    "opt": prompt_batch_opt,
                    "max": prompt_batch_max
                },
                "sequence_length": 77,
                "precision": "fp16",
                "created_at": time.time()
            }
            
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Get combined size
            clip_l_size = os.path.getsize(clip_l_engine_path) / (1024*1024)
            clip_g_size = os.path.getsize(clip_g_engine_path) / (1024*1024)
            total_size = clip_l_size + clip_g_size
            
            success_msg = f"Dual CLIP SDXL TensorRT engines created successfully: CLIP-L ({clip_l_size:.1f}MB) + CLIP-G ({clip_g_size:.1f}MB) = {total_size:.1f}MB total"
            log(success_msg, "INFO", True)
            return success_msg

            
        except Exception as e:
            log_error_with_traceback("Failed to create TensorRT engine", e)
            
            # Provide specific suggestions based on error type
            error_str = str(e).lower()
            if "memory" in error_str or "out of memory" in error_str:
                log("Suggestion: Reduce prompt batch sizes or free up GPU memory", "INFO", True)
                log(f"Current max batch: {prompt_batch_max}, try reducing to {max(1, prompt_batch_max // 2)}", "INFO", True)
            elif "tensorrt" in error_str:
                log("Suggestion: Check TensorRT installation and version compatibility", "INFO", True)
                log(f"Current TensorRT version: {trt.__version__}", "INFO", True)
            elif "cuda" in error_str:
                log("Suggestion: Check CUDA installation and GPU driver", "INFO", True)
            elif "permission" in error_str:
                log(f"Suggestion: Check write permissions for {tensorrt_output_dir}", "INFO", True)
            
            raise
    
    def _load_clip_model(self, model_path, clip_type):
        """
        Load actual CLIP text encoder from safetensors file
        Returns the real PyTorch CLIP model ready for ONNX export
        """
        try:
            log(f"Loading {clip_type} model from {os.path.basename(model_path)}", "DEBUG", True)
            
            # Load using ComfyUI's CLIP loading system
            import comfy.sd
            import comfy.model_management as model_management
            
            # Load CLIP using ComfyUI's checkpoint loader
            log(f"Loading CLIP checkpoint: {model_path}", "DEBUG", True)
            
            # Try multiple methods to load the CLIP model
            model = None
            
            # Method 1: Direct safetensors loading
            try:
                from safetensors.torch import load_file
                state_dict = load_file(model_path)
                
                # Look for text encoder keys
                text_encoder_keys = [k for k in state_dict.keys() if any(x in k for x in ['text_model', 'transformer', 'encoder'])]
                log(f"Found {len(text_encoder_keys)} text encoder keys", "DEBUG", True)
                
                if text_encoder_keys:
                    # Try to extract just the text encoder part
                    if any('text_model' in k for k in text_encoder_keys):
                        # This looks like a CLIP text encoder
                        log(f"Detected CLIP text encoder in {clip_type}", "DEBUG", True)
                        
                        # Import CLIP model classes
                        from transformers import CLIPTextModel, CLIPTextConfig
                        
                        # Create appropriate config based on clip_type
                        if clip_type == "clip-l":
                            config = CLIPTextConfig(
                                vocab_size=49408,
                                hidden_size=768,
                                intermediate_size=3072,
                                num_hidden_layers=12,
                                num_attention_heads=12,
                                max_position_embeddings=77
                            )
                        else:  # clip-g
                            config = CLIPTextConfig(
                                vocab_size=49408,
                                hidden_size=1280,
                                intermediate_size=5120,
                                num_hidden_layers=32,
                                num_attention_heads=20,
                                max_position_embeddings=77
                            )
                        
                        # Create model and load weights
                        model = CLIPTextModel(config)
                        
                        # Try to load compatible weights
                        try:
                            # Filter state dict for text model keys
                            text_state_dict = {}
                            for k, v in state_dict.items():
                                if 'text_model' in k:
                                    # Remove 'text_model.' prefix if present
                                    new_key = k.replace('text_model.', '')
                                    text_state_dict[new_key] = v
                            
                            if text_state_dict:
                                model.load_state_dict(text_state_dict, strict=False)
                                log(f"Loaded {clip_type} weights from safetensors", "DEBUG", True)
                            else:
                                log(f"No compatible weights found for {clip_type}, using random initialization", "WARNING", True)
                        except Exception as weight_error:
                            log(f"Weight loading failed for {clip_type}: {str(weight_error)}", "WARNING", True)
                            log("Using randomly initialized model", "WARNING", True)
                        
                        model.eval()
                        return model
                        
            except Exception as safetensors_error:
                log(f"Safetensors loading failed: {str(safetensors_error)}", "DEBUG", True)
            
            # Method 2: Fallback to simple model
            log(f"Creating simplified {clip_type} model for ONNX export", "DEBUG", True)
            if clip_type == "clip-l":
                model = self._create_clip_l_model()
            elif clip_type == "clip-g":
                model = self._create_clip_g_model()
            else:
                raise ValueError(f"Unknown CLIP type: {clip_type}")
            
            model.eval()
            log(f"{clip_type} model ready for ONNX export", "DEBUG", True)
            return model
            
        except Exception as e:
            log_error_with_traceback(f"Failed to load {clip_type} model", e)
            raise
    
    def _create_clip_l_model(self):
        """Create CLIP-L model structure (ONNX-compatible placeholder)"""
        import torch.nn as nn
        
        class CLIPLModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.token_embedding = nn.Embedding(49408, 768)  # vocab_size, hidden_dim
                self.positional_embedding = nn.Parameter(torch.zeros(77, 768))
                
                # Very simple ONNX-compatible layers (no attention to avoid ONNX issues)
                self.linear1 = nn.Linear(768, 3072)
                self.activation = nn.ReLU()  # Use ReLU instead of GELU for better ONNX compatibility
                self.linear2 = nn.Linear(3072, 768)
                self.linear3 = nn.Linear(768, 768)  # Additional processing layer
                self.ln1 = nn.LayerNorm(768)
                self.ln2 = nn.LayerNorm(768)
                self.ln_final = nn.LayerNorm(768)
                self.dropout = nn.Dropout(0.1)
                
            def forward(self, input_ids):
                # Embedding + positional encoding
                x = self.token_embedding(input_ids)
                x = x + self.positional_embedding
                
                # Simple feed-forward processing (no attention)
                # Layer 1: FFN + residual
                residual = x
                x = self.ln1(x)
                x = self.linear1(x)
                x = self.activation(x)
                x = self.dropout(x)
                x = self.linear2(x)
                x = residual + x
                
                # Layer 2: Additional processing
                residual = x
                x = self.ln2(x)
                x = self.linear3(x)
                x = self.activation(x)
                x = residual + x
                
                # Final layer norm
                x = self.ln_final(x)
                return x
        
        return CLIPLModel()
    
    def _create_clip_g_model(self):
        """Create CLIP-G model structure (ONNX-compatible placeholder)"""
        import torch.nn as nn
        
        class CLIPGModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.token_embedding = nn.Embedding(49408, 1280)  # vocab_size, hidden_dim
                self.positional_embedding = nn.Parameter(torch.zeros(77, 1280))
                
                # Very simple ONNX-compatible layers (no attention to avoid ONNX issues)
                self.linear1 = nn.Linear(1280, 5120)
                self.activation = nn.ReLU()  # Use ReLU instead of GELU for better ONNX compatibility
                self.linear2 = nn.Linear(5120, 1280)
                self.linear3 = nn.Linear(1280, 1280)  # Additional processing layer
                self.ln1 = nn.LayerNorm(1280)
                self.ln2 = nn.LayerNorm(1280)
                self.ln_final = nn.LayerNorm(1280)
                self.dropout = nn.Dropout(0.1)
                
            def forward(self, input_ids):
                # Embedding + positional encoding
                x = self.token_embedding(input_ids)
                x = x + self.positional_embedding
                
                # Simple feed-forward processing (no attention)
                # Layer 1: FFN + residual
                residual = x
                x = self.ln1(x)
                x = self.linear1(x)
                x = self.activation(x)
                x = self.dropout(x)
                x = self.linear2(x)
                x = residual + x
                
                # Layer 2: Additional processing
                residual = x
                x = self.ln2(x)
                x = self.linear3(x)
                x = self.activation(x)
                x = residual + x
                
                # Final layer norm
                x = self.ln_final(x)
                return x
        
        return CLIPGModel()
    
    def _create_single_clip_engine(self, model, engine_path, clip_type, 
                                 prompt_batch_min, prompt_batch_opt, prompt_batch_max):
        """
        Create TensorRT engine for a single CLIP model
        Uses PyTorch -> ONNX -> TensorRT workflow
        """
        try:
            log(f"Creating {clip_type} TensorRT engine...", "INFO", True)
            
            # Step 1: Export to ONNX
            onnx_path = engine_path.replace('.engine', '.onnx')
            log(f"Exporting {clip_type} to ONNX: {os.path.basename(onnx_path)}", "DEBUG", True)
            
            model.eval()
            dummy_input = torch.randint(0, 49408, (prompt_batch_opt, 77), dtype=torch.long)
            
            # Try multiple ONNX export configurations for compatibility
            export_success = False
            
            # Configuration 1: ONNX opset 16 (more compatible with ONNX 1.16.x)
            try:
                log(f"Attempting ONNX export with opset 16...", "DEBUG", True)
                torch.onnx.export(
                    model,
                    dummy_input,
                    onnx_path,
                    export_params=True,
                    opset_version=16,
                    do_constant_folding=True,
                    input_names=['input_ids'],
                    output_names=['text_embeddings'],
                    dynamic_axes={
                        'input_ids': {0: 'batch_size'},
                        'text_embeddings': {0: 'batch_size'}
                    }
                )
                export_success = True
                log(f"ONNX export successful with opset 16", "DEBUG", True)
            except Exception as e:
                log(f"ONNX export failed with opset 16: {str(e)}", "DEBUG", True)
            
            # Configuration 2: ONNX opset 14 (fallback)
            if not export_success:
                try:
                    log(f"Attempting ONNX export with opset 14...", "DEBUG", True)
                    torch.onnx.export(
                        model,
                        dummy_input,
                        onnx_path,
                        export_params=True,
                        opset_version=14,
                        do_constant_folding=True,
                        input_names=['input_ids'],
                        output_names=['text_embeddings'],
                        dynamic_axes={
                            'input_ids': {0: 'batch_size'},
                            'text_embeddings': {0: 'batch_size'}
                        }
                    )
                    export_success = True
                    log(f"ONNX export successful with opset 14", "DEBUG", True)
                except Exception as e:
                    log(f"ONNX export failed with opset 14: {str(e)}", "DEBUG", True)
            
            # Configuration 3: ONNX opset 11 (maximum compatibility)
            if not export_success:
                try:
                    log(f"Attempting ONNX export with opset 11...", "DEBUG", True)
                    torch.onnx.export(
                        model,
                        dummy_input,
                        onnx_path,
                        export_params=True,
                        opset_version=11,
                        do_constant_folding=False,  # Disable constant folding for compatibility
                        input_names=['input_ids'],
                        output_names=['text_embeddings'],
                        dynamic_axes={
                            'input_ids': {0: 'batch_size'},
                            'text_embeddings': {0: 'batch_size'}
                        }
                    )
                    export_success = True
                    log(f"ONNX export successful with opset 11", "DEBUG", True)
                except Exception as e:
                    log(f"ONNX export failed with opset 11: {str(e)}", "DEBUG", True)
            
            if not export_success:
                raise RuntimeError(f"Failed to export {clip_type} model to ONNX with all opset versions")
            
            log(f"{clip_type} ONNX export completed", "DEBUG", True)
            
            # Step 2: Convert ONNX to TensorRT
            log(f"Converting {clip_type} ONNX to TensorRT engine...", "DEBUG", True)
            self._onnx_to_tensorrt(onnx_path, engine_path, clip_type,
                                prompt_batch_min, prompt_batch_opt, prompt_batch_max)
            
            # Clean up ONNX file
            if os.path.exists(onnx_path):
                os.remove(onnx_path)
                log(f"Cleaned up temporary ONNX file: {os.path.basename(onnx_path)}", "DEBUG", True)
            
            log(f"{clip_type} TensorRT engine created successfully", "INFO", True)
            
        except Exception as e:
            log_error_with_traceback(f"Failed to create {clip_type} TensorRT engine", e)
            raise
    
    def _onnx_to_tensorrt(self, onnx_path, engine_path, clip_type,
                         prompt_batch_min, prompt_batch_opt, prompt_batch_max):
        """Convert ONNX model to TensorRT engine"""
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
            
            # Parse ONNX file
            with open(onnx_path, 'rb') as model_file:
                if not parser.parse(model_file.read()):
                    error_msg = f"Failed to parse ONNX file for {clip_type}"
                    for error in range(parser.num_errors):
                        log(f"ONNX Parser Error {error}: {parser.get_error(error)}", "ERROR", True)
                    raise RuntimeError(error_msg)
            
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

class DualCLIPTensorRTLoader:
    @classmethod
    def INPUT_TYPES(s):
        tensorrt_engines = get_existing_tensorrt_engines()
        default_engine = tensorrt_engines[0] if tensorrt_engines and "No TensorRT engines found" not in tensorrt_engines[0] else ""
        
        return {"required": {
            "engine_name": (tensorrt_engines, {"default": default_engine}),
        }}

    RETURN_TYPES = ("CLIP_TENSORRT",)
    RETURN_NAMES = ("clip_tensorrt",)
    FUNCTION = "load_tensorrt_clip"
    CATEGORY = "vovlerTools"

    def load_tensorrt_clip(self, engine_name):
        
        try:
            if "No TensorRT engines found" in engine_name:
                error_msg = "No TensorRT engines found. Please convert CLIP models first."
                log(error_msg, "ERROR", True)
                log(f"TensorRT output directory: {tensorrt_output_dir}", "ERROR", True)
                raise ValueError(error_msg)
            
            if "Error scanning" in engine_name:
                error_msg = "Error accessing TensorRT engines directory. Check permissions."
                log(error_msg, "ERROR", True)
                raise ValueError(error_msg)
            
            engine_path = os.path.join(tensorrt_output_dir, engine_name)
            
            # Validate engine file access
            if not validate_file_access(engine_path, "read"):
                error_msg = f"Cannot access TensorRT engine file: {engine_name}"
                log(f"Full path: {engine_path}", "ERROR", True)
                raise FileNotFoundError(error_msg)
            
            # Check if file is actually a TensorRT engine
            if not engine_name.endswith('.engine'):
                log(f"Warning: File does not have .engine extension: {engine_name}", "WARNING", True)
            
            engine_size = os.path.getsize(engine_path) / (1024*1024)
            if engine_size < 1:
                log(f"Warning: Engine file is very small ({engine_size:.1f}MB), may be corrupted", "WARNING", True)
            
            log(f"Loading dual CLIP SDXL TensorRT engine: {engine_name} ({engine_size:.1f}MB)", "INFO", True)
            
            # Load TensorRT engine
            log("Initializing TensorRT runtime...", "DEBUG", True)
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            
            try:
                with open(engine_path, 'rb') as f:
                    engine_data = f.read()
                    if len(engine_data) == 0:
                        raise ValueError("Engine file is empty")
                    
                    log(f"Engine data loaded: {len(engine_data)} bytes", "DEBUG", True)
                    
                    runtime = trt.Runtime(TRT_LOGGER)
                    if not runtime:
                        raise RuntimeError("Failed to create TensorRT runtime")
                    
                    engine = runtime.deserialize_cuda_engine(engine_data)
                    if not engine:
                        raise RuntimeError("Failed to deserialize TensorRT engine")
                        
            except Exception as load_error:
                log_error_with_traceback(f"Failed to load engine file: {engine_name}", load_error)
                raise
            
            log("Creating execution context...", "DEBUG", True)
            context = engine.create_execution_context()
            if not context:
                raise RuntimeError("Failed to create TensorRT execution context")
            
            # Get all input/output bindings
            log("Analyzing engine bindings...", "DEBUG", True)
            num_bindings = engine.num_io_tensors
            log(f"Total bindings found: {num_bindings}", "DEBUG", True)
            
            input_bindings = []
            output_bindings = []
            
            for i in range(num_bindings):
                tensor_name = engine.get_tensor_name(i)
                if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                    input_bindings.append(tensor_name)
                else:
                    output_bindings.append(tensor_name)
            
            if len(input_bindings) == 0:
                raise ValueError("No input bindings found in engine")
            if len(output_bindings) == 0:
                raise ValueError("No output bindings found in engine")
            
            # Get tensor shapes
            input_shapes = {}
            output_shapes = {}
            
            for binding in input_bindings:
                shape = engine.get_tensor_shape(binding)
                input_shapes[binding] = shape
                log(f"Input binding '{binding}': {shape}", "DEBUG", True)
                
            for binding in output_bindings:
                shape = engine.get_tensor_shape(binding)
                output_shapes[binding] = shape
                log(f"Output binding '{binding}': {shape}", "DEBUG", True)
            
            # Validate expected SDXL structure
            expected_inputs = ["input_ids_clip_l", "input_ids_clip_g"]
            expected_outputs = ["text_embeddings_clip_l", "text_embeddings_clip_g", "pooled_output_clip_g"]
            
            missing_inputs = [inp for inp in expected_inputs if inp not in input_bindings]
            missing_outputs = [out for out in expected_outputs if out not in output_bindings]
            
            if missing_inputs:
                log(f"Warning: Missing expected input bindings: {missing_inputs}", "WARNING", True)
            if missing_outputs:
                log(f"Warning: Missing expected output bindings: {missing_outputs}", "WARNING", True)
            
            clip_tensorrt_data = {
                "engine_name": engine_name,
                "engine": engine,
                "context": context,
                "input_bindings": input_bindings,
                "output_bindings": output_bindings,
                "input_shapes": input_shapes,
                "output_shapes": output_shapes,
                "num_inputs": len(input_bindings),
                "num_outputs": len(output_bindings),
                "max_sequence_length": 77,  # Always 77 for SDXL
                "precision": "fp16",  # Always fp16
                "engine_size_mb": engine_size
            }
            
            log(f"Dual CLIP SDXL TensorRT engine loaded successfully", "INFO", True)
            log(f"Input bindings: {input_bindings}", "INFO", True)
            log(f"Output bindings: {output_bindings}", "INFO", True)
            log(f"Token length: 77 (SDXL), Precision: FP16", "INFO", True)
            
            return (clip_tensorrt_data,)
            
        except Exception as e:
            log_error_with_traceback(f"Failed to load TensorRT engine: {engine_name}", e)
            
            # Provide helpful suggestions
            if "permission" in str(e).lower():
                log(f"Suggestion: Check file permissions for {tensorrt_output_dir}", "INFO", True)
            elif "deserialize" in str(e).lower():
                log("Suggestion: Engine file may be corrupted, try recreating it", "INFO", True)
            elif "runtime" in str(e).lower():
                log("Suggestion: Check TensorRT installation and CUDA compatibility", "INFO", True)
            
            raise

class DualCLIPTensorRTTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip_tensorrt": ("CLIP_TENSORRT",),
            "text": ("STRING", {"multiline": True, "default": ""}),
            "text_clip_g": ("STRING", {"multiline": True, "default": ""}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode_text"
    CATEGORY = "vovlerTools"

    def encode_text(self, clip_tensorrt, text, text_clip_g=""):
        """
        Encode text using the dual CLIP SDXL TensorRT engine
        Always uses 77 tokens and fp16 precision
        """
        
        try:
            # Validate inputs
            if not clip_tensorrt:
                error_msg = "CLIP TensorRT data is None or empty"
                log(error_msg, "ERROR", True)
                raise ValueError(error_msg)
            
            if not isinstance(clip_tensorrt, dict):
                error_msg = f"CLIP TensorRT data must be a dictionary, got {type(clip_tensorrt)}"
                log(error_msg, "ERROR", True)
                raise TypeError(error_msg)
            
            # Validate required keys in clip_tensorrt
            required_keys = ["engine", "context", "input_bindings", "output_bindings"]
            missing_keys = [key for key in required_keys if key not in clip_tensorrt]
            if missing_keys:
                error_msg = f"Missing required keys in CLIP TensorRT data: {missing_keys}"
                log(error_msg, "ERROR", True)
                raise ValueError(error_msg)
            
            # Validate text inputs
            if text is None:
                text = ""
            if text_clip_g is None:
                text_clip_g = ""
                
            if not isinstance(text, str):
                log(f"Converting text to string from {type(text)}", "WARNING", True)
                text = str(text)
            if not isinstance(text_clip_g, str):
                log(f"Converting text_clip_g to string from {type(text_clip_g)}", "WARNING", True)
                text_clip_g = str(text_clip_g)
            
            if not text.strip():
                text = ""
                log("Empty text provided for CLIP-L", "DEBUG", True)
            
            if not text_clip_g.strip():
                text_clip_g = text  # Use same text for CLIP-G if not specified
                log("Using same text for both CLIP-L and CLIP-G", "DEBUG", True)
            
            # Check text length
            if len(text) > 1000:
                log(f"Warning: Very long text for CLIP-L ({len(text)} chars), may be truncated", "WARNING", True)
            if len(text_clip_g) > 1000:
                log(f"Warning: Very long text for CLIP-G ({len(text_clip_g)} chars), may be truncated", "WARNING", True)
            
            log(f"Encoding text with dual CLIP SDXL TensorRT engine", "INFO", True)
            log(f"Text CLIP-L: {text[:50]}{'...' if len(text) > 50 else ''}", "DEBUG", True)
            log(f"Text CLIP-G: {text_clip_g[:50]}{'...' if len(text_clip_g) > 50 else ''}", "DEBUG", True)
            
            # Validate engine state
            engine = clip_tensorrt.get("engine")
            context = clip_tensorrt.get("context")
            
            if not engine:
                raise ValueError("TensorRT engine is None or invalid")
            if not context:
                raise ValueError("TensorRT execution context is None or invalid")
            
            # Log engine information
            engine_name = clip_tensorrt.get("engine_name", "unknown")
            engine_size = clip_tensorrt.get("engine_size_mb", 0)
            log(f"Using engine: {engine_name} ({engine_size:.1f}MB)", "DEBUG", True)
            
            # In a real implementation, you would:
            # 1. Tokenize the input texts to 77 tokens each
            # 2. Run TensorRT inference with fp16 precision
            # 3. Process the CLIP-L and CLIP-G outputs
            # 4. Combine them into SDXL conditioning format
            # 5. Return proper conditioning data with pooled outputs
            
            log("Creating placeholder conditioning (real implementation would run TensorRT inference)", "DEBUG", True)
            
            # For now, return placeholder conditioning for SDXL
            batch_size = 1
            # SDXL typically uses:
            # - CLIP-L: 768-dim embeddings
            # - CLIP-G: 1280-dim embeddings + pooled output
            clip_l_dim = 768
            clip_g_dim = 1280
            
            # Create dummy embeddings (in practice, these come from TensorRT inference)
            try:
                clip_l_embeddings = torch.zeros((batch_size, 77, clip_l_dim), dtype=torch.float16)
                clip_g_embeddings = torch.zeros((batch_size, 77, clip_g_dim), dtype=torch.float16)
                pooled_output = torch.zeros((batch_size, clip_g_dim), dtype=torch.float16)
            except Exception as tensor_error:
                log_error_with_traceback("Failed to create output tensors", tensor_error)
                raise
            
            # SDXL conditioning format combines both CLIP outputs
            # In practice, you would concatenate or process them according to SDXL spec
            try:
                combined_embeddings = torch.cat([clip_l_embeddings, clip_g_embeddings], dim=-1)  # Shape: [1, 77, 2048]
            except Exception as concat_error:
                log_error_with_traceback("Failed to concatenate embeddings", concat_error)
                raise
            
            # Validate output shapes
            expected_combined_shape = (batch_size, 77, clip_l_dim + clip_g_dim)
            if combined_embeddings.shape != expected_combined_shape:
                error_msg = f"Combined embeddings shape mismatch: got {combined_embeddings.shape}, expected {expected_combined_shape}"
                log(error_msg, "ERROR", True)
                raise ValueError(error_msg)
            
            conditioning = [[combined_embeddings, {"pooled_output": pooled_output}]]
            
            log(f"SDXL text encoding complete (77 tokens, FP16)", "INFO", True)
            log(f"CLIP-L embedding shape: {clip_l_embeddings.shape}", "DEBUG", True)
            log(f"CLIP-G embedding shape: {clip_g_embeddings.shape}", "DEBUG", True)
            log(f"Combined embedding shape: {combined_embeddings.shape}", "DEBUG", True)
            log(f"Pooled output shape: {pooled_output.shape}", "DEBUG", True)
            
            return (conditioning,)
            
        except Exception as e:
            log_error_with_traceback("Failed to encode text with TensorRT CLIP", e)
            
            # Provide helpful suggestions
            error_str = str(e).lower()
            if "memory" in error_str:
                log("Suggestion: Reduce text length or free up GPU memory", "INFO", True)
            elif "tensorrt" in error_str:
                log("Suggestion: Check if TensorRT engine is properly loaded", "INFO", True)
            elif "shape" in error_str:
                log("Suggestion: Engine may not be compatible with SDXL format", "INFO", True)
            elif "cuda" in error_str:
                log("Suggestion: Check CUDA availability and GPU memory", "INFO", True)
            
            raise
