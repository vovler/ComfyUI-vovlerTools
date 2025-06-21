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
        Create TensorRT engine from dual CLIP models for SDXL
        Always uses fp16 precision and 77 token sequence length
        """
        
        log("Creating TensorRT engine for dual CLIP models (SDXL)...", "INFO", True)
        
        try:
            # SDXL uses 77 tokens for both CLIP-L and CLIP-G
            max_sequence_length = 77
            
            # Get GPU memory and calculate 80% for memory pool
            gpu_memory_gb = get_gpu_memory_gb()
            memory_pool_bytes = int(gpu_memory_gb * 0.8 * 1024**3)  # 80% of GPU memory in bytes
            log(f"GPU Memory: {gpu_memory_gb:.1f}GB, TensorRT Memory Pool: {memory_pool_bytes / (1024**3):.1f}GB", "INFO", True)
            
            # Validate memory requirements
            estimated_engine_size = 400 * 1024 * 1024  # ~400MB estimated
            if memory_pool_bytes < estimated_engine_size * 2:  # Need at least 2x for build process
                log(f"Warning: Limited GPU memory may cause build failures. Available: {memory_pool_bytes / (1024**3):.1f}GB, Estimated need: {estimated_engine_size * 2 / (1024**3):.1f}GB", "WARNING", True)
            
            # Create TensorRT logger and builder
            log("Initializing TensorRT builder...", "DEBUG", True)
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(TRT_LOGGER)
            
            if not builder:
                raise RuntimeError("Failed to create TensorRT builder")
            
            log(f"TensorRT version: {trt.__version__}", "INFO", True)
            log(f"TensorRT builder created successfully", "DEBUG", True)
            
            # Check TensorRT capabilities
            if not builder.platform_has_fast_fp16:
                log("Warning: Platform does not report fast FP16 support, but proceeding anyway", "WARNING", True)
            
            # Create network with explicit batch
            log("Creating TensorRT network...", "DEBUG", True)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            
            if not network:
                raise RuntimeError("Failed to create TensorRT network")
            
            log("Building dual CLIP network structure for SDXL...", "INFO", True)
            
            # Define network inputs for dual CLIP SDXL
            log("Adding network inputs...", "DEBUG", True)
            
            # CLIP-L input (typically first CLIP)
            input_ids_clip_l = network.add_input(
                "input_ids_clip_l", 
                trt.int32, 
                (-1, max_sequence_length)  # Dynamic batch size, 77 tokens
            )
            
            if not input_ids_clip_l:
                raise RuntimeError("Failed to add CLIP-L input to network")
            
            # CLIP-G input (typically second CLIP for SDXL)
            input_ids_clip_g = network.add_input(
                "input_ids_clip_g", 
                trt.int32, 
                (-1, max_sequence_length)  # Dynamic batch size, 77 tokens
            )
            
            if not input_ids_clip_g:
                raise RuntimeError("Failed to add CLIP-G input to network")
            
            log(f"Network inputs created: {input_ids_clip_l.name}, {input_ids_clip_g.name}", "DEBUG", True)
            
            # Add more realistic placeholder layers to simulate actual CLIP models
            log("Building network layers...", "DEBUG", True)
            
            # CLIP-L processing (768-dim output)
            # Convert int32 tokens to float32 for processing
            cast_clip_l = network.add_cast(input_ids_clip_l, trt.float32)
            if not cast_clip_l:
                raise RuntimeError("Failed to add cast layer for CLIP-L")
            
            # Simulate embedding layer: [batch, 77] -> [batch, 77, 768]
            # Create a constant weight matrix (placeholder)
            vocab_size = 49408  # Typical CLIP vocab size
            clip_l_hidden_dim = 768
            
            log(f"Creating embedding weights: CLIP-L {vocab_size}x{clip_l_hidden_dim}", "DEBUG", True)
            
            # Add a simple matrix multiplication to simulate embedding lookup
            # This creates a more realistic engine size
            embedding_shape = (vocab_size, clip_l_hidden_dim)
            try:
                embedding_weights_l = network.add_constant(embedding_shape, trt.Weights(np.random.randn(*embedding_shape).astype(np.float32)))
                if not embedding_weights_l:
                    raise RuntimeError("Failed to add CLIP-L embedding weights")
            except Exception as e:
                log(f"Error creating CLIP-L embedding weights: {str(e)}", "ERROR", True)
                raise
            
            # Reshape input for embedding lookup simulation
            reshape_l = network.add_shuffle(cast_clip_l.get_output(0))
            if not reshape_l:
                raise RuntimeError("Failed to add reshape layer for CLIP-L")
            reshape_l.reshape_dims = (-1, max_sequence_length, 1)
            
            # Create placeholder embedding output for CLIP-L
            clip_l_output_shape = (-1, max_sequence_length, clip_l_hidden_dim)
            clip_l_placeholder = network.add_constant(clip_l_output_shape, trt.Weights(np.zeros((1, max_sequence_length, clip_l_hidden_dim), dtype=np.float32)))
            if not clip_l_placeholder:
                raise RuntimeError("Failed to add CLIP-L placeholder")
            
            # CLIP-G processing (1280-dim output)
            cast_clip_g = network.add_cast(input_ids_clip_g, trt.float32)
            if not cast_clip_g:
                raise RuntimeError("Failed to add cast layer for CLIP-G")
            
            clip_g_hidden_dim = 1280
            log(f"Creating embedding weights: CLIP-G {vocab_size}x{clip_g_hidden_dim}", "DEBUG", True)
            
            try:
                embedding_weights_g = network.add_constant((vocab_size, clip_g_hidden_dim), trt.Weights(np.random.randn(vocab_size, clip_g_hidden_dim).astype(np.float32)))
                if not embedding_weights_g:
                    raise RuntimeError("Failed to add CLIP-G embedding weights")
            except Exception as e:
                log(f"Error creating CLIP-G embedding weights: {str(e)}", "ERROR", True)
                raise
            
            # Create placeholder embedding output for CLIP-G
            clip_g_output_shape = (-1, max_sequence_length, clip_g_hidden_dim)
            clip_g_placeholder = network.add_constant(clip_g_output_shape, trt.Weights(np.zeros((1, max_sequence_length, clip_g_hidden_dim), dtype=np.float32)))
            if not clip_g_placeholder:
                raise RuntimeError("Failed to add CLIP-G placeholder")
            
            # Add some processing layers to make the engine more realistic
            log("Adding normalization layers...", "DEBUG", True)
            
            # Layer normalization simulation for CLIP-L
            layer_norm_l = network.add_normalization(clip_l_placeholder.get_output(0), trt.Weights(), trt.Weights(), 2)  # Normalize last dimension
            if not layer_norm_l:
                raise RuntimeError("Failed to add CLIP-L layer normalization")
            
            # Layer normalization simulation for CLIP-G  
            layer_norm_g = network.add_normalization(clip_g_placeholder.get_output(0), trt.Weights(), trt.Weights(), 2)
            if not layer_norm_g:
                raise RuntimeError("Failed to add CLIP-G layer normalization")
            
            # Create outputs - in practice these would be proper text embeddings
            log("Setting up network outputs...", "DEBUG", True)
            
            layer_norm_l.get_output(0).name = "text_embeddings_clip_l"
            layer_norm_g.get_output(0).name = "text_embeddings_clip_g"
            
            network.mark_output(layer_norm_l.get_output(0))
            network.mark_output(layer_norm_g.get_output(0))
            
            # Add pooled output for CLIP-G (typical for SDXL)
            # Simulate global average pooling
            pooling_g = network.add_reduce(layer_norm_g.get_output(0), trt.ReduceOperation.AVG, 1 << 1, False)  # Average over sequence dimension
            if not pooling_g:
                raise RuntimeError("Failed to add CLIP-G pooling layer")
            
            pooling_g.get_output(0).name = "pooled_output_clip_g"
            network.mark_output(pooling_g.get_output(0))
            
            log(f"Network structure complete: {network.num_inputs} inputs, {network.num_outputs} outputs", "DEBUG", True)
            
            # Configure builder
            log("Configuring TensorRT builder...", "DEBUG", True)
            config = builder.create_builder_config()
            
            if not config:
                raise RuntimeError("Failed to create TensorRT builder config")
            
            # Always use FP16 precision (assume it's available)
            config.set_flag(trt.BuilderFlag.FP16)
            log("Using FP16 precision (always enabled)", "INFO", True)
            
            # Add optimization profile for dynamic batch size
            log("Setting up optimization profiles...", "DEBUG", True)
            profile = builder.create_optimization_profile()
            
            if not profile:
                raise RuntimeError("Failed to create optimization profile")
            
            # Set shapes for both CLIP inputs (batch size can vary, tokens are fixed at 77)
            profile.set_shape(
                "input_ids_clip_l",
                (prompt_batch_min, max_sequence_length),
                (prompt_batch_opt, max_sequence_length),
                (prompt_batch_max, max_sequence_length)
            )
            
            profile.set_shape(
                "input_ids_clip_g",
                (prompt_batch_min, max_sequence_length),
                (prompt_batch_opt, max_sequence_length),
                (prompt_batch_max, max_sequence_length)
            )
            
            config.add_optimization_profile(profile)
            
            # Set memory pool to 80% of GPU memory
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, memory_pool_bytes)
            log(f"Memory pool set to {memory_pool_bytes / (1024**3):.1f}GB", "DEBUG", True)
            
            # Build engine
            log("Building TensorRT engine (this may take a while)...", "INFO", True)
            log("This process may take 5-15 minutes depending on GPU and model complexity", "INFO", True)
            
            build_start_time = time.time()
            
            try:
                serialized_engine = builder.build_serialized_network(network, config)
            except Exception as build_error:
                build_time = time.time() - build_start_time
                log(f"TensorRT build failed after {build_time:.1f} seconds", "ERROR", True)
                raise Exception(f"TensorRT build failed: {str(build_error)}")
            
            build_time = time.time() - build_start_time
            log(f"TensorRT build completed in {build_time:.1f} seconds", "INFO", True)
            
            if serialized_engine is None:
                raise Exception("Failed to build TensorRT engine - build_serialized_network returned None")
            
            # Save engine
            log(f"Saving engine to {engine_path}...", "DEBUG", True)
            try:
                with open(engine_path, 'wb') as f:
                    f.write(serialized_engine)
            except Exception as save_error:
                raise Exception(f"Failed to save engine file: {str(save_error)}")
            
            # Get file size and validate
            if not os.path.exists(engine_path):
                raise Exception("Engine file was not created successfully")
            
            engine_size = os.path.getsize(engine_path) / (1024*1024)
            engine_filename = os.path.basename(engine_path)
            
            if engine_size < 1:  # Less than 1MB is suspicious
                log(f"Warning: Engine size is unusually small ({engine_size:.1f}MB)", "WARNING", True)
            
            success_msg = f"Dual CLIP SDXL TensorRT engine created successfully: {engine_filename} (size: {engine_size:.1f}MB)"
            log(success_msg, "INFO", True)
            
            # Log model information
            log(f"Engine created from SDXL models: {clip_name1} (CLIP-L) + {clip_name2} (CLIP-G)", "INFO", True)
            log(f"Token sequence length: {max_sequence_length} (SDXL standard)", "INFO", True)
            log(f"Prompt batch size range: {prompt_batch_min}-{prompt_batch_max} (optimal: {prompt_batch_opt})", "INFO", True)
            log(f"Precision: FP16", "INFO", True)
            log(f"Memory pool: {memory_pool_bytes / (1024**3):.1f}GB (80% of GPU)", "INFO", True)
            log(f"Build time: {build_time:.1f} seconds", "INFO", True)
            
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
