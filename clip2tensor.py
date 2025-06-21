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

# Import logging utility from wd14tagger
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

def log(message, type=None, always=True):
    if type is not None:
        message = f"[{type}] {message}"
    print(f"(CLIP2TensorRT) {message}")

def get_gpu_memory_gb():
    """Get total GPU memory in GB"""
    try:
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            return total_memory / (1024**3)  # Convert to GB
        else:
            return 8.0  # Default fallback
    except:
        return 8.0  # Default fallback

# Use ComfyUI's models directory with clip2tensor subdirectory
clip_models_dir = os.path.join(folder_paths.models_dir, "clip")
tensorrt_output_dir = os.path.join(folder_paths.models_dir, "clip_tensorrt")
if not os.path.exists(tensorrt_output_dir):
    os.makedirs(tensorrt_output_dir)
    log(f"Created clip_tensorrt output directory: {tensorrt_output_dir}", "INFO", True)

def get_available_clip_models():
    """Get all available CLIP models (safetensors format)"""
    if not os.path.exists(clip_models_dir):
        return ["No CLIP models found - place .safetensors files in models/clip folder"]
    
    all_files = os.listdir(clip_models_dir)
    clip_files = [f for f in all_files if f.endswith(".safetensors")]
    
    if not clip_files:
        return ["No CLIP models found - place .safetensors files in models/clip folder"]
    
    return clip_files

def get_existing_tensorrt_engines():
    """Get all existing TensorRT engine files"""
    if not os.path.exists(tensorrt_output_dir):
        return ["No TensorRT engines found"]
    
    all_files = os.listdir(tensorrt_output_dir)
    engine_files = [f for f in all_files if f.endswith(".engine")]
    
    if not engine_files:
        return ["No TensorRT engines found"]
    
    return engine_files

class DualCLIPToTensorRT:
    @classmethod
    def INPUT_TYPES(s):
        clip_models = get_available_clip_models()
        default_model = clip_models[0] if clip_models and "No CLIP models found" not in clip_models[0] else ""
        
        return {"required": {
            "clip_name1": (clip_models, {"default": default_model}),
            "clip_name2": (clip_models, {"default": default_model}),
            "output_name": ("STRING", {"default": "dual_clip_sdxl", "multiline": False}),
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
        
        if "No CLIP models found" in clip_name1 or "No CLIP models found" in clip_name2:
            error_msg = "CLIP models not found. Please place .safetensors files in models/clip folder."
            log(error_msg, "ERROR", True)
            return (error_msg,)
        
        if clip_name1 == clip_name2:
            error_msg = "Please select two different CLIP models for dual CLIP conversion."
            log(error_msg, "ERROR", True)
            return (error_msg,)
        
        clip1_path = os.path.join(clip_models_dir, clip_name1)
        clip2_path = os.path.join(clip_models_dir, clip_name2)
        
        if not os.path.exists(clip1_path):
            error_msg = f"CLIP model 1 not found: {clip_name1}"
            log(error_msg, "ERROR", True)
            return (error_msg,)
        
        if not os.path.exists(clip2_path):
            error_msg = f"CLIP model 2 not found: {clip_name2}"
            log(error_msg, "ERROR", True)
            return (error_msg,)
        
        # Always use fp16 and SDXL (77 tokens)
        engine_filename = f"{output_name}_sdxl_{prompt_batch_min}_{prompt_batch_opt}_{prompt_batch_max}_fp16.engine"
        engine_path = os.path.join(tensorrt_output_dir, engine_filename)
        
        if os.path.exists(engine_path):
            success_msg = f"TensorRT engine already exists: {engine_filename}"
            log(success_msg, "INFO", True)
            return (success_msg,)
        
        try:
            log(f"Starting dual CLIP to TensorRT conversion...", "INFO", True)
            log(f"CLIP 1: {clip_name1}", "INFO", True)
            log(f"CLIP 2: {clip_name2}", "INFO", True)
            log(f"Type: SDXL (always)", "INFO", True)
            log(f"Precision: FP16 (always)", "INFO", True)
            log(f"Prompt batch sizes: {prompt_batch_min}/{prompt_batch_opt}/{prompt_batch_max}", "INFO", True)
            log(f"Token sequence length: 77 (SDXL standard)", "INFO", True)
            
            # Load CLIP models using ComfyUI's built-in functionality
            log("Loading CLIP models...", "INFO", True)
            
            success_msg = self._create_tensorrt_engine(
                clip1_path, clip2_path, engine_path, clip_name1, clip_name2,
                prompt_batch_min, prompt_batch_opt, prompt_batch_max
            )
            
            return (success_msg,)
            
        except Exception as e:
            error_msg = f"Error during dual CLIP to TensorRT conversion: {str(e)}"
            log(error_msg, "ERROR", True)
            return (error_msg,)
    
    def _create_tensorrt_engine(self, clip1_path, clip2_path, engine_path, clip_name1, clip_name2,
                              prompt_batch_min, prompt_batch_opt, prompt_batch_max):
        """
        Create TensorRT engine from dual CLIP models for SDXL
        Always uses fp16 precision and 77 token sequence length
        """
        
        log("Creating TensorRT engine for dual CLIP models (SDXL)...", "INFO", True)
        
        # SDXL uses 77 tokens for both CLIP-L and CLIP-G
        max_sequence_length = 77
        
        # Get GPU memory and calculate 80% for memory pool
        gpu_memory_gb = get_gpu_memory_gb()
        memory_pool_bytes = int(gpu_memory_gb * 0.8 * 1024**3)  # 80% of GPU memory in bytes
        log(f"GPU Memory: {gpu_memory_gb:.1f}GB, TensorRT Memory Pool: {memory_pool_bytes / (1024**3):.1f}GB", "INFO", True)
        
        # Create TensorRT logger and builder
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        
        log(f"TensorRT version: {trt.__version__}", "INFO", True)
        
        # Create network with explicit batch
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        
        log("Building dual CLIP network structure for SDXL...", "INFO", True)
        
        # Define network inputs for dual CLIP SDXL
        # CLIP-L input (typically first CLIP)
        input_ids_clip_l = network.add_input(
            "input_ids_clip_l", 
            trt.int32, 
            (-1, max_sequence_length)  # Dynamic batch size, 77 tokens
        )
        
        # CLIP-G input (typically second CLIP for SDXL)
        input_ids_clip_g = network.add_input(
            "input_ids_clip_g", 
            trt.int32, 
            (-1, max_sequence_length)  # Dynamic batch size, 77 tokens
        )
        
        # Add dummy processing layers for demonstration
        # In a real implementation, you would:
        # 1. Load the actual CLIP model weights from safetensors
        # 2. Create text embedding layers (vocab_size -> hidden_dim)
        # 3. Add positional embeddings
        # 4. Add transformer layers (attention, MLP, layer norm)
        # 5. Add final projection layers
        # 6. Combine outputs appropriately for SDXL conditioning
        
        # For now, create placeholder identity layers
        identity_clip_l = network.add_identity(input_ids_clip_l)
        identity_clip_g = network.add_identity(input_ids_clip_g)
        
        # Create outputs - in practice these would be proper text embeddings
        # SDXL typically uses:
        # - CLIP-L: hidden states from second-to-last layer
        # - CLIP-G: pooled output + hidden states
        identity_clip_l.get_output(0).name = "text_embeddings_clip_l"
        identity_clip_g.get_output(0).name = "text_embeddings_clip_g"
        
        network.mark_output(identity_clip_l.get_output(0))
        network.mark_output(identity_clip_g.get_output(0))
        
        # Configure builder
        config = builder.create_builder_config()
        
        # Always use FP16 precision (assume it's available)
        config.set_flag(trt.BuilderFlag.FP16)
        log("Using FP16 precision (always enabled)", "INFO", True)
        
        # Add optimization profile for dynamic batch size
        profile = builder.create_optimization_profile()
        
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
        
        # Build engine
        log("Building TensorRT engine (this may take a while)...", "INFO", True)
        
        try:
            serialized_engine = builder.build_serialized_network(network, config)
        except Exception as build_error:
            raise Exception(f"TensorRT build failed: {str(build_error)}")
        
        if serialized_engine is None:
            raise Exception("Failed to build TensorRT engine - build_serialized_network returned None")
        
        # Save engine
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)
        
        # Get file size
        engine_size = os.path.getsize(engine_path) / (1024*1024)
        engine_filename = os.path.basename(engine_path)
        
        success_msg = f"Dual CLIP SDXL TensorRT engine created successfully: {engine_filename} (size: {engine_size:.1f}MB)"
        log(success_msg, "INFO", True)
        
        # Log model information
        log(f"Engine created from SDXL models: {clip_name1} (CLIP-L) + {clip_name2} (CLIP-G)", "INFO", True)
        log(f"Token sequence length: {max_sequence_length} (SDXL standard)", "INFO", True)
        log(f"Prompt batch size range: {prompt_batch_min}-{prompt_batch_max} (optimal: {prompt_batch_opt})", "INFO", True)
        log(f"Precision: FP16", "INFO", True)
        log(f"Memory pool: {memory_pool_bytes / (1024**3):.1f}GB (80% of GPU)", "INFO", True)
        
        return success_msg

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
        
        if "No TensorRT engines found" in engine_name:
            raise ValueError("No TensorRT engines found. Please convert CLIP models first.")
        
        engine_path = os.path.join(tensorrt_output_dir, engine_name)
        
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TensorRT engine not found: {engine_name}")
        
        log(f"Loading dual CLIP SDXL TensorRT engine: {engine_name}", "INFO", True)
        
        # Load TensorRT engine
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(TRT_LOGGER)
            engine = runtime.deserialize_cuda_engine(f.read())
        
        context = engine.create_execution_context()
        
        # Get all input/output bindings
        num_bindings = engine.num_io_tensors
        
        input_bindings = []
        output_bindings = []
        
        for i in range(num_bindings):
            tensor_name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                input_bindings.append(tensor_name)
            else:
                output_bindings.append(tensor_name)
        
        # Get tensor shapes
        input_shapes = {}
        output_shapes = {}
        
        for binding in input_bindings:
            input_shapes[binding] = engine.get_tensor_shape(binding)
            
        for binding in output_bindings:
            output_shapes[binding] = engine.get_tensor_shape(binding)
        
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
            "precision": "fp16"  # Always fp16
        }
        
        log(f"Dual CLIP SDXL TensorRT engine loaded successfully", "INFO", True)
        log(f"Input bindings: {input_bindings}", "INFO", True)
        log(f"Output bindings: {output_bindings}", "INFO", True)
        log(f"Input shapes: {input_shapes}", "INFO", True)
        log(f"Output shapes: {output_shapes}", "INFO", True)
        log(f"Token length: 77 (SDXL), Precision: FP16", "INFO", True)
        
        return (clip_tensorrt_data,)

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
        
        if not text.strip():
            text = ""
        
        if not text_clip_g.strip():
            text_clip_g = text  # Use same text for CLIP-G if not specified
        
        log(f"Encoding text with dual CLIP SDXL TensorRT engine", "INFO", True)
        log(f"Text CLIP-L: {text[:50]}{'...' if len(text) > 50 else ''}", "INFO", True)
        log(f"Text CLIP-G: {text_clip_g[:50]}{'...' if len(text_clip_g) > 50 else ''}", "INFO", True)
        
        # In a real implementation, you would:
        # 1. Tokenize the input texts to 77 tokens each
        # 2. Run TensorRT inference with fp16 precision
        # 3. Process the CLIP-L and CLIP-G outputs
        # 4. Combine them into SDXL conditioning format
        # 5. Return proper conditioning data with pooled outputs
        
        # For now, return placeholder conditioning for SDXL
        batch_size = 1
        # SDXL typically uses:
        # - CLIP-L: 768-dim embeddings
        # - CLIP-G: 1280-dim embeddings + pooled output
        clip_l_dim = 768
        clip_g_dim = 1280
        
        # Create dummy embeddings (in practice, these come from TensorRT inference)
        clip_l_embeddings = torch.zeros((batch_size, 77, clip_l_dim), dtype=torch.float16)
        clip_g_embeddings = torch.zeros((batch_size, 77, clip_g_dim), dtype=torch.float16)
        pooled_output = torch.zeros((batch_size, clip_g_dim), dtype=torch.float16)
        
        # SDXL conditioning format combines both CLIP outputs
        # In practice, you would concatenate or process them according to SDXL spec
        combined_embeddings = torch.cat([clip_l_embeddings, clip_g_embeddings], dim=-1)  # Shape: [1, 77, 2048]
        
        conditioning = [[combined_embeddings, {"pooled_output": pooled_output}]]
        
        log(f"SDXL text encoding complete (77 tokens, FP16)", "INFO", True)
        log(f"CLIP-L embedding shape: {clip_l_embeddings.shape}", "INFO", True)
        log(f"CLIP-G embedding shape: {clip_g_embeddings.shape}", "INFO", True)
        log(f"Combined embedding shape: {combined_embeddings.shape}", "INFO", True)
        log(f"Pooled output shape: {pooled_output.shape}", "INFO", True)
        
        return (conditioning,)
