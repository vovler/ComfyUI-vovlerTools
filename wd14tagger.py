import comfy.utils
import numpy as np
import csv
import os
import sys
import time
from PIL import Image
import folder_paths
import torch
import tensorrt as trt

# ========== UTILITY FUNCTIONS ==========

def get_ext_dir(subpath=None, mkdir=False):
    dir = os.path.dirname(__file__)
    if subpath is not None:
        dir = os.path.join(dir, subpath)
    dir = os.path.abspath(dir)
    if mkdir and not os.path.exists(dir):
        os.makedirs(dir)
    return dir

def log(message, type=None, always=True):
    if type is not None:
        message = f"[{type}] {message}"
    print(f"(WD14Tagger) {message}")





# ========== END UTILITY FUNCTIONS ==========


sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))



# Use ComfyUI's models directory with wdtagger subdirectory
models_dir = os.path.join(folder_paths.models_dir, "wdtagger")
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    log(f"Created wdtagger models directory: {models_dir}", "INFO", True)


def get_installed_models_onnx():
    """Get all ONNX models that have corresponding CSV files"""
    all_files = os.listdir(models_dir)
    onnx_files = [f for f in all_files if f.endswith(".onnx")]
    models = [m for m in onnx_files if os.path.exists(os.path.join(models_dir, os.path.splitext(m)[0] + ".csv"))]
    return models

def get_installed_models_tensorrt():
    """Get all TensorRT models that have corresponding CSV files"""
    import glob
    # Find all .engine files
    pattern = os.path.join(models_dir, "*.engine")
    engines = glob.glob(pattern)
    valid_engines = []
    
    for engine_path in engines:
        engine_filename = os.path.basename(engine_path)
        # Extract base model name (remove batch size and precision suffix and .engine extension)
        # Example: "model_1_1_4_fp16.engine" -> "model"
        if "_" in engine_filename:
            # Split by underscore and remove last 4 parts (batch sizes + precision) and .engine
            parts = engine_filename.replace(".engine", "").split("_")
            if len(parts) >= 4:
                base_name = "_".join(parts[:-4])  # Remove last 4 parts (batch sizes + precision)
            elif len(parts) >= 3:
                # Legacy format without precision suffix
                base_name = "_".join(parts[:-3])  # Remove last 3 parts (batch sizes)
            else:
                base_name = parts[0]  # Fallback if not enough parts
        else:
            base_name = engine_filename.replace(".engine", "")
        
        # Check if corresponding CSV exists
        csv_path = os.path.join(models_dir, base_name + ".csv")
        if os.path.exists(csv_path):
            valid_engines.append(engine_filename)
    
    return valid_engines


class WD14TensorRTModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        # Show exact TensorRT engine filenames
        tensorrt_models = get_installed_models_tensorrt()        
        if not tensorrt_models:
            tensorrt_models = ["Empty (Convert ONNX to TensorRT || Check If CSV exists)"]
        default_model = tensorrt_models[0]
        return {"required": {
            "model": (tensorrt_models, { "default": default_model }),
        }}

    RETURN_TYPES = ("WD14_MODEL", "WD14_CSV_DATA")
    FUNCTION = "load_model"
    CATEGORY = "vovlerTools"

    def load_model(self, model):
        engine_filename = model
        engine_path = os.path.join(models_dir, engine_filename)
        
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TensorRT engine not found: {engine_filename}")
        
        # Extract base model name for CSV lookup
        if "_" in engine_filename:
            parts = engine_filename.replace(".engine", "").split("_")
            if len(parts) >= 4:
                model_name = "_".join(parts[:-4])  # Remove batch sizes + precision
            elif len(parts) >= 3:
                # Legacy format without precision suffix
                model_name = "_".join(parts[:-3])  # Remove batch sizes only
            else:
                model_name = parts[0]
        else:
            model_name = engine_filename.replace(".engine", "")
        
        model_data = {
            "model_name": model_name,
            "height": None
        }
        
        # Load CSV tags separately
        csv_start_time = time.time()
        csv_data = []
        
        with open(os.path.join(models_dir, model_name + ".csv")) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                csv_data.append(row[1])  # Store original tags
        
        csv_time = time.time() - csv_start_time
        log(f"Model {model_name} - CSV loaded: {csv_time*1000:.2f}ms ({len(csv_data)} tags)", "INFO", True)
        
        # Load TensorRT model
        log(f"Loading TensorRT engine: {engine_filename}", "INFO", True)
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(TRT_LOGGER)
            engine = runtime.deserialize_cuda_engine(f.read())
        
        context = engine.create_execution_context()
        
        # Get input/output info
        input_binding = engine.get_tensor_name(0)
        output_binding = engine.get_tensor_name(1)
        input_shape = engine.get_tensor_shape(input_binding)
        
        if input_shape[1] == 3:  # NCHW format
            model_data["height"] = input_shape[2]
        else:  # NHWC format
            model_data["height"] = input_shape[1]
        
        model_data["engine"] = engine
        model_data["context"] = context
        model_data["input_binding"] = input_binding
        model_data["output_binding"] = output_binding
        model_data["input_shape"] = input_shape
        
        log(f"TensorRT model {model_name} loaded successfully", "INFO", True)
        return (model_data, csv_data)


class WD14BlackListLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "csv_data": ("WD14_CSV_DATA",),
            "blacklisted_tags": ("STRING", {"multiline": True, "default": ""}),
        }}

    RETURN_TYPES = ("WD14_BLACKLIST",)
    RETURN_NAMES = ("blacklist",)
    OUTPUT_IS_LIST = (False,)
    FUNCTION = "load_blacklist"
    CATEGORY = "vovlerTools"

    def load_blacklist(self, csv_data, blacklisted_tags=""):
        blacklist_start_time = time.time()
        
        # Process blacklisted tags - convert to lowercase and split by comma
        blacklist_names = set()
        if blacklisted_tags.strip():
            blacklist_names = set(tag.strip().lower().replace("_", " ") for tag in blacklisted_tags.split(",") if tag.strip())
        
        # Pre-compute blacklisted tag indices for fast lookup
        blacklist_indices = set()
        
        if blacklist_names:
            for i, tag in enumerate(csv_data):
                processed_tag = tag.lower().replace("_", " ")
                if processed_tag in blacklist_names:
                    blacklist_indices.add(i)
        
        # Create optimized blacklist data structure (indices only)
        blacklist_data = blacklist_indices
        
        blacklist_time = time.time() - blacklist_start_time
        log(f"Blacklist compiled: {blacklist_time*1000:.2f}ms ({len(blacklist_indices)} indices from {len(blacklist_names)} input tags)", "INFO", True)

        return (blacklist_data,)


class WD14TaggerAndImageFilterer:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE", ),
            "model": ("WD14_MODEL",),
            "csv_data": ("WD14_CSV_DATA",),
            "blacklist": ("WD14_BLACKLIST",),
            "threshold": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1, "step": 0.05}),
            "resize_method": (["BILINEAR", "LANCZOS", "BICUBIC"], {"default": "BILINEAR"}),
            "enable_print_image_tags": ("BOOLEAN", {"default": False}),
            "bool_bypass_node": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("filtered_images",)
    OUTPUT_IS_LIST = (False,)
    FUNCTION = "filter_images"
    OUTPUT_NODE = True
    CATEGORY = "vovlerTools"

    def filter_images(self, image, model, csv_data, blacklist, threshold, resize_method="BILINEAR", enable_print_image_tags=False, bool_bypass_node=False):
        # Check if bypass is enabled - return images immediately
        if bool_bypass_node:
            log("Node bypass enabled - returning all input images without processing", "INFO", True)
            return (image,)
        
        # Use model and tags separately instead of combining them
        tags_list = csv_data
        
        # Keep original image tensor (in [0,1] range) for output
        original_images = image
        log(f"Input tensor device: {original_images.device}, contiguous: {original_images.is_contiguous()}, shape: {original_images.shape}", "INFO", True)
        
        # Create processing tensor (in [0,255] range) for inference
        tensor = image*255
        tensor = np.array(tensor, dtype=np.uint8)
        batch_size = tensor.shape[0]
        height = model["height"]
        
        if len(blacklist) == 0:
            raise ValueError("No blacklist indices found")
        
        # Group images by dimensions for batch processing
        image_shapes = [tensor[i].shape for i in range(batch_size)]
        
        # Preprocess all images
        total_preprocess_start = time.time()
        
        if resize_method in ["BICUBIC", "BILINEAR"]:
            # Group images by their dimensions for batch processing
            from collections import defaultdict
            shape_groups = defaultdict(list)
            for i, shape in enumerate(image_shapes):
                shape_groups[shape].append(i)
            
            preprocessed_images = [None] * batch_size  # Preallocate list to maintain order
            
            # Process each size group as a batch
            for shape, indices in shape_groups.items():
                if len(indices) > 1:
                    log(f"Batch resizing {len(indices)} images of size {shape} using {resize_method}...", "INFO", True)
                    # Extract images of this size
                    group_tensor = np.array([tensor[i] for i in indices])
                    # Batch process this group
                    group_processed = self._batch_resize_gpu(group_tensor, height, resize_method)
                    # Put results back in correct positions
                    for j, idx in enumerate(indices):
                        preprocessed_images[idx] = group_processed[j]
                else:
                    # Single image - still use batch method for consistency
                    log(f"Processing single image of size {shape} using {resize_method}...", "INFO", True)
                    idx = indices[0]
                    single_tensor = np.array([tensor[idx]])
                    single_processed = self._batch_resize_gpu(single_tensor, height, resize_method)
                    preprocessed_images[idx] = single_processed[0]
        else:
            # Use individual processing for LANCZOS
            log(f"Individual resizing {batch_size} images using {resize_method}...", "INFO", True)
            preprocessed_images = self._individual_resize(tensor, height, resize_method, batch_size)
        
        total_preprocess_time = time.time() - total_preprocess_start
        log(f"Total preprocessing time for {batch_size} images: {total_preprocess_time*1000:.2f}ms", "INFO", True)
        
        # Stack all preprocessed images into a batch
        batch_images = np.stack(preprocessed_images, axis=0)  # Shape: (batch_size, height, width, 3)
        
        # Run batched inference
        inference_start_time = time.time()
        log(f"Running TensorRT inference on batch of {batch_size} images...", "INFO", True)
        
        # TensorRT inference using PyTorch tensors with pre-loaded model
        input_tensor = torch.from_numpy(batch_images.copy()).cuda()
        
        # Prepare input data (convert to NCHW if needed)
        input_shape = model["input_shape"]
        if len(input_shape) == 4 and input_shape[1] == 3:  # NCHW format
            input_tensor = input_tensor.permute(0, 3, 1, 2)  # NHWC -> NCHW
        
        # Convert to appropriate dtype
        engine = model["engine"]
        context = model["context"]
        input_binding = model["input_binding"]
        output_binding = model["output_binding"]
        
        input_dtype = engine.get_tensor_dtype(input_binding)
        if input_dtype == trt.float16:
            input_tensor = input_tensor.half()
        else:
            input_tensor = input_tensor.float()
        
        # Set the input shape for the actual batch size
        if input_shape[0] == -1:
            actual_input_shape = (batch_size,) + input_shape[1:]
            context.set_input_shape(input_binding, actual_input_shape)
        
        # Get output shape and create output tensor
        output_shape = engine.get_tensor_shape(output_binding)
        if output_shape[0] == -1:
            output_shape = (batch_size,) + output_shape[1:]
        
        output_dtype = engine.get_tensor_dtype(output_binding)
        if output_dtype == trt.float16:
            output_tensor = torch.empty(output_shape, dtype=torch.float16, device='cuda')
        else:
            output_tensor = torch.empty(output_shape, dtype=torch.float32, device='cuda')
        
        # Set tensor addresses
        context.set_tensor_address(input_binding, input_tensor.data_ptr())
        context.set_tensor_address(output_binding, output_tensor.data_ptr())
        
        # Run inference with torch CUDA stream
        stream = torch.cuda.default_stream()
        context.execute_async_v3(stream_handle=stream.cuda_stream)
        
        # Convert back to numpy on CPU
        batch_probs = output_tensor.cpu().numpy()  # Shape: (batch_size, num_tags)

        inference_time = time.time() - inference_start_time
        log(f"TensorRT batch inference time: {inference_time*1000:.2f}ms ({inference_time*1000/batch_size:.2f}ms per image)", "INFO", True)

        # Process results and filter images
        post_process_start = time.time()
        
        # Pre-compute optimizations based on whether we need tag output
        processed_tags_for_comparison = None
        
        if enable_print_image_tags:
            # Pre-compute processed tags for comparison when we need tag output
            processed_tags_for_comparison = [tag.lower().replace("_", " ") for tag in tags_list]
        
        filtered_images = []
        filtered_tag_indices = []  # Store indices instead of full strings initially
        
        for i in range(batch_size):
            if not enable_print_image_tags:
                # Fast path: only check precompiled blacklisted tag indices, early exit on first match
                has_blacklisted_tag = False
                detected_blacklisted_tag = None
                for tag_idx in blacklist:
                    prob = batch_probs[i][tag_idx]
                    # Check threshold for all tags (no distinction between general/character)
                    if prob > threshold:
                        has_blacklisted_tag = True
                        detected_blacklisted_tag = tags_list[tag_idx]
                        break
                
                if not has_blacklisted_tag:
                    # Image passes filter
                    single_image = original_images[i]
                    log(f"Adding image {i+1} with shape: {single_image.shape}, dtype: {single_image.dtype}, range: {single_image.min():.3f}-{single_image.max():.3f}, device: {single_image.device}", "INFO", True)
                    filtered_images.append(single_image)
                    log(f"Image {i+1}: PASSED", "INFO", True)
                else:
                    log(f"Image {i+1}: FILTERED OUT - Blacklisted tag detected: {detected_blacklisted_tag}", "INFO", True)
            else:
                # Full path: process all tags when we need tag output
                if processed_tags_for_comparison is None:
                    # Compute on-demand if not pre-computed
                    processed_tags_for_comparison = [tag.lower().replace("_", " ") for tag in tags_list]
                
                result = list(zip(processed_tags_for_comparison, batch_probs[i]))

                # Get all tags above threshold (no distinction between general/character)
                detected_above_threshold = [item for item in result if item[1] > threshold]
                
                # Get all detected tags for this image
                detected_tags = set(item[0] for item in detected_above_threshold)
                
                # Check if any detected tags are in the blacklist using indices (more performant)
                detected_tag_indices = {j for j, tag in enumerate(processed_tags_for_comparison) if tag in detected_tags}
                has_blacklisted_tag = bool(detected_tag_indices & blacklist)
                
                if not has_blacklisted_tag:
                    # Image passes filter - add original ComfyUI tensor slice (in [0,1] range)
                    single_image = original_images[i]
                    log(f"Adding image {i+1} with shape: {single_image.shape}, dtype: {single_image.dtype}, range: {single_image.min():.3f}-{single_image.max():.3f}, device: {single_image.device}", "INFO", True)
                    filtered_images.append(single_image)
                    
                    # Store the detected tag indices for console printing
                    detected_tag_indices_for_output = [j for j, tag in enumerate(processed_tags_for_comparison) if tag in detected_tags]
                    filtered_tag_indices.append(detected_tag_indices_for_output)
                    
                    log(f"Image {i+1}: PASSED", "INFO", True)
                else:
                    # Image has blacklisted tags - find which specific tags caused the filtering
                    blacklisted_indices_found = detected_tag_indices & blacklist
                    blacklisted_tag_names = [tags_list[idx] for idx in blacklisted_indices_found]
                    log(f"Image {i+1}: FILTERED OUT - Blacklisted tags found: {', '.join(blacklisted_tag_names[:3])}", "INFO", True)
        
        post_process_time = time.time() - post_process_start
        total_time = total_preprocess_time + inference_time + post_process_time
        
        # Stack filtered images if any passed
        if filtered_images:
            final_images = torch.stack(filtered_images, dim=0)
            log(f"Final output tensor - shape: {final_images.shape}, dtype: {final_images.dtype}, range: {final_images.min():.3f}-{final_images.max():.3f}, device: {final_images.device}, contiguous: {final_images.is_contiguous()}", "INFO", True)
        else:
            # Return empty tensor with proper shape if no images passed
            final_images = torch.empty((0, original_images.shape[1], original_images.shape[2], original_images.shape[3]))
            log(f"No images passed filter - returning empty tensor with shape: {final_images.shape}", "INFO", True)
        
        # Print formatted tag string for each filtered image to console
        if enable_print_image_tags and len(filtered_tag_indices) > 0:
            # Create formatted output for each filtered image
            image_tag_blocks = []
            for i, tag_indices in enumerate(filtered_tag_indices):
                # Get the original tags for this image
                image_tags = [tags_list[j] for j in tag_indices]
                tag_string = ", ".join(image_tags)
                
                # Format as requested
                image_block = f"######\nIMAGE_{i+1}: {tag_string}\n######"
                image_tag_blocks.append(image_block)
            
            # Combine all image blocks with double newlines and print to console
            combined_tags = "\n\n".join(image_tag_blocks)
            log(f"Detected tags for filtered images:\n{combined_tags}", "INFO", True)
        
        log(f"Filtering complete - {len(filtered_images)}/{batch_size} images passed filter", "INFO", True)
        log(f"Total processing time: {total_time*1000:.2f}ms", "INFO", True)
        log(f"Performance breakdown - Preprocessing: {total_preprocess_time*1000:.2f}ms | TensorRT: {inference_time*1000:.2f}ms | Post-processing: {post_process_time*1000:.2f}ms", "INFO", True)
        
        if len(filtered_images) > 0:
            log(f"Average per filtered image: {total_time*1000/len(filtered_images):.2f}ms", "INFO", True)
        
        return (final_images,)

    def _batch_resize_gpu(self, tensor, target_height, resize_method):
        """Batch resize images using GPU-accelerated PyTorch"""
        # Convert to PyTorch tensor and move to GPU
        batch_tensor = torch.from_numpy(tensor).float().cuda()  # Shape: (batch, height, width, 3)
        batch_tensor = batch_tensor.permute(0, 3, 1, 2)  # Convert to NCHW: (batch, 3, height, width)
        
        # Calculate resize parameters for all images (assuming they're all the same size)
        original_height, original_width = batch_tensor.shape[2], batch_tensor.shape[3]
        ratio = float(target_height) / max(original_height, original_width)
        new_height = int(original_height * ratio)
        new_width = int(original_width * ratio)
        
        # Choose interpolation mode
        if resize_method == "BICUBIC":
            mode = 'bicubic'
        elif resize_method == "BILINEAR":
            mode = 'bilinear'
        else:
            mode = 'bilinear'  # fallback
        
        # Batch resize using PyTorch
        resized_batch = torch.nn.functional.interpolate(
            batch_tensor, 
            size=(new_height, new_width), 
            mode=mode, 
            align_corners=False
        )
        
        # Convert back to NHWC and move to CPU
        resized_batch = resized_batch.permute(0, 2, 3, 1)  # NCHW -> NHWC
        resized_batch = resized_batch.cpu().numpy().astype(np.float32)
        
        # Pad each image to square and convert RGB->BGR
        preprocessed_images = []
        for i in range(resized_batch.shape[0]):
            # Create white square canvas
            square = np.full((target_height, target_height, 3), 255.0, dtype=np.float32)
            
            # Calculate paste position (center)
            paste_y = (target_height - new_height) // 2
            paste_x = (target_height - new_width) // 2
            
            # Paste resized image
            square[paste_y:paste_y+new_height, paste_x:paste_x+new_width] = resized_batch[i]
            
            # Convert RGB to BGR
            square_bgr = square[:, :, ::-1]
            preprocessed_images.append(square_bgr)
        
        return preprocessed_images

    def _individual_resize(self, tensor, target_height, resize_method, batch_size):
        """Individual resize using PIL (for LANCZOS) or PyTorch (for BICUBIC/BILINEAR)"""
        preprocessed_images = []
        
        # Map resize methods
        if resize_method == "LANCZOS":
            pil_filter = Image.LANCZOS
            use_pil = True
        elif resize_method == "BICUBIC":
            torch_mode = 'bicubic'
            use_pil = False
        elif resize_method == "BILINEAR":
            torch_mode = 'bilinear'
            use_pil = False
        else:
            pil_filter = Image.LANCZOS  # fallback
            use_pil = True
        
        for i in range(batch_size):
            if use_pil:
                # Use PIL for LANCZOS
                image_pil = Image.fromarray(tensor[i])
                
                # Reduce to max size and pad with white
                ratio = float(target_height)/max(image_pil.size)
                new_size = tuple([int(x*ratio) for x in image_pil.size])
                image_pil = image_pil.resize(new_size, pil_filter)
                square = Image.new("RGB", (target_height, target_height), (255, 255, 255))
                square.paste(image_pil, ((target_height-new_size[0])//2, (target_height-new_size[1])//2))

                image_np = np.array(square).astype(np.float32)
                image_np = image_np[:, :, ::-1]  # RGB -> BGR
            else:
                # Use PyTorch for BICUBIC/BILINEAR
                image_tensor = torch.from_numpy(tensor[i]).float().cuda()  # Shape: (height, width, 3)
                image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # Convert to NCHW: (1, 3, height, width)
                
                # Calculate new size
                original_height, original_width = image_tensor.shape[2], image_tensor.shape[3]
                ratio = float(target_height) / max(original_height, original_width)
                new_height = int(original_height * ratio)
                new_width = int(original_width * ratio)
                
                # Resize using PyTorch
                resized_tensor = torch.nn.functional.interpolate(
                    image_tensor, 
                    size=(new_height, new_width), 
                    mode=torch_mode, 
                    align_corners=False
                )
                
                # Convert back to numpy and create square canvas
                resized_image = resized_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
                
                # Create white square canvas
                square = np.full((target_height, target_height, 3), 255.0, dtype=np.float32)
                
                # Calculate paste position (center)
                paste_y = (target_height - new_height) // 2
                paste_x = (target_height - new_width) // 2
                
                # Paste resized image
                square[paste_y:paste_y+new_height, paste_x:paste_x+new_width] = resized_image
                
                # Convert RGB to BGR
                image_np = square[:, :, ::-1]
            
            preprocessed_images.append(image_np)
        
        return preprocessed_images


class WDTaggerONNXtoTENSORRT:
    @classmethod
    def INPUT_TYPES(s):
        # Only show ONNX models for conversion
        onnx_models = [os.path.splitext(m)[0] for m in get_installed_models_onnx()]
        if not onnx_models:
            onnx_models = ["No ONNX models found - place .onnx and .csv files in models folder"]
        
        default_model = onnx_models[0] if onnx_models and "No ONNX models found" not in onnx_models[0] else ""
        return {"required": {
            "model": (onnx_models, { "default": default_model }),
            "batch_size_min": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1}),
            "batch_size_opt": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1}),
            "batch_size_max": ("INT", {"default": 4, "min": 1, "max": 16, "step": 1}),
            "precision": (["float16", "float32"], {"default": "float16"}),
        }}

    RETURN_TYPES = ()
    FUNCTION = "convert_to_tensorrt"
    OUTPUT_NODE = True
    CATEGORY = "vovlerTools"

    def convert_to_tensorrt(self, model, batch_size_min, batch_size_opt, batch_size_max, precision):
        # Ensure model name has .onnx extension for source
        if not model.endswith(".onnx"):
            model = model + ".onnx"
        
        onnx_path = os.path.join(models_dir, model)
        model_base = model.replace(".onnx", "")
        precision_suffix = "fp16" if precision == "float16" else "fp32"
        engine_filename = f"{model_base}_{batch_size_min}_{batch_size_opt}_{batch_size_max}_{precision_suffix}.engine"
        engine_path = os.path.join(models_dir, engine_filename)
        
        if not os.path.exists(onnx_path):
            log(f"ONNX model not found: {onnx_path}", "ERROR", True)
            return {}
        
        if os.path.exists(engine_path):
            log(f"TensorRT engine already exists: {engine_filename}", "INFO", True)
            return {}
        
        try:
            log(f"Converting {model} to TensorRT format...", "INFO", True)
            
            # Create TensorRT logger and builder with verbose logging
            TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
            builder = trt.Builder(TRT_LOGGER)
            
            # Log TensorRT and GPU info
            log(f"TensorRT version: {trt.__version__}", "INFO", True)
            
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, TRT_LOGGER)
            
            # Parse ONNX model
            log("Parsing ONNX model...", "INFO", True)
            with open(onnx_path, 'rb') as model_file:
                model_data = model_file.read()
                log(f"ONNX model size: {len(model_data)} bytes", "INFO", True)
                
                if not parser.parse(model_data):
                    log("Failed to parse ONNX model", "ERROR", True)
                    for error in range(parser.num_errors):
                        log(f"Parser error: {parser.get_error(error)}", "ERROR", True)
                    return {}
            
            log("ONNX model parsed successfully", "INFO", True)
            log(f"Network inputs: {network.num_inputs}", "INFO", True)
            log(f"Network outputs: {network.num_outputs}", "INFO", True)
            
            # Log input details
            for i in range(network.num_inputs):
                input_tensor = network.get_input(i)
                log(f"Input {i}: {input_tensor.name}, shape: {input_tensor.shape}, dtype: {input_tensor.dtype}", "INFO", True)
            
            # Configure builder
            config = builder.create_builder_config()
            # Configure precision based on user selection
            if precision == "float16":
                if builder.platform_has_fast_fp16:
                    config.set_flag(trt.BuilderFlag.FP16)
                    log("Using FP16 precision", "INFO", True)
                else:
                    log("FP16 not supported on this platform", "ERROR", True)
                    raise RuntimeError("FP16 precision requested but not supported on this platform. Use float32 instead.")
            else:
                log("Using FP32 precision", "INFO", True)
            
            # Add optimization profile for dynamic shapes if needed
            profile = builder.create_optimization_profile()
            for i in range(network.num_inputs):
                input_tensor = network.get_input(i)
                input_shape = list(input_tensor.shape)
                log(f"Setting optimization profile for input {input_tensor.name}: {input_shape}", "INFO", True)
                
                # Handle dynamic batch size (typically -1 in first dimension)
                if input_shape[0] == -1:  # Dynamic batch size
                    log(f"Dynamic batch size detected for {input_tensor.name}, setting batch size range {batch_size_min}-{batch_size_max}, optimal={batch_size_opt}", "INFO", True)
                    min_shape = [batch_size_min] + input_shape[1:]
                    opt_shape = [batch_size_opt] + input_shape[1:]
                    max_shape = [batch_size_max] + input_shape[1:]
                    profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
                elif any(dim <= 0 for dim in input_shape):
                    log(f"Other dynamic dimensions detected for {input_tensor.name}: {input_shape}", "WARNING", True)
                    # Handle other dynamic dimensions - for now, skip this input
                    continue
                else:
                    # Fixed shape
                    profile.set_shape(input_tensor.name, input_shape, input_shape, input_shape)
            
            config.add_optimization_profile(profile)
            
            # Build the engine
            log("Building TensorRT engine... This may take a while.", "INFO", True)
            try:
                serialized_engine = builder.build_serialized_network(network, config)
            except Exception as build_error:
                log(f"TensorRT build exception: {str(build_error)}", "ERROR", True)
                return {}
            
            if serialized_engine is None:
                log("Failed to build TensorRT engine - build_serialized_network returned None", "ERROR", True)
                log("Common causes:", "ERROR", True)
                log("1. Unsupported ONNX operators", "ERROR", True)
                log("2. GPU memory insufficient", "ERROR", True)
                log("3. CUDA/TensorRT version compatibility issues", "ERROR", True)
                log("4. Model requires dynamic shapes not properly configured", "ERROR", True)
                return {}
            
            # Save the engine
            with open(engine_path, 'wb') as f:
                f.write(serialized_engine)
            
            # Get file size after writing
            engine_size = os.path.getsize(engine_path) / (1024*1024)
            log(f"TensorRT engine saved to: {engine_filename} (size: {engine_size:.1f}MB)", "INFO", True)
            
        except Exception as e:
            log(f"Error converting to TensorRT: {str(e)}", "ERROR", True)
            return {}
        
        return {}