# -*- coding: utf-8 -*-
# HiDream Sampler Node for ComfyUI
# Version: 2024-07-29c (NF4/FP8/BNB Support, Final Call Arg Fix)
#
# Required Dependencies:
# - transformers, diffusers, torch, numpy, Pillow
# - For NF4 models: optimum, accelerate, auto-gptq (`pip install optimum accelerate auto-gptq`)
# - For non-NF4/FP8 models (4-bit): bitsandbytes (`pip install bitsandbytes`)
# - Ensure hi_diffusers library is locally available or hdi1 package is installed.
import torch
import numpy as np
from PIL import Image
import comfy.model_management # Ensure this is imported
import comfy.utils
import gc
import os # For checking paths if needed
import huggingface_hub
from safetensors.torch import load_file

# --- Check for Torch and Transformers version issues
try:
    import packaging.version
    import torch
    import transformers
    
    torch_version = torch.__version__
    transformers_version = transformers.__version__
    
    if packaging.version.parse(torch_version) < packaging.version.parse("2.1.0"):
        print("⚠️ WARNING: Your PyTorch version is older than 2.1.0")
        print("   Windows users may need PyTorch 2.1.0+ for NF4 models")
        print("   Get the correct version at pytorch.org for your CUDA version")
    
    if packaging.version.parse(transformers_version) < packaging.version.parse("4.36.0"):
        print("⚠️ WARNING: Your transformers version is older than 4.36.0")
        print("   May have compatibility issues with NF4 models: pip install --upgrade transformers>=4.36.0")
except (ImportError, AttributeError):
    pass  # Skip version check if packaging module not available

# --- Attention Implementation Detection ---
try:
    import flash_attn
    flash_attn_available = True
    print("Flash Attention 2 is available.")
except ImportError:
    flash_attn_available = False
    print("Flash Attention 2 is not available, will use PyTorch's native attention if possible.")

# Check for scaled_dot_product_attention (available in PyTorch 2.0+)
sdpa_available = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
if sdpa_available:
    print("PyTorch SDPA (Scaled Dot Product Attention) is available.")
else:
    print("PyTorch SDPA not available. Will use eager attention.")

try:
    import accelerate
    accelerate_available = True
except ImportError:
    accelerate_available = False
    print("Warning: accelerate not installed. device_map='auto' for GPTQ models will not be available.")
# Robust GPTQ support detection
gptqmodel_available = False
autogptq_available = False
gptq_support_available = False
try:
    from transformers import GPTQConfig
    gptqmodel_available = True
    gptq_support_available = True
    print("GPTQModel (transformers) support is available (recommended).")
except ImportError:
    print("No GPTQModel support in transformers.")
try:
    import auto_gptq
    autogptq_available = True
    gptq_support_available = True
    print("auto_gptq support is available (legacy).")
except ImportError:
    print("auto_gptq not available.")
try:
    import optimum
    optimum_available = True
except ImportError:
    optimum_available = False
print(f"GPTQ support composite: (GPTQModel: {gptqmodel_available} | auto_gptq: {autogptq_available}) -> {gptq_support_available}")
try:
    import optimum
    optimum_available = True
except ImportError:
    optimum_available = False
    # Add a warning if NF4 models are selected later.
try:
    from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
    from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
    bnb_available = True
except ImportError:
    # This case was handled before, just confirm variable state
    bnb_available = False
    TransformersBitsAndBytesConfig = None
    DiffusersBitsAndBytesConfig = None
    print("Warning: bitsandbytes not installed. 4-bit BNB quantization will not be available.")
# --- Core Imports ---
from transformers import LlamaForCausalLM, AutoTokenizer # Use AutoTokenizer
# --- HiDream Specific Imports ---
# Attempt local import first, then fallback (which might fail)
try:
    # Assuming hi_diffusers is cloned into this custom_node's directory
    from .hi_diffusers.models.transformers.transformer_hidream_image import HiDreamImageTransformer2DModel
    from .hi_diffusers.pipelines.hidream_image.pipeline_hidream_image import HiDreamImagePipeline
    from .hi_diffusers.pipelines.hidream_image.pipeline_hidream_image_to_image import HiDreamImageToImagePipeline
    from .hi_diffusers.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
    from .hi_diffusers.schedulers.flash_flow_match import FlashFlowMatchEulerDiscreteScheduler
    hidream_classes_loaded = True
except ImportError as e:
    print("--------------------------------------------------------------------")
    print(f"ComfyUI-HiDream-Sampler: Could not import local hi_diffusers ({e}).")
    print("Please ensure hi_diffusers library is inside ComfyUI-HiDream-Sampler,")
    print("or hdi1 package is installed in the ComfyUI environment.")
    print("Node may fail to load models.")
    print("--------------------------------------------------------------------")
    # Define placeholders so the script doesn't crash immediately
    HiDreamImageTransformer2DModel = None
    HiDreamImagePipeline = None
    FlowUniPCMultistepScheduler = None
    FlashFlowMatchEulerDiscreteScheduler = None
    hidream_classes_loaded = False
# --- Model Paths ---
ORIGINAL_MODEL_PREFIX = "HiDream-ai"
NF4_MODEL_PREFIX = "azaneko"
ORIGINAL_LLAMA_MODEL_NAME = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1" # For original/FP8
NF4_LLAMA_MODEL_NAME = "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4" # For NF4
# Add alternate model paths (using the same model as NF4 since it's less censored)
ALTERNATE_LLAMA_MODEL_NAME = "akhbar/Meta-Llama-3.1-8B-Instruct-abliterated-GPTQ" 
ALTERNATE_NF4_LLAMA_MODEL_NAME = "akhbar/Meta-Llama-3.1-8B-Instruct-abliterated-GPTQ"
# --- Model Configurations ---
# Added flags for dependency checks
MODEL_CONFIGS = {
    # --- NF4 Models ---
    "full-nf4": {
        "path": f"{NF4_MODEL_PREFIX}/HiDream-I1-Full-nf4",
        "guidance_scale": 5.0, "num_inference_steps": 50, "shift": 3.0,
        "scheduler_class": "FlowUniPCMultistepScheduler", # Use string names for dynamic import
        "is_nf4": True, "is_fp8": False, "requires_bnb": False, "requires_gptq_deps": True
    },
    "dev-nf4": {
        "path": f"{NF4_MODEL_PREFIX}/HiDream-I1-Dev-nf4",
        "guidance_scale": 0.0, "num_inference_steps": 28, "shift": 6.0,
        "scheduler_class": "FlashFlowMatchEulerDiscreteScheduler",
        "is_nf4": True, "is_fp8": False, "requires_bnb": False, "requires_gptq_deps": True
    },
    "fast-nf4": {
        "path": f"{NF4_MODEL_PREFIX}/HiDream-I1-Fast-nf4",
        "guidance_scale": 0.0, "num_inference_steps": 16, "shift": 3.0,
        "scheduler_class": "FlashFlowMatchEulerDiscreteScheduler",
        "is_nf4": True, "is_fp8": False, "requires_bnb": False, "requires_gptq_deps": True
    },
    # --- Original/BNB Models ---
     "full": {
        "path": f"{ORIGINAL_MODEL_PREFIX}/HiDream-I1-Full",
        "guidance_scale": 5.0, "num_inference_steps": 50, "shift": 3.0,
        "scheduler_class": "FlowUniPCMultistepScheduler",
        "is_nf4": False, "is_fp8": False, "requires_bnb": True, "requires_gptq_deps": False
    },
    "dev": {
        "path": f"{ORIGINAL_MODEL_PREFIX}/HiDream-I1-Dev",
        "guidance_scale": 0.0, "num_inference_steps": 28, "shift": 6.0,
        "scheduler_class": "FlashFlowMatchEulerDiscreteScheduler",
        "is_nf4": False, "is_fp8": False, "requires_bnb": True, "requires_gptq_deps": False
    },
    "fast": {
        "path": f"{ORIGINAL_MODEL_PREFIX}/HiDream-I1-Fast",
        "guidance_scale": 0.0, "num_inference_steps": 16, "shift": 3.0,
        "scheduler_class": "FlashFlowMatchEulerDiscreteScheduler",
        "is_nf4": False, "is_fp8": False, "requires_bnb": True, "requires_gptq_deps": False
    }
}
# --- Filter models based on available dependencies ---
# (Keep filtering logic the same)
original_model_count = len(MODEL_CONFIGS)
if not bnb_available:
    MODEL_CONFIGS = {k: v for k, v in MODEL_CONFIGS.items() if not v.get("requires_bnb", False)}
if not optimum_available or not gptq_support_available:
    print(f"Warning: GPTQ support missing (optimum: {optimum_available}, GPTQ: {gptq_support_available})")
    MODEL_CONFIGS = {k: v for k, v in MODEL_CONFIGS.items() if not v.get("requires_gptq_deps", False)}
else:
    print("GPTQ dependencies available - all models should work")
if not hidream_classes_loaded: MODEL_CONFIGS = {}
filtered_model_count = len(MODEL_CONFIGS)
if filtered_model_count == 0: print("*"*70 + "\nCRITICAL ERROR: No HiDream models available...\n" + "*"*70)
elif filtered_model_count < original_model_count: print("*"*70 + "\nWarning: Some HiDream models disabled...\n" + "*"*70)
# Define BitsAndBytes configs (if available)
# (Keep definitions the same)
bnb_llm_config = None; bnb_transformer_4bit_config = None
if bnb_available: bnb_llm_config = TransformersBitsAndBytesConfig(load_in_4bit=True); bnb_transformer_4bit_config = DiffusersBitsAndBytesConfig(load_in_4bit=True)
model_dtype = torch.bfloat16
# Get available scheduler classes
# (Keep definitions the same)
available_schedulers = {};
if hidream_classes_loaded: available_schedulers = {"FlowUniPCMultistepScheduler": FlowUniPCMultistepScheduler, "FlashFlowMatchEulerDiscreteScheduler": FlashFlowMatchEulerDiscreteScheduler}

DEBUG_CACHE = True  # Set to False in production

# Use a more aggressive global cleanup
def global_cleanup():
    """Global cleanup function for use with multiple HiDream nodes"""
    print("HiDream: Performing global cleanup...")
    
    # Clear any pending operations
    torch.cuda.synchronize()
    
    # Get current memory stats
    if torch.cuda.is_available():
        before_mem = torch.cuda.memory_allocated() / 1024**2
        print(f"  Memory before cleanup: {before_mem:.2f} MB")
    
    # Perform HiDreamSampler cleanup
    HiDreamSampler.cleanup_models()
    
    # Additional cleanup
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        after_mem = torch.cuda.memory_allocated() / 1024**2
        print(f"  Memory after cleanup: {after_mem:.2f} MB")
    
    return True

# --- Helper: Get Scheduler Instance ---
def get_scheduler_instance(scheduler_name, shift_value):
    if not available_schedulers: raise RuntimeError("No schedulers available...")
    scheduler_class = available_schedulers.get(scheduler_name)
    if scheduler_class is None: raise ValueError(f"Scheduler class '{scheduler_name}' not found...")
    return scheduler_class(num_train_timesteps=1000, shift=shift_value, use_dynamic_shifting=False)
# --- Loading Function (Handles NF4 and default BNB) ---
def load_models(model_type, use_alternate_llm=False):
    if not hidream_classes_loaded:
        raise ImportError("Cannot load models: HiDream classes failed to import.")
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown or incompatible model_type: {model_type}")

    config = MODEL_CONFIGS[model_type]
    model_path = config["path"]
    is_nf4 = config.get("is_nf4", False)
    scheduler_name = config["scheduler_class"]
    shift = config["shift"]
    requires_bnb = config.get("requires_bnb", False)
    requires_gptq_deps = config.get("requires_gptq_deps", False)

    if requires_bnb and not bnb_available:
        raise ImportError(f"Model '{model_type}' requires BitsAndBytes...")
    if requires_gptq_deps and (not optimum_available or not gptq_support_available):
        raise ImportError(f"Model '{model_type}' requires Optimum & GPTQ support...")

    print(f"--- Loading Model Type: {model_type} ---")
    print(f"Model Path: {model_path}")
    print(f"NF4: {is_nf4}, Requires BNB: {requires_bnb}, Requires GPTQ deps: {requires_gptq_deps}")
    print(f"Using alternate LLM: {use_alternate_llm}")
    start_mem = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    print(f"(Start VRAM: {start_mem:.2f} MB)")

    # Create a standardized cache key used by all nodes
    cache_key = f"{model_type}_{'alternate' if use_alternate_llm else 'standard'}"

    # Check cache with debug info
    if DEBUG_CACHE:
        print(f"Cache check for key: {cache_key}")
        print(f"Cache contains: {list(HiDreamSampler._model_cache.keys())}")

    if cache_key in HiDreamSampler._model_cache:
        pipe, stored_config = HiDreamSampler._model_cache[cache_key]
        if pipe is not None and hasattr(pipe, 'transformer') and pipe.transformer is not None:
            print(f"Using cached model for {cache_key}")
            return pipe, MODEL_CONFIGS[model_type]  # Always return original config dict
        else:
            print(f"Cache entry invalid for {cache_key}, reloading")
            HiDreamSampler._model_cache.pop(cache_key, None)

    # --- 1. Load LLM (Conditional) ---
    text_encoder_load_kwargs = {"low_cpu_mem_usage": True, "torch_dtype": model_dtype}
    if is_nf4:
        # Choose alternate model if requested, but keep loading process identical
        if use_alternate_llm:
            llama_model_name = ALTERNATE_NF4_LLAMA_MODEL_NAME
            print(f"\n[1a] Preparing Alternate LLM (GPTQ): {llama_model_name}")
        else:
            llama_model_name = NF4_LLAMA_MODEL_NAME
            print(f"\n[1a] Preparing LLM (GPTQ): {llama_model_name}")

        # Device map for accelerate
        if accelerate_available:
            if hasattr(torch.cuda, 'get_device_properties') and torch.cuda.is_available():
                total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                max_mem = int(total_mem * 0.4)
                text_encoder_load_kwargs["max_memory"] = {0: f"{max_mem}GiB"}
                print(f"     Setting max memory limit: {max_mem}GiB of {total_mem:.1f}GiB")
            text_encoder_load_kwargs["device_map"] = "auto"
            print("     Using device_map='auto'.")
        else:
            print("     accelerate not found, attempting manual placement.")

    else:
        # For non-NF4 models, choose alternate if requested
        if use_alternate_llm:
            llama_model_name = ALTERNATE_LLAMA_MODEL_NAME
            print(f"\n[1a] Preparing Alternate LLM (4-bit BNB): {llama_model_name}")
        else:
            llama_model_name = ORIGINAL_LLAMA_MODEL_NAME
            print(f"\n[1a] Preparing LLM (4-bit BNB): {llama_model_name}")

        # Apply BNB config, set attn impl
        if bnb_llm_config:
            text_encoder_load_kwargs["quantization_config"] = bnb_llm_config
            print("     Using 4-bit BNB.")
        else:
            raise ImportError("BNB config required for standard LLM.")

        if flash_attn_available:
            text_encoder_load_kwargs["attn_implementation"] = "flash_attention_2"
            print("     Using Flash Attention 2.")
        elif sdpa_available:
            text_encoder_load_kwargs["attn_implementation"] = "sdpa"
            print("     Using PyTorch SDPA attention.")
        else:
            text_encoder_load_kwargs["attn_implementation"] = "eager"
            print("     Using standard eager attention.")

    print(f"[1b] Loading Tokenizer: {llama_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(llama_model_name, use_fast=False)
    print("     Tokenizer loaded.")

    # --- 1c. LLM Model Loading (Special handling for NF4/GPTQ) ---
    if is_nf4:
        # Rope scaling fix
        try:
            from transformers import AutoConfig
            config_obj = AutoConfig.from_pretrained(llama_model_name)
            config_obj.rope_scaling = {"type": "linear", "factor": 1.0}
            print(f"     ✅ Fixed rope_scaling to: {config_obj.rope_scaling}")
            text_encoder_load_kwargs["config"] = config_obj
            text_encoder_load_kwargs["low_cpu_mem_usage"] = True
        except Exception as e:
            print(f"     ⚠️ Failed to patch config: {e}")

        # --- NEW GPTQModel LOGIC ---
        try:
            if gptqmodel_available:
                from transformers import AutoModelForCausalLM, GPTQConfig
                gptq_kwargs = dict(bits=4)
                text_encoder_load_kwargs["quantization_config"] = GPTQConfig(**gptq_kwargs)
                print("[GPTQ] Using transformers.GPTQConfig for quantization")
                text_encoder = AutoModelForCausalLM.from_pretrained(llama_model_name, **text_encoder_load_kwargs)
            elif autogptq_available:
                from auto_gptq import AutoGPTQForCausalLM
                print("[GPTQ] Using auto_gptq (legacy fallback)")
                text_encoder = AutoGPTQForCausalLM.from_pretrained(llama_model_name, **text_encoder_load_kwargs)
            else:
                raise ImportError("No supported GPTQModel backend for INT4/NF4 (need transformers.GPTQConfig or auto_gptq).")
        except Exception as e:
            print(f"     ⚠️ GPTQ model load failed: {e}")
            raise
    else:
        # Standard BNB/FP8 path (no GPTQ)
        print(f"[1c] Loading Text Encoder: {llama_model_name}... (May download files)")
        text_encoder = LlamaForCausalLM.from_pretrained(llama_model_name, **text_encoder_load_kwargs)

    if "device_map" not in text_encoder_load_kwargs:
        print("     Moving text encoder to CUDA...")
        text_encoder.to("cuda")

    step1_mem = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    print(f"✅ Text encoder loaded! (VRAM: {step1_mem:.2f} MB)")

    # --- 2. Load Transformer (Conditional) ---
    print(f"\n[2] Preparing Transformer from: {model_path}")
    transformer_load_kwargs = {"subfolder": "transformer", "torch_dtype": model_dtype, "low_cpu_mem_usage": True}
    if is_nf4:
        print("     Type: NF4")
    else:
        print("     Type: Standard (Applying 4-bit BNB quantization)")
        if bnb_transformer_4bit_config:
            transformer_load_kwargs["quantization_config"] = bnb_transformer_4bit_config
        else:
            raise ImportError("BNB config required for transformer but unavailable.")

    print("     Loading Transformer... (May download files)")
    transformer = HiDreamImageTransformer2DModel.from_pretrained(model_path, **transformer_load_kwargs)
    print("     Moving Transformer to CUDA...")
    transformer.to("cuda")
    step2_mem = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    print(f"✅ Transformer loaded! (VRAM: {step2_mem:.2f} MB)")

    # --- 3. Load Scheduler ---
    print(f"\n[3] Preparing Scheduler: {scheduler_name}")
    scheduler = get_scheduler_instance(scheduler_name, shift)
    print(f"     Using Scheduler: {scheduler_name}")

    # --- 4. Load Pipeline ---
    print(f"\n[4] Loading Pipeline from: {model_path}")
    print("     Passing pre-loaded components...")
    pipe = HiDreamImagePipeline.from_pretrained(
        model_path,
        scheduler=scheduler,
        tokenizer_4=tokenizer,
        text_encoder_4=text_encoder,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True
    )
    print("     Pipeline structure loaded.")

    # --- 5. Final Setup ---
    print("\n[5] Finalizing Pipeline...")
    print("     Assigning transformer...")
    pipe.transformer = transformer
    print("     Moving pipeline object to CUDA (final check)...")
    try:
        pipe.to("cuda")
    except Exception as e:
        print(f"     Warning: Could not move pipeline object to CUDA: {e}.")
    if is_nf4:
        print("     Attempting CPU offload for NF4...")
        if hasattr(pipe, "enable_sequential_cpu_offload"):
            try:
                pipe.enable_sequential_cpu_offload()
                print("     ✅ CPU offload enabled.")
            except Exception as e:
                print(f"     ⚠️ Failed CPU offload: {e}")
        else:
            print("     ⚠️ enable_sequential_cpu_offload() not found.")

    final_mem = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    print(f"✅ Pipeline ready! (VRAM: {final_mem:.2f} MB)")
    return pipe, MODEL_CONFIGS[model_type]
    
# --- Resolution Parsing & Tensor Conversion ---
RESOLUTION_OPTIONS = [ # (Keep list the same)
    "1024 × 1024 (Square)","768 × 1360 (Portrait)","1360 × 768 (Landscape)",
    "880 × 1168 (Portrait)","1168 × 880 (Landscape)","1248 × 832 (Landscape)",
    "832 × 1248 (Portrait)"
]
def parse_resolution(resolution_str):
    """Parse resolution string into height and width dimensions."""
    try:
        # Extract the resolution part before the parenthesis
        res_part = resolution_str.split(" (")[0].strip()
        # Replace 'x' with '×' for consistency if needed
        parts = res_part.replace('x', '×').split("×")
        
        if len(parts) != 2:
            raise ValueError(f"Expected format 'width × height', got '{res_part}'")
            
        width_str = parts[0].strip()
        height_str = parts[1].strip()
        
        width = int(width_str)
        height = int(height_str)
        print(f"Successfully parsed resolution: {width}x{height}")
        return height, width
    except Exception as e:
        print(f"Error parsing resolution '{resolution_str}': {e}. Falling back to 1024x1024.")
        return 1024, 1024
def pil2tensor(image: Image.Image):
    """Convert PIL image to tensor with better error handling"""
    if image is None:
        print("pil2tensor: Image is None")
        return None
    
    try:
        # Debug image properties
        print(f"pil2tensor: Image mode={image.mode}, size={image.size}")
        
        # Ensure image is in RGB mode
        if image.mode != 'RGB':
            print(f"Converting image from {image.mode} to RGB")
            image = image.convert('RGB')
        
        # Convert to numpy array with explicit steps
        np_array = np.array(image)
        print(f"Numpy array shape={np_array.shape}, dtype={np_array.dtype}")
        
        # Convert to float32 and normalize
        np_array = np_array.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(np_array)
        tensor = tensor.unsqueeze(0)
        print(f"Final tensor shape={tensor.shape}")
        
        return tensor
    except Exception as e:
        print(f"Error in pil2tensor: {e}")
        import traceback
        traceback.print_exc()
        
        # Try ComfyUI's own conversion if ours fails
        try:
            print("Trying ComfyUI's own conversion...")
            tensor = comfy.utils.pil2tensor(image)
            print(f"ComfyUI conversion successful: {tensor.shape}")
            return tensor
        except Exception as e2:
            print(f"ComfyUI conversion also failed: {e2}")
            return None

# --- ComfyUI Node Definition ---
class HiDreamSampler:
    _model_cache = {}
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "HiDream"
    
    @classmethod
    def cleanup_models(cls):
        """Clean up all cached models - can be called by external memory management"""
        print("HiDream: Cleaning up all cached models...")
        keys_to_del = list(cls._model_cache.keys())
        for key in keys_to_del:
            print(f"  Removing '{key}'...")
            try:
                pipe_to_del, _ = cls._model_cache.pop(key)
                # More aggressive cleanup - clear all major components
                if hasattr(pipe_to_del, 'transformer'):
                    pipe_to_del.transformer = None
                if hasattr(pipe_to_del, 'text_encoder_4'):
                    pipe_to_del.text_encoder_4 = None
                if hasattr(pipe_to_del, 'tokenizer_4'):
                    pipe_to_del.tokenizer_4 = None
                if hasattr(pipe_to_del, 'scheduler'):
                    pipe_to_del.scheduler = None
                del pipe_to_del
            except Exception as e:
                print(f"  Error cleaning up {key}: {e}")
        # Multiple garbage collection passes
        for _ in range(3):
            gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Force synchronization
            torch.cuda.synchronize()
        print("HiDream: Cache cleared")
        return True
        
    @staticmethod
    def parse_aspect_ratio(aspect_ratio_str):
        """Parse aspect ratio string to get width and height"""
        try:
            # Extract dimensions from the parenthesis
            dims_part = aspect_ratio_str.split("(")[1].split(")")[0]
            width, height = dims_part.split("×")
            return int(width), int(height)
        except Exception as e:
            print(f"Error parsing aspect ratio '{aspect_ratio_str}': {e}. Falling back to 1024x1024.")
            return 1024, 1024
            
    @classmethod
    def INPUT_TYPES(s):
        available_model_types = list(MODEL_CONFIGS.keys())
        if not available_model_types:
            return {"required": {"error": ("STRING", {"default": "No models available...", "multiline": True})}}
        default_model = "fast-nf4" if "fast-nf4" in available_model_types else "fast" if "fast" in available_model_types else available_model_types[0]
        
        # Define schedulers
        scheduler_options = [
            "Default for model",
            "UniPC",
            "Euler",
            "Karras Euler",
            "Karras Exponential"
        ]
        
        # Resolution options
        aspect_ratio_options = [
        "1:1 Square 1024 × 1024",
        "1:1 Medium 1280 × 1280",
        "1:1 Large 1536 × 1536",
        "1:2 Tablet 704 x 1408",
        "2:1 Rectangle 1408 x 704",
        "2:3 Portrait 704 × 1344",
        "3:2 Landscape 1344 × 704",
        "2:5 Tall 640 × 1600",
        "5:2 Wide 1600 × 640",
        "12:25 Poster 704 × 1472",
        "25:12 Banner 1472 × 704",
        "3:2 Landscape 1152 × 768",
        "2:3 Portrait 768 × 1152",
        "4:5 Classic 896 × 1120",
        "5:4 Classic 1120 × 896",
        "10:9 Near-Square 960 × 832",
        "9:10 Near-Square 832 × 960",
        "16:9 Widescreen 1344 × 768",
        "9:16 Portrait 768 × 1344",
        "1:3 Vertical Poster 576 × 1728",
        "3:1 Panoramic 1728 × 576",
        "1:4 Ultrawide 512 × 2048",
        "4:1 Banner 2048 × 512",
        ]
        
        return {
            "required": {
                "model_type": (available_model_types, {"default": default_model}),
                "prompt": ("STRING", {"multiline": True, "default": "..."}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "aspect_ratio": (aspect_ratio_options, {"default": "1:1 Square (1024×1024)"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "scheduler": (scheduler_options, {"default": "Default for model"}),
                "override_steps": ("INT", {"default": -1, "min": -1, "max": 100}),
                "override_cfg": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 20.0, "step": 0.1}),
                "use_alternate_llm": ("BOOLEAN", {"default": False})
            }
        }
    
    def generate(self, model_type, prompt, negative_prompt, aspect_ratio, seed, scheduler,
                 override_steps, override_cfg, use_alternate_llm=False, **kwargs):
        # Parse resolution from aspect ratio using the static method
        width, height = HiDreamSampler.parse_aspect_ratio(aspect_ratio)
        print(f"Using resolution: {width}×{height} from aspect ratio: {aspect_ratio}")
        # Make dimensions divisible by 16
        width = (width // 64) * 64
        height = (height // 64) * 64
        # Monitor initial memory usage
        if torch.cuda.is_available():
            initial_mem = torch.cuda.memory_allocated() / 1024**2
            print(f"HiDream: Initial VRAM usage: {initial_mem:.2f} MB")
        if not MODEL_CONFIGS or model_type == "error":
            print("HiDream Error: No models loaded.")
            return (torch.zeros((1, 512, 512, 3)),)
        pipe = None; config = None
        # Create cache key that includes alternate state
        cache_key = f"{model_type}_{'alternate' if use_alternate_llm else 'standard'}"
        # --- Model Loading / Caching ---
        if cache_key in self._model_cache:
            print(f"Checking cache for {cache_key}...")
            pipe, config = self._model_cache[cache_key]
            valid_cache = True
            if pipe is None or config is None or not hasattr(pipe, 'transformer') or pipe.transformer is None:
                valid_cache = False
                print("Invalid cache, reloading...")
                del self._model_cache[cache_key]
                pipe, config = None, None
            if valid_cache:
                print("Using cached model.")
        if pipe is None:
            if self._model_cache:
                print(f"Clearing ALL cache before loading {model_type}...")
                keys_to_del = list(self._model_cache.keys())
                for key in keys_to_del:
                    print(f"  Removing '{key}'...")
                    try:
                        pipe_to_del, _= self._model_cache.pop(key)
                        # More aggressive cleanup - clear all major components
                        if hasattr(pipe_to_del, 'transformer'):
                            pipe_to_del.transformer = None
                        if hasattr(pipe_to_del, 'text_encoder_4'):
                            pipe_to_del.text_encoder_4 = None
                        if hasattr(pipe_to_del, 'tokenizer_4'):
                            pipe_to_del.tokenizer_4 = None
                        if hasattr(pipe_to_del, 'scheduler'):
                            pipe_to_del.scheduler = None
                        del pipe_to_del
                    except Exception as e:
                        print(f"  Error removing {key}: {e}")
                # Multiple garbage collection passes
                for _ in range(3):
                    gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    # Force synchronization
                    torch.cuda.synchronize()
                print("Cache cleared.")
            print(f"Loading model for {model_type}{' (alternate)' if use_alternate_llm else ''}...")
            try:
                pipe, config = load_models(model_type, use_alternate_llm)
                self._model_cache[cache_key] = (pipe, config)
                print(f"Model {model_type}{' (alternate)' if use_alternate_llm else ''} loaded & cached!")
            except Exception as e:
                print(f"!!! ERROR loading {model_type}: {e}")
                import traceback
                traceback.print_exc()
                return (torch.zeros((1, 512, 512, 3)),)
        if pipe is None or config is None:
            print("CRITICAL ERROR: Load failed.")
            return (torch.zeros((1, 512, 512, 3)),)
        # --- Update scheduler if requested ---
        txt2img_pipe, model_config = load_models(model_type, use_alternate_llm)
        original_scheduler_class = model_config["scheduler_class"]
        original_shift = config["shift"]
        if scheduler != "Default for model":
            print(f"Replacing default scheduler ({original_scheduler_class}) with: {scheduler}")
            # Create a completely fresh scheduler instance to avoid any parameter leakage
            if scheduler == "UniPC":
                new_scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=original_shift, use_dynamic_shifting=False)
                pipe.scheduler = new_scheduler
            elif scheduler == "Euler":
                new_scheduler = FlashFlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=original_shift, use_dynamic_shifting=False)
                pipe.scheduler = new_scheduler
            elif scheduler == "Karras Euler":
                new_scheduler = FlashFlowMatchEulerDiscreteScheduler(
                    num_train_timesteps=1000,
                    shift=original_shift,
                    use_dynamic_shifting=False,
                    use_karras_sigmas=True
                )
                pipe.scheduler = new_scheduler
            elif scheduler == "Karras Exponential":
                new_scheduler = FlashFlowMatchEulerDiscreteScheduler(
                    num_train_timesteps=1000,
                    shift=original_shift,
                    use_dynamic_shifting=False,
                    use_exponential_sigmas=True
                )
                pipe.scheduler = new_scheduler
        else:
            # Ensure we're using the original scheduler as specified in the model config
            print(f"Using model's default scheduler: {original_scheduler_class}")
            pipe.scheduler = get_scheduler_instance(original_scheduler_class, original_shift)
        # --- Generation Setup ---
        is_nf4_current = config.get("is_nf4", False)
        num_inference_steps = override_steps if override_steps >= 0 else config["num_inference_steps"]
        guidance_scale = override_cfg if override_cfg >= 0.0 else config["guidance_scale"]
        # Create the progress bar
        pbar = comfy.utils.ProgressBar(num_inference_steps)
        # Set default max sequence lengths
        max_length_clip_l = 77
        max_length_openclip = 150
        max_length_t5 = 256
        max_length_llama = 256

        # Set default encoder weights
        clip_l_weight = 1.0
        openclip_weight = 1.0
        t5_weight = 1.0
        llama_weight = 1.0
                     
        try:
            inference_device = comfy.model_management.get_torch_device()
        except Exception:
            inference_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Creating Generator on: {inference_device}")
        generator = torch.Generator(device=inference_device).manual_seed(seed)
        print(f"\n--- Starting Generation ---")
        print(f"Model: {model_type}{' (alternate)' if use_alternate_llm else ''}, Res: {height}x{width}, Steps: {num_inference_steps}, CFG: {guidance_scale}, Seed: {seed}")
        print(f"Using standard sequence lengths: CLIP-L: {max_length_clip_l}, OpenCLIP: {max_length_openclip}, T5: {max_length_t5}, Llama: {max_length_llama}")
        # --- Run Inference ---
        output_images = None
        try:
            if not is_nf4_current:
                print(f"Ensuring pipe on: {inference_device} (Offload NOT enabled)")
                pipe.to(inference_device)
            else:
                print(f"Skipping pipe.to({inference_device}) (CPU offload enabled).")
            print("Executing pipeline inference...")
            # Call pipeline with individual sequence lengths
            with torch.inference_mode():
                output_images = pipe(
                    prompt=prompt,           # CLIP-L 
                    prompt_2=prompt,         # OpenCLIP - explicitly send same prompt
                    prompt_3=prompt,         # T5 - explicitly send same prompt
                    prompt_4=prompt,         # LLM - explicitly send same prompt
                    negative_prompt=negative_prompt.strip() if negative_prompt else None,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    num_images_per_prompt=1,
                    generator=generator,
                    max_sequence_length_clip_l=max_length_clip_l,
                    max_sequence_length_openclip=max_length_openclip,
                    max_sequence_length_t5=max_length_t5,
                    max_sequence_length_llama=max_length_llama,
                    clip_l_scale=clip_l_weight,
                    openclip_scale=openclip_weight,
                    t5_scale=t5_weight,
                    llama_scale=llama_weight,
                ).images
            print("Pipeline inference finished.")
        except Exception as e:
            print(f"!!! ERROR during execution: {e}")
            import traceback
            traceback.print_exc()
            return (torch.zeros((1, height, width, 3)),)
        finally:
            pbar.update_absolute(num_inference_steps) # Update pbar regardless
        print("--- Generation Complete ---")
        
        # Robust output handling
        if output_images is None or len(output_images) == 0:
            print("ERROR: No images returned. Creating blank image.")
            return (torch.zeros((1, height, width, 3)),)
            
        try:
            print(f"Processing output image. Type: {type(output_images[0])}")
            output_tensor = pil2tensor(output_images[0])
            if output_tensor is None:
                print("ERROR: pil2tensor returned None. Creating blank image.")
                return (torch.zeros((1, height, width, 3)),)
                
            # Fix for any non-float32 tensor issue
            if output_tensor.dtype != torch.float32:
                print(f"Converting {output_tensor.dtype} tensor to float32 for ComfyUI compatibility")
                output_tensor = output_tensor.to(torch.float32)
                
            # Verify tensor shape is valid
            if len(output_tensor.shape) != 4 or output_tensor.shape[0] != 1 or output_tensor.shape[3] != 3:
                print(f"ERROR: Invalid tensor shape {output_tensor.shape}. Creating blank image.")
                return (torch.zeros((1, height, width, 3)),)
                
            print(f"Output tensor shape: {output_tensor.shape}")
            
            # After generating the image, try to clean up any temporary memory
            try:
                import comfy.model_management as model_management
                print("HiDream: Requesting ComfyUI memory cleanup...")
                model_management.soft_empty_cache()
            except Exception as e:
                print(f"HiDream: ComfyUI cleanup failed: {e}")
                
            # Log final memory usage
            if torch.cuda.is_available():
                final_mem = torch.cuda.memory_allocated() / 1024**2
                print(f"HiDream: Final VRAM usage: {final_mem:.2f} MB (Change: {final_mem-initial_mem:.2f} MB)")
                
            return (output_tensor,)
        except Exception as e:
            print(f"Error processing output image: {e}")
            import traceback
            traceback.print_exc()
            return (torch.zeros((1, height, width, 3)),)

# --- ComfyUI Node 2 Definition ---
class HiDreamSamplerAdvanced:
    _model_cache = HiDreamSampler._model_cache
    cleanup_models = HiDreamSampler.cleanup_models
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "HiDream"
    
    @classmethod
    def INPUT_TYPES(s):
        available_model_types = list(MODEL_CONFIGS.keys())
        if not available_model_types:
            return {"required": {"error": ("STRING", {"default": "No models available...", "multiline": True})}}
        default_model = "fast-nf4" if "fast-nf4" in available_model_types else "fast" if "fast" in available_model_types else available_model_types[0]
        
        # Define schedulers
        scheduler_options = [
            "Default for model",
            "UniPC",
            "Euler",
            "Karras Euler",
            "Karras Exponential"
        ]
        
        # Resolution options
        aspect_ratio_options = [
            "1:1 (Square Reso)",
            "1:2 Tablet 704 x 1408",
            "2:1 Rectangle 1408 x 704",
            "2:3 Portrait 704 × 1344",
            "3:2 Landscape 1344 × 704",
            "2:5 Tall 640 × 1600",
            "5:2 Wide 1600 × 640",
            "12:25 Poster 704 × 1472",
            "25:12 Banner 1472 × 704",
            "3:2 Landscape 1152 × 768",
            "2:3 Portrait 768 × 1152",
            "4:5 Classic 896 × 1120",
            "5:4 Classic 1120 × 896",
            "10:9 Near-Square 960 × 832",
            "9:10 Near-Square 832 × 960",
            "16:9 Widescreen 1344 × 768",
            "9:16 Portrait 768 × 1344",
            "1:3 Vertical Poster 576 × 1728",
            "3:1 Panoramic 1728 × 576",
            "1:4 Ultrawide 512 × 2048",
            "4:1 Banner 2048 × 512",
            "Custom"
        ]
        
        return {
            "required": {
                "model_type": (available_model_types, {"default": default_model}),
                "primary_prompt": ("STRING", {"multiline": True, "default": "..."}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "aspect_ratio": (aspect_ratio_options, {"default": "1:1 (Square Reso)"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "scheduler": (scheduler_options, {"default": "Default for model"}),
                "override_steps": ("INT", {"default": -1, "min": -1, "max": 100}),
                "override_cfg": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 20.0, "step": 0.1}),
                "use_alternate_llm": ("BOOLEAN", {"default": False})
            },
            "optional": {
                "clip_l_prompt": ("STRING", {"multiline": True, "default": ""}),
                "openclip_prompt": ("STRING", {"multiline": True, "default": ""}),
                "t5_prompt": ("STRING", {"multiline": True, "default": ""}),
                "llama_prompt": ("STRING", {"multiline": True, "default": ""}),
                "llm_system_prompt": ("STRING", {
                    "multiline": True, 
                    "default": "You are a creative AI assistant that helps create detailed, vivid images based on user descriptions."
                }),
                "clip_l_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "openclip_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "t5_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "llama_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "square_resolution": ("INT", {"default": 1024, "min": 512, "max": 3072, "step": 64}),
                "custom_width": ("INT", {"default": 1024, "min": 512, "max": 3072, "step": 64}),
                "custom_height": ("INT", {"default": 1024, "min": 512, "max": 3072, "step": 64}),
                "max_length_clip_l": ("INT", {"default": 77, "min": 64, "max": 218}),
                "max_length_openclip": ("INT", {"default": 77, "min": 64, "max": 218}),
                "max_length_t5": ("INT", {"default": 128, "min": 64, "max": 512}),
                "max_length_llama": ("INT", {"default": 128, "min": 64, "max": 2048})
            }
        }
    
    @staticmethod
    def parse_dimensions(aspect_ratio_str):
        """Parse dimensions from a string like '9:16 (768×1360)'"""
        try:
            if "Square Reso" in aspect_ratio_str:
                return 1024, 1024  # Default if parsing fails
            dims_part = aspect_ratio_str.split("(")[1].split(")")[0]
            width, height = dims_part.split("×")
            return int(width), int(height)
        except Exception as e:
            print(f"Error parsing aspect ratio '{aspect_ratio_str}': {e}. Falling back to 1024x1024.")
            return 1024, 1024
    
    def generate(self, model_type, primary_prompt, negative_prompt, aspect_ratio, seed, scheduler,
                 override_steps, override_cfg, use_alternate_llm=False,
                 clip_l_prompt="", openclip_prompt="", t5_prompt="", llama_prompt="",
                 llm_system_prompt="You are a creative AI assistant...",
                 square_resolution=1024, custom_width=1024, custom_height=1024,
                 max_length_clip_l=77, max_length_openclip=77, max_length_t5=128, max_length_llama=128,
                 clip_l_weight=1.0, openclip_weight=1.0, t5_weight=1.0, llama_weight=1.0, **kwargs):
        
        # Get width and height based on aspect ratio
        if "Square Reso" in aspect_ratio:
            width, height = square_resolution, square_resolution
            print(f"Using square resolution: {width}×{height}")
        elif aspect_ratio == "Custom":
            width, height = custom_width, custom_height
            print(f"Using custom resolution: {width}×{height}")
        else:
            width, height = self.parse_dimensions(aspect_ratio)
            print(f"Using resolution: {width}×{height} from aspect ratio: {aspect_ratio}")
        
        # Make width and height divisible by 16
        width = (width // 64) * 64
        height = (height // 64) * 64
        
        # Monitor initial memory usage
        if torch.cuda.is_available():
            initial_mem = torch.cuda.memory_allocated() / 1024**2
            print(f"HiDream: Initial VRAM usage: {initial_mem:.2f} MB")
            
        if not MODEL_CONFIGS or model_type == "error":
            print("HiDream Error: No models loaded.")
            return (torch.zeros((1, 512, 512, 3)),)
            
        pipe = None; config = None
        
        # Create cache key that includes alternate state
        cache_key = f"{model_type}_{'alternate' if use_alternate_llm else 'standard'}"
        
        # --- Model Loading / Caching ---
        if cache_key in self._model_cache:
            print(f"Checking cache for {cache_key}...")
            pipe, config = self._model_cache[cache_key]
            valid_cache = True
            if pipe is None or config is None or not hasattr(pipe, 'transformer') or pipe.transformer is None:
                valid_cache = False
                print("Invalid cache, reloading...")
                del self._model_cache[cache_key]
                pipe, config = None, None
            if valid_cache:
                print("Using cached model.")
                
        if pipe is None:
            if self._model_cache:
                print(f"Clearing ALL cache before loading {model_type}...")
                keys_to_del = list(self._model_cache.keys())
                for key in keys_to_del:
                    print(f"  Removing '{key}'...")
                    try:
                        pipe_to_del, _= self._model_cache.pop(key)
                        # More aggressive cleanup - clear all major components
                        if hasattr(pipe_to_del, 'transformer'):
                            pipe_to_del.transformer = None
                        if hasattr(pipe_to_del, 'text_encoder_4'):
                            pipe_to_del.text_encoder_4 = None
                        if hasattr(pipe_to_del, 'tokenizer_4'):
                            pipe_to_del.tokenizer_4 = None
                        if hasattr(pipe_to_del, 'scheduler'):
                            pipe_to_del.scheduler = None
                        del pipe_to_del
                    except Exception as e:
                        print(f"  Error removing {key}: {e}")
                # Multiple garbage collection passes
                for _ in range(3):
                    gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    # Force synchronization
                    torch.cuda.synchronize()
                print("Cache cleared.")
                
            print(f"Loading model for {model_type}{' (alternate)' if use_alternate_llm else ''}...")
            try:
                pipe, config = load_models(model_type, use_alternate_llm)
                self._model_cache[cache_key] = (pipe, config)
                print(f"Model {model_type}{' (alternate)' if use_alternate_llm else ''} loaded & cached!")
            except Exception as e:
                print(f"!!! ERROR loading {model_type}: {e}")
                import traceback
                traceback.print_exc()
                return (torch.zeros((1, 512, 512, 3)),)
                
        if pipe is None or config is None:
            print("CRITICAL ERROR: Load failed.")
            return (torch.zeros((1, 512, 512, 3)),)
            
        # --- Update scheduler if requested ---
        original_scheduler_class = config["scheduler_class"]
        original_shift = config["shift"]
        
        if scheduler != "Default for model":
            print(f"Replacing default scheduler ({original_scheduler_class}) with: {scheduler}")
            
            # Create a completely fresh scheduler instance to avoid any parameter leakage
            if scheduler == "UniPC":
                new_scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=original_shift, use_dynamic_shifting=False)
                pipe.scheduler = new_scheduler
            elif scheduler == "Euler":
                new_scheduler = FlashFlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=original_shift, use_dynamic_shifting=False)
                pipe.scheduler = new_scheduler
            elif scheduler == "Karras Euler":
                new_scheduler = FlashFlowMatchEulerDiscreteScheduler(
                    num_train_timesteps=1000, 
                    shift=original_shift, 
                    use_dynamic_shifting=False,
                    use_karras_sigmas=True
                )
                pipe.scheduler = new_scheduler
            elif scheduler == "Karras Exponential":
                new_scheduler = FlashFlowMatchEulerDiscreteScheduler(
                    num_train_timesteps=1000, 
                    shift=original_shift,
                    use_dynamic_shifting=False,
                    use_exponential_sigmas=True
                )
                pipe.scheduler = new_scheduler
        else:
            # Ensure we're using the original scheduler as specified in the model config
            print(f"Using model's default scheduler: {original_scheduler_class}")
            pipe.scheduler = get_scheduler_instance(original_scheduler_class, original_shift)
                
        # --- Generation Setup ---
        is_nf4_current = config.get("is_nf4", False)
        num_inference_steps = override_steps if override_steps >= 0 else config["num_inference_steps"]
        guidance_scale = override_cfg if override_cfg >= 0.0 else config["guidance_scale"]
        pbar = comfy.utils.ProgressBar(num_inference_steps) # Keep pbar for final update
        
        try:
            inference_device = comfy.model_management.get_torch_device()
        except Exception:
            inference_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        print(f"Creating Generator on: {inference_device}")
        generator = torch.Generator(device=inference_device).manual_seed(seed)
        print(f"\n--- Starting Generation ---")
        print(f"Model: {model_type}{' (alternate)' if use_alternate_llm else ''}, Res: {height}x{width}, Steps: {num_inference_steps}, CFG: {guidance_scale}, Seed: {seed}")
        print(f"Sequence lengths - CLIP-L: {max_length_clip_l}, OpenCLIP: {max_length_openclip}, T5: {max_length_t5}, Llama: {max_length_llama}")
        
        # --- Run Inference ---
        output_images = None
        try:
            if not is_nf4_current:
                print(f"Ensuring pipe on: {inference_device} (Offload NOT enabled)")
                pipe.to(inference_device)
            else:
                print(f"Skipping pipe.to({inference_device}) (CPU offload enabled).")
                
            print("Executing pipeline inference...")
            # Make width and height divisible by 16
            width = (width // 64) * 64
            height = (height // 64) * 64
            
            # Use specific prompts for each encoder, falling back to primary prompt if empty
            prompt_clip_l = clip_l_prompt.strip() if clip_l_prompt.strip() else primary_prompt
            prompt_openclip = openclip_prompt.strip() if openclip_prompt.strip() else primary_prompt
            prompt_t5 = t5_prompt.strip() if t5_prompt.strip() else primary_prompt
            prompt_llama = llama_prompt.strip() if llama_prompt.strip() else primary_prompt
            
            print(f"Using per-encoder prompts:")
            print(f"  CLIP-L ({max_length_clip_l} tokens): {prompt_clip_l[:50]}{'...' if len(prompt_clip_l) > 50 else ''}")
            print(f"  OpenCLIP ({max_length_openclip} tokens): {prompt_openclip[:50]}{'...' if len(prompt_openclip) > 50 else ''}")
            print(f"  T5 ({max_length_t5} tokens): {prompt_t5[:50]}{'...' if len(prompt_t5) > 50 else ''}")
            print(f"  Llama ({max_length_llama} tokens): {prompt_llama[:50]}{'...' if len(prompt_llama) > 50 else ''}")

                        
            # Replace truly blank inputs with minimal period
            if not prompt_clip_l.strip():
                prompt_clip_l = "."
                
            if not prompt_openclip.strip():
                prompt_openclip = "."
                
            if not prompt_t5.strip():
                prompt_t5 = "."
                
            # Custom system prompt for blank LLM prompts to try to prevent LLM output noise
            custom_system_prompt = llm_system_prompt
            if not prompt_llama.strip():
                prompt_llama = "."
                custom_system_prompt = "You will only output a single period as your output '.'\nDo not add any other acknowledgement or extra text or data."

            # Create the progress bar
            pbar = comfy.utils.ProgressBar(num_inference_steps)
            
            # Define a progress callback function that updates the ComfyUI progress bar
            def progress_callback(pipe, i, t, callback_kwargs):
                # Update ComfyUI progress bar
                pbar.update_absolute(i+1)
                return callback_kwargs
            
            # Call pipeline with encoder-specific prompts and system prompt
            with torch.inference_mode():
                output_images = pipe(
                    prompt=prompt_clip_l,         # CLIP-L specific prompt
                    prompt_2=prompt_openclip,     # OpenCLIP specific prompt
                    prompt_3=prompt_t5,           # T5 specific prompt
                    prompt_4=prompt_llama,        # Llama specific prompt
                    negative_prompt=negative_prompt.strip() if negative_prompt else None,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    num_images_per_prompt=1,
                    generator=generator,
                    max_sequence_length_clip_l=max_length_clip_l,
                    max_sequence_length_openclip=max_length_openclip,
                    max_sequence_length_t5=max_length_t5,
                    max_sequence_length_llama=max_length_llama,
                    llm_system_prompt=custom_system_prompt,
                    clip_l_scale=clip_l_weight,
                    openclip_scale=openclip_weight,
                    t5_scale=t5_weight,
                    llama_scale=llama_weight,
                    callback_on_step_end=progress_callback,
                    callback_on_step_end_tensor_inputs=["latents"],
                ).images
            print("Pipeline inference finished.")
        except Exception as e:
            print(f"!!! ERROR during execution: {e}")
            import traceback
            traceback.print_exc()
            return (torch.zeros((1, height, width, 3)),)
        finally:
            pbar.update_absolute(num_inference_steps) # Update pbar regardless
            
        print("--- Generation Complete ---")
        
        # Robust output handling
        if output_images is None or len(output_images) == 0:
            print("ERROR: No images returned. Creating blank image.")
            return (torch.zeros((1, height, width, 3)),)
            
        try:
            print(f"Processing output image. Type: {type(output_images[0])}")
            output_tensor = pil2tensor(output_images[0])
            if output_tensor is None:
                print("ERROR: pil2tensor returned None. Creating blank image.")
                return (torch.zeros((1, height, width, 3)),)
                
            # Fix for bfloat16 tensor issue
            if output_tensor.dtype == torch.bfloat16:
                print("Converting bfloat16 tensor to float32 for ComfyUI compatibility")
                output_tensor = output_tensor.to(torch.float32)
                
            # Verify tensor shape is valid
            if len(output_tensor.shape) != 4 or output_tensor.shape[0] != 1 or output_tensor.shape[3] != 3:
                print(f"ERROR: Invalid tensor shape {output_tensor.shape}. Creating blank image.")
                return (torch.zeros((1, height, width, 3)),)
                
            print(f"Output tensor shape: {output_tensor.shape}")
            
            # After generating the image, try to clean up any temporary memory
            try:
                import comfy.model_management as model_management
                print("HiDream: Requesting ComfyUI memory cleanup...")
                model_management.soft_empty_cache()
            except Exception as e:
                print(f"HiDream: ComfyUI cleanup failed: {e}")
                
            # Log final memory usage
            if torch.cuda.is_available():
                final_mem = torch.cuda.memory_allocated() / 1024**2
                print(f"HiDream: Final VRAM usage: {final_mem:.2f} MB (Change: {final_mem-initial_mem:.2f} MB)")
                
            return (output_tensor,)
        except Exception as e:
            print(f"Error processing output image: {e}")
            import traceback
            traceback.print_exc()
            return (torch.zeros((1, height, width, 3)),)


class HiDreamImg2Img:
    _model_cache = HiDreamSampler._model_cache
    cleanup_models = HiDreamSampler.cleanup_models
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "HiDream"
    
    @classmethod
    def INPUT_TYPES(s):
        available_model_types = list(MODEL_CONFIGS.keys())
        if not available_model_types:
            return {"required": {"error": ("STRING", {"default": "No models available...", "multiline": True})}}
        
        default_model = "fast-nf4" if "fast-nf4" in available_model_types else "fast" if "fast" in available_model_types else available_model_types[0]
        
        # Define schedulers
        scheduler_options = [
            "Default for model",
            "UniPC",
            "Euler",
            "Karras Euler",
            "Karras Exponential"
        ]
        
        return {
            "required": {
                "model_type": (available_model_types, {"default": default_model}),
                "image": ("IMAGE",),
                "denoising_strength": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01}),
                "prompt": ("STRING", {"multiline": True, "default": "..."}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "scheduler": (scheduler_options, {"default": "Default for model"}),
                "override_steps": ("INT", {"default": -1, "min": -1, "max": 100}),
                "override_cfg": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 20.0, "step": 0.1}),
                "use_alternate_llm": ("BOOLEAN", {"default": False})
            },
            "optional": {
                "llm_system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "You are a creative AI assistant that helps create detailed, vivid images based on user descriptions."
                }),
                "clip_l_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "openclip_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "t5_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "llama_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
            }
        }

    def preprocess_image(self, image, target_height=None, target_width=None):
        """Resize and possibly crop input image to match model requirements."""
        import torch.nn.functional as F
        import math
        
        # Get original dimensions
        _, orig_h, orig_w, _ = image.shape
        orig_aspect = orig_w / orig_h
        
        print(f"Original image dimensions: {orig_w}x{orig_h}, aspect ratio: {orig_aspect:.3f}")
        
        # If no target size provided, find closest standard resolution
        if target_height is None or target_width is None:
            # Define standard resolutions (must be divisible by 16)
            standard_resolutions = [
                (1024, 1024),  # 1:1
                (768, 1360),   # 9:16 (portrait)
                (1360, 768),   # 16:9 (landscape)
                (880, 1168),   # 3:4 (portrait)
                (1168, 880),   # 4:3 (landscape)
                (832, 1248),   # 2:3 (portrait)
                (1248, 832),   # 3:2 (landscape)
            ]
            
            # Find closest aspect ratio
            best_diff = float('inf')
            target_width, target_height = standard_resolutions[0]  # Default to square
            
            for w, h in standard_resolutions:
                res_aspect = w / h
                diff = abs(res_aspect - orig_aspect)
                if diff < best_diff:
                    best_diff = diff
                    target_width, target_height = w, h
            
            print(f"Selected target resolution: {target_width}x{target_height}")
        
        # Ensure dimensions are divisible by 16
        target_width = (target_width // 16) * 16
        target_height = (target_height // 16) * 16
        
        # Convert to format expected by F.interpolate [B,C,H,W]
        # ComfyUI typically uses [B,H,W,C]
        x = image.permute(0, 3, 1, 2)
        
        # Calculate resize dimensions preserving aspect ratio
        if orig_aspect > target_width / target_height:  # Image is wider
            new_w = target_width
            new_h = int(new_w / orig_aspect)
            new_h = (new_h // 16) * 16  # Make divisible by 16
        else:  # Image is taller
            new_h = target_height
            new_w = int(new_h * orig_aspect)
            new_w = (new_w // 16) * 16  # Make divisible by 16
        
        # Resize to preserve aspect ratio
        x_resized = F.interpolate(x, size=(new_h, new_w), mode='bicubic', align_corners=False)
        
        # Create target tensor with correct dimensions
        x_result = torch.zeros(1, 3, target_height, target_width, device=x.device, dtype=x.dtype)
        
        # Calculate position for center crop
        y_offset = max(0, (new_h - target_height) // 2)
        x_offset = max(0, (new_w - target_width) // 2)
        
        # Calculate how much to copy
        height_to_copy = min(new_h, target_height)
        width_to_copy = min(new_w, target_width)
        
        # Place the resized image in the center of the target tensor
        target_y_offset = max(0, (target_height - height_to_copy) // 2)
        target_x_offset = max(0, (target_width - width_to_copy) // 2)
        
        x_result[:, :, 
                 target_y_offset:target_y_offset+height_to_copy, 
                 target_x_offset:target_x_offset+width_to_copy] = x_resized[:, :, 
                                                                           y_offset:y_offset+height_to_copy, 
                                                                           x_offset:x_offset+width_to_copy]
        
        print(f"Processed to: {target_width}x{target_height} (divisible by 16)")
        
        # Convert back to ComfyUI format [B,H,W,C]
        return x_result.permute(0, 2, 3, 1)
    
    def generate(self, model_type, image, denoising_strength, prompt, negative_prompt, 
             seed, scheduler, override_steps, override_cfg, use_alternate_llm=False,
             llm_system_prompt="You are a creative AI assistant...",
             clip_l_weight=1.0, openclip_weight=1.0, t5_weight=1.0, llama_weight=1.0, **kwargs):


        # Preprocess the input image to ensure compatible dimensions
        processed_image = self.preprocess_image(image)
        
        # Get dimensions from processed image for the output
        _, height, width, _ = processed_image.shape
                 
        # Monitor initial memory usage
        if torch.cuda.is_available():
            initial_mem = torch.cuda.memory_allocated() / 1024**2
            print(f"HiDream: Initial VRAM usage: {initial_mem:.2f} MB")
            
        if not MODEL_CONFIGS or model_type == "error":
            print("HiDream Error: No models loaded.")
            return (torch.zeros((1, 512, 512, 3)),)
            
        pipe = None
        config = None
        
        # Create cache key that includes alternate state
        cache_key = f"{model_type}_img2img_{'alternate' if use_alternate_llm else 'standard'}"
        
        # Try to reuse from cache first
        if cache_key in self._model_cache:
            print(f"Checking cache for {cache_key}...")
            pipe, config = self._model_cache[cache_key]
            valid_cache = True
            
            if pipe is None or config is None or not hasattr(pipe, 'transformer') or pipe.transformer is None:
                valid_cache = False
                print("Invalid cache, reloading...")
                del self._model_cache[cache_key]
                pipe, config = None, None
                
            if valid_cache:
                print("Using cached model.")
        
        # Load model if needed
        if pipe is None:
            # Clear cache before loading new model
            if self._model_cache:
                print(f"Clearing img2img cache before loading {model_type}...")
                keys_to_del = list(self._model_cache.keys())
                for key in keys_to_del:
                    print(f"  Removing '{key}'...")
                    try:
                        pipe_to_del, _= self._model_cache.pop(key)
                        # More aggressive cleanup
                        if hasattr(pipe_to_del, 'transformer'):
                            pipe_to_del.transformer = None
                        if hasattr(pipe_to_del, 'text_encoder_4'):
                            pipe_to_del.text_encoder_4 = None
                        if hasattr(pipe_to_del, 'tokenizer_4'):
                            pipe_to_del.tokenizer_4 = None
                        if hasattr(pipe_to_del, 'scheduler'):
                            pipe_to_del.scheduler = None
                        del pipe_to_del
                    except Exception as e:
                        print(f"  Error removing {key}: {e}")
                
                # Multiple garbage collection passes
                for _ in range(3):
                    gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    # Force synchronization
                    torch.cuda.synchronize()
                print("Cache cleared.")
            
            print(f"Loading model for {model_type} img2img...")
            try:
                # First load regular model
                txt2img_pipe, config = load_models(model_type, use_alternate_llm)
                
                # Convert to img2img pipeline
                print("Creating img2img pipeline from loaded txt2img pipeline...")
                pipe = HiDreamImageToImagePipeline(
                    scheduler=txt2img_pipe.scheduler,
                    vae=txt2img_pipe.vae,
                    text_encoder=txt2img_pipe.text_encoder,
                    tokenizer=txt2img_pipe.tokenizer,
                    text_encoder_2=txt2img_pipe.text_encoder_2,
                    tokenizer_2=txt2img_pipe.tokenizer_2,
                    text_encoder_3=txt2img_pipe.text_encoder_3,
                    tokenizer_3=txt2img_pipe.tokenizer_3,
                    text_encoder_4=txt2img_pipe.text_encoder_4,
                    tokenizer_4=txt2img_pipe.tokenizer_4,
                )
                
                # Copy transformer and move to right device
                pipe.transformer = txt2img_pipe.transformer
                
                # Cleanup txt2img pipeline references
                txt2img_pipe = None
                
                # Cache the img2img pipeline
                self._model_cache[cache_key] = (pipe, config)
                print(f"Model {model_type} loaded & cached for img2img!")
                
            except Exception as e:
                print(f"!!! ERROR loading {model_type}: {e}")
                import traceback
                traceback.print_exc()
                return (torch.zeros((1, 512, 512, 3)),)
                
        if pipe is None or config is None:
            print("CRITICAL ERROR: Load failed.")
            return (torch.zeros((1, 512, 512, 3)),)
        
        # Update scheduler if requested
        original_scheduler_class = config["scheduler_class"]
        original_shift = config["shift"]
        if scheduler != "Default for model":
            print(f"Replacing default scheduler ({original_scheduler_class}) with: {scheduler}")
            # Create a completely fresh scheduler instance to avoid any parameter leakage
            if scheduler == "UniPC":
                new_scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=original_shift, use_dynamic_shifting=False)
                pipe.scheduler = new_scheduler
            elif scheduler == "Euler":
                new_scheduler = FlashFlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=original_shift, use_dynamic_shifting=False)
                pipe.scheduler = new_scheduler
            elif scheduler == "Karras Euler":
                new_scheduler = FlashFlowMatchEulerDiscreteScheduler(
                    num_train_timesteps=1000,
                    shift=original_shift,
                    use_dynamic_shifting=False,
                    use_karras_sigmas=True
                )
                pipe.scheduler = new_scheduler
            elif scheduler == "Karras Exponential":
                new_scheduler = FlashFlowMatchEulerDiscreteScheduler(
                    num_train_timesteps=1000,
                    shift=original_shift,
                    use_dynamic_shifting=False,
                    use_exponential_sigmas=True
                )
                pipe.scheduler = new_scheduler
        else:
            # Ensure we're using the original scheduler as specified in the model config
            print(f"Using model's default scheduler: {original_scheduler_class}")
            pipe.scheduler = get_scheduler_instance(original_scheduler_class, original_shift)
            
        # Setup generation parameters
        is_nf4_current = config.get("is_nf4", False) 
        num_inference_steps = override_steps if override_steps >= 0 else config["num_inference_steps"]
        guidance_scale = override_cfg if override_cfg >= 0.0 else config["guidance_scale"]
        
        # Create progress bar
        pbar = comfy.utils.ProgressBar(num_inference_steps)
        
        # Define progress callback
        def progress_callback(pipe, i, t, callback_kwargs):
            # Update ComfyUI progress bar
            pbar.update_absolute(i+1)
            return callback_kwargs
            
        try:
            inference_device = comfy.model_management.get_torch_device()
        except Exception:
            inference_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        print(f"Creating Generator on: {inference_device}")
        generator = torch.Generator(device=inference_device).manual_seed(seed)
        
        print(f"\n--- Starting Img2Img Generation ---")
        _, h, w, _ = image.shape
        print(f"Model: {model_type}{' (alternate)' if use_alternate_llm else ''}, Input Size: {h}x{w}")
        print(f"Denoising: {denoising_strength}, Steps: {num_inference_steps}, CFG: {guidance_scale}, Seed: {seed}")
        
        output_images = None
        try:
            if not is_nf4_current:
                print(f"Ensuring pipe on: {inference_device} (Offload NOT enabled)")
                pipe.to(inference_device)
            else:
                print(f"Skipping pipe.to({inference_device}) (CPU offload enabled).")
                
            print("Executing pipeline inference...")
            
            with torch.inference_mode():
                output_images = pipe(
                    prompt=prompt,
                    prompt_2=prompt,  # Same prompt for all encoders
                    prompt_3=prompt,
                    prompt_4=prompt,
                    negative_prompt=negative_prompt.strip() if negative_prompt else None,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    num_images_per_prompt=1,
                    generator=generator,
                    init_image=processed_image,
                    denoising_strength=denoising_strength,
                    llm_system_prompt=llm_system_prompt,
                    clip_l_scale=clip_l_weight,
                    openclip_scale=openclip_weight,
                    t5_scale=t5_weight,
                    llama_scale=llama_weight,
                    callback_on_step_end=progress_callback,
                    callback_on_step_end_tensor_inputs=["latents"],
                ).images
                
            print("Pipeline inference finished.")
            
        except Exception as e:
            print(f"!!! ERROR during execution: {e}")
            import traceback
            traceback.print_exc()
            return (torch.zeros((1, h, w, 3)),)
            
        finally:
            pbar.update_absolute(num_inference_steps) # Update pbar regardless
            
        print("--- Generation Complete ---")
        
        # Robust output handling
        if output_images is None or len(output_images) == 0:
            print("ERROR: No images returned. Creating blank image.")
            return (torch.zeros((1, h, w, 3)),)
            
        try:
            print(f"Processing output image. Type: {type(output_images[0])}")
            output_tensor = pil2tensor(output_images[0])
            
            if output_tensor is None:
                print("ERROR: pil2tensor returned None. Creating blank image.")
                return (torch.zeros((1, h, w, 3)),)
                
            # Fix for bfloat16 tensor issue
            if output_tensor.dtype == torch.bfloat16:
                print("Converting bfloat16 tensor to float32 for ComfyUI compatibility")
                output_tensor = output_tensor.to(torch.float32)
                
            # Verify tensor shape is valid
            if len(output_tensor.shape) != 4 or output_tensor.shape[0] != 1 or output_tensor.shape[3] != 3:
                print(f"ERROR: Invalid tensor shape {output_tensor.shape}. Creating blank image.")
                return (torch.zeros((1, h, w, 3)),)
                
            print(f"Output tensor shape: {output_tensor.shape}")
            
            # After generating the image, try to clean up any temporary memory
            try:
                import comfy.model_management as model_management
                print("HiDream: Requesting ComfyUI memory cleanup...")
                model_management.soft_empty_cache()
            except Exception as e:
                print(f"HiDream: ComfyUI cleanup failed: {e}")
                
            # Log final memory usage
            if torch.cuda.is_available():
                final_mem = torch.cuda.memory_allocated() / 1024**2
                print(f"HiDream: Final VRAM usage: {final_mem:.2f} MB (Change: {final_mem-initial_mem:.2f} MB)")
                
            return (output_tensor,)
            
        except Exception as e:
            print(f"Error processing output image: {e}")
            import traceback
            traceback.print_exc()
            return (torch.zeros((1, h, w, 3)),)

# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "HiDreamSampler": HiDreamSampler,
    "HiDreamSamplerAdvanced": HiDreamSamplerAdvanced,
    "HiDreamImg2Img": HiDreamImg2Img
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HiDreamSampler": "HiDream Sampler",
    "HiDreamSamplerAdvanced": "HiDream Sampler (Advanced)",
    "HiDreamImg2Img": "HiDream Image to Image"
}

# --- Register with ComfyUI's Memory Management ---
try:
    import comfy.model_management as model_management
    
    # Check if we can register a cleanup callback
    if hasattr(model_management, 'unload_all_models'):
        original_unload = model_management.unload_all_models
        
        # Wrap the original function to include our cleanup
        def wrapped_unload():
            print("HiDream: ComfyUI is unloading all models, cleaning HiDream cache...")
            HiDreamSampler.cleanup_models()
            return original_unload()
            
        # Replace the original function with our wrapped version
        model_management.unload_all_models = wrapped_unload
        print("HiDream: Successfully registered with ComfyUI memory management")
except Exception as e:
    print(f"HiDream: Could not register cleanup with model_management: {e}")

print("-" * 50 + "\nHiDream Sampler Node Initialized\nAvailable Models: " + str(list(MODEL_CONFIGS.keys())) + "\n" + "-" * 50)
