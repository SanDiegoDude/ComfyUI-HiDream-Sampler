from .hidreamsampler import HiDreamSampler, HiDreamSamplerAdvanced, HiDreamImg2Img, HiDreamResolutionSelect

NODE_CLASS_MAPPINGS = {
    "HiDreamSampler": HiDreamSampler,
    "HiDreamSamplerAdvanced": HiDreamSamplerAdvanced,
    "HiDreamImg2Img": HiDreamImg2Img,
    "HiDreamResolutionSelect": HiDreamResolutionSelect  # Added new node class
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HiDreamSampler": "HiDream Sampler",
    "HiDreamSamplerAdvanced": "HiDream Sampler (Advanced)",
    "HiDreamImg2Img": "HiDream Image to Image",
    "HiDreamResolutionSelect": "HiDream Resolution Select"  # Added display name for new node
}

WEB_DIRECTORY = "./web" # Assuming you have this directory for web-related files, if any
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
