I've decided to break this fork off into it's own repository since I am making many library changes that could be detrimental to a larger project if pulled upstream. My thanks to https://github.com/lum3on/comfyui_HiDream-Sampler for their initial work on the project!

## Unofficial img2img support! ##

🖼️🤖🖼️

**New capability Unlocked, image2image for both UniPC and flash_flow_euler**

- Fully supported by NF4-Fast/Dev/Full 
- Added new library pipeline 'hideream_image_to_image'
-  - ComfyUI-HiDream-Sampler/hi_diffusers/pipelines/hidream_image/pipeline_hidream_image_to_image.py
- **Added new HiDream node 'HiDream Image to Image'**
- - images are automatically resized and inner cropped to HiDream supported aspect ratio/size.
- - Uses native FFE or UniPC noising, scaling and scheduling for noising the images

![image](https://github.com/user-attachments/assets/e6286e3a-9e3f-491f-9508-f05f498ba210)



# Update Notes 4/10/25 #
**HiDream Sampler node**
- Reverted the simple node to using aspect ratio presets for resolution due to issues with the model throwing black box errors outside of it's standard accepted resolutions.
- added under-hood logic for setting all prompt encoders to scale-weight 1.0 and receiving the same prompt input
- added default system prompt for LLM encoder

**HiDream Sampler (Advanced) node**
- added back the aspect_ratio selector box with the usable presets due to reasoning above
- - advanced mode aspect_ratio box allows for setting custom size inputs
- added a new input 'square_resolution' that defaults to 1024 and allows you to create larger (or smaller) square images. 
- - it starts to get pretty wonky above 3MP, but I was getting good results at 1280x1280 and 1536x1536.
- height/width are now custom_height and custom_width, both default to 1024 and are only enabled if "Custom" is selected on the aspect_ratio drop down.
- Exposed llm_system_prompt with a default 'make it nicer' system prompt
- exposed weights for all encoders, scalable from 0.0 to 5.0
- - Setting scale weight to 0.0 effectively 'turns off' the embeddings for that encoder
- added new handling for blank inputs on the individual encoder prompt channels to try to prevent the LLM from creating hallucinatory noise when it doesn't receive an input prompt.  

# HiDream Library Changes # 
- exposed system prompt of LLM encoder and exposed on library call as llm_system_prompt
- added individual encoder scaling, defaulting to 1.0 multiplier
- exposed multiplier scale weights on library call

-------
# Previous Update #
**Added Many improvements!**
- Added "use_uncensored_llm" option - this currently loads a different llama3.1-8b model that is just as censored as the first model. I will work on setting up a proper LLM replacement here, but may take a few days to get working properly. Until then this is just a "try a different LLM model" button. ** THIS IS STILL A WIP, DON'T @ ME **
- Renamed the existing node to "HiDream Sampler"
- Added new Node "HiDream Sampler (Advanced)"
- Exposed negative prompt for both (Note - only works on Full and Full-nf4 models)
- changed resolution to discrete integers to allow for free-range resolution setting instead of predefined values
-  - Modified library to remove cap on output resolution/scale. Watch out, you can OOM yourself if you go too big. Also, I haven't tested really large images yet, the results will likely be really funky.
- modified the HiDream library to accept different max lengths per encoder model, as the previous 128 limit being enforced for all 4 encoder models was ridiculously low and stupid to put in front of an LLM or t5xxl.
- - For the 'simple' sampler node, I set the defaults to:  CLIP-L: 77, OpenCLIP: 150, T5: 256, Llama: 256 
- Advanced sampler node adds discrete values for max input lengths for each encoder, as well as a prompt box for each encoder.  - - Default behavior is the primary prompt is used for all 4 encoder inputs unless overridden by an individual encoder input prompt. 
- - - you can 'blank out' any encoder you don't want to use by simply leaving the primary prompt blank, then inserting a prompt for only the encoder(s) you want to use, or use different prompts for different encoders.
- - I think there is something wonky going on with the LLM encoder, it seems to have a lot of output 'noise' even when the input prompt is zeroed out, which I suspect is the LLM hallucinating output, will investigate but until then, the LLM encoder is way stronger than all of the other encoders, even when you don't feed it a prompt.

I will continue to add more improvements as I go, this has been fascinating to explore this model. I think there is still a lot of room for improvement (and optimizing).

Forked from original https://github.com/lum3on/comfyui_HiDream-Sampler

- Added NF4 (Full/Dev/Fast) download and load support
- Added better memory handling
- Added more informative CLI output for TQDM

Many thanks to the folks who created this and set it up for Comfy, I just spent a few hours adding better support for consumer GPUs.

- Full/Dev/Fast requires roughly 27GB VRAM
- NF4 requires roughly 15GB VRAM

![workflow (1)](https://github.com/user-attachments/assets/c408e477-2714-434d-acc9-6d49d0476291)


# HiDreamSampler for ComfyUI

A custom ComfyUI node for generating images using the HiDream AI model.

## Features
- Supports `full`, `dev`, and `fast` model types.
- Supports `nf4-full`, `nf4-dev`, and `nf4-fast` model types.
- Configurable resolution and inference steps.
- Uses 4-bit quantization for lower memory usage.

## Installation
Recommended to install Flash Attention, but not required. Falls back to SDPA and eager attention

1. Clone this repository into your `ComfyUI/custom_nodes/` directory:
   ```bash
   git clone https://github.com/SanDiegoDude/ComfyUI-HiDream-Sampler.git
2. Install requirements
    ```bash
    pip install -r requirements.txt

    #optional if you want to use NF4 models
    pip install optimum accelerate auto-gptq

3. Restart ComfyUI.

## Usage
ComfyUI Nodes
HiDream Sampler provides three nodes for different workflows:

### 1. HiDream Sampler (Basic) 
**Simple text-to-image generation with a single prompt.**

Inputs:
   model_type: Choose model variant (fast, dev, full or their NF4 counterparts)
   prompt: Text description of your desired image
   negative_prompt: Elements to avoid in the image
   aspect_ratio: Select from preset aspect ratios (1:1, 16:9, etc.)
   seed: Random seed for reproducible results
   scheduler: Sampling method (Default, UniPC, Euler, etc.)
   override_steps: Custom number of sampling steps (-1 to use model default)
   override_cfg: Custom guidance scale (-1.0 to use model default)
   use_uncensored_llm: Toggle for less filtered language model responses

### 2. HiDream Sampler (Advanced) 
**Fine-grained control over every encoder and generation parameter.**

Inputs:
   All inputs from basic sampler, plus:
   Encoder-specific prompts:
   clip_l_prompt: Text for CLIP-L encoder
   openclip_prompt: Text for OpenCLIP encoder
   t5_prompt: Text for T5 encoder
   llama_prompt: Text for Llama encoder
   llm_system_prompt: System instruction for Llama LLM
   Encoder weights: Control influence of each encoder (clip_l_weight, openclip_weight, t5_weight, llama_weight)
   Resolution controls: square_resolution or custom_width/height
   Sequence length controls: max_length parameters for each encoder

### 3. HiDream Image to Image
**Transform existing images using HiDream models.**

Inputs:
   All inputs from basic sampler, plus:
   image: Input image to transform
   denoising_strength: How much to change from input (0.0=no change, 1.0=complete change)
   Encoder weights: Available to control influence (like in Advanced sampler)
   Example Workflow
   Add the appropriate HiDream node to your workflow
   Configure inputs based on your needs:
   For text-to-image: Enter a detailed prompt
   For image-to-image: Connect an image and set denoising_strength (recommended: 0.7-0.8)
   Connect the output to a PreviewImage or SaveImage node
   Execute and enjoy your results!

### Tips
   NF4 models run better on lower VRAM but require specific dependencies
   Best results come from detailed, descriptive prompts
   For Windows users: Requires PyTorch 2.1.0+ for NF4 models
   Experiment with encoder weights in Advanced mode to emphasize different aspects

## Requirements
- ComfyUI
- CUDA-enabled GPU (for model inference)

