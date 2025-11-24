import os
import gradio as gr
import numpy as np
import spaces
import torch
import random
from PIL import Image
from typing import Iterable
from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

colors.orange_red = colors.Color(
    name="orange_red",
    c50="#FFF0E5",
    c100="#FFE0CC",
    c200="#FFC299",
    c300="#FFA366",
    c400="#FF8533",
    c500="#FF4500",
    c600="#E63E00",
    c700="#CC3700",
    c800="#B33000",
    c900="#992900",
    c950="#802200",
)

class OrangeRedTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.orange_red,
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_secondary_text_color="black",
            button_secondary_text_color_hover="white",
            button_secondary_background_fill="linear-gradient(90deg, *primary_300, *primary_300)",
            button_secondary_background_fill_hover="linear-gradient(90deg, *primary_400, *primary_400)",
            button_secondary_background_fill_dark="linear-gradient(90deg, *primary_500, *primary_600)",
            button_secondary_background_fill_hover_dark="linear-gradient(90deg, *primary_500, *primary_500)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )

orange_red_theme = OrangeRedTheme()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16

from diffusers import FlowMatchEulerDiscreteScheduler
from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
from qwenimage.qwen_fa3_processor import QwenDoubleStreamAttnProcessorFA3

print("Loading Qwen Image Edit Pipeline...")
pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509",
    transformer=QwenImageTransformer2DModel.from_pretrained(
        "linoyts/Qwen-Image-Edit-Rapid-AIO",
        subfolder='transformer',
        torch_dtype=dtype,
        device_map='cuda'
    ),
    torch_dtype=dtype
).to(device)

print("Loading and Fusing Lightning LoRA...")
pipe.load_lora_weights("lightx2v/Qwen-Image-Lightning",
                       weight_name="Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors",
                       adapter_name="lightning")
pipe.fuse_lora(adapter_names=["lightning"], lora_scale=1.0)

print("Loading Task Adapters...")

pipe.load_lora_weights("tarn59/apply_texture_qwen_image_edit_2509", 
                       weight_name="apply_texture_v2_qwen_image_edit_2509.safetensors", 
                       adapter_name="texture")

pipe.load_lora_weights("ostris/qwen_image_edit_inpainting", 
                       weight_name="qwen_image_edit_inpainting.safetensors", 
                       adapter_name="fusion")

pipe.load_lora_weights("ostris/qwen_image_edit_2509_shirt_design", 
                       weight_name="qwen_image_edit_2509_shirt_design.safetensors", 
                       adapter_name="shirt_design")

pipe.load_lora_weights("dx8152/Qwen-Image-Edit-2509-Fusion", 
                       weight_name="溶图.safetensors", 
                       adapter_name="fusion-x")

try:
    pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())
    print("Flash Attention 3 Processor set successfully.")
except Exception as e:
    print(f"Could not set FA3 processor (likely hardware mismatch): {e}. using default attention.")

MAX_SEED = np.iinfo(np.int32).max

def update_dimensions_on_upload(image):
    if image is None:
        return 1024, 1024
    
    original_width, original_height = image.size
    
    if original_width > original_height:
        new_width = 1024
        aspect_ratio = original_height / original_width
        new_height = int(new_width * aspect_ratio)
    else:
        new_height = 1024
        aspect_ratio = original_width / original_height
        new_width = int(new_height * aspect_ratio)
        
    # Ensure dimensions are multiples of 16
    new_width = (new_width // 16) * 16
    new_height = (new_height // 16) * 16
    
    return new_width, new_height

@spaces.GPU(duration=30)
def infer(
    image_1,
    image_2,
    prompt,
    lora_adapter,
    seed,
    randomize_seed,
    guidance_scale,
    steps,
    progress=gr.Progress(track_tqdm=True)
):
    if image_1 is None or image_2 is None:
        raise gr.Error("Please upload both images for Fusion/Texture/FaceSwap tasks.")
    
    if not prompt:
        if lora_adapter == "Cloth-Design-Fuse":
            prompt = "Put this design on their shirt."
        elif lora_adapter == "Texture Edit":
            prompt = "Apply texture to object."
        elif lora_adapter == "Fuse-Objects":
            prompt = "Fuse object into background."
        elif lora_adapter == "Super-Fusion":
            prompt = "Blend the product into the background, correct its perspective and lighting, and make it naturally integrated with the scene."
            
    adapters_map = {
        "Texture Edit": "texture",
        "Fuse-Objects": "fusion",
        "Cloth-Design-Fuse": "shirt_design",
        "Super-Fusion": "fusion-x",
    }
    
    active_adapter = adapters_map.get(lora_adapter)
    
    if active_adapter:
        pipe.set_adapters([active_adapter], adapter_weights=[1.0])
    else:
        pipe.set_adapters([], adapter_weights=[])

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    generator = torch.Generator(device=device).manual_seed(seed)
    negative_prompt = "worst quality, low quality, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry"

    img1_pil = image_1.convert("RGB")
    img2_pil = image_2.convert("RGB")
    
    width, height = update_dimensions_on_upload(img1_pil)

    result = pipe(
        image=[img1_pil, img2_pil],
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        generator=generator,
        true_cfg_scale=guidance_scale,
    ).images[0]

    return result, seed

@spaces.GPU(duration=30)
def infer_example(image_1, image_2, prompt, lora_adapter):
    if image_1 is None or image_2 is None:
        return None, 0
    result, seed = infer(
        image_1.convert("RGB"), 
        image_2.convert("RGB"), 
        prompt, 
        lora_adapter, 
        0,
        True,
        1.0,
        4
    )
    return result, seed

css="""
#col-container {
    margin: 0 auto;
    max-width: 1100px;
}
#main-title h1 {font-size: 2.1em !important;}
"""

with gr.Blocks() as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("# **Qwen-Image-Edit-2509-LoRAs-Fast-Fusion**", elem_id="main-title")
        gr.Markdown("Perform diverse image edits using specialized [LoRA](https://huggingface.co/models?other=base_model:adapter:Qwen/Qwen-Image-Edit-2509) adapters for the [Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) model.")
        with gr.Row(equal_height=True):
          
            with gr.Column(scale=1):
                with gr.Row():
                    image_1 = gr.Image(label="Base Image", type="pil", height=290)
                    image_2 = gr.Image(label="Reference Image", type="pil", height=290)

                prompt = gr.Text(
                    label="Edit Prompt",
                    show_label=True,
                    placeholder="e.g., Apply wood texture to the mug...",
                )
                
                run_button = gr.Button("Edit Image", variant="primary")

                with gr.Accordion("Advanced Settings", open=False, visible=False):
                    seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
                    randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
                    guidance_scale = gr.Slider(label="True Guidance Scale", minimum=1.0, maximum=10.0, step=0.1, value=1.0)
                    steps = gr.Slider(label="Inference Steps", minimum=1, maximum=50, step=1, value=4)

            with gr.Column(scale=1):
                output_image = gr.Image(label="Output Image", interactive=False, format="png", height=350)
                
                with gr.Row():
                    lora_adapter = gr.Dropdown(
                        label="Choose Editing Style",
                        choices=["Texture Edit", "Cloth-Design-Fuse", "Fuse-Objects", "Super-Fusion"],
                        value="Texture Edit",
                    )        
 
        gr.Examples(
            examples=[
                ["examples/Cloth2.jpg", "examples/Design2.png", "Put this design on their shirt.", "Cloth-Design-Fuse"],
                ["examples/Cup1.png", "examples/Wood1.png", "Apply wood texture to mug.", "Texture Edit"],
                ["examples/Cloth1.jpg", "examples/Design1.png", "Put this design on their shirt.", "Cloth-Design-Fuse"],
                ["examples/F3.jpg", "examples/F4.jpg", "Replace her glasses with the new glasses from image 1.", "Super-Fusion"],
                ["examples/F1.jpg", "examples/F2.jpg", "Put the small bottle on the table.", "Super-Fusion"],
                ["examples/Mug1.jpg", "examples/Texture1.jpg", "Apply the design from image 2 to the mug.", "Texture Edit"],
                ["examples/Cat1.jpg", "examples/Glass1.webp", "A cat wearing glasses in image 2.", "Fuse-Objects"],

            ],
            inputs=[image_1, image_2, prompt, lora_adapter],
            outputs=[output_image, seed],
            fn=infer_example,
            cache_examples=False,
            label="Examples"
        )

    run_button.click(
        fn=infer,
        inputs=[image_1, image_2, prompt, lora_adapter, seed, randomize_seed, guidance_scale, steps],
        outputs=[output_image, seed]
    )
    
if __name__ == "__main__":
    demo.queue(max_size=30).launch(css=css, theme=orange_red_theme, mcp_server=True, ssr_mode=False, show_error=True)
