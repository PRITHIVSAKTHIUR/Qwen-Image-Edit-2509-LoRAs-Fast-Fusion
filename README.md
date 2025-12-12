# **Qwen-Image-Edit-2509-LoRAs-Fast-Fusion**

**Qwen-Image-Edit-2509-LoRAs-Fast-Fusion** is a fast, interactive web application built with Gradio that enables advanced image editing using the [Qwen/Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) model from Alibaba's Qwen team. It leverages specialized LoRA adapters for efficient, low-step inference (as few as 4 steps) on tasks like texture application, object fusion, and cloth design transfer. Upload a base image and a reference image, provide a prompt, and select an editing style to generate edited results with Lightning-fast acceleration. This app showcases Qwen-Image-Edit's capabilities in inpainting, outpainting, and style transfer, optimized with Flash Attention 3 (if supported) and a custom orange-red themed interface for a vibrant user experience.

<img width="1525" height="768" alt="mlWbJlO2MGos1B8fnSueK" src="https://github.com/user-attachments/assets/c01d1081-ca1a-4569-8876-ea0cc9567f00" />
<img width="1523" height="771" alt="4a14dQH_wFE7XxeRg_HXX" src="https://github.com/user-attachments/assets/9163b256-4375-425c-9a00-8e41402b6c1d" />

### Key Features
- **Specialized LoRA Adapters**: Choose from "Texture Edit" (apply textures), "Fuse-Objects" (blend elements), or "Cloth-Design-Fuse" (transfer designs to clothing).
- **Lightning Inference**: Uses Qwen-Image-Lightning LoRA for ultra-fast generation (4 steps default).
- **Prompt-Guided Edits**: Natural language prompts guide the fusion (e.g., "Apply wood texture to the mug").
- **Advanced Controls**: Adjustable guidance scale, steps, seed randomization, and auto-resizing for optimal resolution (multiples of 16).
- **GPU Optimization**: Auto-detects CUDA; supports bfloat16 for efficiency.
- **Gradio Interface**: Dual-image upload with annotated outputs, examples, and progress tracking.
- **Custom Theme**: Orange-red gradients and shadows for a dynamic, fiery aesthetic.

## Requirements

To run this app locally, install the dependencies. Note that it requires recent versions of Hugging Face libraries from Git for compatibility with Qwen-Image-Edit.

```bash
pip install torch torchvision numpy gradio spaces sentencepiece huggingface_hub supervision kernels
pip install git+https://github.com/huggingface/accelerate.git
pip install git+https://github.com/huggingface/diffusers.git
pip install git+https://github.com/huggingface/peft.git
pip install git+https://github.com/QwenLM/QwenImage.git  # For qwenimage custom components
```

### Hardware Recommendations
- **GPU**: NVIDIA GPU with CUDA 11.8+ and at least 8GB VRAM (e.g., RTX 30-series or A100) for fast inference. Falls back to CPU (slower).
- **RAM**: 16GB+ recommended.
- **Storage**: ~10GB for model weights and LoRAs (downloaded on first run).

## Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/PRITHIVSAKTHIUR/Qwen-Image-Edit-2509-LoRAs-Fast-Fusion.git
   cd Qwen-Image-Edit-2509-LoRAs-Fast-Fusion
   ```

2. **Install Dependencies**:
   Run the pip commands from the [Requirements](#requirements) section above.

3. **Prepare Examples** (Optional):
   Ensure the `examples/` folder contains sample images like `Cloth1.jpg`, `Design1.png`, `Cup1.png`, `Wood1.png`, etc. Download them if needed or replace with your own.

4. **Run Locally**:
   ```bash
   python app.py
   ```
   - The app launches a local web server (typically at `http://127.0.0.1:7860`).
   - Open the URL in your browser to start editing.

### Deployment on Hugging Face Spaces
- Optimized for HF Spaces with `@spaces.GPU` (30-second duration limit per inference).
- Fork the repo, create a new Space on [Hugging Face](https://huggingface.co/spaces), and link it. Enable GPU in Space settings.
- Models and LoRAs auto-download on startup.

## Hardware & Environment Requirements

| Component | Requirement | Details from Log |
| :--- | :--- | :--- |
| **GPU Model** | NVIDIA Enterprise (H200 / H100 / A100) | High VRAM requirement (MIG or full card) |
| **VRAM** | **80 GB Recommended** (Min 70 GB) | Peak tensor packing reached **60.8 GB** |
| **System RAM** | 128 GB+ | Required for safe model offloading/loading |
| **Disk Space** | ~60 GB+ | Transformer (~21GB) + Text Encoders (~17GB) + LoRAs |
| **Python** | v3.13 | Detected `cpython-313` |
| **PyTorch** | v2.9.1 | `2.9.1+cu128` |
| **CUDA** | v12.8 | `torch29-cxx11-cu128` |

## Usage

1. **Upload Images**: Select a "Base Image" (e.g., a mug) and a "Reference Image" (e.g., wood texture).
2. **Enter Prompt**: Describe the edit (e.g., "Apply wood texture to the mug"). Defaults provided for each style.
3. **Choose Editing Style**: Dropdown for "Texture Edit", "Fuse-Objects", or "Cloth-Design-Fuse".
4. **Tune Settings** (Optional): Adjust seed, guidance scale (default: 1.0), and steps (default: 4) in the Advanced accordion.
5. **Click Edit Image**: Generates the output with progress bar. Randomize seed for variations.
6. **Test Examples**: Use built-in examples for quick demos like shirt design transfer.

### Input/Output Format
- **Inputs**: Two PIL Images (base + reference), Text Prompt, LoRA Style, Seed/Params.
- **Output**: Edited PIL Image (PNG format) + Updated Seed.

### Example Workflows
- **Texture Edit**: Base: Mug photo; Reference: Wood pattern; Prompt: "Apply wood texture to mug."
- **Cloth-Design-Fuse**: Base: Person in shirt; Reference: Graphic design; Prompt: "Put this design on their shirt."
- **Fuse-Objects**: Base: Cat photo; Reference: Glasses; Prompt: "A cat wearing glasses."

## Code Structure

- **`app.py`**: Main Gradio app with pipeline loading, inference function, and UI setup.
- **`examples/`**: Sample image pairs for testing various edits.
- **Custom Theme**: `OrangeRedTheme` class for bold, gradient-based styling.
- **Key Components**:
  - `QwenImageEditPlusPipeline`: Core editing pipeline with LoRA fusion.
  - `infer(...)`: Handles image preprocessing, adapter selection, and generation.
  - LoRAs: Lightning (speed), Texture, Fusion, Shirt Design—loaded and fused dynamically.

### Model Details
- Base: [Qwen/Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509)
- Transformer: [linoyts/Qwen-Image-Edit-Rapid-AIO](https://huggingface.co/linoyts/Qwen-Image-Edit-Rapid-AIO) (subfolder: transformer)
- LoRAs: From [lightx2v](https://huggingface.co/lightx2v), [tarn59](https://huggingface.co/tarn59), [ostris](https://huggingface.co/ostris)

## Troubleshooting

- **Model/LoRA Loading Error**: Verify internet access and Git installs. Clear cache with `huggingface-cli delete-cache`.
- **CUDA/FA3 Issues**: If Flash Attention 3 fails, it uses default attention. Ensure CUDA toolkit matches PyTorch version.
- **Out of Memory**: Reduce steps or image size; use `torch_dtype=torch.float16` for lower VRAM.
- **No Output/Blurry Results**: Increase guidance scale (try 2.0+) or steps (8+). Ensure prompts are descriptive.
- **Gradio Errors**: Update Gradio (`pip install --upgrade gradio`) or check for port conflicts (e.g., `--port 7861`).

## Contributing

Contributions welcome! Fork, branch, PR.

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-orange?logo=github)](https://github.com/PRITHIVSAKTHIUR/Qwen-Image-Edit-2509-LoRAs-Fast-Fusion.git)

## Acknowledgments

- [Qwen Team](https://huggingface.co/Qwen): For the Qwen-Image-Edit model.
- [Gradio](https://www.gradio.app/): Intuitive UI framework.
- [Hugging Face Diffusers/PEFT](https://huggingface.co/docs/diffusers): For pipeline and adapter handling.
- LoRA Creators: [lightx2v](https://huggingface.co/lightx2v), [tarn59](https://huggingface.co/tarn59), [ostris](https://huggingface.co/ostris).

For bugs or ideas, open a GitHub issue. Edit away! 
