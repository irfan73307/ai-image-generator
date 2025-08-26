import torch
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import gradio as gr

MODEL_ID = "runwayml/stable-diffusion-v1-5"

# Try to use GPU if available, else fallback to CPU
if torch.cuda.is_available():
    device = "cuda"
    torch_dtype = torch.float16
else:
    device = "cpu"
    torch_dtype = torch.float32

try:
    # Load pipeline with optimizations
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype,
        safety_checker=None,
    )
    
    # Use a faster scheduler
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    # Enable memory optimizations (important for speed & VRAM saving)
    if device == "cuda":
        pipe.enable_attention_slicing()  
        pipe.enable_xformers_memory_efficient_attention()  # requires xformers installed
        pipe.enable_model_cpu_offload()

    pipe = pipe.to(device)
except Exception as e:
    pipe = None
    error_message = str(e)

def generate_image(prompt):
    if pipe is None:
        return None, f"Model failed to load: {error_message}"
    try:
        # Reduce inference steps (default=50 → faster = 20)
        result = pipe(prompt, num_inference_steps=20)
        image = result.images[0]
        return image, None
    except Exception as e:
        return None, f"Error: {str(e)}"

def gradio_interface(prompt):
    image, error = generate_image(prompt)
    if error:
        return None, error
    return image, ""

with gr.Blocks() as demo:
    gr.Markdown("# 🚀 Fast AI Image Generator (Stable Diffusion)")
    with gr.Row():
        prompt = gr.Textbox(label="Enter your prompt", placeholder="A futuristic city at sunset")
        generate_btn = gr.Button("Generate Image")
    image_output = gr.Image(label="Generated Image")
    error_output = gr.Textbox(label="Error Message", interactive=False)

    generate_btn.click(fn=gradio_interface, inputs=prompt, outputs=[image_output, error_output])

demo.launch(share=True)
