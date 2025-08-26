# Free AI Image Generator (Stable Diffusion)

This is a simple, beginner-friendly AI image generator using Hugging Face Diffusers (Stable Diffusion) and Gradio. No paid services or OpenAI API required.

## Features
- Uses free Stable Diffusion model from Hugging Face (default: `runwayml/stable-diffusion-v1-5`)
- Works on CPU or GPU (automatically detects and falls back to CPU)
- Simple Gradio web interface

## How to Run

1. **Install dependencies**
   
   Open a terminal in this folder and run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app**
   
   ```bash
   python app.py
   ```

3. **Open Gradio link**
   
   After running, Gradio will show a local link (e.g., `http://127.0.0.1:7860`). Open it in your browser.

4. **Generate images**
   
   - Type a prompt (e.g., "A cat astronaut in space")
   - Click "Generate Image"
   - View the generated image

## Notes
- This project uses only free Hugging Face Stable Diffusion models.
- Optimized for free usage (Colab CPU/GPU or local PC).
- For best results, use a machine with a GPU. On CPU, generation will be slower.

## License
MIT
