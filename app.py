import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import matplotlib.pyplot as plt
import streamlit as st

# Clear CUDA cache (optional)
torch.cuda.empty_cache()

# Load the model
model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

# Streamlit UI elements
st.title("Stable Diffusion Image Generator")
prompt = st.text_input("Enter a prompt", "a house in front of the ocean")

# If the user inputs a prompt
if prompt:
    # Generate the image
    with st.spinner("Generating image..."):
        image = pipe(prompt, width=1000, height=1000).images[0]
    
    # Display the generated image
    st.image(image, caption="Generated Image", use_column_width=True)
