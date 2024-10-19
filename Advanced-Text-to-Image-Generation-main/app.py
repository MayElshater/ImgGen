import mlflow
import streamlit as st
from utils.image_utils import generate_images  # Import the image generation logic


# Set MLflow experiment
mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("Ad-Text-to-Image Generation")

# Streamlit App
st.title("🌄 Advanced Text-to-Image Generation with MLflow")
st.markdown("""
    Generate stunning images from text using **DeepInfra API**. 
    Customize the image size, inference steps, and more. All your experiments will be tracked using **MLflow**.
""")

st.write("---")

# User Input for the text description
st.subheader("🔤 Enter Image Description")
text_description = st.text_input(
    "Text Description", 
    value="A beautiful landscape with mountains", 
    help="Describe the scene you want to generate."
)

# Organizing Input Fields in Columns
st.subheader("🛠 Customize Image Parameters")

with st.expander("Advanced Settings", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        width = st.number_input(
            "Image Width", 
            min_value=256, 
            max_value=2048, 
            value=1024, 
            step=64, 
            key="width", 
            help="Specify the width of the image in pixels."
        )
        height = st.number_input(
            "Image Height", 
            min_value=256, 
            max_value=2048, 
            value=1024, 
            step=64, 
            key="height", 
            help="Specify the height of the image in pixels."
        )
        num_images = st.number_input(
            "Number of Images", 
            min_value=1, 
            max_value=10, 
            value=1, 
            key="num_images", 
            help="Specify how many images to generate."
        )
    
    with col2:
        num_inference_steps = st.slider(
            "Number of Inference Steps", 
            min_value=1, 
            max_value=50, 
            value=5, 
            key="num_inference_steps", 
            help="Controls the quality and detail of the image. Higher values result in more detailed images."
        )
        guidance_scale = st.slider(
            "Guidance Scale", 
            min_value=1.0, 
            max_value=20.0, 
            value=1.0, 
            step=0.1, 
            key="guidance_scale", 
            help="Controls how much the model adheres to your prompt. Higher values result in more prompt-faithful images."
        )

# Streamlit button interaction
if st.button("🎨 Generate Image(s)"):
    with st.spinner('🔄 Generating image(s)...'):
        # Generate multiple images if required
        image_paths = generate_images(text_description, 
                                      width=width, 
                                      height=height, 
                                      steps=num_inference_steps, 
                                      guidance_scale=guidance_scale, 
                                      num_images=num_images)

        if image_paths:
            # Display all generated images
            st.subheader(f"🖼 Generated {len(image_paths)} Image(s):")
            for img_path in image_paths:
                st.image(img_path, caption=f"Generated Image", use_column_width=True)
            st.success("✅ Image(s) generation logged to MLflow successfully!")

# Footer with separator
st.write("---")
st.markdown("""
    Created with ❤️ using **Streamlit** and **MLflow**. 
    Track your experiment logs and artifacts in the MLflow UI.
""")