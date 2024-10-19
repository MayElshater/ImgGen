import mlflow
import requests
from PIL import Image
from io import BytesIO
import os
import base64
from datetime import datetime

# Function to generate image(s)
def generate_images(prompt, width=1024, height=1024, steps=5, guidance_scale=1.0, num_images=1):
    DEEPINFRA_TOKEN = os.getenv("DEEPINFRA_TOKEN")
    API_URL = "https://api.deepinfra.com/v1/inference/stabilityai/sdxl-turbo"
    HEADERS = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPINFRA_TOKEN}"
    }
    
    data = {
        "prompt": prompt,
        "num_images": num_images,
        "width": width,
        "height": height,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale
    }
    
    try:
        print("Attempting to start MLflow run...")
        with mlflow.start_run() as run:
            # Example usage of MLflow to track parameters
            mlflow.log_param("prompt", prompt)
            mlflow.log_param("width", width)
            mlflow.log_param("height", height)
            mlflow.log_param("num_inference_steps", steps)
            mlflow.log_param("guidance_scale", guidance_scale)
            mlflow.log_param("num_images", num_images)

            # Make the POST request
            response = requests.post(API_URL, headers=HEADERS, json=data)

            # Check for a successful response
            if response.status_code == 200:
                result = response.json()
                image_paths = []

                # Create output directory once
                os.makedirs('outputs/generated_images/', exist_ok=True)

                # Process each image
                for i, image_base64 in enumerate(result['images']):
                    image_data = base64.b64decode(image_base64.split(",")[1])
                    image = Image.open(BytesIO(image_data))

                    # Save the image
                    image_path = f"outputs/generated_images/generated_{datetime.now().strftime('%Y%m%d%H%M%S')}_{i}.png"
                    image.save(image_path)
                    image_paths.append(image_path)

                    # Log the image artifact in MLflow
                    mlflow.log_artifact(image_path, artifact_path="generated_images")
                    print(f"Image saved to {image_path}")

                return image_paths

            else:
                # Log API errors
                print(f"Error {response.status_code}: {response.text}")
                mlflow.log_metric("API_error", response.status_code)
                mlflow.log_param("API_response", response.text)  # Log response text for debugging
                return None

    except Exception as e:
        # Handle exceptions
        print(f"Error in MLflow Logging or API call: {e}")
        mlflow.log_param("Error", str(e))
        return None
