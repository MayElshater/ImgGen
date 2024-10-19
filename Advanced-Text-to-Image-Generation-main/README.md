
# Advanced Text-to-Image Generation with DeepInfra API & MLflow

This project is a **Streamlit** web application that generates images from text descriptions using the **DeepInfra API**. All image generation parameters and artifacts are tracked and logged using **MLflow**.

## Features

- Enter a text prompt to generate stunning images.
- Customize image parameters such as:
  - **Width** and **Height**
  - **Number of inference steps** (controls detail and quality)
  - **Guidance scale** (controls adherence to the prompt)
  - **Number of images** to generate
- Images are saved locally and logged in MLflow for experiment tracking.

## Model Used

The application uses the **StabilityAI SDXL Turbo** model provided through the **DeepInfra API**. This model is designed for fast and high-quality text-to-image generation. It leverages advanced diffusion models to generate detailed and accurate images based on the provided text prompt. The SDXL Turbo model is optimized for performance and speed, making it suitable for generating multiple images in a short time.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/text-to-image-generator.git
   cd text-to-image-generator
   ```

2. **Install Dependencies**:
   Install the required Python libraries using `requirements.txt`.
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   Launch the Streamlit app.
   ```bash
   streamlit run app.py
   ```

4. **MLflow UI**:
   Start the MLflow UI to view your experiment logs.
   ```bash
   mlflow ui
   ```
   Navigate to `http://localhost:5000` to view logs.

## Project Structure

```
/text-to-image-generator
│
├── /outputs/              # Generated images saved here
│   └── /generated_images/
│
├── /utils/                # Utility functions
│   └── image_utils.py     # Image generation logic
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## API Configuration

The application uses **DeepInfra API** for image generation. You need a valid **DeepInfra API token** to use the service. Update the API token in the `image_utils.py` file:
```python
DEEPINFRA_TOKEN = "your-api-token-here"
```

## Future Improvements

- Add error handling for specific API failures.
- Add support for more image generation models.
- Allow the user to download generated images directly from the interface.
