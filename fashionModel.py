import os
import base64
import json
import re
import pickle
import logging
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from pydantic import BaseModel
from typing import List
from groq import Groq

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define constants
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')
IMAGE_FOLDER = r"E:\VSCode\FashionModelProject\ClothingAttributeDataset\images"
OUTPUT_FOLDER = r"E:\VSCode\FashionModelProject\processedData"
GROQ_API_KEY = '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'

# Initialize the GROQ client
client = Groq(api_key=GROQ_API_KEY)

# Schema classes
class FashionBaseModel(BaseModel):
    clothing_category: str
    color: str
    style: str
    suitable_for_weather: str
    description: str

class FashionModel(BaseModel):
    clothes: List[FashionBaseModel]

# Ensure output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
logging.info(f"Output directory '{OUTPUT_FOLDER}' is ready.")

# Helper function: Get image files
def get_image_files(folder: str, extensions: tuple) -> List[str]:
    return [f for f in os.listdir(folder) if f.endswith(extensions)]

# Helper function: Extract JSON from response
def extract_json_from_response(response: str) -> dict:
    try:
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
    except json.JSONDecodeError as e:
        logging.error(f"JSON decoding failed: {e}")
    return None

# Helper function: Save data to file
def save_to_file(data, file_path: str, mode='json'):
    try:
        if mode == 'json':
            with open(file_path, 'w') as output_file:
                json.dump(data, output_file, indent=4)
        elif mode == 'pkl':
            with open(file_path, 'wb') as pkl_file:
                pickle.dump(data, pkl_file)
        logging.info(f"Data saved to '{file_path}'.")
    except Exception as e:
        logging.error(f"Failed to save data to '{file_path}': {e}")

# Main process
def process_images():
    image_files = get_image_files(IMAGE_FOLDER, IMAGE_EXTENSIONS)
    if not image_files:
        logging.error("No image files found. Exiting...")
        return

    all_features = {}

    for filename in tqdm(image_files, desc="Processing Images"):
        try:
            image_path = os.path.join(IMAGE_FOLDER, filename)
            with Image.open(image_path) as image:
                buffered = BytesIO()
                image.save(buffered, format="JPEG")
                base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

            # Create prompt messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"""You are an expert in fashion and clothes. Extract clothing features from the image. 
                            Return JSON output only if it validates against the provided schema. 
                            The JSON object must use the schema: {json.dumps(FashionModel.model_json_schema(), indent=2)}."""},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ]

            # API call
            completion = client.chat.completions.create(
                model="llama-3.2-11b-vision-preview",
                messages=messages,
                temperature=1,
                max_tokens=1024,
                stream=False,
                response_format={"type": "json_object"}
            )

            # Process response
            response = completion.choices[0].message.content
            logging.info(f"Raw API response for {filename}: {response}")
            features = extract_json_from_response(response)
            if not features:
                logging.warning(f"No valid features extracted for {filename}. Skipping...")
                continue

            # Save individual JSON
            json_output_file = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(filename)[0]}.json")
            save_to_file(features, json_output_file, mode='json')

            # Aggregate features
            all_features[filename] = features

        except Exception as e:
            logging.error(f"Error processing image {filename}: {e}")

    # Save all features to a single file
    pkl_output_file = os.path.join(OUTPUT_FOLDER, "all_features.pkl")
    save_to_file(all_features, pkl_output_file, mode='pkl')

# Entry point
if __name__ == "__main__":
    process_images()
