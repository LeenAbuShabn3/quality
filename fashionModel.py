import os
import base64
import json
import re
import pickle
from PIL import Image
from io import BytesIO
from groq import Groq
from tqdm import tqdm
from pydantic import BaseModel
from typing import List


#Class holding schema
class FashionBaseModel(BaseModel):
    clothing_category: str
    color: str
    style: str
    suitable_for_weather: str
    description: str

#Class with a list of objects
class FashionModel(BaseModel):
    clothes: List[FashionBaseModel]

#Initialize the GROQ client with API key
client = Groq(api_key='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')

#Path to the folder containing images
image_folder = r"E:\VSCode\FashionModelProject\ClothingAttributeDataset\images"

#Path to the folder where features will be saved
output_folder = r"E:\VSCode\FashionModelProject\processedData"

#Create the output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Directory '{output_folder}' created successfully.")
else:
    print(f"Directory '{output_folder}' already exists.")

#Prepare a list of image files
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

#Function to extract JSON object from the API response
def extract_json_from_response(response):
    """Extract the JSON object from the API response."""
    try:
        #Use regex to extract the JSON-like structure
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
    except json.JSONDecodeError as e:
        print(f"JSON decoding failed: {e}")
    return None

#Function to save data to a .pkl file
def save_to_pkl(data, file_path):
    """Save the given data to a .pkl file."""
    try:
        with open(file_path, 'wb') as pkl_file:
            pickle.dump(data, pkl_file)
        print(f"Data saved to '{file_path}'.")
    except Exception as e:
        print(f"Failed to save to .pkl: {e}")

#Dictionary to store all features for a single .pkl file
all_features = {}

#Use tqdm to create a progress bar, buffering the image into ram and chaning it to base64(string)
for filename in tqdm(image_files, desc="Processing Images"):
    image_path = os.path.join(image_folder, filename)
    image = Image.open(image_path)
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

    #Prompt request with schema
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"""You are an expert in fashion and clothes, you feature extract clothes within images. Return JSON output only if it validates against the provided schema. Do not generate schemas, or anything other than a JSON array of clothing features. The JSON object must use the schema: {json.dumps(FashionModel.model_json_schema(), indent=2)}"""},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        }
    ]

    #Determines the response. Example: temp is creativity, stream: determines whether the response is delivered in chunks or all at once.
    try:
        completion = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=messages,
            temperature=1,
            max_tokens=1024,
            stream=False,
            stop=None,
            response_format={"type": "json_object"}
        )

        #Extract raw response
        response = completion.choices[0].message.content
        print(f"Raw API response for {filename}: {response}")

        #Extract and validate JSON
        features = extract_json_from_response(response)
        if not features:
            print(f"Could not extract valid JSON for {filename}. Skipping...")
            continue

        print(f"Formatted features for image {filename}: {features}")

        #Save to JSON file
        json_output_file = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.json")
        with open(json_output_file, 'w') as output_file:
            json.dump(features, output_file, indent=4)
        print(f"Features saved to '{json_output_file}'.")

        #Save to PKL file
        all_features[filename] = features

    except Exception as e:
        print(f"An error occurred while processing image {filename}: {e}")

#Save all features to a single .pkl file
pkl_output_file = os.path.join(output_folder, "all_features.pkl")
save_to_pkl(all_features, pkl_output_file)
