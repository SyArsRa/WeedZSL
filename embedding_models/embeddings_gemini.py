import json
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
genai.configure(api_key=os.getenv("GENAI_API_KEY"))

def gemini_api_call(main_crop, classes):
    print(f"--- Calling Gemini API for main crop: {main_crop} ---")
    print(f"Detected Classes: {classes}")

    prompt = f"""
You are given a list of plant classes: {classes}.  
The target crop is: "{main_crop}".  
Identify the class or classes that visually resemble the target crop (without fruit just the simple the crop) the most when viewed from the top — including any variants, growth stages, or phenotypic variations.  

Return ONLY a valid JSON list of class names (as strings).  
DO NOT return any explanation, comments, or additional text.  
DO NOT return an empty list under any circumstances.  
    """

    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt, generation_config={"temperature": 0})

        if response and response.parts:
            value = response.text.strip()
            if value.startswith("```json"):
                value = value[7:]
            if value.endswith("```"):
                value = value[:-3]
            value = value.strip()

            print(f"API Raw Response: {value}")
            class_list = json.loads(value)

            if isinstance(class_list, list) and all(isinstance(x, str) for x in class_list):
                print(f"Selected Classes: {class_list}")
                return class_list
            else:
                print("Warning: Invalid API format. Returning fallback.")
                return [main_crop] if main_crop in classes else []

        else:
            print("Warning: Empty Gemini response.")
            return [main_crop] if main_crop in classes else []

    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return [main_crop] if main_crop in classes else []

    except Exception as e:
        print(f"Gemini API error: {e}")
        return [main_crop] if main_crop in classes else []


def get_similar_classes_gemini(all_detected_classes, main_crop, threshold=0.8):
    """
    Determines which detected classes resemble the main crop.
    Uses Gemini API if the main crop is not directly in the known class list.
    """
    selected_classes = []

    try:
        selected_classes = gemini_api_call(main_crop, all_detected_classes)
        if not isinstance(selected_classes, list) or not all(isinstance(item, str) for item in selected_classes):
            print("Warning: API did not return a valid list. Defaulting to [main_crop].")
            selected_classes = [main_crop] if main_crop in all_detected_classes else []
    except Exception as e:
        print(f"Error during API call: {e}. Falling back to [main_crop].")
        selected_classes = [main_crop] if main_crop in all_detected_classes else []

    return selected_classes
