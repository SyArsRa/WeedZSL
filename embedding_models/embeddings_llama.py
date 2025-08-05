import json
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from huggingface_hub import login
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


# Authenticate with Hugging Face
token = HF_TOKEN  # Replace with your actual token
login(token)

# Model ID and setup
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
processor = AutoProcessor.from_pretrained(model_id, token=token)
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    device_map="auto",                      # Auto-dispatch to GPU
    torch_dtype=torch.bfloat16,             # Use float16/bfloat16 on GPU
    token=token
)

def llama32_api_call(main_crop, classes):
    print(f"--- Calling LLaMA 3.2 API for main crop: {main_crop} ---")
    print(f"Detected Classes: {classes}")

    prompt = f"""
Given the following plant classes found in an image: {classes}.
The main crop being grown is {main_crop}.
Which of these classes visually resemble {main_crop} at early-to-mid growth stages?

Return ONLY a JSON list of class names (strings) that look like the main crop — including any variants or growth stages.
DO NOT RETURN ANY EXPLANATION OR TEXT OUTSIDE THE JSON LIST.
    """.strip()

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }
    ]

    try:
        # Step 1: Apply chat template (returns string)
        chat_prompt = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )

        # Step 2: Tokenize
        inputs = processor.tokenizer(
            chat_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(model.device)  # Send inputs to same device as model

        # Step 3: Generate
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=256,
        )

        # Step 4: Decode the new tokens only
        response = processor.tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )[0].strip()

        print(f"API Raw Response: {response}")

        # Step 5: Clean and parse JSON output
        if response.startswith("```json"):
            response = response[7:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()

        class_list = json.loads(response)

        if isinstance(class_list, list) and all(isinstance(x, str) for x in class_list):
            print(f"Selected Classes: {class_list}")
            return class_list
        else:
            print("Warning: Invalid format. Returning fallback.")
            return [main_crop] if main_crop in classes else []

    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return [main_crop] if main_crop in classes else []

    except Exception as e:
        print(f"LLaMA 3.2 API error: {e}")
        return [main_crop] if main_crop in classes else []

def compare_classes(all_detected_classes, main_crop, threshold=None):
    """
    Determines which detected classes resemble the main crop.
    Uses LLaMA 3.2 API if the main crop is not directly in the known class list.
    """
    selected_classes = []

    try:
        selected_classes = llama32_api_call(main_crop, all_detected_classes)
        if not isinstance(selected_classes, list) or not all(isinstance(item, str) for item in selected_classes):
            print("Warning: API did not return a valid list. Defaulting to [main_crop].")
            selected_classes = [main_crop] if main_crop in all_detected_classes else []
    except Exception as e:
        print(f"Error during API call: {e}. Falling back to [main_crop].")
        selected_classes = [main_crop] if main_crop in all_detected_classes else []

    return selected_classes
