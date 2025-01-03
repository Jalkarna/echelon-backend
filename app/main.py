from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import json
from typing import List, Dict
import google.generativeai as genai
import os
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:3000", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def print_predefined_sample():
    sample_size = min(5, len(predefined_data))
    logger.info(f"Sample of predefined data (first {sample_size} items):")
    for item in predefined_data[:sample_size]:
        logger.info(f"Word: {item.get('sign_text', 'N/A')}, Expression: {item.get('facial_expression', 'N/A')}")

# Load pre-defined data from JSON file
try:
    with open("data/pre_defined.json", "r") as f:
        predefined_data = json.load(f)
    print_predefined_sample()
except FileNotFoundError:  # This is for the handling the error whic is generating through the json file
    logger.error("pre_defined.json file not found.")
    raise RuntimeError("pre_defined.json file not found. Please ensure it exists in the same directory as this script.")

# Configure Google Generative AI
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    logger.error("GOOGLE_API_KEY not found in environment variables.")
    raise RuntimeError("GOOGLE_API_KEY not found. Please set it in your .env file.")

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-pro')

# Input data model
class TextInput(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Welcome to the Text-to-Sign Language API!"}

@app.post("/convert")
async def convert_text(input_data: TextInput):
    """
    Convert input text to sign grammar and retrieve movement data.
    """
    try:
        logger.info(f"Received input text: {input_data.text}")

        # Step 1: Send input text to Gemini 1.5 for sign grammar conversion
        prompt = f"""
        Convert the following text to American Sign Language (ASL) grammar. 
        Provide ONLY the ASL gloss (the written representation of ASL signs) without any explanation.
        Use uppercase letters for each sign. Separate signs with spaces.
        Example: "How are you?" should be converted to "HOW YOU".
        
        Text to convert: '{input_data.text}'
        """
        logger.debug(f"Sending prompt to Gemini: {prompt}")
        response = await model.generate_content_async(prompt)
        sign_grammar_text = response.text.strip().upper()
        logger.info(f"Received sign grammar text: {sign_grammar_text}")

        # Step 2: Split the sign grammar text into individual words
        words = sign_grammar_text.split()
        logger.debug(f"Split words: {words}")

        # Step 3: Retrieve movement data for each word
        movements = []
        for word in words:
            # Case-insensitive search and strip whitespace
            movement = next((item for item in predefined_data if item.get("sign_text", "").strip().upper() == word.strip()), None)
            if movement:
                movements.append(movement)
                logger.info(f"Found movement for word: {word}")
                logger.debug(f"Movement data: {movement}")
            else:
                # Log the exact word we're looking for and a sample of predefined data
                logger.warning(f"No movement found for word: '{word}'")
                logger.debug(f"Sample of predefined data: {predefined_data[:2]}")
                default_movement = {
                    "sign_text": word,
                    "facial_expression": "DEFAULT_EXPRESSION",
                    "hand_movement": {"type": "DEFAULT", "movement": "DEFAULT", "duration": 1},
                    "frames_json": "[]"
                }
                movements.append(default_movement)
                logger.info(f"Added default movement for word: {word}")
                logger.debug(f"Default movement data: {default_movement}")

        # Log the final movements list
        logger.info(f"Final movements list: {movements}")

        # Step 4: Aggregate and send the data to the frontend
        response_data = {
            "original_text": input_data.text,
            "sign_grammar_text": sign_grammar_text,
            "movements": movements
        }
        logger.info("Successfully processed request")
        return response_data
    except Exception as e:  
        logger.exception(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

