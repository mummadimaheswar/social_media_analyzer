import os
import json
from google import genai
from google.genai import types
from pydantic import BaseModel

class PostAnalysis(BaseModel):
    tone: str
    intent: str
    communication_style: str
    summary: str

def analyze_post(post_text: str) -> str:
    """
    Analyzes a social media post and returns the classification 
    in a structured JSON format.
    """
    # The client automatically picks up the GEMINI_API_KEY environment variable.
    client = genai.Client()
    
    prompt = f"""
    Analyze the following social media post text and classify its tone, intent, and communication style.
    Provide a brief summary.
    
    Post:
    {post_text}
    """
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=PostAnalysis,
        ),
    )
    
    return response.text
