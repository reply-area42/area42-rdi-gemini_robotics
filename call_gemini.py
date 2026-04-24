from google import genai
from google.genai import types
import cv2
import numpy as np
import json
from secrets import API_KEY
import position_calculator as pc
from PIL import ImageColor, ImageDraw, ImageFont, Image

import drawing_utils as du


PROMPT = """
        Detect max 3 objects. Return JSON: {"obj": name, 
        "box_2d": [ymin, xmin, ymax, xmax], 
        "polygon": [y1, x1, y2, x2, ...]}. Use normalized 0-1000 coordinates. No prose.
        """

client = genai.Client(api_key=API_KEY)


def call_model(image_bytes, prompt):

    prompt = prompt if prompt is not None else PROMPT

    image_response = client.models.generate_content(
        model="gemini-robotics-er-1.6-preview",
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type='image/png',
            ),
            prompt
        ],
        config=types.GenerateContentConfig(
            temperature=1.0,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json",
        )
    )

    print(image_response.text)

    return image_response



