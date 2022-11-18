# Author: MathleyNeutron
import json
import base64
import io
from PIL import Image
from model_handler import ModelHandler
import numpy as np


def init_context(context):
    context.logger.info("Init context...  0%")

    # Load model
    model = ModelHandler()
    context.user_data.model = model

    context.logger.info("Init context...100%")

def handler(context, event):
    context.logger.info("Run object detection model")

    # Load image
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    image = Image.open(buf)
    image = np.array(image)
    h, w = image.shape[:2]

    # Get bounding boxes from model
    model = context.user_data.model
    bouding_boxes = model.handler(image)
    
    # Convert to CVAT format
    encoded_results = []
    for idx in range(len(bouding_boxes)):
        label, xcenter, y_center, width, height, conf = bouding_boxes[idx]
        encoded_results.append({
            'confidence': conf,
            'label': label,
            'points': [
                (xcenter - width / 2) * w,
                (y_center - height / 2) * h,
                (xcenter + width / 2) * w,
                (y_center + height / 2) * h
            ],
            'type': 'rectangle'
        })
        
    return context.Response(body=json.dumps(encoded_results), headers={},
        content_type='application/json', status_code=200)

