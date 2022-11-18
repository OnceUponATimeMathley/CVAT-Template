# Author: MathleyNeutron
import json
import base64
import io
from PIL import Image
from model_handler import ModelHandler



def init_context(context):
    context.logger.info("Init context...  0%")

    # Load model
    model = ModelHandler()
    context.user_data.model = model

    context.logger.info("Init context...100%")

def handler(context, event):
    context.logger.info("Run polyline model")

    # Load image
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    image = Image.open(buf)

    # Get bounding boxes from model
    model = context.user_data.model
    response = model.handler(image)

    # Convert to CVAT format
    encoded_results = []
    for idx in range(len(response)):
        label, points = response[idx]
        encoded_results.append({
            'label': label,
            'points': points,
            'type': 'polyline'
        })

    return context.Response(body=json.dumps(encoded_results), headers={},
        content_type='application/json', status_code=200)

