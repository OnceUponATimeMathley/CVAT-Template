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
    context.logger.info("Run Skeleton Model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    img_data = Image.open(buf).convert('RGB') # Fix png image: 4 channels

    # Get configs and model
    model = context.user_data.model

    template = model.handle(img_data)

    # template: version, flags, shapes, imgHeight, imgWidth
    # shapes: label, points, landmark, status, group_id
    encoded_results = []
    for idx in range(len(template['shapes'])):
        infos = template['shapes'][idx]
        label = infos['label']
        elements = []
        for i in range(len(infos['points'])):
            element = {
                "type": "points",
                "points": infos['points'][i]
            }
            elements.append(element)
        skeleton = {
        "label": label,
        "type": "skeleton",
        "occluded": False,
        "outside": False,
        "elements": elements
        }
        encoded_results.append(skeleton)

    print("Complete!")

    return context.Response(body=json.dumps(encoded_results), headers={},
        content_type='application/json', status_code=200)
