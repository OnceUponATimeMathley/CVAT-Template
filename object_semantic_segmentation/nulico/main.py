# Author: MathleyNeutron
import json
import base64
import io
from PIL import Image
from model_handler import ModelHandler
import numpy as np
from skimage.measure import find_contours, approximate_polygon

def init_context(context):
    context.logger.info("Init context...  0%")

    # Load model
    model = ModelHandler()
    context.user_data.model = model

    # Load configs
    configs = {}
    configs["TOLERANCE"] = 3
    configs["NUM_POINTS_OF_CONTOUR_THRES"] = 15
    configs["NUM_MASK_POINT_THRES"] = 300
    configs["MASK_THRESHOLD"] = 0.5
    setattr(context.user_data, "configs", configs)


    context.logger.info("Init context...100%")

def handler(context, event):
    context.logger.info("Run Skeleton Model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    img_data = Image.open(buf).convert('RGB') # Fix png image: 4 channels


    # Get configs and model
    configs = context.user_data.configs
    model = context.user_data.model

    cluster_pred, classes = model.handle(img_data)

    # Get mask image of each class
    masks = get_masks(configs, cluster_pred, len(classes))

    # Get contours
    all_contour = get_contours(configs, masks)

    encoded_results = []

    for idx, contours in all_contour.items():
       for contour in contours:
           contour = np.flip(contour, axis=1) - 1
           # Approximate the contour and reduce the number of points
           contour = approximate_polygon(contour, tolerance=configs["TOLERANCE"])
           if len(contour) < configs["NUM_POINTS_OF_CONTOUR_THRES"]:
               continue

           encoded_results.append({
           "label" : classes[idx],
           "points" : contour.ravel().tolist(),
           "type" : "polygon"
           })

    return context.Response(body=json.dumps(encoded_results), headers={},
        content_type='application/json', status_code=200)


def get_masks(configs, cluster_pred, num_classes):
    masks = []
    for i in range(num_classes):
        mask = np.where(cluster_pred == i, 1, 0)
        if not np.any(mask):
            continue

        if np.sum(mask) > configs["NUM_MASK_POINT_THRES"]:
            masks.append([np.pad(mask, [1, 1], 'constant', constant_values=(0, 0)), i])

    return masks # List of mask image for each classs and the class index


def get_contours(configs, masks):
    all_contour = {}

    for i in range(len(masks)):
        mask_img, idx = masks[i]
        contours = find_contours(mask_img, configs["MASK_THRESHOLD"]) # List of contour for mask_img
        all_contour[idx] = contours

    return all_contour