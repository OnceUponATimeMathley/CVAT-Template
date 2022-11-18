import json
import base64
from PIL import Image
import io
from model_handler import ModelHandler
import numpy as np
from skimage.measure import find_contours, approximate_polygon
from skimage import draw


def init_context(context):
    context.logger.info("Init context...  0%")

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
    context.logger.info("Run interactive segmentation model")

    # Load image
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    img = Image.open(buf)

    pos_points = data["pos_points"]
    # Get configs and model
    configs = context.user_data.configs
    model = context.user_data.model

    encoded_results = {}
    postprocess_all_contour = []

    if len(pos_points) == 1:
        cluster_pred, classes = model.handle(img)

        # Get mask image of each class
        masks = get_masks(configs, cluster_pred, len(classes))
        # Get contours
        all_contour = get_contours(configs, masks)
        # Post processing contour with approximation
        for idx, contours in all_contour.items():
            for contour in contours:
                contour = np.flip(contour, axis=1) - 1
                # Approximate the contour and reduce the number of points
                contour = approximate_polygon(contour, tolerance=configs["TOLERANCE"])
                if len(contour) < configs["NUM_POINTS_OF_CONTOUR_THRES"]:
                    continue
                postprocess_all_contour.append([contour, idx])

        result = None
        # Check which contour a click point is in
        for index in range(len(postprocess_all_contour)):
            if check_point_in_polygon(pos_points[0], postprocess_all_contour[index][0]):
                polygon, class_idx = postprocess_all_contour[index]
                mask = poly2mask(polygon[:, 1], polygon[:, 0], cluster_pred.shape)
                result =  polygon, class_idx, mask

        if result:
            encoded_results  = {
                "points" : (result[0].astype(np.int32)).tolist(),
                "mask" : (result[2].astype(np.uint8)).tolist()
            }

    print("Complete!")

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

def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask

def check_point_in_polygon(point, polygon):
    if type(polygon) is np.ndarray:
        polygon = polygon.tolist()

    n = len(polygon) # Number of points in the polygon
    count = 0
    x = point[0]
    y = point[1]

    for i in range(n - 1):
        f_point = polygon[i]
        s_point = polygon[i + 1]

        x1 = f_point[0]
        y1 = f_point[1]
        x2 = s_point[0]
        y2 = s_point[1]

        if ((y < y1) != (y < y2)) and (x < ((x2 - x1) * (y - y1) / (y2 - y1) + x1)):
            count += 1
            # print("Count: ", count)

    return (False if count % 2 == 0 else True)
