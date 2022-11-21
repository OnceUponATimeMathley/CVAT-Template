# Import nescessary library and package

# import sys
# sys.path.append('/opt/nuclio/stego/src')  
import yaml
from base_handler import BaseHandler



class ModelHandler(BaseHandler):
    def __init__(self):
        self.net = None
        self.configs = {}

        with open("./configs.yaml", 'r') as stream:
            try:
                parsed_yaml = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise Exception("Configs file error: ", exc)

        # Init another necessary variables

    def handle(self, image):
        """
            INPUT:
                image: The original image, the output of Image.open(). You need to
                convert it to numpy or tensor array,... corresponding to the input type of
                your model.
            OUTPUT:
                - The predicted mask tensor with shape H, W. Each pixel has the predicted 
                class index from 0 to number of classes - 1
                - The list of class: ['person', 'background',....]
                return predict_mask, classes
        """


        # Handle here
        pass

