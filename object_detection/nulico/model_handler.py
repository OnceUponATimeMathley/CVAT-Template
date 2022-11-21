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
                The list of elements contain:
                    - label: string type
                    - xcenter, ycenter, width, height: the value in [0, 1]
                    - conf: float type
        """


        # Handle here
        pass

