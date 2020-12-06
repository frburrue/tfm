from PIL import Image, ImageDraw
from io import BytesIO


class RekognitionTextParser:
    """Encapsulates an Amazon Rekognition text element."""
    def __init__(self, text_data):
        """
        Initializes the text object.

        :param text_data: Text data, in the format returned by Amazon Rekognition
                          functions.
        """
        self.text = text_data.get('DetectedText')
        self.kind = text_data.get('Type')
        self.id = text_data.get('Id')
        self.parent_id = text_data.get('ParentId')
        self.confidence = text_data.get('Confidence')
        self.geometry = text_data.get('Geometry')
        
    def to_dict(self):
        """
        Renders some of the text data to a dict.

        :return: A dict that contains the text data.
        """
        rendering = {}
        if self.text is not None:
            rendering['text'] = self.text
        if self.kind is not None:
            rendering['kind'] = self.kind
        if self.geometry is not None:
            rendering['polygon'] = self.geometry.get('Polygon')
        return rendering
        
        
def show_rekognition_polygons(image_bytes, polygons, color):
        """
        Draws polygons on an image and shows it with the default image viewer.
    
        :param image_bytes: The image to draw, as bytes.
        :param polygons: The list of polygons to draw on the image.
        :param color: The color to use to draw the polygons.
        """
        image = Image.open(BytesIO(image_bytes))
        draw = ImageDraw.Draw(image)
        for polygon in polygons:
            draw.polygon([
                (image.width * point['X'], image.height * point['Y']) for point in polygon],
                outline=color)
        return image