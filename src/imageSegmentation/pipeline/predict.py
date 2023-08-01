import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import numpy as np
import base64
from PIL import Image
from io import BytesIO
import tensorflow as tf
from tensorflow.keras import backend as K



class LoasAndMetrics:
    def bce_jaccard_loss(y_train, y_valid, smooth=1e-7):
        y_train_flat = K.flatten(y_train)
        y_valid_flat = K.flatten(y_valid)

        bce_loss = K.binary_crossentropy(y_train_flat, y_valid_flat)

        intersection = K.sum(y_train_flat * y_valid_flat)
        union = K.sum(y_train_flat) + K.sum(y_valid_flat) - intersection
        jaccard_loss = 1.0 - (intersection + smooth) / (union + smooth)
        total_loss = bce_loss + jaccard_loss
        return total_loss

    def iou_score(y_train, y_valid, smooth=1e-7):

        y_train_flat = K.flatten(y_train)
        y_valid_flat = K.flatten(y_valid)

        intersection = K.sum(y_train_flat * y_valid_flat)
        union = K.sum(y_train_flat) + K.sum(y_valid_flat) - intersection

        iou = (intersection + smooth) / (union + smooth)
        return iou

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        imagename = self.filename
        loss_and_metrics = LoasAndMetrics()
        custom_objects = {
            'bce_jaccard_loss': loss_and_metrics.bce_jaccard_loss,
            'iou_score':loss_and_metrics.iou_score
        }

        # model = load_model(os.path.join("artifacts", "segmentation_model", "model.h5"), custom_objects = custom_objects)
        model = load_model(os.path.join("artifacts", "segmentation_model", "base_model.h5"), custom_objects = custom_objects)
        test_image = image.load_img(imagename, target_size=(512, 512))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = model.predict(test_image)

        predicted_labels = np.argmax(result, axis=-1)

        colors = [
            [255, 235, 0],    # Class 0: Yellow
            [0, 0, 0],        # Class 1: Black
            [27, 71, 151],    # Class 2: Dark Blue
            [111, 48, 253],   # Class 3: Purple
            [137, 126, 126],  # Class 4: Gray
            [201, 19, 223],   # Class 5: Magenta
            [254, 233, 3],    # Class 6: Yellowish
            [255, 0, 29],     # Class 7: Red
            [255, 159, 0],    # Class 8: Orange
            [255, 160, 1]     # Class 9: Orangeish
        ]

        # Create an RGB image from the predicted labels
        rgb_image = np.zeros((predicted_labels.shape[1], predicted_labels.shape[2], 3), dtype=np.uint8)
        for class_idx, color in enumerate(colors):
            mask = (predicted_labels[0] == class_idx)
            rgb_image[mask] = color

        # Convert the RGB image to a base64-encoded string
        img_pil = Image.fromarray(rgb_image)
        buffered = BytesIO()
        img_pil.save(buffered, format="PNG")
        processed_string = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return processed_string


