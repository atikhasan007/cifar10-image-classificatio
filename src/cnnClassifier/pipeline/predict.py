import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


class PredictionPipeline:

    def __init__(self, filename):
        self.filename = filename

        self.model = load_model(
            r"C:\Users\Hp\Desktop\cifar10-image-classificatio\artifacts\training\model.h5"
        )

        self.class_names = [
            "Airplane",
            "Automobile",
            "Bird",
            "Cat",
            "Deer",
            "Dog",
            "Frog",
            "Horse",
            "Ship",
            "Truck"
        ]

    def predict(self):

        img = image.load_img(
            self.filename,
            target_size=(32, 32)
        )

        img = image.img_to_array(img)

       
        img = img.astype(np.float32)

        img = np.expand_dims(img, axis=0)

        prediction = self.model.predict(img, verbose=0)

        class_index = np.argmax(prediction[0])

        confidence = float(prediction[0][class_index])

        return [{
            "class": self.class_names[class_index],
            "confidence": f"{confidence*100:.2f}%"
        }]