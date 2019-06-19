import tensorflow as tf
import cv2
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model


class Predict:

    def __init__(self):
        self.CATEGORIES = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship',
                           'Truck']  # will use this to convert prediction num to string value
        self.model = tf.keras.models.load_model("static/cnn_10_classifier_loss0.465acc0.8839.model")

    def load_image(self, path):
        img = load_img(path, target_size=(32, 32))
        img = img_to_array(img)
        img = img.reshape(1, 32, 32, 3)
        img = img.astype('float32')
        img = img / 255.0

        return img

    def prediction(self, image):
        predation = self.model.predict(image)
        class_name = str(self.CATEGORIES[np.argmax(predation[0])])
        class_pr = str(predation[0][np.argmax(predation[0])] * 100) + "%"
        if int(predation[0][np.argmax(predation[0])] * 100) > 50:
            return class_name + "   " + class_pr
        else:
            return 'this UNKNOWN object  but it maybe a/an ' + class_name

    # print(predction[0][0])
