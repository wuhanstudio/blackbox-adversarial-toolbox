r"""
This module implements the DeepAPI client for pretrained VGG16 model on CIFAR-10 dataset.
"""

import requests

import numpy as np
np.set_printoptions(suppress=True)

from PIL import Image

from io import BytesIO
import base64

from sklearn.preprocessing import LabelEncoder

class VGG16Cifar10:
    def __init__(self, url):
        """
        - url: DeepAPI server URL
        """
        self.url = url

        # cifar10 labels
        cifar10_labels = np.array(['frog', 'deer', 'cat', 'bird', 'dog', 'truck', 'ship', 'airplane', 'horse', 'automobile'])

        # integer encode
        self.__label_encoder__ = LabelEncoder()
        integer_encoded = self.__label_encoder__.fit_transform(cifar10_labels)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

        # Map each label to an integer. ['frog'] --> 6
        self.label_map = dict(zip(cifar10_labels, integer_encoded))

    def predict(self, X):
        """
        - X: numpy array of shape (N, 3, W, H)
        """
        y_pred = []
        try:
            y_pred_temp = np.zeros([10])
            for x in X:
                # Load the input image and construct the payload for the request
                image = Image.fromarray(np.uint8(x * 255.0))
                buff = BytesIO()
                image.save(buff, format="JPEG")

                data = {'file': base64.b64encode(buff.getvalue()).decode("utf-8")}
                res = requests.post(self.url, json=data).json()['predictions']

                for r in res:
                    y_pred_temp[self.label_map[r['label']][0]] = r['probability']

                y_pred.append(y_pred_temp)

        except Exception as e:
            print(e)

        return np.array(y_pred)

    def print(self, y):
        """
        Print the prediction result.
        """
        print()
        for i in range(0, len(y)):
            print('{:<15s}{:.5f}'.format(self.__label_encoder__.inverse_transform([i])[0], y[i]))

    def get_class_name(self, i):
        """
        Get the class name from the prediction label 0-10.
        """
        return self.__label_encoder__.inverse_transform([i])[0]
