import os
import unittest

import numpy as np
from PIL import Image

from bat.apis.deepapi import DeepAPI_VGG16_Cifar10

class TestDeepAPI(unittest.TestCase):

    def test_vgg_cifar10_dog(self):
        # Load Image
        x = np.asarray(Image.open("tests/dog.jpg").convert('RGB'))

        # Initialize the Cloud API Model
        assert os.environ['DEEPAPI_URL']
        model = DeepAPI_VGG16_Cifar10(os.environ['DEEPAPI_URL'])

        # Get Preditction
        y_pred = model.predict(np.array([x]))[0]

        # Print result
        model.print(y_pred)
        print('Prediction', np.argmax(y_pred), model.get_class_name(np.argmax(y_pred)))

        assert np.argmax(y_pred) == 4

    def test_vgg_cifar10_cat(self):
        # Load Image
        x = np.asarray(Image.open("tests/cat.jpg").convert('RGB'))

        # Initialize the Cloud API Model
        assert os.environ['DEEPAPI_URL']
        model = DeepAPI_VGG16_Cifar10(os.environ['DEEPAPI_URL'])

        # Get Preditction
        y_pred = model.predict(np.array([x]))[0]

        # Print result
        model.print(y_pred)
        print('Prediction', np.argmax(y_pred), model.get_class_name(np.argmax(y_pred)))

        assert np.argmax(y_pred) == 2

if __name__ == '__main__':
    unittest.main()
