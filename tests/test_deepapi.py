import os
import unittest

import numpy as np
from PIL import Image

from bat.apis.deepapi import VGG16Cifar10

class TestDeepAPI(unittest.TestCase):

    def test_vgg_cifar10(self):
        # Load Image [0.0, 1.0]
        x = np.asarray(Image.open("tests/dog.jpg").resize((32, 32))) / 255.0

        # Initialize the Cloud API Model
        assert os.environ['DEEPAPI_URL']
        model = VGG16Cifar10(os.environ['DEEPAPI_URL'] + "/vgg16_cifar10")

        # Get Preditction
        y_pred = model.predict(np.array([x]))[0]

        # Print result
        model.print(y_pred)
        print('Prediction', np.argmax(y_pred), model.get_class_name(np.argmax(y_pred)))

        assert np.argmax(y_pred) == 5

if __name__ == '__main__':
    unittest.main()