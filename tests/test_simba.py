import os
import unittest

import numpy as np
from PIL import Image

# Load the Cloud API Model
from bat.apis.deepapi import DeepAPI_VGG16_Cifar10

# Load the SimBA Attack
from bat.attacks import SimBA

class TestSimBA(unittest.TestCase):

    def test_simba_cat(self):
        # Load Image
        x = np.asarray(Image.open("tests/cat.jpg").convert('RGB'))
        x = np.array([x])

        # Initialize API Model
        assert os.environ['DEEPAPI_URL']
        model = DeepAPI_VGG16_Cifar10(os.environ['DEEPAPI_URL'])

        # Get Preditction
        y_pred = model.predict(x)[0]

        print('\nOriginal Prediction:', model.get_class_name(np.argmax(y_pred)))
        assert (np.argmax(y_pred) == 2)

        # SimBA Attack
        simba = SimBA(model)
        x_adv = simba.attack(x, np.argmax(y_pred), epsilon=0.05, max_it=10)
        y_adv = model.predict(x_adv)[0]

        print('Adversarial Output:', model.get_class_name(np.argmax(y_adv)))

        # assert (np.argmax(y_adv) != np.argmax(y_pred))

    def test_simba_dog(self):
        # Load Image
        x = np.asarray(Image.open("tests/dog.jpg").convert('RGB'))
        x = np.array([x])

        # Initialize API Model
        assert os.environ['DEEPAPI_URL']
        model = DeepAPI_VGG16_Cifar10(os.environ['DEEPAPI_URL'])

        # Get Preditction
        y_pred = model.predict(x)[0]

        print('\nOriginal Prediction:', model.get_class_name(np.argmax(y_pred)))
        assert (np.argmax(y_pred) == 4)

        # SimBA Attack
        simba = SimBA(model)
        x_adv = simba.attack(x, np.argmax(y_pred), epsilon=0.05, max_it=10)
        y_adv = model.predict(x_adv)[0]

        print('Adversarial Output:', model.get_class_name(np.argmax(y_adv)))

        # assert (np.argmax(y_adv) != np.argmax(y_pred))

if __name__ == '__main__':
    unittest.main()
