import unittest

import numpy as np
from PIL import Image

# Load the Cloud API Model
from bat.apis.deepapi import VGG16Cifar10

# Load the SimBA Attack
from bat.attacks import SimBA

class TestSimBA(unittest.TestCase):

    def test_simba(self):
        # Load Image [0.0, 1.0]
        x = np.asarray(Image.open("tests/cat.jpg").resize((32, 32))) / 255.0

        # Initialize API Model
        model = VGG16Cifar10("https://api.wuhanstudio.uk" + "/vgg16_cifar10")

        # Get Preditction
        y_pred = model.predict(np.array([x]))[0]

        assert (np.argmax(y_pred) == 3)

        # SimBA Attack
        simba = SimBA(model)
        x_adv = simba.attack(x, epsilon=0.1, max_it=1000)
        y_adv = model.predict(np.array([x_adv]))[0]

        assert (np.argmax(y_adv) != np.argmax(y_pred))

if __name__ == '__main__':
    unittest.main()