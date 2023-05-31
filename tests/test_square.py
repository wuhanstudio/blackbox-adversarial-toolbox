import os
import unittest

import numpy as np
from PIL import Image

# Load the Cloud API Model
from bat.apis.deepapi import DeepAPI_VGG16_Cifar10

# Load the Square Attack
from bat.attacks.square_attack import SquareAttack

def dense_to_onehot(y, n_classes):
    y_onehot = np.zeros([len(y), n_classes], dtype=bool)
    y_onehot[np.arange(len(y)), y] = True
    return y_onehot

class TestSquare(unittest.TestCase):

    def test_square_cat(self):
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

        y_target_onehot = dense_to_onehot(np.array([np.argmax(y_pred)]), n_classes=len(y_pred))

        # SimBA Attack
        square_attack = SquareAttack(model)
        x_adv, _ = square_attack.attack(x, y_target_onehot, False, epsilon = 0.05, max_it=3000, concurrency=8)
        y_adv = model.predict(x_adv)[0]

        print('Adversarial Output:', model.get_class_name(np.argmax(y_adv)))

        # assert (np.argmax(y_adv) != np.argmax(y_pred))

    def test_square_dog(self):
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

        y_target_onehot = dense_to_onehot(np.array([np.argmax(y_pred)]), n_classes=len(y_pred))

        # SimBA Attack
        square_attack = SquareAttack(model)
        x_adv, _ = square_attack.attack(x, y_target_onehot, False, epsilon = 0.05, max_it=3000, concurrency=8)
        y_adv = model.predict(x_adv)[0]

        print('Adversarial Output:', model.get_class_name(np.argmax(y_adv)))

        # assert (np.argmax(y_adv) != np.argmax(y_pred))

if __name__ == '__main__':
    unittest.main()
