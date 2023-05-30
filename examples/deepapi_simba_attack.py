import argparse

import numpy as np
np.set_printoptions(suppress=True)

from PIL import Image

# Load the Cloud API Model
from bat.apis.deepapi import DeepAPI_VGG16_Cifar10

# Load the SimBA Attack
from bat.attacks import SimBA

if __name__ == '__main__':

    # Load Image [0.0, 1.0]
    x = np.asarray(Image.open('../tests/dog.jpg').resize((32, 32)).convert('RGB'))
    x = np.array([x])

    # Initialize API Model
    model = DeepAPI_VGG16_Cifar10('http://localhost:8080')

    # Get Preditction
    y_pred = model.predict(x)[0]

    # Print result
    model.print(y_pred)
    print('Prediction', np.argmax(y_pred), model.get_class_name(np.argmax(y_pred)))
    print()

    # SimBA Attack
    simba = SimBA(model)
    x_adv = simba.attack(x, np.argmax(y_pred), epsilon=0.05, max_it=3000, concurrency=4)

    # Print result after attack
    y_pred = model.predict(x_adv)[0]
    model.print(y_pred)
    print('Prediction', np.argmax(y_pred), model.get_class_name(np.argmax(y_pred)))

    # Save image
    Image.fromarray((x_adv[0]).astype(np.uint8)).save('result.jpg')
