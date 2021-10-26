import argparse

import numpy as np
np.set_printoptions(suppress=True)

from PIL import Image

# Load the Cloud API Model
from bat.apis.deepapi import VGG16Cifar10

# Load the SimBA Attack
from bat.attacks import SimBA

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DeepAPI SimBA Attack example')

    parser.add_argument(
        'url',
        type=str,
        help='API root url (e.g. https://api.wuhanstudio.uk)',
    )
    parser.add_argument(
        'image',
        type=str,
        help='image file'
    )
    args = parser.parse_args()

    # Load Image [0.0, 1.0]
    x = np.asarray(Image.open(args.image).resize((32, 32))) / 255.0

    # Initialize API Model
    model = VGG16Cifar10(args.url + "/vgg16_cifar10")

    # Get Preditction
    y_pred = model.predict(np.array([x]))[0]

    # Print result
    model.print(y_pred)
    print('Prediction', np.argmax(y_pred), model.get_class_name(np.argmax(y_pred)))
    print()

    # SimBA Attack
    simba = SimBA(model)
    x_adv = simba.attack(x, epsilon=0.1, max_it=1000, distributed=True , batch=50, max_workers=10)

    # Print result after attack
    y_pred = model.predict(np.array([x_adv]))[0]
    model.print(y_pred)
    print('Prediction', np.argmax(y_pred), model.get_class_name(np.argmax(y_pred)))

    # Save image
    Image.fromarray((x_adv * 255).astype(np.uint8)).save('result.jpg')
