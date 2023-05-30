import validators

import numpy as np
np.set_printoptions(suppress=True)

from PIL import Image

from bat.apis.deepapi import bat_deepapi_model_list
from bat.attacks import SimBA

def simba_attack_deepapi():

    for i, (_, model) in enumerate(bat_deepapi_model_list.items(), start=1):
        print(i, ':', model[0])

    try:
        # Get the model type
        index = input(f"Please input the model index (default: 1): ")
        if len(index) == 0:
            index = 1
        else:
            while not index.isdigit() or int(index) > len(bat_deepapi_model_list):
                index = input(f"Model [{index}] does not exist. Please try again: ")

        # Get the DeepAPI server url
        deepapi_url = input(f"Please input the DeepAPI URL (default: http://localhost:8080): ")
        if len(deepapi_url) == 0:
            deepapi_url = 'http://localhost:8080'
        else:
            while not validators.url(deepapi_url):
                deepapi_url = input(f"Invalid URL. Please try again: ")

        # Get the image file
        try:
            file = input(f"Please input the image file: ")
            while len(file) == 0:
                file = input(f"Please input the image file: ")
            image = Image.open(file).convert('RGB')
            if index == 1:
                image = image.resize((32, 32))
            x = np.array(image)
            x = np.array([x])
        except Exception as e:
            print(e)
            return

        # DeepAPI Model
        deepapi_model = bat_deepapi_model_list[int(index)][1](deepapi_url)
    
        # Make predictions
        y_pred = deepapi_model.predict(x)[0]

        if y_pred is not None:
            deepapi_model.print(y_pred)
            print('Prediction', np.argmax(y_pred), deepapi_model.get_class_name(np.argmax(y_pred)))
            print()
        
        # SimBA Attack
        simba = SimBA(deepapi_model)
        x_adv = simba.attack(x, np.argmax(y_pred), epsilon=0.05, max_it=3000, concurrency=4)

        # Print result after attack
        y_adv = deepapi_model.predict(x_adv)[0]
        deepapi_model.print(y_adv)
        print('Prediction', np.argmax(y_adv), deepapi_model.get_class_name(np.argmax(y_adv)))
        print()

        # Save image
        Image.fromarray((x_adv[0]).astype(np.uint8)).save('result.jpg', subsampling=0, quality=100)
        print("The adversarial image is saved as result.jpg")

    except KeyboardInterrupt as e:
        print()
        return
