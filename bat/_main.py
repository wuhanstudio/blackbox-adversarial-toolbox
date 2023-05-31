import click
import validators

import numpy as np
from PIL import Image

from bat.apis.imagga import Imagga
from bat.apis.google import CloudVision
from bat.apis.deepapi import bat_deepapi_model_list

from bat.examples.simba_attack_deepapi import simba_attack_deepapi
from bat.examples.bandits_attack_deepapi import bandits_attack_deepapi
from bat.examples.square_attack_deepapi import square_attack_deepapi

bat_api_list = {
    'deepapi': 'An open-source image classification cloud service for research on black-box adversarial attacks.',
    'google': 'Google Cloud Vision AI.',
    'imagga': 'Imagga automatic tagging API.'
}

bat_attack_list = [
    ('SimBA', 'Local Search', 'A Simple Black-box Adversarial Attacks'),
    ('Square Attack', 'Local Search', 'A query-efficient black-box adversarial attack via random search.'),
    ('Bandits Atack', 'Gradient Estimation', 'Black-Box Adversarial Attacks with Bandits and Priors')
]

bat_example_list = [
    ('simba_deepapi', 'SimBA Attack against DeepAPI'),
    ('bandits_deepapi', 'Bandits Attack against DeepAPI'),
    ('square_deepapi', 'Square Attack against DeepAPI'),
]

# Main CLI (bat)
@click.group()
def main_cli():
    """The CLI tool for Black-box Adversarial Toolbox (BAT)."""
    pass

# bat api
@click.group()
def api():
    """Manage Cloud APIs"""
    pass

# bat api list
@api.command('list')
def api_list():
    """List supported Cloud APIs"""
    max_len = max([len(x) for x in bat_api_list.keys()])
    for i, api in enumerate(bat_api_list.keys(), start=1):
        print('{} : {:<{w}s}\t{}'.format(i, api, bat_api_list[api], w=max_len+1))

# bat api run
@api.group('run')
def api_run():
    """Run supported Cloud APIs"""
    pass

# bat api run deepapi
@api_run.command('deepapi')
def api_run_deepapi():
    """Send an image to DeepAPI"""
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
            x = np.array(image)
        except Exception as e:
            print(e)
            return

        deepapi_model = bat_deepapi_model_list[int(index)][1](deepapi_url)
    
        y = deepapi_model.predict(np.array([x]))[0]

        if y is not None:
            deepapi_model.print(y)

    except KeyboardInterrupt as e:
        print()
        return

# bat api run imagga
@api_run.command('imagga')
def api_run_deepapi():
    """Send an image to Imagga auto-tagging API"""
    api_key = input(f"Please input the Imagga API Key: ")
    api_secret = input(f"Please input the Imagga API Secret: ")
    imagga_client = Imagga(api_key, api_secret, concurrency=2)

    # Get the image file
    try:
        file = input(f"Please input the image file: ")
        while len(file) == 0:
            file = input(f"Please input the image file: ")
    except Exception as e:
        print(e)
        return

    # Make predictions
    y = imagga_client.predict(file)

    # Print results
    if y is not None:
        imagga_client.print(y)

# bat api run google
@api_run.command('google')
def api_run_google():
    """Send an image to Google Cloud Vision"""
    vision_client = CloudVision()

    # Get the image file
    try:
        file = input(f"Please input the image file: ")
        while len(file) == 0:
            file = input(f"Please input the image file: ")
    except Exception as e:
        print(e)
        return

    # Make predictions
    y = vision_client.predict(file)
    
    # Print resuilts
    if y is not None:
        vision_client.print(y)

# bat attack
@click.group()
def attack():
    """Manage Attacks"""
    pass

# bat attack list
@attack.command('list')
def attack_list():
    """List supported Attacks"""
    max_len = max([len(x[0]) for x in bat_attack_list])
    for i, attack in enumerate(bat_attack_list, start=1):
        print('{} : {:<{w}s}\t{}'.format(i, attack[0], attack[1], w=max_len))

# bat example
@click.group()
def example():  
    """Manage Examples"""
    pass

# bat example list
@example.command('list')
def example_list():
    """List examples"""
    max_len = max([len(x[0]) for x in bat_example_list])
    for i, example in enumerate(bat_example_list, start=1):
        print('{} : {:<{w}s}\t{}'.format(i, example[0], example[1], w=max_len))

# bat exmaple run
@example.group('run')
def example_run():
    """Run examples"""
    pass

# bat exmaple run simba_deepapi
@example_run.command('simba_deepapi')
def example_run_simba_deepapi():
    """SimBA Attack against DeepAPI"""
    simba_attack_deepapi()

# bat exmaple run bandits_deepapi
@example_run.command('bandits_deepapi')
def example_run_bandits_deepapi():
    """Bandits Attack against DeepAPI"""
    bandits_attack_deepapi()

# bat exmaple run square_deepapi
@example_run.command('square_deepapi')
def example_run_bandits_deepapi():
    """Square Attack against DeepAPI"""
    square_attack_deepapi()

def main():
    main_cli.add_command(api)
    main_cli.add_command(attack)
    main_cli.add_command(example)

    api.add_command(api_list)
    api.add_command(api_run)

    attack.add_command(attack_list)

    example.add_command(example_list)
    example.add_command(example_run)

    return main_cli()

if __name__ == "__main__":

    main()
