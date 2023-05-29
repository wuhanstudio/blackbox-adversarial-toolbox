import click
import validators

import numpy as np
from PIL import Image

from bat.apis.deepapi import DeepAPI_VGG16_Cifar10, DeepAPI_VGG16_ImageNet, DeepAPI_Resnet50_ImageNet, DeepAPI_Inceptionv3_ImageNet

bat_api_list = [
    'deepapi',
    'google',
    'imagga'
]

bat_deepapi_model_list = {
    1: ['vgg16_cifar10', DeepAPI_VGG16_Cifar10],
    2: ['vgg16_imagenet', DeepAPI_VGG16_ImageNet],
    3: ['resnet50_imagenet', DeepAPI_Resnet50_ImageNet],
    4: ['inceptionv3_imagenet', DeepAPI_Inceptionv3_ImageNet]
}

bat_attack_list = [
    'simba',
    'square',
    'bandits'
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
    click.echo(bat_api_list)

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
        deepapi_model.print(y)

    except KeyboardInterrupt as e:
        print()
        return
    

# bat attack
@click.group()
def attack():
    """Manage Attacks"""
    pass

# bat attack list
@attack.command('list')
def attack_list():
    """List supported Attacks"""
    click.echo(bat_attack_list)

# bat example
@click.group()
def example():  
    """Manage Examples"""
    pass

# bat example list
@example.command('list')
def example_list():
    """List examples"""
    pass

# bat exmaple run
@example.command('run')
def example_run():
    """Run examples"""
    pass

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
