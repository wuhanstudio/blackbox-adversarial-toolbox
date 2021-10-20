<img src="docs/images/bat.png" width=300px align="left">

# Black-box Adversarial Toolbox (BAT)

Black-box Adversarial Toolbox (BAT) - Python Library for Deep Learning Security that focuses on Black-box attacks.



## Installation

```python
pip install blackbox-adversarial-toolbox
```



## Usage



```python
from bat.attacks import SimBA

# Load Image [0.0, 1.0]
x = np.asarray(Image.open(args.image).resize((32, 32))) / 255.0

# Initialize the Cloud API Model
model = VGG16Cifar10(args.url + "/vgg16_cifar10")

# SimBA Attack
simba = SimBA()
x_adv = simba.attack(x, model)
```

