r'''
This module implements several API clients for Deep Learning cloud API services, so that we can use `model.predict()` to get prediction results from API servers.

- [DeepAPI](https://github.com/wuhanstudio/deepapi)

```python
from bat.apis.deepapi import VGG16Cifar10

# Load Image [0.0, 1.0]
x = np.asarray(Image.open("dog.jpg").resize((32, 32))) / 255.0

# Initialize API Model
model = VGG16Cifar10("https://api.wuhanstudio.uk" + "/vgg16_cifar10")

# Get Preditction
y_pred = model.predict(np.array([x]))[0]
```

'''


from bat.apis import deepapi
