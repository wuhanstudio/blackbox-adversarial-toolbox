r'''
This module implements several API clients for Deep Learning cloud API services, so that we can use `model.predict()` to get prediction results from API servers.


### bat.apis.deepapi

[DeepAPI](https://github.com/wuhanstudio/deepapi) is an open source image classification cloud service for research on distributed black-box attacks.

```python
from bat.apis.deepapi import DeepAPI_VGG16_Cifar10

# Load Image
x = np.sarray(Image.open("dog.jpg"))
x = np.array([x])

# Initialize API Model
model = DeepAPI_VGG16_Cifar10("http://localhost:8080" )

# Get Preditctions
y_pred = model.predict(np.array([x]))[0]

# Print Predictions
model.print(y_pred)
```

### bat.apis.google

- [Google Cloud Vision](https://cloud.google.com/vision)

```python
from bat.apis.google import CloudVision

# Initialize API Model
model = CloudVision()

# Get Preditctions
y_pred = model.predict("dog.jpg")

# Print Predictions
model.print(y_pred)
```

### bat.apis.imagga

- [Imagga](https://docs.imagga.com/#tags)

```python
from bat.apis.imagga import Imagga

# Initialize API Model
api_key = input(f"Please input the Imagga API Key: ")
api_secret = input(f"Please input the Imagga API Secret: ")

model = Imagga(api_key, api_secret, concurrency=2)

# Get Preditctions
y_pred = model.predict("dog.jpg")

# Print Predictions
model.print(y_pred)
```

'''

from bat.apis import deepapi
from bat.apis import google
from bat.apis import imagga
