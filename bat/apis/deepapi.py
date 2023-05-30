r"""
This module implements the DeepAPI client for pretrained VGG16 model on CIFAR-10 dataset.
"""

import requests
from urllib.parse import urlparse
import urllib.request, json 
from urllib.parse import urljoin

import numpy as np
np.set_printoptions(suppress=True)

from PIL import Image
from io import BytesIO
import base64

import concurrent.futures

class DeepAPIBase:
    def __init__(self, concurrency=8):
        self.concurrency = concurrency
        pass

    def predict(self, X):
        n_targets = 0
        if type(X) == list:
            n_targets = len(X)
        elif type(X) == np.ndarray:
            n_targets = X.shape[0]
        else:
            raise ValueError('Input type not supported...')

        return self.predictX(X) if n_targets > 1 else self.predictO(X)

    def predictX(self, X):
        """
        - X: numpy array of shape (N, H, W, C)
        """
        if isinstance(X, list):
            for x in X:
                assert len(x.shape) == 3, 'Expecting a 3D tensor'
        else:
            if len(X.shape) != 4 or X.shape[3] != 3:
                raise ValueError(
                    "`predict` expects "
                    "a batch of images "
                    "(i.e. a 4D array of shape (samples, 224, 224, 3)). "
                    "Found array with shape: " + str(X.shape)
                )

        # Single thread
        def send_request(url, data):
            y_pred_temp = np.zeros(len(self.labels))
            for attempt in range(5):
                try:
                    res = requests.post(url, json=data, timeout=10).json()['predictions']
                    for r in res:
                        y_pred_temp[self.labels.index(r['label'])] = r['probability']
                except Exception as e:
                    print('\nError:', e, 'Retrying...', attempt, '\n')
                    continue

                break

            return y_pred_temp

        y_preds = []
        y_index = []
        y_executors = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            for i, x in enumerate(X):
                # Load the input image and construct the payload for the request
                image = Image.fromarray(np.uint8(x))
                buff = BytesIO()
                image.save(buff, format="JPEG", subsampling=0, quality=100)

                data = {'file': base64.b64encode(buff.getvalue()).decode("utf-8")}
                y_executors[executor.submit(send_request, url=self.url, data=data)] = i

            for y_executor in concurrent.futures.as_completed(y_executors):
                y_index.append(y_executors[y_executor])
                y_preds.append(y_executor.result())

            y_preds = [y for _, y in sorted(zip(y_index, y_preds))]

        return np.array(y_preds)


    def predictO(self, X):
        """
        - X: numpy array of shape (N, H, W, C)
        """
        if isinstance(X, list):
            for x in X:
                assert len(x.shape) == 3, 'Expecting a 3D tensor'
        else:
            if len(X.shape) != 4 or X.shape[3] != 3:
                raise ValueError(
                    "`predict` expects "
                    "a batch of images "
                    "(i.e. a 4D array of shape (samples, 224, 224, 3)). "
                    "Found array with shape: " + str(X.shape)
                )

        y_preds = []
        try:
            for x in X:
                y_pred_temp = np.zeros(len(self.labels))
                # Load the input image and construct the payload for the request
                image = Image.fromarray(np.uint8(x))
                buff = BytesIO()
                image.save(buff, format="JPEG")

                data = {'file': base64.b64encode(buff.getvalue()).decode("utf-8")}

                for attempt in range(5):
                    try:
                        res = requests.post(self.url, json=data, timeout=10).json()['predictions']
                        for r in res:
                            y_pred_temp[self.labels.index(r['label'])] = r['probability']
                    except Exception as e:
                        print('\nError:', e, 'Retrying...', attempt, '\n')
                        continue

                    break

                y_preds.append(y_pred_temp)

        except Exception as e:
            print(e)

        return np.array(y_preds)

    def print(self, y):
        """
        Print the prediction result.
        """
        max_len = max([len(x) for x in self.labels])

        print()

        # Unsorted
        # for i in range(0, len(y)):
        #     print('{:<{w}s}{:.5f}'.format(self.labels[i], y[i], w=max_len+1))

        # Sorted
        for p, l in sorted(zip(y, self.labels), reverse=True):
            print('{:<{w}s}{:.5f}'.format(l, p, w=max_len+1))

    def get_class_name(self, i):
        """
        Get the class name from the prediction label 0-10.
        """
        return self.labels[i]


class DeepAPI_VGG16_Cifar10(DeepAPIBase):

    def __init__(self, url, concurrency=8):
        """
        - url: DeepAPI server URL
        """
        super().__init__(concurrency)

        url_parse = urlparse(url)
        self.url = urljoin(url_parse.scheme + '://' + url_parse.netloc, 'vgg16_cifar10')

        # cifar10 labels
        cifar10_labels = ['frog', 'deer', 'cat', 'bird', 'dog', 'truck', 'ship', 'airplane', 'horse', 'automobile']
        self.labels = cifar10_labels


class DeepAPI_ImageNet(DeepAPIBase):
    def __init__(self, concurrency=8):
        super().__init__(concurrency)

        # Load the Keras application labels 
        with urllib.request.urlopen("https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json") as l_url:
            imagenet_json = json.load(l_url)

            # imagenet_json['134'] = ["n02012849", "crane"],
            imagenet_json['517'] = ['n02012849', 'crane_bird']

            # imagenet_json['638'] = ['n03710637', 'maillot']
            imagenet_json['639'] = ['n03710721', 'maillot_tank_suit']

            imagenet_labels = [imagenet_json[str(k)][1] for k in range(len(imagenet_json))]

            self.labels = imagenet_labels


class DeepAPI_Inceptionv3_ImageNet(DeepAPI_ImageNet):
    def __init__(self, url, concurrency=8):
        """
        - url: DeepAPI server URL
        """
        super().__init__(concurrency)

        url_parse = urlparse(url)
        self.url = urljoin(url_parse.scheme + '://' + url_parse.netloc, 'inceptionv3')


class DeepAPI_Resnet50_ImageNet(DeepAPI_ImageNet):
    def __init__(self, url, concurrency=8):
        """
        - url: DeepAPI server URL
        """
        super().__init__(concurrency)

        url_parse = urlparse(url)
        self.url = urljoin(url_parse.scheme + '://' + url_parse.netloc, 'resnet50')


class DeepAPI_VGG16_ImageNet(DeepAPI_ImageNet):
    def __init__(self, url, concurrency=8):
        """
        - url: DeepAPI server URL
        """
        super().__init__(concurrency)

        url_parse = urlparse(url)
        self.url = urljoin(url_parse.scheme + '://' + url_parse.netloc, 'vgg16')


bat_deepapi_model_list = {
    1: ['vgg16_cifar10', DeepAPI_VGG16_Cifar10],
    2: ['vgg16_imagenet', DeepAPI_VGG16_ImageNet],
    3: ['resnet50_imagenet', DeepAPI_Resnet50_ImageNet],
    4: ['inceptionv3_imagenet', DeepAPI_Inceptionv3_ImageNet]
}
