import io
from google.cloud import vision

import concurrent.futures

class CloudVision:
    def __init__(self, concurrency=1):
        self.concurrency = concurrency

    def predict(self, image_path):
        try:
            with io.open(image_path, 'rb') as image_file:
                content = image_file.read()

            image = vision.Image(content=content)

            client = vision.ImageAnnotatorClient()

            response = client.label_detection(image=image)
            labels = response.label_annotations

        except Exception as e:
            print(e)
            return

        return labels

    def predictX(self, image_paths):
        y_preds = []
        y_index = []
        y_executors = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            for i, image_path in enumerate(image_paths):
                # Load the input image and construct the payload for the request
                y_executors[executor.submit(self.predict, image_path)] = i

            for y_executor in concurrent.futures.as_completed(y_executors):
                y_index.append(y_executors[y_executor])
                y_preds.append(y_executor.result())

            y_preds = [y for _, y in sorted(zip(y_index, y_preds))]

        return y_preds

    def print(self, y_preds):
        max_mid_len = 0
        max_desc_len = 0

        for y in y_preds:
            if len(y.description) > max_desc_len:
                max_desc_len = len(y.description)
            if len(y.mid) > max_mid_len:
                max_mid_len = len(y.mid)

        for y in y_preds:
            print('{:<{w_id}s} {:<{w_desc}s} {:.5f}'.format(y.mid, y.description, y.score, w_id=max_mid_len, w_desc=max_desc_len))
