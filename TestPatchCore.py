import os
import cv2 as cv
import numpy as np
from anomalib.deploy import OpenVINOInferencer

class AnomalyDetector:
    def __init__(self, weights, metadata, task='segmentation', device='GPU'):
        self.weights = weights
        self.metadata = metadata
        self.task = task
        self.device = device
        self.model = OpenVINOInferencer(path=self.weights, metadata=self.metadata, task=self.task, device=self.device)

    def infer_from_image(self, image):
        predictions = self.model.pre_process(image)
        predictions = self.model.forward(predictions)
        predictions = self.model.post_process(predictions)
        return predictions

    def infer_from_image_path(self, image_path):
        predictions = cv.imread(image_path)
        predictions = cv.cvtColor(predictions, cv.COLOR_BGR2RGB)
        predictions = self.model.pre_process(predictions)
        predictions = self.model.forward(predictions)
        predictions = self.model.post_process(predictions)
        return predictions

def test_patchcore(img_path, weights, metadata, task, device='GPU'):
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    model = OpenVINOInferencer(path=weights, metadata=metadata, task=task, device=device)
    predictions = model.pre_process(img)
    predictions = model.forward(predictions)
    predictions = model.post_process(predictions)
    return predictions

WEIGHTS = r'results\patchcore\OringClassification\run\weights\onnx\model.onnx'
METADATA = r'results\patchcore\OringClassification\run\weights\onnx\metadata.json'
DEVICE = 'GPU'
TASK = 'segmentation'

FOLDER = r'anomalib\src\anomalib\models\patchcore\patchcore_data2_test'
YES_FOLDER = os.path.join(FOLDER, 'oring_present')
NO_FOLDER = os.path.join(FOLDER, 'oring_absent')
DESTINATION = os.path.join(FOLDER, 'patchcore_result')

os.makedirs(DESTINATION, exist_ok=True)
os.makedirs(os.path.join(DESTINATION, 'oring_present'), exist_ok=True)
os.makedirs(os.path.join(DESTINATION, 'oring_absent'), exist_ok=True)

model = AnomalyDetector(WEIGHTS, METADATA, TASK, DEVICE)

yes_img_paths = [os.path.join(YES_FOLDER, img_path) for img_path in sorted(os.listdir(YES_FOLDER))]
for img_path in yes_img_paths:
    result = model.infer_from_image_path(img_path)
    mask = result['pred_mask']
    mask = np.stack([mask, mask, mask], axis=-1) * 255
    cv.imwrite(os.path.join(DESTINATION, 'oring_present', os.path.basename(img_path)), mask)

no_img_paths = [os.path.join(NO_FOLDER, img_path) for img_path in sorted(os.listdir(NO_FOLDER))]
for img_path in no_img_paths:
    result = model.infer_from_image_path(img_path)
    mask = result['pred_mask']
    mask = np.stack([mask, mask, mask], axis=-1) * 255
    cv.imwrite(os.path.join(DESTINATION, 'oring_absent', os.path.basename(img_path)), mask)

FOLDER = 'Crops'
cm = np.zeros(shape=(2, 2))
for oring in ['YES', 'NO']:
    images = [os.path.join(FOLDER, oring, i) for i in sorted(os.listdir(os.path.join(FOLDER, oring)))[:50]]
    for img_path in images:
        anomaly = test_patchcore(img_path, WEIGHTS, METADATA, TASK, MODE)
        if anomaly and oring == 'NO':
            cm[0, 0] = cm[0, 0] + 1
        elif not anomaly and oring == 'YES':
            cm[1, 1] = cm[1, 1] + 1
        elif anomaly and oring == 'YES':
            cm[1, 0] = cm[1, 0] + 1
        else:
            cm[0, 1] = cm[0, 1] + 1
print(cm)
