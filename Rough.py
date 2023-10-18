import cv2 as cv
from anomalib.deploy import OpenVINOInferencer

img_path = r'anomalib\src\anomalib\models\patchcore\patchcore_data2_test\oring_absent\52.png'
img = cv.imread(img_path)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
model = OpenVINOInferencer(path=r'results\patchcore\OringClassification\run\weights\onnx\model.onnx', metadata=r'results\patchcore\OringClassification\run\weights\onnx\metadata.json', task='segmentation', device='GPU')
img = model.pre_process(img)
img = model.forward(img)
img = model.post_process(img)
print(img['anomaly_map'].shape, img['anomaly_map'].min(), img['anomaly_map'].max())
print(img['pred_label'])
print(img['pred_score'])
print(img['pred_mask'].shape, img['pred_mask'].min(), img['pred_mask'].max())
print(img['pred_boxes'])
print(img['box_labels'])

cv.imshow('ANOMALY MAP', img['anomaly_map'] * 255)
cv.imshow('PRED MASK', img['pred_mask'] * 255)
if cv.waitKey(0) == 27:
    cv.destroyAllWindows()