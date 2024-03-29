from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import cv2

#config_file = "e2e_faster_rcnn_X_101_32x8d_FPN_1x_ava.yaml"
config_file = "../configs/e2e_faster_rcnn_X_101_32x8d_FPN_1x.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
#cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
cfg.merge_from_list(["MODEL.WEIGHT", "e2e_faster_rcnn_X_101_32x8d_FPN_1x.pth"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)
# load image and then run prediction
img_name = "KVq6If6ozMY.1360.jpg"
image = cv2.imread(img_name)
predictions = coco_demo.run_on_opencv_image(image)
cv2.imwrite('output.jpg', predictions)
cv2.imshow('Predictions', predictions)
cv2.waitKey(0)
