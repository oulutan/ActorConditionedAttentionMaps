from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

#config_file = "../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"
#config_file = "../configs/caffe2/e2e_faster_rcnn_X_101_32x8d_FPN_1x_caffe2.yaml"
#config_file = "../configs/e2e_mask_rcnn_X_101_32x8d_FPN_1x.yaml"
config_file = "../configs/e2e_faster_rcnn_X_101_32x8d_FPN_1x.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
#cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
cfg.merge_from_list(["MODEL.WEIGHT", "e2e_faster_rcnn_X_101_32x8d_FPN_1x.pth"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.70,
)
# load image and then run prediction
import cv2
#image = cv2.imread('/home/oytun/work/ActorConditionedAttentionMaps/data/AVA/segments/val/midframes/1j20qq1JyX4/1798.jpg')
image = cv2.imread('/home/oytun/work/maskrcnn_benchmark/maskrcnn-benchmark/demo/test_images/COCO_train2014_000000035110.jpg')
predictions = coco_demo.run_on_opencv_image(image)
cv2.imshow('preds',predictions)
cv2.waitKey(0)
pred_boxes = coco_demo.compute_prediction(image)
