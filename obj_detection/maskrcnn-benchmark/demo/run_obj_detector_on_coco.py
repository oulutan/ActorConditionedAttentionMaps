from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

import os
idx_no = 2
GPUS = ['0', '2', '3']
#os.environ['CUDA_VISIBLE_DEVICES']=GPUS[idx_no]
os.environ['CUDA_VISIBLE_DEVICES']="1"


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
    confidence_threshold=0.50,
)
# load image and then run prediction
import cv2
#image = cv2.imread('/home/oytun/work/ActorConditionedAttentionMaps/data/AVA/segments/val/midframes/1j20qq1JyX4/1798.jpg')
#image = cv2.imread('/home/oytun/work/maskrcnn_benchmark/maskrcnn-benchmark/demo/test_images/COCO_train2014_000000035110.jpg')
#predictions = coco_demo.run_on_opencv_image(image)
#cv2.imshow('preds',predictions)
#cv2.waitKey(0)

#predictions = coco_demo.compute_prediction(image)
#scores = predictions.get_field("scores").tolist()
#labels = predictions.get_field("labels").tolist()
#labels = [self.CATEGORIES[i] for i in labels]
#boxes = predictions.bbox

from tqdm import tqdm
import json
CATEGORIES = [
        "__background",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]

def get_3_decimal_float(infloat):
    return float("%.3f" % infloat)

#split = "train"
split = "val"

images_path = "./vader/%s2014/" % split

all_files = os.listdir(images_path)
all_images = [f for f in all_files if f.endswith('.jpg')]
all_images.sort()

# split into 3
n = len(all_images)
n1 = n//3
n2 = 2 * n // 3
#start, end = 0, n1
#start, end = n1, n2
#start, end = n2, n
#start, end = [(0,n1), (n1,n2), (n2,n)][idx_no]
start, end = 0, n

print("working on %i - %i" % (start, end))
all_images = all_images[start:end]

for image_name in tqdm(all_images):
    image_full_path = images_path + image_name
    cur_img = cv2.imread(image_full_path)
    H,W,C = cur_img.shape

    predictions = coco_demo.compute_prediction(cur_img)
    scores = predictions.get_field("scores").tolist()
    labels = predictions.get_field("labels").tolist()
    #labels = [self.CATEGORIES[i] for i in labels]
    boxes = predictions.bbox.tolist()
    
    cur_results = {'image_name':image_name, 'H':H, 'W':W, 'detections':[]}
    for ii in range(len(boxes)):
        #box_coords = boxes[0,ii] # hard code batch dimension to 1
        #score = scores[0,ii]
        #class_no = int(classes[0,ii])
        left, top, right, bottom = boxes[ii]
        box_coords = [top, left, bottom, right]
        score = scores[ii]
        class_no = int(labels[ii])

        box_coords = [get_3_decimal_float(c) for c in box_coords]
        score = get_3_decimal_float(score)
        class_str = CATEGORIES[class_no]
        cur_box = {'class_no':class_no, 'score':score, 'class_str':class_str, 'box_coords':box_coords}
        cur_results['detections'].append(cur_box)


    image_name_no_extension = image_name.split('.')[0]
    output_path = "./vader/new_object_results/%s/%s.json" % (split, image_name_no_extension)

    with open(output_path, 'w') as fp:
        json.dump(cur_results, fp)

    tqdm.write("%s is written!" % output_path)

