import cv2
import os
import json
import numpy as np
import argparse
import torch
from tqdm import tqdm

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

IMAGES_FOLDER = "./images/"
OUTPUT_FOLDER = "./outputs/"
DETECTION_TH = 0.01
    

def main():
    parser = argparse.ArgumentParser()

    # total_no_sets and current_set is used for splitting the data and running multiple processes manually on different gpus or machines
    # if total no sets is 1 and then current_set can only be 0 and will run normally(default mode)
    # if total no sets is 2 lets say, then you have to run this script with current_set=0 and current_set=1 which will split the total data into 2 points and
    # allows running separately 
    parser.add_argument('-t', '--total_no_sets', type=int, required=False, default=1)
    parser.add_argument('-c', '--current_set', type=int, required=False, default=0)
    parser.add_argument('-g', '--gpu', type=str, required=False, default='0')
    # NO_GPUS = 4
    # CUR_GPU = 0 # zero based
    #parser.add_argument('-g', '--gpu', type=str, required=True)

    args = parser.parse_args()

    gpu = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    total_no_sets = args.total_no_sets
    current_set = args.current_set

    print('SET no %i (0 based) of %i SETS'%(current_set, total_no_sets))

    all_files = os.listdir(IMAGES_FOLDER)
    all_files.sort()

    no_segments = len(all_files)
    vid_per_set = no_segments / total_no_sets
    start_idx = current_set * vid_per_set
    start_idx = np.floor(start_idx).astype(int)
    end_idx = (current_set+1) * vid_per_set
    end_idx = no_segments if (current_set+1 == total_no_sets) else np.ceil(end_idx).astype(int)

    print('Working on images: [%i - %i)' % (start_idx, end_idx))
    cur_files = all_files[start_idx:end_idx]


    config_file = "../configs/e2e_faster_rcnn_X_101_32x8d_FPN_1x.yaml"
    
    # update the config options with the config file
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(["MODEL.WEIGHT", "e2e_faster_rcnn_X_101_32x8d_FPN_1x.pth"])
    #cfg.merge_from_list(["MODEL.WEIGHT", "faster_rcnn_ava_model_0255000.pth"])
    
    coco_demo = COCODemo(
        cfg,
        min_image_size=800,
        confidence_threshold=DETECTION_TH,
    )



    for mm, img_file in enumerate(tqdm(cur_files)):
        print('\n\n Working on %s, %i/%i \n\n' % (img_file, mm, len(cur_files)))
        img_path = os.path.join(IMAGES_FOLDER, img_file)
        img = cv2.imread(img_path)
        pred_boxes = coco_demo.compute_prediction(img)

        H,W,C = img.shape

        boxes = pred_boxes.bbox / torch.tensor([W, H, W, H], dtype=torch.float)
        box_list = boxes.tolist()
        scores = pred_boxes.get_field("scores")
        classes = pred_boxes.get_field("labels")
        num_boxes = len(box_list)
        # clean up organize
        segment_detections = []
        for bb in range(num_boxes):
            left, top, right, bottom = [get_3_decimal_float(coord) for coord in box_list[bb]] # xyxy : left top right bottom
            cur_box = [top, left, bottom, right] 

            cur_score = get_3_decimal_float(scores[bb])
            cur_class_no = int(classes[bb])
            cur_class_str = coco_demo.CATEGORIES[cur_class_no]
            cur_detection = {'box':cur_box, 'score':cur_score, 'class_str':cur_class_str, 'class_no':cur_class_no}
            segment_detections.append(cur_detection)

        

        #cur_detections = object_detections

        results_dict= { 'segment_name':img_file,
                        'detections':segment_detections,
                        'height':H, 'width':W,}

        save_results_json(results_dict)
        #print('Timestamp done : %s' %timestamp)

        #print('\n\nMovie done %s\n\n' % movie)
        tqdm.write('\n\nImage done %s\n\n' % img_file)





# def filter_detection_results(detection_list):
#     boxes_batch,scores_batch,classes_batch,num_detections_batch = detection_list
#     mini_batch_size = len(boxes_batch)
# 
#     object_detections_batch = []
#     for bb in range(mini_batch_size):
#         boxes,scores,classes,num_detections = boxes_batch[bb],scores_batch[bb],classes_batch[bb],num_detections_batch[bb]
#         cur_object_detections = []
#         for ii in range(len(boxes)):
#             score = get_3_decimal_float(scores[ii])
#             if score < DETECTION_TH:
#                 continue
#             box = boxes[ii]
#             box = [get_3_decimal_float(coord) for coord in box]
#             class_no = int(classes[ii])
#             class_str = object_detection_wrapper.get_object_name(class_no)
# 
#             detection = {'box':box, 'score':score, 'class_no':class_no, 'class_str':class_str}
# 
#             cur_object_detections.append(detection)
# 
#         object_detections_batch.append(cur_object_detections)
#     return object_detections_batch
# 
# def combine_info_with_detections(object_detections, info, H, W):
#     mini_batch_size = len(info)
# 
#     init_info = info[0]
#     detection_results = {   'movie_name':init_info['movie_name'],
#                             'timestamp':init_info['timestamp'],
#                             'detections':[],
#                             'frame_nos':[],
#                             'height':H, 'width':W,}
# 
#     for bb in range(mini_batch_size):
#         cur_object_detections = object_detections[bb]
#         cur_info = info[bb]
#         
#         detection_results['detections'].append(cur_object_detections)
#         detection_results['frame_nos'].append(cur_info['frame_no'])
#         
# 
#     return detection_results

def save_results_json(combined_results):
    img_file = combined_results['segment_name']
    json_path = os.path.join(OUTPUT_FOLDER, '%s.json' %img_file)

    with open(json_path, 'w') as fp:
        json.dump(combined_results, fp)

def get_3_decimal_float(infloat):
    return float('%.3f' % infloat)



if __name__ == '__main__':
    # _test_fps_of_segments()
    main()
