import cv2
import os
import json
import numpy as np
import argparse
import torch
from tqdm import tqdm

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

SPLIT = 'train'
#SPLIT = 'val'
#SPLIT = 'test'
#OBJ_DETECTION_FREQ = 0.5 # seconds
DETECTION_TH = 0.01

AVA_FOLDER = os.environ['ACAM_DIR'] + '/data/AVA'
SEGMENTS_FOLDER = AVA_FOLDER + '/segments/'
DATA_FOLDER = AVA_FOLDER + '/data/'
#SEGMENT_ANN_FILE = DATA_FOLDER + 'segment_annotations_%s.json' % SPLIT
SEGMENT_ANN_FILE = DATA_FOLDER + 'segment_annotations_v22_%s.json' % SPLIT

# OBJECT_DETECTIONS_FOLDER = AVA_FOLDER + '/objects/' + SPLIT + '/'
#OBJECT_DETECTIONS_FOLDER = AVA_FOLDER + '/objects_mrcnn/' + SPLIT + '/'
OBJECT_DETECTIONS_FOLDER = AVA_FOLDER + '/objects_finetuned_mrcnn/' + SPLIT + '/'
if not os.path.exists(OBJECT_DETECTIONS_FOLDER):
    os.makedirs(OBJECT_DETECTIONS_FOLDER)

#BATCH_SIZE = 2

    
def read_keyframe(segment_key):
    movie_name, timestamp = segment_key.split('.')
    keyframe_path = os.path.join(SEGMENTS_FOLDER, SPLIT, 'midframes', movie_name, '%s.jpg' %timestamp)

    keyframe_img = cv2.imread(keyframe_path)
    return keyframe_img

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--total_no_sets', type=int, required=True)
    parser.add_argument('-c', '--current_set', type=int, required=True)
    parser.add_argument('-g', '--gpu', type=str, required=True)
    # NO_GPUS = 4
    # CUR_GPU = 0 # zero based
    #parser.add_argument('-g', '--gpu', type=str, required=True)

    args = parser.parse_args()

    gpu = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    total_no_sets = args.total_no_sets
    current_set = args.current_set

    print('SET no %i (0 based) of %i SETS'%(current_set, total_no_sets))

    with open(SEGMENT_ANN_FILE) as fp:
        annotations = json.load(fp)

    segment_keys = annotations.keys()
    segment_keys.sort()
    # -5KQ66BBWC4.0902


    movie_timestamp_mapping = {}
    #for segment_key in current_segment_keys:
    for segment_key in segment_keys:
        movie_name, timestamp = segment_key.split('.')
        if movie_name in movie_timestamp_mapping.keys():
            movie_timestamp_mapping[movie_name].append(segment_key)
        else:
            movie_timestamp_mapping[movie_name] = [segment_key]

    movies = movie_timestamp_mapping.keys()
    movies.sort()

    no_segments = len(movies)
    vid_per_set = no_segments / total_no_sets
    start_idx = current_set * vid_per_set
    start_idx = np.floor(start_idx).astype(int)
    end_idx = (current_set+1) * vid_per_set
    end_idx = no_segments if (current_set+1 == total_no_sets) else np.ceil(end_idx).astype(int)

    print('Working on movies: [%i - %i)' % (start_idx, end_idx))
    cur_movies = movies[start_idx:end_idx]


    config_file = "../configs/e2e_faster_rcnn_X_101_32x8d_FPN_1x.yaml"
    
    # update the config options with the config file
    cfg.merge_from_file(config_file)
    #cfg.merge_from_list(["MODEL.WEIGHT", "e2e_faster_rcnn_X_101_32x8d_FPN_1x.pth"])
    cfg.merge_from_list(["MODEL.WEIGHT", "faster_rcnn_ava_model_0255000.pth"])
    
    coco_demo = COCODemo(
        cfg,
        min_image_size=800,
        confidence_threshold=DETECTION_TH,
    )



    for mm, movie in enumerate(tqdm(cur_movies)):
        print('\n\n Working on %s, %i/%i \n\n' % (movie, mm, len(cur_movies)))
        for segment_key in movie_timestamp_mapping[movie]:
            midframe = read_keyframe(segment_key)
            pred_boxes = coco_demo.compute_prediction(midframe)

            H,W,C = midframe.shape

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

            

            movie_name, timestamp = segment_key.split('.')
            #cur_detections = object_detections

            results_dict= { 'movie_name':movie_name,
                            'timestamp':timestamp,
                            'detections':segment_detections,
                            'height':H, 'width':W,}

            save_results_json(results_dict)
            #print('Timestamp done : %s' %timestamp)
            tqdm.write('Timestamp done : %s' %timestamp)

        #print('\n\nMovie done %s\n\n' % movie)
        tqdm.write('\n\nMovie done %s\n\n' % movie)





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
    movie_name, timestamp = combined_results['movie_name'], combined_results['timestamp']
    movie_path = os.path.join(OBJECT_DETECTIONS_FOLDER, movie_name)
    if not os.path.exists(movie_path):
        os.mkdir(movie_path)
    json_path = os.path.join(movie_path, '%s.json' %timestamp)

    with open(json_path, 'w') as fp:
        json.dump(combined_results, fp)

def get_3_decimal_float(infloat):
    return float('%.3f' % infloat)



if __name__ == '__main__':
    # _test_fps_of_segments()
    main()
