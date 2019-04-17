import tensorflow as tf
import cv2
import os
import json
import numpy as np
import argparse

import object_detection_wrapper


#OBJ_DETECT_PATH = os.environ['AVA_DIR'] + '/tensorflow_object/'
OBJ_DETECT_PATH = '/home/oytun/work/tensorflow_object/'
## Faster
# OBJ_DETECT_GRAPH_PATH = OBJ_DETECT_PATH + '/zoo/batched_zoo/faster_rcnn_nas_lowproposals_coco_2018_01_28/batched_graph/frozen_inference_graph.pb'
## Better
# OBJ_DETECT_GRAPH_PATH = OBJ_DETECT_PATH + '/zoo/batched_zoo/faster_rcnn_nas_coco_2018_01_28/batched_graph/frozen_inference_graph.pb'
## Different obj detector
OBJ_DETECT_GRAPH_PATH = OBJ_DETECT_PATH + '/zoo/batched_zoo/faster_rcnn_resnet101_coco_2018_01_28_lowth/batched_graph/frozen_inference_graph.pb'
# SPLIT = 'train'
SPLIT = 'val'
# SPLIT = 'test'
OBJ_DETECTION_FREQ = 0.5 # seconds
DETECTION_TH = 0.01

AVA_FOLDER = os.environ['AVA_DIR'] + '/AVA'
SEGMENTS_FOLDER = AVA_FOLDER + '/segments/'
DATA_FOLDER = AVA_FOLDER + '/data/'
SEGMENT_ANN_FILE = DATA_FOLDER + 'segment_annotations_%s.json' % SPLIT

# OBJECT_DETECTIONS_FOLDER = AVA_FOLDER + '/objects/' + SPLIT + '/'
OBJECT_DETECTIONS_FOLDER = AVA_FOLDER + '/objects_faster_rcnn_resnet_coco/' + SPLIT + '/'
if not os.path.exists(OBJECT_DETECTIONS_FOLDER):
    os.makedirs(OBJECT_DETECTIONS_FOLDER)

# BATCH_SIZE = 2
BATCH_SIZE = 4


# def get_detection_frames(segment_key):
#     movie_name, timestamp = segment_key.split('.')
#     video_path = os.path.join(SEGMENTS_FOLDER, SPLIT, 'clips', movie_name, '%s.mp4' %timestamp)

#     vcap = cv2.VideoCapture(video_path)

#     # Video Properties
#     vidfps = vcap.get(cv2.CAP_PROP_FPS)

#     W = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
#     H = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float

#     T_frames = np.ceil(vidfps * OBJ_DETECTION_FREQ)

#     returned_count = 0
#     frame_count = 0
#     while vcap.isOpened():
#         ret, frame = vcap.read()
#         # resize for speed
#         # frame = cv2.resize(frame,None,fx=0.5, fy=0.5)
#         if not ret:
#             break

#         # only yield frames to run detections on
#         if frame_count % T_frames == 0:
#             yield {'frame': frame, 'movie_name':movie_name, 'timestamp':timestamp, 'detection_frame_no':returned_count, 'frame_no':frame_count}
#             returned_count += 1

#         frame_count += 1

# def get_batch(segment_list):
#     for segment in segment_list:
#         frame_gen = get_detection_frames(segment)
#         batch = []
#         batch_info = []

#         for frame_info in frame_gen:
#             batch.append(frame_info['frame'])
#             del frame_info['frame']
#             batch_info.append(frame_info)

#         # import pdb;pdb.set_trace()
#         batch_np = np.stack(batch)
#         yield {'batch': batch_np, 'info': batch_info}

    
def read_keyframe(segment_key):
    movie_name, timestamp = segment_key.split('.')
    keyframe_path = os.path.join(SEGMENTS_FOLDER, SPLIT, 'midframes', movie_name, '%s.jpg' %timestamp)

    keyframe_img = cv2.imread(keyframe_path)
    return keyframe_img

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--total_no_sets', type=int, required=True)
    parser.add_argument('-c', '--current_set', type=int, required=True)
    # NO_GPUS = 4
    # CUR_GPU = 0 # zero based
    #parser.add_argument('-g', '--gpu', type=str, required=True)

    args = parser.parse_args()

    #gpu = args.gpu
    #os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    total_no_sets = args.total_no_sets
    current_set = args.current_set

    print('SET no %i (0 based) of %i SETS'%(current_set, total_no_sets))

    with open(SEGMENT_ANN_FILE) as fp:
        annotations = json.load(fp)

    segment_keys = annotations.keys()
    segment_keys.sort()
    # -5KQ66BBWC4.0902

    # segment_keys = ["z-fsLpGHq6o.1415",
    # 				"z-fsLpGHq6o.1416",
    # 				"b-YoBU0XT90.1455",
    # 				"b-YoBU0XT90.1448",
    # 				"b-YoBU0XT90.1449",
    # 				"b-YoBU0XT90.1450",
    # 				"b-YoBU0XT90.1447",
    # 				"b-YoBU0XT90.1451",
    # 				"KHHgQ_Pe4cI.1047",
    # 				"KHHgQ_Pe4cI.1046",
    # 				"z-fsLpGHq6o.1413",
    # 				"z-fsLpGHq6o.1417",
    # 				"z-fsLpGHq6o.1414",
    # 				"KHHgQ_Pe4cI.1045",
    # 				"KHHgQ_Pe4cI.1044",
    # 				"KHHgQ_Pe4cI.1048"]

    # current_segment_keys = current_segment_keys[len(current_segment_keys)//2:]##debug
    # current_segment_keys = current_segment_keys[1256:2257]

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


    
    detection_graph = object_detection_wrapper.generate_graph(OBJ_DETECT_GRAPH_PATH)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.Session(graph=detection_graph, config=config)


    for mm, movie in enumerate(cur_movies):
        print('\n\n Working on %s, %i/%i \n\n' % (movie, mm, len(cur_movies)))
        key_gen = iter(movie_timestamp_mapping[movie])
        done = False
        while not done:
            batch_keys = []
            for _ in range(BATCH_SIZE):
                try:
                    segment_key = key_gen.next()
                    batch_keys.append(segment_key)   
                except StopIteration:
                    done = True
                    break
            
            if batch_keys:
                batch = [np.expand_dims(read_keyframe(batch_key), axis=0) for batch_key in batch_keys]
                batch_np = np.concatenate(batch, axis=0)
                B, H, W, C = batch_np.shape

                detection_list = object_detection_wrapper.get_detections(detection_graph, sess, batch_np)
                object_detections = filter_detection_results(detection_list)
                
                # process the information
                for ii in range(len(batch_keys)):
                    cur_key = batch_keys[ii]
                    movie_name, timestamp = cur_key.split('.')
                    cur_detections = object_detections[ii]

                    results_dict= { 'movie_name':movie_name,
                                    'timestamp':timestamp,
                                    'detections':cur_detections,
                                    'height':H, 'width':W,}

                    save_results_json(results_dict)
                    print('Timestamp done : %s' %timestamp)

        print('\n\nMovie done %s\n\n' % movie)



def filter_detection_results(detection_list):
    boxes_batch,scores_batch,classes_batch,num_detections_batch = detection_list
    mini_batch_size = len(boxes_batch)

    object_detections_batch = []
    for bb in range(mini_batch_size):
        boxes,scores,classes,num_detections = boxes_batch[bb],scores_batch[bb],classes_batch[bb],num_detections_batch[bb]
        cur_object_detections = []
        for ii in range(len(boxes)):
            score = get_3_decimal_float(scores[ii])
            if score < DETECTION_TH:
                continue
            box = boxes[ii]
            box = [get_3_decimal_float(coord) for coord in box]
            class_no = int(classes[ii])
            class_str = object_detection_wrapper.get_object_name(class_no)

            detection = {'box':box, 'score':score, 'class_no':class_no, 'class_str':class_str}

            cur_object_detections.append(detection)

        object_detections_batch.append(cur_object_detections)
    return object_detections_batch

def combine_info_with_detections(object_detections, info, H, W):
    mini_batch_size = len(info)

    init_info = info[0]
    detection_results = {   'movie_name':init_info['movie_name'],
                            'timestamp':init_info['timestamp'],
                            'detections':[],
                            'frame_nos':[],
                            'height':H, 'width':W,}

    for bb in range(mini_batch_size):
        cur_object_detections = object_detections[bb]
        cur_info = info[bb]
        
        detection_results['detections'].append(cur_object_detections)
        detection_results['frame_nos'].append(cur_info['frame_no'])
        

    return detection_results

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


def _test_fps_of_segments():
    with open(SEGMENT_ANN_FILE) as fp:
        annotations = json.load(fp)

    segment_keys = annotations.keys()
    segment_keys.sort()
    # -5KQ66BBWC4.0902

    summary = []
    prev_movie_name = ''
    for segment_key in segment_keys:
        movie_name, timestamp = segment_key.split('.')
        if movie_name == prev_movie_name:
            continue
        video_path = os.path.join(SEGMENTS_FOLDER, SPLIT, 'clips', movie_name, '%s.mp4' %timestamp)
        vcap = cv2.VideoCapture(video_path)

        # Video Properties
        vidfps = vcap.get(cv2.CAP_PROP_FPS)

        W = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        H = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float

        T_frames = np.ceil(vidfps * OBJ_DETECTION_FREQ)
        
        returned_count = 0
        frame_count = 0
        while vcap.isOpened():
            ret, frame = vcap.read()
            # resize for speed
            # frame = cv2.resize(frame,None,fx=0.5, fy=0.5)
            if not ret:
                break

            # only yield frames to run detections on
            if frame_count % T_frames == 0:
                # yield {'frame': frame, 'movie_name':movie_name, 'timestamp':timestamp, 'count':returned_count}
                returned_count += 1

            frame_count += 1

        message = '%s - %.2f - %i x %i - %i frames, midframe %i, returned %i, total %i' % (segment_key, vidfps, H, W, T_frames, 2*T_frames, returned_count, frame_count)
        print(message)

        summary.append(message)


        prev_movie_name = movie_name

    summary_file = DATA_FOLDER+'video_summary_%s.txt'%SPLIT
    with open(summary_file, 'w' ) as fp:
        fp.write('\n'.join(summary))
    print('Summary file %s written!' % summary_file)


if __name__ == '__main__':
    # _test_fps_of_segments()
    main()
