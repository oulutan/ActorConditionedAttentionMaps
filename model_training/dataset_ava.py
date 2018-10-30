import os

# import __main__ as main
# if os.path.basename(main.__file__) == 'result_validation.py' or os.path.basename(main.__file__) == 'visualize_ava_style_results.py':
#     print('Called from %s, not importing tensorflow' % os.path.basename(main.__file__))
# else:
#     import tensorflow as tf

import numpy as np
import json
import cv2
import imageio

import logging
import subprocess

# import models.extra_layers as extra_layers
 
INPUT_H = 400
INPUT_W = 400
INPUT_T = 32
# INPUT_T = 16
 
KEY_FRAME_INDEX = 2
 
NUM_CLASSES = 60
 
import socket
HOSTNAME = socket.gethostname()
 
# AVA_FOLDER = '/media/sidious_data/AVA'
ACAM_FOLDER = os.environ['ACAM_DIR']

AVA_FOLDER = ACAM_FOLDER + '/data/AVA' 
SEGMENTS_FOLDER = AVA_FOLDER + '/segments/'
DATA_FOLDER = AVA_FOLDER + '/data/'

# AVA_FOLDER = ACAM_FOLDER + '/data/AVA' 

MODEL_SAVER_PATH = AVA_FOLDER + '/ckpts/model_ckpt'
RESULT_SAVE_PATH = AVA_FOLDER + '/ActionResults/'
 
# max amount of rois in a single image
# this initializes the roi vector sizes as well
MAX_ROIS = 100
# TEMP_RESOLUTION = 32

USE_GROUND_TRUTH_BOXES = False

MAX_ROIS_IN_TRAINING = 20
 
# USE_TRACKER_TEMPORAL_ROIS = False
 
 
# data is also available on skywalker
# if HOSTNAME == 'skywalker':
#     SEGMENTS_FOLDER = '/media/ssd1/oytun/data/AVA/segments/'
#     print('Reading from skywalker')
 
# if HOSTNAME == 'vader':
#     SEGMENTS_FOLDER = '/media/ssd1/oytun/AVA/segments/'
#     print('Reading from vader')
 
 
with open(DATA_FOLDER+'label_conversions.json') as fp:
    LABEL_CONVERSIONS = json.load(fp)
 
ANN2TRAIN = LABEL_CONVERSIONS['ann2train']
TRAIN2ANN = LABEL_CONVERSIONS['train2ann']
 
_split = 'train'
annotations_path = DATA_FOLDER + 'segment_annotations_%s.json' % _split
with open(annotations_path) as fp:
    ANNOS_TRAIN = json.load(fp)
 
_split = 'val'
annotations_path = DATA_FOLDER + 'segment_annotations_%s.json' % _split
with open(annotations_path) as fp:
    ANNOS_VAL = json.load(fp)
 

_split = 'test'
annotations_path = DATA_FOLDER + 'segment_annotations_%s.json' % _split
with open(annotations_path) as fp:
    ANNOS_TEST = json.load(fp)

def get_train_list():
    # with open(DATA_FOLDER + 'segment_keys_train_detections_only_th_020.json') as fp:
    with open(DATA_FOLDER + 'segment_keys_train_detections_only.json') as fp:
        train_detection_segments = json.load(fp)
    return train_detection_segments
 
def get_val_list():
    with open(DATA_FOLDER + 'segment_keys_val_detections_only.json') as fp:
        val_detection_segments = json.load(fp)
    return val_detection_segments

def get_test_list():
    with open(DATA_FOLDER + 'segment_keys_test_detections_only.json') as fp:
        test_detection_segments = json.load(fp)
    return test_detection_segments

# Python Functions
def get_data(segment_key, split):
 
    sample = get_video_frames(segment_key,split)
    labels_np, rois_np, no_det = get_labels(segment_key,split)
 
    return sample, labels_np, rois_np, no_det, segment_key
 
# def get_video_frames(segment_key, split):
#     # Uses opencv, stupid opencv wont build with video support on the server
#     # split = sample_info['split']
#     # segment_key = sample_info['segment_key']
#     movie_key, timestamp = segment_key.split('.')
 
#     vid_path = os.path.join(SEGMENTS_FOLDER, split, 'clips', movie_key, '%s.mp4' % timestamp)
 
#     vcap = cv2.VideoCapture(vid_path)
 
#     vidfps = vcap.get(cv2.CAP_PROP_FPS)
#     vid_W = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
#     vid_H = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
#     no_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
 
#     f_timesteps, f_H, f_W, f_C = [INPUT_T, INPUT_H, INPUT_W, 3]
 
#     slope = (no_frames-1) / float(f_timesteps - 1)
#     indices = (slope * np.arange(f_timesteps)).astype(np.int64)
 
#     sample = np.zeros([f_timesteps, f_H, f_W, f_C], np.uint8)
 
#     timestep = 0
#     for vid_idx in range(no_frames):
#         if vid_idx not in indices:
#             ret, frame = vcap.read()
         
#         else:
#             ret, frame = vcap.read()
#             # bgr2rgb
#             frame = frame[:,:,::-1]
#             # reshaped = cv2.resize(frame, (f_H, f_W))
#             reshaped = cv2.resize(frame, (f_W, f_H))
#             # repeat the frame if necessary
#             for _ in range(indices.tolist().count(vid_idx)):
#                 sample[timestep, :, :, :] = reshaped
#                 timestep += 1
 
#     vcap.release()
#     return sample

def get_video_frames(segment_key, split):
    # Uses imageio - ffmpeg
    # split = sample_info['split']
    # segment_key = sample_info['segment_key']
    movie_key, timestamp = segment_key.split('.')
 
    vid_path = os.path.join(SEGMENTS_FOLDER, split, 'clips', movie_key, '%s.mp4' % timestamp)
 

    video = imageio.get_reader(vid_path, 'ffmpeg')
    # vcap = cv2.VideoCapture(vid_path)
 
    # vidfps = vcap.get(cv2.CAP_PROP_FPS)
    # vid_W = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    # vid_H = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
    # no_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
    vidinfo = video.get_meta_data()
    # vidfps = vidinfo['fps']
    # vid_W, vid_H = vidinfo['size']
    no_frames = vidinfo['nframes']-1 # last frames seem to be bugged
 
    f_timesteps, f_H, f_W, f_C = [INPUT_T, INPUT_H, INPUT_W, 3]
 
    slope = (no_frames-1) / float(f_timesteps - 1)
    indices = (slope * np.arange(f_timesteps)).astype(np.int64)
 
    sample = np.zeros([f_timesteps, f_H, f_W, f_C], np.uint8)
 
    timestep = 0
    # for vid_idx in range(no_frames):
    for vid_idx in indices.tolist():
        frame = video.get_data(vid_idx)
        # frame = frame[:,:,::-1] #opencv reads bgr, i3d trained with rgb
        reshaped = cv2.resize(frame, (f_W, f_H))
        sample[timestep, :, :, :] = reshaped
        timestep += 1
 
    video.close()
    return sample.astype(np.float32)
 
def get_labels(segment_key,split):
    # sample_annotations = sample_info['annotations']
    # segment_key = sample_info['segment_key']
    if split == 'train': annos = ANNOS_TRAIN
    elif split == 'val': annos = ANNOS_VAL
    elif split == 'test': annos = ANNOS_TEST
    else: raise KeyError
     
    sample_annotations = annos[segment_key]
     
    
    # if not USE_TRACKER_TEMPORAL_ROIS:
    #     detections = get_obj_detection_results(segment_key,split)
    #     labels_np, rois_np, no_det = match_annos_with_detections(sample_annotations, detections, split)
 
    # if USE_TRACKER_TEMPORAL_ROIS:
    #     obj_detections = get_obj_detection_results(segment_key,split) # I need this because tracker bboxes are in pixel values and height and width are saved in object detections
    #     keyframe_detections, rois_np, no_det = get_tracker_rois(segment_key,split, obj_detections)
    #     labels_np, _, _ = match_annos_with_detections(sample_annotations, keyframe_detections, split)

    detections = get_obj_detection_results(segment_key,split)
    labels_np, rois_np, no_det = match_annos_with_detections(sample_annotations, detections, split)
     
    return labels_np, rois_np, no_det
 
def get_obj_detection_results(segment_key,split):
    # split = sample_info['split']
    # segment_key = sample_info['segment_key']
 
    # KEY_FRAME_INDEX = 2
    movie_key, timestamp = segment_key.split('.')
    ## object detection results
    obj_detects_folder = AVA_FOLDER + '/objects/' + split + '/'
    ## ava detection results
    #obj_detects_folder = AVA_FOLDER + '/ava_trained_persons/' + split + '/object_detections/'
    segment_objects_path = os.path.join(obj_detects_folder, movie_key, '%s.json' %timestamp )
     
    with open(segment_objects_path) as fp:
        results = json.load(fp)
 
    detections = results['detections']
    H = results['height']
    W = results['width']
 
    # filter out non-person detections
    detections = [det for det in detections if det['class_str'] == 'person']
    # filter out detection confidences so that i can train efficiently
    if split == 'train':
        # detections = [det for det in detections if det['score'] > 0.20]
        if len(detections) > MAX_ROIS_IN_TRAINING:
            # they are sorted by confidence already, take top #k
            detections = detections[:MAX_ROIS_IN_TRAINING]

    # just so I can use these in get_tracker
    # detections[0]['height'] = H 
    # detections[0]['width'] = W
    # detections[0]['frame_nos'] = results['frame_nos']
    return detections
    # detections = [{"box": [0.07, 0.006, 0.981, 0.317], "class_str": "person", "score": 0.979, "class_no": 1},
 
 
MATCHING_IOU = 0.5
def match_annos_with_detections(annotations, detections, split):
    gt_boxes = []
    for ann in annotations:
        # left, top, right, bottom
        # [0.07, 0.141, 0.684, 1.0]
        cur_box = ann['bbox'] 
        gt_boxes.append(cur_box)
 
    det_boxes = []
    for detection in detections:
        # top, left, bottom, right
        # [0.07, 0.006, 0.981, 0.317]
        top, left, bottom, right = detection['box'] 
        box = left, top, right, bottom
        class_label = detection['class_str']
        det_boxes.append(box)
 
    if USE_GROUND_TRUTH_BOXES and split != 'test':
        # FOR DEBUGGING PURPOSES
        det_boxes = gt_boxes
 
    no_gt = len(gt_boxes)
    no_det = len(det_boxes)
 
    iou_mtx = np.zeros([no_gt, no_det])
 
    for gg in range(no_gt):
        gt_box = gt_boxes[gg]
        for dd in range(no_det):
            dt_box = det_boxes[dd]
            iou_mtx[gg,dd] = IoU_box(gt_box, dt_box)
 
    # assume less than #MAX_ROIS boxes in each image
    if no_det > MAX_ROIS: print('MORE DETECTIONS THAN MAX ROIS!!')
    labels_np = np.zeros([MAX_ROIS, NUM_CLASSES], np.int32)
    rois_np = np.zeros([MAX_ROIS, 4], np.float32) # the 0th index will be used as the featmap index
 
    # TODO if no_gt or no_det is 0 this will give error
    # This is fixed within functions calling this
    if split != 'test':
        if no_gt != 0 and no_det != 0:
            max_iou_for_each_det = np.max(iou_mtx, axis=0)
            index_for_each_det = np.argmax(iou_mtx, axis=0)
             
 
            for dd in range(no_det):
                cur_max_iou = max_iou_for_each_det[dd]
 
                if not USE_GROUND_TRUTH_BOXES:
                    top, left, bottom, right = detections[dd]['box']
                else:
                    ## DEBUGGING
                    left, top, right, bottom = annotations[dd]['bbox']
 
                # # region of interest layer expects
                # # regions of interest as lists of:
                # # feature map index, upper left, bottom right coordinates
                rois_np[dd,:] = [top, left, bottom, right]
                if cur_max_iou < MATCHING_IOU:
                    continue
                matched_ann = annotations[index_for_each_det[dd]]
                actions = matched_ann['actions'] # "actions": ["14", "17", "79"]
                benchmark_actions = [ANN2TRAIN[action]['train_id'] for action in actions if action in ANN2TRAIN.keys()]
                for b_action in benchmark_actions:
                    labels_np[dd,b_action] = 1 # could be multiple 1s
    else:
        for dd in range(no_det):
            top, left, bottom, right = detections[dd]['box']
            rois_np[dd,:] = [top, left, bottom, right]
 
    return labels_np, rois_np, no_det
 
 
def IoU_box(box1, box2):
    '''
    left1, top1, right1, bottom1 = box1
    left2, top2, right2, bottom2 = box2
 
    returns intersection over union
    '''
    left1, top1, right1, bottom1 = box1
    left2, top2, right2, bottom2 = box2
 
    left_int = max(left1, left2)
    top_int = max(top1, top2)
 
    right_int = min(right1, right2)
    bottom_int = min(bottom1, bottom2)
 
    areaIntersection = max(0, right_int - left_int) * max(0, bottom_int - top_int)
 
    area1 = (right1 - left1) * (bottom1 - top1)
    area2 = (right2 - left2) * (bottom2 - top2)
     
    IoU = areaIntersection / float(area1 + area2 - areaIntersection)
    return IoU
  

def process_evaluation_results( res_name):
    
    logging.info('Generating ava style results')
    subprocess.call(['python', ACAM_FOLDER + '/model_training/result_validation.py', '--result_name', res_name])
    
    logging.info('Calculating final AP values')
    subprocess.call(['bash', ACAM_FOLDER + '/evaluation/run_ava_detection.sh', res_name])
    
    logging.info('Done!')



# Testing functions
def _test_dataset_cropping():
    import tensorflow as tf
    split = 'train'
    # temp_rois = USE_TRACKER_TEMPORAL_ROIS
    annotations_path = DATA_FOLDER + 'segment_annotations_%s.json' % split
    with open(annotations_path) as fp:
        annotations = json.load(fp)
 
    segment_keys = annotations.keys()
    segment_keys.sort()
    # with open('segment_keys_%s_detections_only.json' % split) as fp:
    #     val_detection_segments = json.load(fp)
    # from tqdm import tqdm
    # segment_keys = val_detection_segments
     
    np.random.seed(5)
    np.random.shuffle(segment_keys)
 
    no_gpus = 5
    batch_size = 6
 
    output_types = [tf.uint8, tf.int32, tf.float32, tf.int64, tf.string]
    dataset = tf.data.Dataset.from_tensor_slices((segment_keys,[split]*len(segment_keys)))
    dataset = dataset.map(lambda seg_key, c_split: tuple(tf.py_func(get_data, [seg_key,c_split], output_types)), num_parallel_calls=8)
    dataset = dataset.batch(batch_size=batch_size*no_gpus)
    dataset = dataset.prefetch(buffer_size=50)
 
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
 
    rgb_seq = tf.cast(next_element[0], tf.float32)
    rgb_seq.set_shape([batch_size*no_gpus,32,400,400,3])
 
    _, labels, rois, no_dets,_ = next_element
 
    rois_nonzero, labels_nonzero, batch_indices_nonzero = combine_batch_rois(rois, labels)
 
    boxes = temporal_roi_cropping(rgb_seq, rois_nonzero, batch_indices_nonzero, [50,50])
 
 
 
    init_op = tf.global_variables_initializer()
 
    sess = tf.Session()
 
    sess.run(init_op)
 
    # test dataset iterator:
    for _ in range(100):
        sample, labels_np, rois_np, no_dets_np, segment_key, np_boxes, rois_nonzero_np = sess.run(list(next_element)+[boxes, rois_nonzero])
        np_boxes = np.uint8(np_boxes)
        _visualize(0, np_boxes)
        import pdb;pdb.set_trace()
 
def _visualize(index, np_boxes):
    images = np_boxes[index,:]
    images = images[::5,:]
    img_to_show = np.reshape(images, [-1,50,3])
    # cv2.imshow('rois', img_to_show)
    cv2.imwrite('rois.jpg', img_to_show)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
 

# I had to check for corrupted files after a Hard drive failure
def _find_error():
    split = 'test'
    with open('segment_keys_%s_detections_only.json' % split) as fp:
        val_detection_segments = json.load(fp)
    # from tqdm import tqdm
    segment_keys = val_detection_segments
 
    # segment_keys = ["WMFTBgYWJS8.0940", "WMFTBgYWJS8.0941", "WMFTBgYWJS8.0944", "WMFTBgYWJS8.0946", "WMFTBgYWJS8.0951", "WMFTBgYWJS8.0969"]
    from joblib import Parallel, delayed
    Parallel(n_jobs=40)(delayed(_exception_wrapper)(segment_key) for segment_key in segment_keys[53000:])
     
    # segment_keys = ["WMFTBgYWJS8.0940", "WMFTBgYWJS8.0941", "WMFTBgYWJS8.0944", "WMFTBgYWJS8.0946", "WMFTBgYWJS8.0951", "WMFTBgYWJS8.0969"]
    # for segment_key in segment_keys:
        # _exception_wrapper(segment_key)     
def _exception_wrapper(segment_key):
    try:
        print('Done with %s' % segment_key)
        sample, labels_np, rois_np, no_det, segment_key = get_data(segment_key, 'test')
        if np.all(sample == 0):
            print('\t\tError with %s' % segment_key)    
    except TypeError:
        print('Error with %s' % segment_key)
        # import pdb;pdb.set_trace()
         
 
if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    _test_dataset_cropping()
    # _find_error()
