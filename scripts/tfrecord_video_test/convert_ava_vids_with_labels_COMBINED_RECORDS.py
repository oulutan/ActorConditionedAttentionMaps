import tensorflow as tf
import imageio
import numpy as np
import cv2
import os
import json
#from tqdm import tqdm
from joblib import Parallel, delayed

def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _bytes_list_feature(values):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


SPLIT = 'train'
input_dir = "/home/ulutan/work/ActorConditionedAttentionMaps/data/AVA/segments/%s/clips/" % SPLIT
output_dir = "/data/ulutan/AVA/tfrecords_labels/%s_tfrecords/" % SPLIT
#input_dir = "/home/ulutan/work/ActorConditionedAttentionMaps/data/AVA/segments/train/clips/"
#output_dir = "/home/ulutan/work/train_tfrecords/"
#input_dir = "/home/ulutan/work/ActorConditionedAttentionMaps/data/AVA/segments/val/clips/"
#input_dir = "/home/ulutan/work/ActorConditionedAttentionMaps/data/AVA/segments/test/clips/"
#output_dir = "/home/ulutan/work/val_tfrecords/"
#output_dir = "/data/ulutan/AVA/tfrecords/val_tfrecords/"
#output_dir = "/data/ulutan/AVA/tfrecords/test_tfrecords/"

MAX_ROIS = 50
NUM_CLASSES = 60
ACAM_FOLDER = os.environ['ACAM_DIR']

AVA_FOLDER = ACAM_FOLDER + '/data/AVA' 
DATA_FOLDER = AVA_FOLDER + '/data/'



with open(DATA_FOLDER+'label_conversions.json') as fp:
    LABEL_CONVERSIONS = json.load(fp)
 
ANN2TRAIN = LABEL_CONVERSIONS['ann2train']
TRAIN2ANN = LABEL_CONVERSIONS['train2ann']
 
#_split = 'train'
#annotations_path = DATA_FOLDER + 'segment_annotations_v22_%s.json' % _split
#with open(annotations_path) as fp:
#    ANNOS_TRAIN = json.load(fp)
# 
#_split = 'val'
#annotations_path = DATA_FOLDER + 'segment_annotations_v22_%s.json' % _split
#with open(annotations_path) as fp:
#    ANNOS_VAL = json.load(fp)
# 
#
#_split = 'test'
#annotations_path = DATA_FOLDER + 'segment_annotations_v22_%s.json' % _split
#with open(annotations_path) as fp:
#    ANNOS_TEST = json.load(fp)
ann_path = DATA_FOLDER + 'segment_annotations_v22_%s.json' % SPLIT
with open(ann_path) as fp:
    ANNOS = json.load(fp)

ALL_KEYS = ANNOS.keys()
MOVIE_SET = set([key.split('.')[0] for key in ALL_KEYS])


movies = os.listdir(input_dir)
movies.sort()
#for movie in tqdm(movies):



def get_labels_wrapper(movie, segment): #, split):
    segment_key = "%s.%s" % (movie, segment)
    #split = check_split(segment_key)
    split = SPLIT

    labels_np, rois_np, no_det = get_labels(segment_key,split)
    return labels_np, rois_np, no_det, segment_key

def get_labels(segment_key,split):
    # sample_annotations = sample_info['annotations']
    # segment_key = sample_info['segment_key']
    #if split == 'train': annos = ANNOS_TRAIN
    #elif split == 'val': annos = ANNOS_VAL
    #elif split == 'test': annos = ANNOS_TEST
    #else: raise KeyError
    annos = ANNOS
     
    sample_annotations = annos[segment_key]
     
    
    # if not USE_TRACKER_TEMPORAL_ROIS:
    #     detections = get_obj_detection_results(segment_key,split)
    #     labels_np, rois_np, no_det = match_annos_with_detections(sample_annotations, detections, split)
 
    # if USE_TRACKER_TEMPORAL_ROIS:
    #     obj_detections = get_obj_detection_results(segment_key,split) # I need this because tracker bboxes are in pixel values and height and width are saved in object detections
    #     keyframe_detections, rois_np, no_det = get_tracker_rois(segment_key,split, obj_detections)
    #     labels_np, _, _ = match_annos_with_detections(sample_annotations, keyframe_detections, split)

    detections, H, W = get_obj_detection_results(segment_key,split)
    
    labels_np, rois_np, no_det = match_annos_with_detections(sample_annotations, detections, split)
     
    return labels_np, rois_np, no_det



def get_obj_detection_results(segment_key,split):
    # split = sample_info['split']
    # segment_key = sample_info['segment_key']
 
    # KEY_FRAME_INDEX = 2
    movie_key, timestamp = segment_key.split('.')
    ## object detection results
    # obj_detects_folder = AVA_FOLDER + '/objects/' + split + '/'
    # finetuned
    obj_detects_folder = AVA_FOLDER + '/objects_finetuned_mrcnn/' + split + '/'
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
    if len(detections) > MAX_ROIS:
        # they are sorted by confidence already, take top #k
        detections = detections[:MAX_ROIS]
    # filter out detection confidences so that i can train efficiently
    #if split == 'train':
    #    #detections = [det for det in detections if det['score'] > 0.90]
    #    if len(detections) > MAX_ROIS_IN_TRAINING:
    #        # they are sorted by confidence already, take top #k
    #        detections = detections[:MAX_ROIS_IN_TRAINING]
    #else:
    #    #detections = [det for det in detections if det['score'] > 0.70]
    #    if len(detections) > MAX_ROIS:
    #        # they are sorted by confidence already, take top #k
    #        detections = detections[:MAX_ROIS]
        

    # just so I can use these in get_tracker
    #detections[0]['height'] = H 
    #detections[0]['width'] = W
    # detections[0]['frame_nos'] = results['frame_nos']
    return detections, H, W
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
 
    #if USE_GROUND_TRUTH_BOXES and split != 'test':
    #    # FOR DEBUGGING PURPOSES
    #    det_boxes = gt_boxes
 
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
 
                #if not USE_GROUND_TRUTH_BOXES:
                top, left, bottom, right = detections[dd]['box']
                #else:
                #    ## DEBUGGING
                #    left, top, right, bottom = annotations[dd]['bbox']
 
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




def generate_tfrecord(movie):
    if movie not in MOVIE_SET:
        return
    segments = os.listdir(input_dir+movie)
    segments.sort()
    #os.mkdir(output_dir + movie)
    for segment in segments:
        segment_no_ext = segment.split(".")[0]
        #output_file = os.path.join(output_dir, movie, "%s.tfrecord" % segment_no_ext)
        input_file = os.path.join(input_dir, movie, segment)
        #with tf.python_io.TFRecordWriter(output_file) as writer:
        if True:

def create_tf_example(segment_key):
	movie, segment_no_ext = segment_key.split('.')
	input_file = os.path.join(input_dir, movie, segment)
    # Read and resize all video frames, np.uint8 of size [N,H,W,3]
    video = imageio.get_reader(input_file)
    #frame_gen = video.iter_data()
    #frame_list = []
    #for frame in frame_gen:
    #    resized_frame = cv2.resize(frame, (400,400))
    #    frame_list.append(resized_frame)
      
    #frames = np.stack(frame_list, axis=0)
    vidinfo = video.get_meta_data()
    # vidfps = vidinfo['fps']
    # vid_W, vid_H = vidinfo['size']
    no_frames = vidinfo['nframes']-1 # last frames seem to be bugged
 
    f_timesteps, f_H, f_W, f_C = [32, 400, 400, 3]
 
    slope = (no_frames-1) / float(f_timesteps - 1)
    indices = (slope * np.arange(f_timesteps)).astype(np.int64)
 
    frames = np.zeros([f_timesteps, f_H, f_W, f_C], np.uint8)
 
    timestep = 0
    # for vid_idx in range(no_frames):
    for vid_idx in indices.tolist():
        frame = video.get_data(vid_idx)
        # frame = frame[:,:,::-1] #opencv reads bgr, i3d trained with rgb
        reshaped = cv2.resize(frame, (f_W, f_H))
        frames[timestep, :, :, :] = reshaped
        timestep += 1
 
    video.close()

    if "%s.%s" % (movie, segment_no_ext) not in ANNOS:
        continue

    # get labels
    labels_np, rois_np, no_det, segment_key = get_labels_wrapper(movie, segment_no_ext)


    
    features = {}
    features['num_frames']  = _int64_feature(frames.shape[0])
    features['height']      = _int64_feature(frames.shape[1])
    features['width']       = _int64_feature(frames.shape[2])
    features['channels']    = _int64_feature(frames.shape[3])
    #features['class_label'] = _int64_feature(example['class_id'])
    #features['class_text']  = _bytes_feature(tf.compat.as_bytes(example['class_label']))
    #features['filename']    = _bytes_feature(tf.compat.as_bytes(example['video_id']))
    #features['filename']    = _bytes_feature(tf.compat.as_bytes(fname))
    features['movie']    = _bytes_feature(tf.compat.as_bytes(movie))
    features['segment']    = _bytes_feature(tf.compat.as_bytes(segment_no_ext))
    
    # Compress the frames using JPG and store in as a list of strings in 'frames'
    encoded_frames = [tf.compat.as_bytes(cv2.imencode(".jpg", frame)[1].tobytes())
                      for frame in frames]
    features['frames'] = _bytes_list_feature(encoded_frames)
    
    #labels
    features['no_det'] = _int64_feature(no_det)
    features['segment_key'] = _bytes_feature(segment_key)
    features['labels'] = _int64_feature(labels_np.reshape([-1]).tolist())
    features['rois'] = _float_feature(rois_np.reshape([-1]).tolist())
    
    tfrecord_example = tf.train.Example(features=tf.train.Features(feature=features))
	return tfrecord_example
    #writer.write(tfrecord_example.SerializeToString())
    #tqdm.write("Output file %s written!" % output_file)
    #print("Output file %s written!" % output_file)

#Parallel(n_jobs=20)(delayed(generate_tfrecord)(movie) for movie in movies)
#for movie in movies:
#    generate_tfrecord(movie)
#output_file = 
#writer = tf.python_io.TFRecordWriter(output_file)

import contextlib2
from object_detection.dataset_tools import tf_record_creation_util

num_shards=10
output_filebase='/path/to/train_dataset.record'

with contextlib2.ExitStack() as tf_record_close_stack:
  output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
      tf_record_close_stack, output_filebase, num_shards)
  for index, example in examples:
    tf_example = create_tf_example(example)
    output_shard_index = index % num_shards
    output_tfrecords[output_shard_index].write(tf_example.SerializeToString())
