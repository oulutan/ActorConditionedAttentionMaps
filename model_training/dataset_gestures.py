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
import tensorflow as tf
import random
 
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# import models.extra_layers as extra_layers


## During training randomly select a midframe and just use its bbox
# during eval, evaluate for every frame in the video. 

INPUT_H = 400
INPUT_W = 400
INPUT_T = 32
# INPUT_T = 16
 
KEY_FRAME_INDEX = 2
 
NUM_CLASSES = 8 # 1 for background

ACAM_FOLDER = os.environ['ACAM_DIR']

GESTURES_FOLDER = ACAM_FOLDER + '/data/GESTURES' 

HUMANS_FOLDER = GESTURES_FOLDER + '/HumanSegments/'
SYNTH_FOLDER = GESTURES_FOLDER + '/SyntheticData/'
OBJECT_DETECTIONS_FOLDER = GESTURES_FOLDER + '/object_detections/'
OBJECT_DETECTIONS_SYNTH_FOLDER = GESTURES_FOLDER + '/object_detections_synth/'
RESULTS_FOLDER = GESTURES_FOLDER + '/ActionResults/'
DATA_FOLDER = GESTURES_FOLDER + '/data/'


MODEL_SAVER_PATH = GESTURES_FOLDER + '/ckpts/model_ckpt'
RESULT_SAVE_PATH = GESTURES_FOLDER + '/ActionResults/' 
 
# max amount of rois in a single image
# this initializes the roi vector sizes as well
MAX_ROIS = 20
MAX_ROIS_IN_TRAINING = 20


# jhmdb learning rates for cosine
# lr_max = 0.001
# lr_min = 0.0001

ALL_HUMAN_SAMPLES = DATA_FOLDER + 'all_samples.json'
with open(ALL_HUMAN_SAMPLES) as fp:
    ALL_HUMAN_VIDS = json.load(fp)

HUMAN_ACTORS = ["s00", "s01", "s02", "s02b", "s03", "s04", "s05", "s06", "s07", "s08", "s09", "s10", "s11", "s12"]

#VAL_ACTORS = [0, 13] # s00 and s12 as val
#VAL_ACTORS = [9, 10, 11, 12, 13] 
VAL_ACTORS = [5, 6, 7, 8, 9, 10, 11, 12, 13] 
TRAIN_ACTORS = [ii for ii in range(len(HUMAN_ACTORS)) if ii not in VAL_ACTORS]


# synthetic
USE_SYNTH = True
SYNTHS = ["FemaleCivilian", "FemaleMilitary", "MaleCivilian", "MaleMilitary"]
ALL_SYNTH_SAMPLES = DATA_FOLDER + "all_synth_samples.json"
with open(ALL_SYNTH_SAMPLES) as fp:
    ALL_SYNTH_VIDS = json.load(fp)


ACT_STR_TO_NO = {
    'Idle': 0,
    'Advance': 1,
    'Attention':2,
    'Rally':3,
    'MoveForward':4,
    'Halt':5,
    'FollowMe':6,
    'MoveInReverse':7,
}

ACT_NO_TO_STR = {ACT_STR_TO_NO[strkey]:strkey for strkey in ACT_STR_TO_NO.keys()} 


def process_evaluation_results(res_name):
    print("Results for %s "% res_name)
    res_path = RESULTS_FOLDER + "%s.txt" % res_name


    with open(res_path) as fp:
        res = json.load(fp)

    y_true = []
    y_pred = []
    for r in res:
        y_true.append(np.argmax(r[2]))
        y_pred.append(np.argmax(r[3]))

    import logging

    print(classification_report(y_true, y_pred, target_names=['Idle', 'Advance', 'Attention', 'Rally', 'Forward', 'Halt', 'FollowMe', 'Reverse']))
    logging.info(classification_report(y_true, y_pred, target_names=['Idle', 'Advance', 'Attention', 'Rally', 'Forward', 'Halt', 'FollowMe', 'Reverse']))

    print(confusion_matrix(y_true, y_pred))
    logging.info(confusion_matrix(y_true, y_pred))



def get_train_list():
    print("Training on %s" % (",".join([HUMAN_ACTORS[ii] for ii in TRAIN_ACTORS])))
    train_list = []
    for ii in TRAIN_ACTORS:
        train_list.extend(ALL_HUMAN_VIDS[ii])

    if USE_SYNTH: 
        random.seed(5)
        synth_list = []
        for sublist in ALL_SYNTH_VIDS:
            samples = random.sample(sublist, 3000)
            synth_list.extend(samples)
        train_list.extend(synth_list)

    return train_list

# during validation we will go through all frames individually. 
def get_val_list():
    print("Validating on %s" % (",".join([HUMAN_ACTORS[ii] for ii in VAL_ACTORS])))
    val_list = []
    for ii in VAL_ACTORS:
        val_list.extend(ALL_HUMAN_VIDS[ii])
    return val_list

## filters samples with no detected people!!!!
def filter_no_detections(sample, labels_np, rois_np, no_det, segment_key):
    rois_bool = tf.cast(rois_np, tf.bool)
    return tf.reduce_any(rois_bool)

def get_data(segment_key, split):
    # its a synth sample
    if "-" in segment_key:
        sample = get_video_frames_synth(segment_key, split)
        labels_np, rois_np, no_det = get_labels_synth(segment_key,split )
    # its a human sample
    else:
        sample = get_video_frames(segment_key, split)
        labels_np, rois_np, no_det = get_labels(segment_key,split )

    return sample, labels_np, rois_np, no_det, segment_key

def get_video_frames(segment_key, split):

    actor, vidname = segment_key.split('/')
    

    vid_path = os.path.join(HUMANS_FOLDER, actor, vidname)
    video = imageio.get_reader(vid_path, 'ffmpeg')
    no_frames = video.get_length()


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

def get_obj_detection_results(segment_key,split):
    # split = sample_info['split']
    # segment_key = sample_info['segment_key']
 
    # KEY_FRAME_INDEX = 2
    actor, vidname = segment_key.split('/')
    vidname_noext = vidname.split('.')[0]
    ## object detection results
    #obj_detects_folder = OBJECT_DETECTIONS_FOLDER
    ## ava detection results
    #obj_detects_folder = AVA_FOLDER + '/ava_trained_persons/' + split + '/object_detections/'
    segment_objects_path = os.path.join(OBJECT_DETECTIONS_FOLDER , '%s.json' %vidname_noext )
     
    with open(segment_objects_path) as fp:
        results = json.load(fp)
 
    detections = results['detections']
    #H = results['height']
    #W = results['width']
 
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

def get_labels(segment_key, split):
    actor, vidname = segment_key.split('/')
    action_idx = int(vidname.split('_')[-1].split('.')[0])

    #sample_annotations = [{'bbox':[0.25, 0.25, 0.75, 0.75], 'actions':[action_idx]}]
    sample_annotations = [[0.25, 0.25, 0.75, 0.75]]


    detection_boxes = get_obj_detection_results(segment_key, split)

    labels_np, rois_np, no_det = match_annos_with_detections(sample_annotations, detection_boxes, action_idx)

    return labels_np, rois_np, no_det

def get_video_frames_synth(segment_key, split):

    synth_model, vidname = segment_key.split('-')
    vidname_noext = vidname.split('.')[0]

    vid_path = os.path.join(SYNTH_FOLDER, synth_model, vidname)
    video = imageio.get_reader(vid_path, 'ffmpeg')
    no_frames = video.get_length() -5 # last frame gives issue


    f_timesteps, f_H, f_W, f_C = [INPUT_T, INPUT_H, INPUT_W, 3]
 
    slope = (no_frames-1) / float(f_timesteps - 1)
    indices = (slope * np.arange(f_timesteps)).astype(np.int64)
 
    sample = np.zeros([f_timesteps, f_H, f_W, f_C], np.uint8)
 
    timestep = 0
    # for vid_idx in range(no_frames):
    for vid_idx in indices.tolist():
        try:
            frame = video.get_data(vid_idx)
        except KeyboardInterrupt:
            raise
        except:
            print("Error at segment %s" % segment_key)
            break

        # frame = frame[:,:,::-1] #opencv reads bgr, i3d trained with rgb
        reshaped = cv2.resize(frame, (f_W, f_H))
        sample[timestep, :, :, :] = reshaped
        timestep += 1
 
    video.close()
    return sample.astype(np.float32)

def get_obj_detection_results_synth(segment_key, split): 
    synth_model, vidname = segment_key.split('-')
    vidname_noext = vidname.split('.')[0]

    segment_objects_path = os.path.join(OBJECT_DETECTIONS_SYNTH_FOLDER , '%s-%s.json' % (synth_model, vidname_noext) )

    with open(segment_objects_path) as fp:
        results = json.load(fp)
 
    detections = results['detections']
    #H = results['height']
    #W = results['width']
 
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

def get_labels_synth(segment_key, split):
    synth_model, vidname = segment_key.split('-')
    act_str = segment_key.split('_')[1]
    action_idx = ACT_STR_TO_NO[act_str]

    #sample_annotations = [{'bbox':[0.25, 0.25, 0.75, 0.75], 'actions':[action_idx]}]
    sample_annotations = [[0.25, 0.25, 0.75, 0.75]]


    detection_boxes = get_obj_detection_results_synth(segment_key, split)

    labels_np, rois_np, no_det = match_annos_with_detections(sample_annotations, detection_boxes, action_idx)

    return labels_np, rois_np, no_det



MATCHING_IOU = 0.1
def match_annos_with_detections(annotations, detections, action):
    # gt_boxes = []
    # for ann in annotations:
    #     # left, top, right, bottom
    #     # [0.07, 0.141, 0.684, 1.0]
    #     cur_box = ann['bbox'] 
    #     gt_boxes.append(cur_box)
    gt_boxes = annotations
 
    det_boxes = []
    for detection in detections:
        # top, left, bottom, right
        # [0.07, 0.006, 0.981, 0.317]
        # top, left, bottom, right = detection['box'] 
        # box = left, top, right, bottom
        box = detection['box_coords']
        class_label = detection['class_str']
        det_boxes.append(box)
 
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
    if no_gt != 0 and no_det != 0:
        max_iou_for_each_det = np.max(iou_mtx, axis=0)
        index_for_each_det = np.argmax(iou_mtx, axis=0)
            

        for dd in range(no_det):
            cur_max_iou = max_iou_for_each_det[dd]

            
            top, left, bottom, right = detections[dd]['box_coords']

            # # region of interest layer expects
            # # regions of interest as lists of:
            # # feature map index, upper left, bottom right coordinates
            rois_np[dd,:] = [top, left, bottom, right]
            if cur_max_iou < MATCHING_IOU:
                labels_np[dd, 0] = 1 # bg class for softmax
            else:
            #matched_ann = annotations[index_for_each_det[dd]]
                #labels_np[dd, ACT_STR_TO_NO[action]] = 1
                labels_np[dd, action] = 1

 
    return labels_np, rois_np, no_det



def IoU_box(box1, box2):
    '''
    top1, left1, bottom1, right1 = box1
    top2, left2, bottom2, right2 = box2
 
    returns intersection over union
    '''
    # left1, top1, right1, bottom1 = box1
    # left2, top2, right2, bottom2 = box2
    top1, left1, bottom1, right1 = box1
    top2, left2, bottom2, right2 = box2
 
    left_int = max(left1, left2)
    top_int = max(top1, top2)
 
    right_int = min(right1, right2)
    bottom_int = min(bottom1, bottom2)
 
    areaIntersection = max(0, right_int - left_int) * max(0, bottom_int - top_int)
 
    area1 = (right1 - left1) * (bottom1 - top1)
    area2 = (right2 - left2) * (bottom2 - top2)
     
    IoU = areaIntersection / float(area1 + area2 - areaIntersection)
    return IoU
    

def get_per_class_AP(results_list):
    '''
    results_list is a list where each
    result = ['path' [multilabel-binary labels] [probs vector]]

    returns per class_AP vector with class average precisions
    '''
    class_results = [{'truth':[], 'pred':[]} for _ in range(NUM_CLASSES)]

    for result in results_list:
        cur_key = result[0]
        cur_roi_id = result[1]
        cur_truths = result[2]
        cur_preds = result[3]
        
        # cur_preds = np.random.uniform(size=40)
        # cur_preds = [0 for _ in range(40)]

        for cc in range(NUM_CLASSES):
            class_results[cc]['truth'].append(cur_truths[cc])
            class_results[cc]['pred'].append(cur_preds[cc])

    ground_truth_count = []
    class_AP = []
    for cc in range(NUM_CLASSES):
        y_truth = class_results[cc]['truth']
        y_pred = class_results[cc]['pred']
        AP = average_precision_score(y_truth, y_pred)

        # print(AP)
        # plot_pr_curve(y_truth, y_pred)
        # import pdb; pdb.set_trace()

        if np.isnan(AP): AP = 0

        class_AP.append(AP)
        ground_truth_count.append(sum(y_truth))
        
    # import pdb; pdb.set_trace()
    return class_AP, ground_truth_count

def get_class_AP_str(class_AP, cnt):
    ''' Returns a printable string'''
    ap_str = ''
    for cc in range(NUM_CLASSES):
        class_str = ACT_NO_TO_STR[cc][0:15] # just take the first 15 chars, some of them are too long
        class_cnt = cnt[cc]
        AP = class_AP[cc]
        # AP = AP if not np.isnan(AP) else 0
        # cur_row = '%s:    %i%% \n' %(class_str, AP*100)#class_str + ':    ' + str(tools.get_3_decimal_float(AP)) + '\n'
        cur_row = class_str + '(%i)' % class_cnt + ':'
        cur_row += (' ' * (25 -len(cur_row))) + '%.1f%%\n' % (AP*100.0)
        ap_str += cur_row
    class_avg = np.mean(class_AP)
    # class_avg = class_avg if not np.isnan(class_avg) else 0
    ap_str += '\n' + 'Average:' + (' '*17) + '%.1f%%\n' % (class_avg*100.0)
    return ap_str

def get_AP_str(results_list):
    class_AP, cnt = get_per_class_AP(results_list)
    ap_str = get_class_AP_str(class_AP, cnt)
    return ap_str
