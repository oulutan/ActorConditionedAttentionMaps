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
 
from sklearn.metrics import average_precision_score
# import models.extra_layers as extra_layers


## During training randomly select a midframe and just use its bbox
# during eval, evaluate for every frame in the video. 

INPUT_H = 400
INPUT_W = 400
INPUT_T = 32
# INPUT_T = 16
 
KEY_FRAME_INDEX = 2
 
NUM_CLASSES = 21

ACAM_FOLDER = os.environ['ACAM_DIR']

JHMDB_FOLDER = ACAM_FOLDER + '/data/JHMDB' 

VIDEOS_FOLDER = JHMDB_FOLDER + '/ReCompress_Videos/'
OBJECT_DETECTIONS_FOLDER = JHMDB_FOLDER + '/objects/'
DATA_FOLDER = JHMDB_FOLDER + '/data/'
SPLIT_INFO_FOLDER = JHMDB_FOLDER + '/splits/'

SPLIT_NO = 1

MODEL_SAVER_PATH = JHMDB_FOLDER + '/ckpts/split_%i_model_ckpt' % SPLIT_NO
RESULT_SAVE_PATH = JHMDB_FOLDER + '/ActionResults/split_%i' % SPLIT_NO
 
# max amount of rois in a single image
# this initializes the roi vector sizes as well
MAX_ROIS = 100
MAX_ROIS_IN_TRAINING = 20


ALL_VIDS_FILE = DATA_FOLDER + 'all_vids.txt'
with open(ALL_VIDS_FILE) as fp:
    ALL_VIDS = fp.readlines()
ALL_ACTIONS = list(set([v.split(" ")[0] for v in ALL_VIDS]))
ALL_ACTIONS.sort()

ACT_STR_TO_NO = {
    'brush_hair':0,
    'catch':1,
    'clap':2,
    'climb_stairs':3,
    'golf':4,
    'jump':5,
    'kick_ball':6,
    'pick':7,
    'pour':8,
    'pullup':9,
    'push':10,
    'run':11,
    'shoot_ball':12,
    'shoot_bow':13,
    'shoot_gun':14,
    'sit':15,
    'stand':16,
    'swing_baseball':17,
    'throw':18,
    'walk':19,
    'wave':20
}

ACT_NO_TO_STR = {ACT_STR_TO_NO[strkey]:strkey for strkey in ACT_STR_TO_NO.keys()} 

ANNOTATIONS_FILE = DATA_FOLDER + 'segment_annotations.json'
with open(ANNOTATIONS_FILE) as fp:
    ANNOTATIONS = json.load(fp)


# during training a frame will be randomly selected so 
# add the total no of frames as the current frame
# I can use that number to randomly select the frame
def get_train_list():
    train_segments = []
    for act in ALL_ACTIONS:
        fname = SPLIT_INFO_FOLDER + '%s_test_split%i.txt' % (act, SPLIT_NO)
        with open(fname) as fp:
            vids_info = fp.readlines()
        vids_info = [v.strip() for v in vids_info]
        # vidname 1: 1 means training
        # train_vids = ["%s %s 0" % (act, v.split(" ")[0]) for v in vids_info if v.split(" ")[1] == '1']
        train_vids = []
        for v in vids_info:
            vidname, train_test = v.split(" ")
            if train_test == '1':
                vid_str = "%s %i" % (vidname, ANNOTATIONS[vidname]['nframes'])
                train_vids.append(vid_str)

        train_segments.extend(train_vids)
    return train_segments * 20

# during validation we will go through all frames individually. 
def get_val_list():
    val_segments = []
    for act in ALL_ACTIONS:
        fname = SPLIT_INFO_FOLDER + '%s_test_split%i.txt' % (act, SPLIT_NO)
        with open(fname) as fp:
            vids_info = fp.readlines()
        vids_info = [v.strip() for v in vids_info]
        # vidname 1: 1 means training
        # val_vids = ["%s %s" % (act, v.split(" ")[0]) for v in vids_info if v.split(" ")[1] == '1']
        val_vids = []
        for v in vids_info:
            vidname, train_test = v.split(" ")
            if train_test == '2':
                for ii in range(ANNOTATIONS[vidname]['nframes']):
                    vid_str = "%s %i" % (vidname, ii)
                    val_vids.append(vid_str)

        val_segments.extend(val_vids)
    return val_segments


def get_data(segment_key, split):
    
    sample, center_frame = get_video_frames(segment_key, split)
    labels_np, rois_np, no_det = get_labels(segment_key,split, center_frame)

    return sample, labels_np, rois_np, no_det, segment_key

def get_video_frames(segment_key, split):

    vidname, frame_info = segment_key.split(' ')
    action = ANNOTATIONS[vidname]['action']
    if split == 'train':
        # if its train frame info is total number of frames in the segments
        center_frame = np.random.randint(low=0, high=int(frame_info))
    else:
        center_frame = int(frame_info)
    

    vid_path = os.path.join(VIDEOS_FOLDER, action, vidname)
    video = imageio.get_reader(vid_path, 'ffmpeg')

    sample = np.zeros([INPUT_T, INPUT_H, INPUT_W, 3], np.uint8)

    for ii in range(INPUT_T):
        cur_frame_idx = center_frame - INPUT_T // 2 + ii
        if cur_frame_idx < 0 or cur_frame_idx >= ANNOTATIONS[vidname]['nframes'] :
            continue
        else:
            frame = video.get_data(cur_frame_idx)
            # frame = frame[:,:,::-1] #opencv reads bgr, i3d trained with rgb
            reshaped = cv2.resize(frame, (INPUT_W, INPUT_H))
            sample[ii,:,:,:] = reshaped

    video.close()

    return sample.astype(np.float32), center_frame

def get_labels(segment_key, split, center_frame):
    vidname, frame_info = segment_key.split(' ')
    sample_annotations = ANNOTATIONS[vidname]
    action = sample_annotations['action']

    ann_box = sample_annotations['frame_boxes'][center_frame]

    detection_results_file = os.path.join(OBJECT_DETECTIONS_FOLDER, action, "%s.json" % vidname)
    with open(detection_results_file) as fp:
        detection_results = json.load(fp)
    detection_boxes = detection_results['frame_objects'][center_frame]
    detection_boxes = [detbox for detbox in detection_boxes if detbox['class_str'] == 'person']
    
    if split == 'train':
        # detections = [det for det in detections if det['score'] > 0.20]
        if len(detection_boxes) > MAX_ROIS_IN_TRAINING:
            # they are sorted by confidence already, take top #k
            detection_boxes = detection_boxes[:MAX_ROIS_IN_TRAINING]

    labels_np, rois_np, no_det = match_annos_with_detections([ann_box], detection_boxes, action)

    return labels_np, rois_np, no_det



MATCHING_IOU = 0.5
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
        box = detection['box']
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

            
            top, left, bottom, right = detections[dd]['box']

            # # region of interest layer expects
            # # regions of interest as lists of:
            # # feature map index, upper left, bottom right coordinates
            rois_np[dd,:] = [top, left, bottom, right]
            if cur_max_iou < MATCHING_IOU:
                continue
            matched_ann = annotations[index_for_each_det[dd]]
            
            labels_np[dd, ACT_STR_TO_NO[action]] = 1

 
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
