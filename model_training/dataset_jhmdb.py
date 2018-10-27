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
DATA_FOLDER = JHMDB_FOLDER + '/data/'
SPLIT_INFO_FOLDER = JHMDB_FOLDER + '/splits/'

SPLIT_NO = 1

MODEL_SAVER_PATH = JHMDB_FOLDER + '/ckpts/split_%i_model_ckpt' % SPLIT_NO
RESULT_SAVE_PATH = JHMDB_FOLDER + '/ActionResults/split_%i' % SPLIT_NO
 
# max amount of rois in a single image
# this initializes the roi vector sizes as well
MAX_ROIS = 100


ALL_VIDS_FILE = DATA_FOLDER + 'all_vids.txt'
with open(ALL_VIDS_FILE) as fp:
    ALL_VIDS = fp.readlines()
ALL_ACTIONS = list(set([v.split(" ")[0] for v in ALL_VIDS]))

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
        # vidname 1: 1 means training
        # train_vids = ["%s %s 0" % (act, v.split(" ")[0]) for v in vids_info if v.split(" ")[1] == '1']
        train_vids = []
        for v in vids_info:
            vidname, train_test = v.split(" ")
            if train_test == '1':
                vid_str = "%s %i" % (vidname, ANNOTATIONS[vidname]['nframes'])
                train_vids.append(vid_str)

        train_segments.extend(train_vids)
    return train_segments

# during validation we will go through all frames individually. 
def get_val_list():
    val_segments = []
    for act in ALL_ACTIONS:
        fname = SPLIT_INFO_FOLDER + '%s_test_split%i.txt' % (act, SPLIT_NO)
        with open(fname) as fp:
            vids_info = fp.readlines()
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
    
    sample = get_video_frames(segment_key, split)
    labels_np, rois_np, no_det = get_labels(segment_key,split)

    return sample, labels_np, rois_np, no_det, segment_key

def get_video_frames(segment_key, split):
