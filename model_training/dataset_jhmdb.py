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

MODEL_SAVER_PATH = JHMDB_FOLDER + '/ckpts/model_ckpt'
RESULT_SAVE_PATH = JHMDB_FOLDER + '/ActionResults/'
 
# max amount of rois in a single image
# this initializes the roi vector sizes as well
MAX_ROIS = 100

