import os
import shutil
from tqdm import tqdm
import json

SPLIT = "train"
#SPLIT = "val"
#SPLIT = "test"

AVA_FOLDER = os.environ['ACAM_DIR'] + '/data/AVA'
SEGMENTS_FOLDER = AVA_FOLDER + '/segments/'
COMBINED_MIDFRAMES_FOLDER = AVA_FOLDER + '/combined_midframes/'
DATA_FOLDER = AVA_FOLDER + '/data/'
SEGMENT_ANN_FILE = DATA_FOLDER + 'segment_annotations_v22_%s.json' % SPLIT

with open(SEGMENT_ANN_FILE) as fp:
    annotations = json.load(fp)

segment_keys = annotations.keys()
segment_keys.sort()

for segment_key in tqdm(segment_keys):
    movie_name, timestamp = segment_key.split('.')
    keyframe_path = os.path.join(SEGMENTS_FOLDER, SPLIT, 'midframes', movie_name, '%s.jpg' %timestamp)

    new_path = os.path.join(COMBINED_MIDFRAMES_FOLDER, SPLIT, "%s.jpg" % segment_key)

    shutil.copy(keyframe_path, new_path)
    tqdm.write("Copied %s ---> %s" % (keyframe_path, new_path))
