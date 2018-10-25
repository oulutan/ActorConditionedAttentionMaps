import os
import scipy.io as sio
import imageio
from tqdm import tqdm
import numpy as np
import json


ACAM_FOLDER = os.environ['ACAM_DIR']

JHMDB_FOLDER = ACAM_FOLDER + '/data/JHMDB' 

VIDEOS_FOLDER = JHMDB_FOLDER + '/ReCompress_Videos/'
JOINTS_FOLDER = JHMDB_FOLDER + '/joint_positions/'
DATA_FOLDER = JHMDB_FOLDER + '/data/'




ALL_VIDS_FILE = DATA_FOLDER + 'all_vids.txt'
with open(ALL_VIDS_FILE) as fp:
    all_vids = fp.readlines()
# parse the filelist
all_vids = [vid.strip() for vid in all_vids]

annotations_dict = {}
for vid_key in tqdm(all_vids):
    action, vid_id = vid_key.split(" ")
    
    # read joints file
    joints_file = os.path.join(JOINTS_FOLDER, action, vid_id, 'joint_positions.mat')
    joints = sio.loadmat(joints_file)['pos_img']

    # get video information
    video_path = os.path.join(VIDS_FOLDER, action, vid_id)
    video = imageio.get_reader(video_path)
    vidinfo = video.get_meta_data()

    no_frames = vidinfo['nframes']
    assert no_frames == joints.shape[-1]
    W, H = vidinfo['size']

    bboxes = [] 
    # generate bounding boxes
    for ii in range(no_frames):
        cur_joints = joints[:,:,ii]
        #[0,:,:] is in width, [1,:,:] is in height
        top = np.min(cur_joints[1])
        left = np.min(cur_joints[0])
        bottom = np.max(cur_joints[1])
        right = np.max(cur_joints[0])

        # joints can be outside image boundaries
        top = top if top > 0 else 0
        left = left if left > 0 else 0

        bottom = bottom if bottom < H else H
        right = right if right < W else W

        # map the coords to normalized float
        topn = float(top) / float(H)
        bottomn = float(bottom) / float(H)

        leftn = float(left) / float(W)
        rightn = float(right) / float(W)

        bboxes.append([topn, leftn, bottomn, rightn])

    print('%s done!' % vid_key)
    annotations_dict[vid_id] = {'vid_id': vid_id, 
                                'action':action,
                                'height':H,
                                'width': W,
                                'nframes': no_frames,
                                'frame_boxes': bboxes}

annotation_file = DATA_FOLDER + 'segment_annotations.json'

with open(annotation_file, 'w') as fp:
    json.dump(annotations_dict, fp)
print("%s written!" % annotation_file)




