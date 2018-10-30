import os
import scipy.io as sio
import imageio
from tqdm import tqdm
import numpy as np
import json
import cv2


ACAM_FOLDER = os.environ['ACAM_DIR']

JHMDB_FOLDER = ACAM_FOLDER + '/data/JHMDB' 

VIDEOS_FOLDER = JHMDB_FOLDER + '/ReCompress_Videos/'
JOINTS_FOLDER = JHMDB_FOLDER + '/joint_positions/'
DATA_FOLDER = JHMDB_FOLDER + '/data/'




ALL_VIDS_FILE = DATA_FOLDER + 'all_vids.txt'
def generate_annotation_file():
    with open(ALL_VIDS_FILE) as fp:
        all_vids = fp.readlines()
    # parse the filelist
    all_vids = [vid.strip() for vid in all_vids]

    annotations_dict = {}
    for vid_key in tqdm(all_vids):
        action, vid_id_avi = vid_key.split(" ")
        vid_id = vid_id_avi.split(".avi")[0]
        
        # read joints file
        joints_file = os.path.join(JOINTS_FOLDER, action, vid_id, 'joint_positions.mat')
        joints = sio.loadmat(joints_file)['pos_img']

        # get video information
        video_path = os.path.join(VIDEOS_FOLDER, action, vid_id_avi)
        video = imageio.get_reader(video_path)
        vidinfo = video.get_meta_data()

        no_frames = vidinfo['nframes']
        no_anns = joints.shape[-1]
        #assert no_frames == joints.shape[-1]
        #if no_frames != joints.shape[-1]:
        #    import pdb;pdb.set_trace()
        
        
        W, H = vidinfo['size']

        bboxes = [] 
        # generate bounding boxes
        for ii in range(no_anns):
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
        
        # videos have more frames than annotations
        # strecth them out
        interpolated_bboxes = []
        indices = (no_anns-1) * np.arange(no_frames) // (no_frames-1)
        for idx in indices:
            interpolated_bboxes.append(bboxes[idx])

        tqdm.write('%s done!' % vid_key)
        annotations_dict[vid_id_avi] = {'vid_id': vid_id, 
                                    'action':action,
                                    'height':H,
                                    'width': W,
                                    'nframes': no_frames,
                                    'frame_boxes': interpolated_bboxes}

    annotation_file = DATA_FOLDER + 'segment_annotations.json'

    with open(annotation_file, 'w') as fp:
        json.dump(annotations_dict, fp)
    print("%s written!" % annotation_file)


def get_frame_generator(vid_key):
    action, vid_id = vid_key.split(" ")
    video_path = os.path.join(VIDEOS_FOLDER, action, vid_id)

    video = imageio.get_reader(video_path)
    gen = video.iter_data()
    return gen

COLORS = np.random.randint(0, 255, [100, 3])
def draw_objects(frame, detections):
    H,W,C = frame.shape
    for bbid, bbox in enumerate(detections):
        color = COLORS[bbid,:]
        current_bbox = bbox

        top, left, bottom, right = current_bbox
        left = int(W * left)
        right = int(W * right)

        top = int(H * top)
        bottom = int(H * bottom)

        # conf = bbox['score']
        conf = 1.0
        if conf < 0.20:
            continue
        # label = bbox['class_str']
        label = 'Annotation'
        message = label + ' %.2f' % conf

        cv2.rectangle(frame, (left,top), (right,bottom), color, 2)

        font_size =  max(0.5,(right - left)/50.0/float(len(message)))
        cv2.rectangle(frame, (left, top-int(font_size*40)), (right,top), color, -1)
        cv2.putText(frame, message, (left, top-12), 0, font_size, (255,255,255)-color, 1)

def visualize_annotations():
    with open(ALL_VIDS_FILE) as fp:
        all_vids = fp.readlines()
    # parse the filelist
    all_vids = [vid.strip() for vid in all_vids]

    annotation_file = DATA_FOLDER + 'segment_annotations.json'

    with open(annotation_file) as fp:
        annotations_dict = json.load(fp)

    for vid_key in tqdm(all_vids):
        action, vid_id_avi = vid_key.split(" ")
        vid_id = vid_id_avi.split(".avi")[0]
        
        # read joints file
        # joints_file = os.path.join(JOINTS_FOLDER, action, vid_id, 'joint_positions.mat')
        # joints = sio.loadmat(joints_file)['pos_img']
        ann = annotations_dict[vid_id]

        # get video information
        vid_gen = get_frame_generator(vid_key)

        for ff in range(ann['nframes']):
            frame = vid_gen.next()
            current_box = ann['frame_boxes'][ff]
            disp_frame = frame.copy()
            draw_objects(disp_frame, [current_box])
            cv2.imshow('anns', disp_frame)
            cv2.waitKey(0)




if __name__ == '__main__':
    generate_annotation_file()
    # visualize_annotations()
