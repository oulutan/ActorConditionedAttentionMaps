
import cv2
import os
import json
import numpy as np
import argparse
import imageio

import object_detection_wrapper


DETECTION_TH = 0.01

#AVA_FOLDER = os.environ['AVA_DIR'] + '/AVA'
#SEGMENTS_FOLDER = AVA_FOLDER + '/segments/'
#DATA_FOLDER = AVA_FOLDER + '/data/'
#SEGMENT_ANN_FILE = DATA_FOLDER + 'segment_annotations_%s.json' % SPLIT
VIDS_FOLDER = "/media/ssd1/oytun/JHMDB/JHMDB_video/ReCompress_Videos/"
ALL_VIDS_FILE = "/media/ssd1/oytun/JHMDB/all_vids.txt"

OBJECT_DETECTIONS_FOLDER = "/media/ssd1/oytun/JHMDB/objects/"

def get_frame_generator(vid_key):
    action, vid_id = vid_key.split(" ")
    video_path = os.path.join(VIDS_FOLDER, action, vid_id)

    video = imageio.get_reader(video_path)
    gen = video.iter_data()
    return gen

COLORS = np.random.randint(0, 255, [100, 3])
def draw_objects(frame, detections):
    H,W,C = frame.shape
    for bbid, bbox in enumerate(detections):
        color = COLORS[bbid,:]
        current_bbox = bbox['box']

        top, left, bottom, right = current_bbox
        left = int(W * left)
        right = int(W * right)

        top = int(H * top)
        bottom = int(H * bottom)

        conf = bbox['score']
        if conf < 0.20:
            continue
        label = bbox['class_str']
        message = label + ' %.2f' % conf

        cv2.rectangle(frame, (left,top), (right,bottom), color, 2)

        font_size =  max(0.5,(right - left)/50.0/float(len(message)))
        cv2.rectangle(frame, (left, top-int(font_size*40)), (right,top), color, -1)
        cv2.putText(frame, message, (left, top-12), 0, font_size, (255,255,255)-color, 1)


with open(ALL_VIDS_FILE) as fp:
    all_vids = fp.readlines()
# parse the filelist
all_vids = [vid.strip() for vid in all_vids]


for vid in all_vids:
    print("Working on %s" % vid)
    frame_gen = get_frame_generator(vid)
    act, vid_id = vid.split(" ")
    detection_results_file = os.path.join(OBJECT_DETECTIONS_FOLDER, act, "%s.json" % vid_id)
    with open(detection_results_file) as fp:
       results = json.load(fp)

    for ff, frame in enumerate(frame_gen):
        detection_results = results['frame_objects'][ff]
        H, W = results['height'], results['width']

        frame_to_visualize = frame.copy()
        draw_objects(frame_to_visualize, detection_results)

        cv2.imshow('results', frame_to_visualize)
        k = cv2.waitKey(0)
        if k == ord('q'):
            os._exit(0)
        if k == ord('n'):
            break

        
            

