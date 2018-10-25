import numpy as np
import json


ACAM_FOLDER = os.environ['ACAM_DIR']

JHMDB_FOLDER = ACAM_FOLDER + '/data/JHMDB' 

VIDEOS_FOLDER = JHMDB_FOLDER + '/ReCompress_Videos/'
OBJECTS_FOLDER = JHMDB_FOLDER + '/objects/'
DATA_FOLDER = JHMDB_FOLDER + '/data/'

ALL_VIDS_FILE = DATA_FOLDER + 'all_vids.txt'

def IoU_box(box1, box2):
    '''
    returns intersection over union
    '''
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


def link_boxes():
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
    
        ann = annotations_dict[vid_id]

        detection_file = os.path.join(OBJECTS_FOLDER, action, "%s.json" % vid_id_avi)
        with open(detection_file) as fp:
            detection_results = json.load(fp)

        frame_objects = detection_results['frame_objects']
        nframes = ann['nframes']

        tracks = []
        