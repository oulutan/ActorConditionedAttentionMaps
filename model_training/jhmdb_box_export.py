import pickle
import json
from tqdm import tqdm
import numpy as np

import dataset_jhmdb

#filename = "../data/JHMDB/ActionResults/split_1VALIDATION_Results_soft_attn_roipooled_06"
filename = "../data/JHMDB/ActionResults/split_1VALIDATION_Results_i3d_tail_71"

with open("%s.txt" % filename ) as fp:
    all_results = json.load(fp)

H = 240
W = 320
split = "val"

output_pickle = {}


for result in tqdm(all_results):
    cur_key = result[0]
    cur_video_id, cur_timestamp = cur_key.split('.')
    cur_roi_id = result[1]
    cur_truths = result[2]
    cur_preds = result[3]

    vid_name, center_frame = cur_key.split(" ")
    vid_name_no_ext = vid_name.split(".")[0]
    center_frame = int(center_frame)
    labels_np, rois_np, no_det = dataset_jhmdb.get_labels(cur_key, split, center_frame)
     
    
    cur_detection = rois_np[cur_roi_id]
    # top, left, bottom, right
    # [0.07, 0.006, 0.981, 0.317]
    top, left, bottom, right = cur_detection.tolist()
    #object_prob = cur_detection['score']

    #j_style_box = [left,top,right,bottom]
    j_style_box = [left*W,top*H,right*W,bottom*H]
    #j_style_box = [int(coord) for coord in j_style_box]


    #cur_class = np.argmax(cur_truths)
    #class_str = dataset_jhmdb.ACT_NO_TO_STR[cur_class]
    class_str = dataset_jhmdb.ANNOTATIONS[vid_name]['action']
    output_key = "%s/%s/%.5d.png" % (class_str, vid_name_no_ext, center_frame)

    if output_key in output_pickle:
        for cc in range(1,22):
            cur_pred = cur_preds[cc-1] # 1 based to 0 based
            output_pickle[output_key][cc][cur_roi_id] = j_style_box + [cur_pred]
    else:
        output_pickle[output_key]={}
        for cc in range(1,22):
            cur_pred = cur_preds[cc-1] # 1 based to 0 based
            output_pickle[output_key][cc] = np.zeros([no_det, 5])
            output_pickle[output_key][cc][cur_roi_id] = j_style_box + [cur_pred]

with open('tail_res_pickle.pkl', 'wb') as fp:
    pickle.dump(output_pickle, fp)

    


