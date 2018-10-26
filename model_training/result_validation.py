import numpy as np
import json
import dataset_ava
import os
import argparse
from tqdm import tqdm


# RESULT_NAME = 'VALIDATION_Results_I3DTail_5gpu_20'
# RESULT_NAME = 'VALIDATION_Results_v100sI3DTail_32t_35'
# RESULT_NAME = 'VALIDATION_Results_nonlocal_i3d_tail_31'
# RESULT_NAME = 'VALIDATION_Results_nonlocal_v2_i3d_tail_35'
RESULT_NAME = 'VALIDATION_Results_I3DTail_32t_36'
# RESULT_NAME = 'VALIDATION_Results_multi_nonlocal_i3d_03'
# RESULT_NAME = 'VALIDATION_Results_REGUk80sI3DTail_37'
# RESULT_NAME = 'VALIDATION_Results_k80sI3DTail_36'
# RESULT_NAME = 'VALIDATION_Results_t32_10batch_v100_basic_27'
# RESULT_NAME = 'VALIDATION_Results_focal_I3DTail_32t_18'
# RESULT_NAME = 'VALIDATION_Results_lowlrI3DTail_21'
SPLIT = 'val'
#RESULT_NAME = 'TEST_Results_detboxes_basicmodel_firsttry_test_12'
#RESULT_NAME = 'TEST_Results_basicmodel_augment_balanced_tuning_sgd_42'
#SPLIT = 'test'

ACAM_FOLDER = os.environ['ACAM_DIR']
AVA_FOLDER = ACAM_FOLDER + '/data/AVA'
RESULTS_FOLDER = AVA_FOLDER + '/ActionResults/'
RESULTS_PATH = RESULTS_FOLDER + '%s.txt' % RESULT_NAME
#RESULTS_PATH = '/media/sidious_data/AVA/important_results/%s.txt' % RESULT_NAME
OUTPUT_FOLDER = AVA_FOLDER + '/ava_style_results/'
OUTPUT_PATH = OUTPUT_FOLDER + '%s.csv' % RESULT_NAME

OBJECT_ACTION_CORRELATION = False

if OBJECT_ACTION_CORRELATION:
    OBJECTS_FOLDER = "/media/sidious_data/AVA/objects/"
    cors = np.load("../scripts/actions_objects_train.npy")
    # remove person class which is index 0
    cors = cors[:,1:]
    normcors = cors.T / np.sum(cors, axis=1) # normalize histograms
    normcors = normcors.T
    ACTION_OBJ_COR_MTX = normcors


def read_serialized_results(file_path):
    with open(file_path) as fp:
        data = json.load(fp)
    return data

# def nms_on_obj_detections(all_results, split):
#     keys = set()
#     for result in all_results:
#         keys.add(result[0])

#     keys = list(keys)
#     nms_dict = {}
#     print('Nms on object detection!')
#     for cur_key in tqdm(keys):
#         detections = dataset_ava.get_obj_detection_results(cur_key, split)

#         dets = []
#         for cur_det in detections:
#             top, left, bottom, right = cur_det['box']
#             dets.append({'box':[left, top, right, bottom], 'prob':cur_det['score']})
#         # dets = [{'box':cur_det['box'], 'prob':cur_det['score']} for cur_det in detections]

#         nms_results = non_max_suppression(dets)

#         nms_box_list = [nms_res['box'] for nms_res in nms_results]

#         nms_dict[cur_key] = [box.tolist() for box in nms_box_list]

#     print('Done with nms!')
#     return nms_dict

# def nms_convert_results(all_results, split='val'):
#     '''
#     all_results is a list where each
#     result = ['path' [multilabel-binary labels] [probs vector]]

#     returns per class_AP vector with class average precisions
    
#     [["1j20qq1JyX4.0902", 0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0.0, 0.    0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.041, 0.542, 0.0, 0.01, 0.0, 0.008, 0.0, 0.0, 0.0, 0.0, 0.001, 0.001, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.002, 0.0, 0.0, 0.0, 0.0    1, 0.0, 0.0, 0.0, 0.0, 0.001, 0.001, 0.003, 0.002, 0.0, 0.0, 0.001, 0.0, 0.0, 0.475, 0.001, 0.003, 0.0, 0.021, 0.391]]
#     '''
#     output_strings = []
#     nms_dict = nms_on_obj_detections(all_results, split)
#     print_flag = True
#     for result in tqdm(all_results):
#         cur_key = result[0]
#         cur_video_id, cur_timestamp = cur_key.split('.')
#         cur_roi_id = result[1]
#         cur_truths = result[2]
#         cur_preds = result[3]

        
#         detections = dataset_ava.get_obj_detection_results(cur_key, split)
        
        
#         cur_detection = detections[cur_roi_id]
#         # top, left, bottom, right
#         # [0.07, 0.006, 0.981, 0.317]
#         top, left, bottom, right = cur_detection['box']
#         object_prob = cur_detection['score']

#         # DEBUGGING
#         # if print_flag: print('USING GROUND TRUTH BOXES!!!!'); print_flag=False
#         # annotations = dataset_ava.ANNOS_VAL[cur_key]
#         # left, top, right, bottom = annotations[cur_roi_id]['bbox']

#         # ava style
#         cur_box = [left, top, right, bottom]
#         if cur_box not in nms_dict[cur_key]:
#             continue

#         for cc in range(dataset_ava.NUM_CLASSES):
#             cur_probability = cur_preds[cc] * object_prob
#             # cur_probability = cur_preds[cc]
#             # cur_probability = cur_truths[cc] # what if we have every label correct

#             cur_action_id = dataset_ava.TRAIN2ANN[str(cc)]['ann_id'] # this is string
#             if cur_probability < 0.001:
#             # if cur_probability < 0.005:
#             # if cur_probability < 0.01:
#                 continue
#             else:
#                 current_string = ''
#                 current_string += '%s,' % cur_video_id
#                 current_string += '%s,' % cur_timestamp
#                 current_string += '%.3f,%.3f,%.3f,%.3f,' % tuple(cur_box)
#                 current_string += '%s,' % cur_action_id
#                 current_string += '%.3f'% cur_probability
#                 output_strings.append(current_string)

#     return output_strings

def convert_results(all_results, split='val'):
    '''
    all_results is a list where each
    result = ['path' [multilabel-binary labels] [probs vector]]

    returns per class_AP vector with class average precisions
    
    [["1j20qq1JyX4.0902", 0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0.0, 0.    0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.041, 0.542, 0.0, 0.01, 0.0, 0.008, 0.0, 0.0, 0.0, 0.0, 0.001, 0.001, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.002, 0.0, 0.0, 0.0, 0.0    1, 0.0, 0.0, 0.0, 0.0, 0.001, 0.001, 0.003, 0.002, 0.0, 0.0, 0.001, 0.0, 0.0, 0.475, 0.001, 0.003, 0.0, 0.021, 0.391]]
    '''
    output_strings = []
    print_flag = True
    for result in tqdm(all_results):
        cur_key = result[0]
        cur_video_id, cur_timestamp = cur_key.split('.')
        cur_roi_id = result[1]
        cur_truths = result[2]
        cur_preds = result[3]

        
        detections = dataset_ava.get_obj_detection_results(cur_key, split)
        
        
        cur_detection = detections[cur_roi_id]
        # top, left, bottom, right
        # [0.07, 0.006, 0.981, 0.317]
        top, left, bottom, right = cur_detection['box']
        object_prob = cur_detection['score']

        # DEBUGGING
        # if print_flag: print('USING GROUND TRUTH BOXES!!!!'); print_flag=False
        # annotations = dataset_ava.ANNOS_VAL[cur_key]
        # left, top, right, bottom = annotations[cur_roi_id]['bbox']

        # ava style
        cur_box = [left, top, right, bottom]


        for cc in range(dataset_ava.NUM_CLASSES):
            cur_probability = cur_preds[cc] * object_prob
            # cur_probability = cur_preds[cc]
            # cur_probability = cur_truths[cc] # what if we have every label correct

            cur_action_id = dataset_ava.TRAIN2ANN[str(cc)]['ann_id'] # this is string
            if cur_probability < 0.001:
            # if cur_probability < 0.005:
            # if cur_probability < 0.01:
                continue
            else:
                current_string = ''
                current_string += '%s,' % cur_video_id
                current_string += '%s,' % cur_timestamp
                current_string += '%.3f,%.3f,%.3f,%.3f,' % tuple(cur_box)
                current_string += '%s,' % cur_action_id
                current_string += '%.3f'% cur_probability
                output_strings.append(current_string)

    return output_strings

def filter_results_nms(result_strings):
    filtered_strings = []
    cur_video_results = {}
    prev_video_key = ''
    for result in tqdm(result_strings):
        cur_video_id, cur_timestamp, left, top, right, bottom, cur_action_id, cur_probability = result.split(',')
        left, top, right, bottom, cur_probability = [float(val) for val in [left, top, right, bottom, cur_probability]]
        cur_video_key = '%s.%s' % (cur_video_id, cur_timestamp)

        if prev_video_key != cur_video_key:
            # perform nms on collected results and set up for the next video
            # nms
            for act_id in cur_video_results.keys():
                result_list = cur_video_results[act_id]
                nms_list = non_max_suppression(result_list)
                supp_cnt = len(result_list) - len(nms_list)
                
                for nms_res in nms_list:
                    prev_vid_id, prev_timestamp = prev_video_key.split('.')
                    nms_box = nms_res['box']
                    nms_prob = nms_res['prob']

                    current_string = ''
                    current_string += '%s,' % prev_vid_id
                    current_string += '%s,' % prev_timestamp
                    current_string += '%.3f,%.3f,%.3f,%.3f,' % tuple(nms_box)
                    current_string += '%s,' % act_id
                    current_string += '%.3f'% nms_prob
                    filtered_strings.append(current_string)

            # set up for the next video
            cur_video_results = {}
            prev_video_key = cur_video_key
        
        if cur_action_id in cur_video_results.keys():
            cur_video_results[cur_action_id].append({'prob':cur_probability, 'act_id':cur_action_id, 'box':[left,top,right,bottom]})
        else:
            cur_video_results[cur_action_id] = [{'prob':cur_probability, 'act_id':cur_action_id, 'box':[left,top,right,bottom]}]
    # repeat nms for the last video
    for act_id in cur_video_results.keys():
        result_list = cur_video_results[act_id]
        nms_list = non_max_suppression(result_list)
        
        for nms_res in nms_list:
            prev_vid_id, prev_timestamp = prev_video_key.split('.')
            nms_box = nms_res['box']
            nms_prob = nms_res['prob']

            current_string = ''
            current_string += '%s,' % prev_vid_id
            current_string += '%s,' % prev_timestamp
            current_string += '%.3f,%.3f,%.3f,%.3f,' % tuple(nms_box)
            current_string += '%s,' % act_id
            current_string += '%.3f'% nms_prob
            filtered_strings.append(current_string)

    return filtered_strings
def non_max_suppression(result_list):
    # result list is a list of dicts for that particular action and video
    IoU_th = 0.5 # any IoU above is filtered out
    probs = np.array([result['prob'] for result in result_list])
    boxes = np.array([result['box'] for result in result_list])

    sorted_indices = np.argsort(probs)[::-1] # larger to smaller
    sorted_probs = probs[sorted_indices]
    sorted_boxes = boxes[sorted_indices, :]

    picks = []
    for ii in range(sorted_probs.shape[0]):
        cur_prob = sorted_probs[ii]
        cur_box = sorted_boxes[ii]

        max_IoU = 0.0
        for pp in range(len(picks)):
            pick_box = picks[pp]['box']
            IoU = dataset_ava.IoU_box(cur_box, pick_box)
            if IoU > max_IoU: max_IoU = IoU

        if max_IoU < IoU_th:
            picks.append({'box':cur_box, 'prob':cur_prob})

    return picks
            




def read_and_convert_results(results_path, output_path):
    print('Reading %s' %results_path)
    # results_path = RESULTS_PATH
    results = read_serialized_results(results_path)
    # results = results[:1000] # debug
    output_strings = convert_results(results, SPLIT)
    
    print('%.i samples before filtering' % len(output_strings))

    # with open(output_path, 'w') as fp:
    #     fp.write('\n'.join(output_strings))
    # print('Output file %s is written' % output_path)
    
    print('Filtering results with NMS')
    # print('%.i samples before filtering' % len(output_strings))
    filtered_strings = filter_results_nms(output_strings)
    print('%.i samples after filtering' % len(filtered_strings))

    # ### import pdb;pdb.set_trace()

    with open(output_path, 'w') as fp:
       fp.write('\n'.join(filtered_strings))
    print('Output file %s is written' % output_path)




def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--result_name', type=str, default=RESULT_NAME, required=False)
    args = parser.parse_args()
    result_name = args.result_name

    results_path = RESULTS_FOLDER + '%s.txt' % result_name
    output_path = OUTPUT_FOLDER + '%s.csv' % result_name
    read_and_convert_results(results_path, output_path)

    # read_and_convert_results(RESULTS_PATH, OUTPUT_PATH)

# def run_on_runname(result_name):


if __name__ == '__main__':
    main()
