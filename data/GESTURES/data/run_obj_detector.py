import obj_wrapper as obj
import json
import os
from tqdm import tqdm
import imageio

#segments_dir = "../HumanSegments/"
segments_dir = "../SyntheticData/"
#objects_dir = "../object_detections/"
objects_dir = "../object_detections_synth/"


#with open('all_samples.json') as fp:
with open('all_synth_samples.json') as fp:
    per_actor_samples = json.load(fp)

#all_samples = []
#for actorsamples in per_actor_samples:
#    all_samples.extend(actorsamples)

#actor_set = ["s00", "s01", "s02", "s02b", "s03", "s04", "s05", "s06", "s07", "s08", "s09", "s10", "s11", "s12"]
actor_set = ["FemaleCivilian", "FemaleMilitary", "MaleCivilian", "MaleMilitary"]

ckpt_path = "/home/oytun/work/tf_object_wrapper/weights/faster_rcnn_nas_coco_2018_01_28/frozen_inference_graph.pb"
detector = obj.Object_Detector(ckpt_path)

#for actid, actor_name in enumerate(tqdm(actor_set)):
#for sample_name in tqdm(all_samples):
for actid, actor_name in enumerate(actor_set):
#  if actid<3: continue
  for sample_name in tqdm(per_actor_samples[actid]):
    #actor_name = sample_name.split('_')[0]
    vid_path = os.path.join(segments_dir, actor_name, sample_name)
    try:
        reader = imageio.get_reader(vid_path)
    except IOError:
        continue
    n = reader.get_length()

    midframe = reader.get_data(n//2)
    boxes,scores,classes,num_detections = detector.detect_objects_in_np(midframe, expand=True)

    H,W,C = midframe.shape

    cur_results = {'sample_name':sample_name, 'actor_name':actor_name, 'H':H, 'W':W, 'detections':[]}

    for ii in range(num_detections[0]):
        box_coords = boxes[0,ii] # hard code batch dimension to 1
        score = scores[0,ii]
        class_no = int(classes[0,ii])
        box_coords = [obj.get_3_decimal_float(c) for c in box_coords.tolist()]
        score = obj.get_3_decimal_float(score)
        class_str = obj.object_id2str[class_no]['name']
        cur_box = {'class_no':class_no, 'score':score, 'class_str':class_str, 'box_coords':box_coords}
        cur_results['detections'].append(cur_box)

    sample_name_no_ext = sample_name.split('.')[0]
    #output_path = objects_dir + '%s.json' % sample_name_no_ext
    output_path = objects_dir + '%s-%s.json' % (actor_name, sample_name_no_ext) # for synth

    with open(output_path, 'w') as fp:
        json.dump(cur_results, fp)

    tqdm.write("%s is written!" % output_path)
    reader.close()





