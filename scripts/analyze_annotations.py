import json

# split = 'train'
split = 'val'

annotations_folder = '../AVA/annotations/'
DATA_FOLDER = '../AVA/data/'

with open(DATA_FOLDER+'label_conversions.json') as fp:
    LABEL_CONVERSIONS = json.load(fp)
ANN2TRAIN = LABEL_CONVERSIONS['ann2train']
TRAIN2ANN = LABEL_CONVERSIONS['train2ann']

annotation_file = annotations_folder + 'ava_%s_v2.1.csv' % split
with open(annotation_file) as fp:
    file_name_list = fp.read().splitlines()

action_dict = {str(ii):[] for ii in range(1,81)}

for anno_index, seg_info in enumerate(file_name_list):
    movie_key, timestamp, left, top, right, bottom, action_id  = seg_info.split(',')
    segment_key = '%s.%s' % (movie_key, timestamp)

    action_dict[action_id].append([segment_key, (left,top,right,bottom)])

action_sizes = []
for ii in ANN2TRAIN.keys():
    action_str = ANN2TRAIN[ii]['class_str']
    action_sizes.append((action_str, len(action_dict[ii])))

action_sizes.sort(key=lambda act_size: act_size[1])

print('\n'.join(['%s - %i ' % action_item for action_item in action_sizes]))

with open(DATA_FOLDER+'action_lists_%s.json' % split, 'w') as fp:
    json.dump(action_dict, fp)
