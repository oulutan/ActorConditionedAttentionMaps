# test annotation bboxes with extracted midframes and clips

import cv2
import os
import json
import numpy as np



AVA_FOLDER = os.environ['AVA_DIR'] + '/AVA'
segments_folder = AVA_FOLDER + '/segments/segments/'
annotations_folder = AVA_FOLDER + '/annotations/'
data_folder = AVA_FOLDER + '/data/'
objects_folder = AVA_FOLDER + '/objects/'
# objects_folder = AVA_FOLDER + '/ava_trained_persons/'

split = 'train'
seg_anno_file = data_folder + 'segment_annotations_%s.json' %split



def main():

    with open(seg_anno_file) as fp:
        annotations = json.load(fp)

    seg_keys = annotations.keys()
    seg_keys.sort()

    last_mov = ''
    for seg_key in seg_keys:
    # for seg_key in ['-5KQ66BBWC4.0902']:
        print('Working on %s' %seg_key)
        cur_annos = annotations[seg_key]

        movie_key, timestamp = seg_key.split('.')
        if movie_key == last_mov:
            continue
        last_mov = movie_key
        midframe_file = os.path.join(segments_folder, split, 'midframes', movie_key, timestamp+'.jpg')

        object_detection_file = os.path.join(objects_folder, split, movie_key, '%s.json' %timestamp)
        with open(object_detection_file) as fp:
            object_results = json.load(fp)

        midframe = cv2.imread(midframe_file)

        anno_frame = np.copy(midframe)
        draw_anno(anno_frame, cur_annos)

       
        obj_frame = np.copy(midframe)
        draw_objects(obj_frame, object_results['detections'])

        img_to_show = np.concatenate([obj_frame, anno_frame], axis=1)

        cv2.imshow('result', img_to_show)
        k = cv2.waitKey(0)
        if k == ord('q'):
            os._exit(0)



def draw_objects(frame, detections):
    H,W,C = frame.shape
    for bbid, bbox in enumerate(detections):
        color = colors[bbid,:]
        current_bbox = bbox['box']

        top, left, bottom, right = current_bbox
        left = int(W * left)
        right = int(W * right)

        top = int(H * top)
        bottom = int(H * bottom)

        conf = bbox['score']
        label = bbox['class_str']
        message = label + ' %.2f' % conf

        cv2.rectangle(frame, (left,top), (right,bottom), color, 2)

        font_size =  max(0.5,(right - left)/50.0/float(len(message)))
        cv2.rectangle(frame, (left, top-int(font_size*40)), (right,top), color, -1)
        cv2.putText(frame, message, (left, top-12), 0, font_size, (255,255,255)-color, 1)


colors = np.random.randint(0, 255, [100, 3])
def draw_anno(frame, bboxes):
    H,W,C = frame.shape
    for bbid, bbox in enumerate(bboxes):
        color = colors[bbid,:]
        current_bbox = bbox['bbox']

        left, top, right, bottom = current_bbox
        left = int(W * left)
        right = int(W * right)

        top = int(H * top)
        bottom = int(H * bottom)

        cv2.rectangle(frame, (left,top), (right,bottom), color, 2)


if __name__ == '__main__':
    main()
