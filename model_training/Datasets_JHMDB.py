import os

# import __main__ as main
# if os.path.basename(main.__file__) == 'result_validation.py' or os.path.basename(main.__file__) == 'visualize_ava_style_results.py':
#     print('Called from %s, not importing tensorflow' % os.path.basename(main.__file__))
# else:
#     import tensorflow as tf

import numpy as np
import json
import cv2
import imageio
import tensorflow as tf
 
from sklearn.metrics import average_precision_score

class Data_JHMDB:

    def __init__(self, batch_size=1, no_gpus=1, run_test = False):
        self.batch_size = batch_size
        self.no_gpus = no_gpus
        self.run_test = run_test

        self.PREPROCESS_CORES = 10
        self.BUFFER_SIZE = 1

        #INPUT_H = 400
        #INPUT_W = 400
        self.INPUT_H = 240
        self.INPUT_W = 320
        self.INPUT_T = 32
        # INPUT_T = 16
        
        self.KEY_FRAME_INDEX = 2
        
        self.NUM_CLASSES = 21 + 1 # 1 for background

        self.ACAM_FOLDER = os.environ['ACAM_DIR']

        self.JHMDB_FOLDER = self.ACAM_FOLDER + '/data/JHMDB' 

        self.VIDEOS_FOLDER = self.JHMDB_FOLDER + '/ReCompress_Videos/'
        self.OBJECT_DETECTIONS_FOLDER = self.JHMDB_FOLDER + '/objects/'
        self.DATA_FOLDER = self.JHMDB_FOLDER + '/data/'
        self.SPLIT_INFO_FOLDER = self.JHMDB_FOLDER + '/splits/'

        #SPLIT_NO = 1
        self.SPLIT_NO = 1

        self.MODEL_SAVER_PATH = self.JHMDB_FOLDER + '/ckpts/split_%i_model_ckpt' % self.SPLIT_NO
        self.RESULT_SAVE_PATH = self.JHMDB_FOLDER + '/ActionResults/split_%i' % self.SPLIT_NO
        
        # max amount of rois in a single image
        # this initializes the roi vector sizes as well
        self.MAX_ROIS = 100
        self.MAX_ROIS_IN_TRAINING = 20

        self.final_layer = 'softmax' # sigmoid or softmax


        # jhmdb learning rates for cosine
        # lr_max = 0.001
        # lr_min = 0.0001


        self.ALL_VIDS_FILE = self.DATA_FOLDER + 'all_vids.txt'
        with open(self.ALL_VIDS_FILE) as fp:
            self.ALL_VIDS = fp.readlines()
        self.ALL_ACTIONS = list(set([v.split(" ")[0] for v in self.ALL_VIDS]))
        self.ALL_ACTIONS.sort()

        self.ACT_STR_TO_NO = {
            'brush_hair':0,
            'catch':1,
            'clap':2,
            'climb_stairs':3,
            'golf':4,
            'jump':5,
            'kick_ball':6,
            'pick':7,
            'pour':8,
            'pullup':9,
            'push':10,
            'run':11,
            'shoot_ball':12,
            'shoot_bow':13,
            'shoot_gun':14,
            'sit':15,
            'stand':16,
            'swing_baseball':17,
            'throw':18,
            'walk':19,
            'wave':20,
            'background': 21
        }

        self.ACT_NO_TO_STR = {self.ACT_STR_TO_NO[strkey]:strkey for strkey in self.ACT_STR_TO_NO.keys()} 

        self.ANNOTATIONS_FILE = self.DATA_FOLDER + 'segment_annotations.json'
        with open(self.ANNOTATIONS_FILE) as fp:
            self.ANNOTATIONS = json.load(fp)

    def process_evaluation_results(self, res_name):
        print("Results for %s "% res_name)


    # during training a frame will be randomly selected so 
    # add the total no of frames as the current frame
    # I can use that number to randomly select the frame
    def get_train_list(self):
        train_segments = []
        for act in self.ALL_ACTIONS:
            fname = self.SPLIT_INFO_FOLDER + '%s_test_split%i.txt' % (act, self.SPLIT_NO)
            with open(fname) as fp:
                vids_info = fp.readlines()
            vids_info = [v.strip() for v in vids_info]
            # vidname 1: 1 means training
            # train_vids = ["%s %s 0" % (act, v.split(" ")[0]) for v in vids_info if v.split(" ")[1] == '1']
            train_vids = []
            for v in vids_info:
                vidname, train_test = v.split(" ")
                if train_test == '1':
                    vid_str = "%s %i" % (vidname, self.ANNOTATIONS[vidname]['nframes'])
                    train_vids.append(vid_str)

            train_segments.extend(train_vids)
        return train_segments * 20

    # during validation we will go through all frames individually. 
    def get_val_list(self):
        val_segments = []
        for act in self.ALL_ACTIONS:
            fname = self.SPLIT_INFO_FOLDER + '%s_test_split%i.txt' % (act, self.SPLIT_NO)
            with open(fname) as fp:
                vids_info = fp.readlines()
            vids_info = [v.strip() for v in vids_info]
            # vidname 1: 1 means training
            # val_vids = ["%s %s" % (act, v.split(" ")[0]) for v in vids_info if v.split(" ")[1] == '1']
            val_vids = []
            for v in vids_info:
                vidname, train_test = v.split(" ")
                if train_test == '2':
                    for ii in range(8, self.ANNOTATIONS[vidname]['nframes'] - 8):
                        vid_str = "%s %i" % (vidname, ii)
                        val_vids.append(vid_str)

            val_segments.extend(val_vids)
        return val_segments

    ## filters samples with no detected people!!!!
    def filter_no_detections(sample, labels_np, rois_np, no_det, segment_key):
        rois_bool = tf.cast(rois_np, tf.bool)
        return tf.reduce_any(rois_bool)

    def get_data(segment_key, split):
        
        sample, center_frame = self.get_video_frames(segment_key, split)
        labels_np, rois_np, no_det = self.get_labels(segment_key,split, center_frame)

        return sample, labels_np, rois_np, no_det, segment_key

    def get_video_frames(self, segment_key, split):

        vidname, frame_info = segment_key.split(' ')
        action = self.ANNOTATIONS[vidname]['action']
        if split == 'train':
            # if its train frame info is total number of frames in the segments
            center_frame = np.random.randint(low=8, high=int(frame_info)-8)
        else:
            center_frame = int(frame_info)
        

        vid_path = os.path.join(self.VIDEOS_FOLDER, action, vidname)
        try: 
            video = imageio.get_reader(vid_path, 'ffmpeg')

            sample = np.zeros([self.INPUT_T, self.INPUT_H, self.INPUT_W, 3], np.uint8)

            for ii in range(self.INPUT_T):
                cur_frame_idx = center_frame - self.INPUT_T // 2 + ii
                if cur_frame_idx < 0 or cur_frame_idx >= self.ANNOTATIONS[vidname]['nframes'] :
                    continue
                else:
                    frame = video.get_data(cur_frame_idx)
                    # frame = frame[:,:,::-1] #opencv reads bgr, i3d trained with rgb
                    #reshaped = cv2.resize(frame, (INPUT_W, INPUT_H))
                    reshaped = frame
                    sample[ii,:,:,:] = reshaped

            video.close()
        except IOError: 
            sample = np.zeros([self.INPUT_T, self.INPUT_H, self.INPUT_W, 3], np.uint8)
            print("IO ERROR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1")

        return sample.astype(np.float32), center_frame

    def get_labels(self, segment_key, split, center_frame):
        vidname, frame_info = segment_key.split(' ')
        sample_annotations = self.ANNOTATIONS[vidname]
        action = sample_annotations['action']

        ann_box = sample_annotations['frame_boxes'][center_frame]

        detection_results_file = os.path.join(self.OBJECT_DETECTIONS_FOLDER, action, "%s.json" % vidname)
        with open(detection_results_file) as fp:
            detection_results = json.load(fp)
        detection_boxes = detection_results['frame_objects'][center_frame]
        detection_boxes = [detbox for detbox in detection_boxes if detbox['class_str'] == 'person']
        
        if split == 'train':
            # detections = [det for det in detections if det['score'] > 0.20]
            if len(detection_boxes) > self.MAX_ROIS_IN_TRAINING:
                # they are sorted by confidence already, take top #k
                detection_boxes = detection_boxes[:self.MAX_ROIS_IN_TRAINING]

        labels_np, rois_np, no_det = self.match_annos_with_detections([ann_box], detection_boxes, action)

        return labels_np, rois_np, no_det

    def setup_tfdatasets(self):
        train_detection_segments = self.get_train_list()    
        split = 'train'

        #               [sample, labels_np, rois_np, no_det, segment_key] 
        output_types = [tf.uint8, tf.int64, tf.float32, tf.int64, tf.string]
        #output_types = [tf.float32, tf.int32, tf.float32, tf.int64, tf.string]

        # shuffle the list outside tf so I know the order. 
        #np.random.seed(5)
        # np.random.seed(7)
        np.random.shuffle(train_detection_segments)


        dataset = tf.data.Dataset.from_tensor_slices((train_detection_segments,[split]*len(train_detection_segments)))
        dataset = dataset.shuffle(len(train_detection_segments)//8)
        dataset = dataset.repeat()# repeat infinitely
        dataset = dataset.map(lambda seg_key, c_split: 
                tuple(tf.py_func(self.get_data, [seg_key,c_split], output_types)),
                num_parallel_calls=self.PREPROCESS_CORES)

        # dataset = dataset.interleave(lambda x: dataset.from_tensors(x).repeat(2),
        #                                 cycle_length=10, block_length=1)
        dataset = dataset.filter(self.filter_no_detections)
        #dataset = dataset.shuffle(self.batch_size * self.no_gpus * 200)
        #dataset = dataset.prefetch(buffer_size=BUFFER_SIZE * self.no_gpus)
        dataset = dataset.batch(batch_size=self.batch_size*self.no_gpus)
        dataset = dataset.prefetch(buffer_size=self.BUFFER_SIZE)
        self.training_dataset = dataset
        self.train_detection_segments = train_detection_segments
        
        if not self.run_test:
            val_detection_segments = self.get_val_list()
            split = 'val'
                
        else:
            val_detection_segments = self.get_test_list()
            split = 'test'
        

        #               [sample, labels_np, rois_np, no_det, segment_key] 
        # output_types = [tf.uint8, tf.int32, tf.float32, tf.int64, tf.string]
        #output_types = [tf.float32, tf.int32, tf.float32, tf.int64, tf.string]

        # shuffle with a known seed so that we always get the same samples while validating on like first 500 samples
        #if not self.evaluate:
        #if True:
        #    np.random.seed(10)
        #    np.random.shuffle(val_detection_segments)


        dataset = tf.data.Dataset.from_tensor_slices((val_detection_segments,[split]*len(val_detection_segments)))
        dataset = dataset.repeat()# repeat infinitely
        dataset = dataset.map(lambda seg_key, c_split: 
                tuple(tf.py_func(self.get_data, [seg_key,c_split], output_types)),
                num_parallel_calls=self.PREPROCESS_CORES * self.no_gpus)

            
        # dataset = dataset.prefetch(buffer_size=BUFFER_SIZE)
        dataset = dataset.filter(self.filter_no_detections)
        dataset = dataset.batch(batch_size=self.batch_size*self.no_gpus)
        dataset = dataset.prefetch(buffer_size=self.BUFFER_SIZE)
        self.validation_dataset = dataset
        self.val_detection_segments = val_detection_segments

        # skip validation
        # self.val_detection_segments = val_detection_segments[:200]
        


        #### configure the input selector
        self.input_handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle( self.input_handle, self.training_dataset.output_types, self.training_dataset.output_shapes)

        next_element = iterator.get_next()

        # Define shapes of the inputs coming from python functions
        input_batch, labels, rois, no_dets, segment_keys = next_element

        # input_batch = tf.cast(input_batch, tf.float32)

        input_batch.set_shape([None, self.INPUT_T, self.INPUT_H, self.INPUT_W, 3])
        #labels.set_shape([None, self.dataset_fcn.MAX_ROIS, self.dataset_fcn.NUM_CLASSES])
        labels.set_shape([None, self.MAX_ROIS_IN_TRAINING, self.NUM_CLASSES])
        #rois.set_shape([None, self.dataset_fcn.MAX_ROIS, 4])
        rois.set_shape([None, self.MAX_ROIS_IN_TRAINING, 4])
        no_dets.set_shape([None])
        segment_keys.set_shape([None])
        

        self.input_batch = input_batch
        self.labels =labels
        self.rois = rois
        self.no_dets = no_dets
        self.segment_keys = segment_keys

        return input_batch, labels, rois, no_dets, segment_keys

        
    def initialize_data_iterators(self, sess):
        self.training_iterator = self.training_dataset.make_one_shot_iterator()
        self.validation_iterator = self.validation_dataset.make_initializable_iterator()

        # The `Iterator.string_handle()` method returns a tensor that can be evaluated
        # and used to feed the `handle` placeholder.
        self.training_handle = sess.run(self.training_iterator.string_handle())
        self.validation_handle = sess.run(self.validation_iterator.string_handle())
        
        sess.run(self.validation_iterator.initializer)

    def select_iterator(self, feed_dict, is_training):
        if is_training:
            feed_dict[self.input_handle] = self.training_handle
        else:
            feed_dict[self.input_handle] = self.validation_handle



    
    def match_annos_with_detections(self, annotations, detections, action):
        # gt_boxes = []
        # for ann in annotations:
        #     # left, top, right, bottom
        #     # [0.07, 0.141, 0.684, 1.0]
        #     cur_box = ann['bbox'] 
        #     gt_boxes.append(cur_box)
        MATCHING_IOU = 0.5
        gt_boxes = annotations
    
        det_boxes = []
        for detection in detections:
            # top, left, bottom, right
            # [0.07, 0.006, 0.981, 0.317]
            # top, left, bottom, right = detection['box'] 
            # box = left, top, right, bottom
            box = detection['box']
            class_label = detection['class_str']
            det_boxes.append(box)
    
        no_gt = len(gt_boxes)
        no_det = len(det_boxes)
    
        iou_mtx = np.zeros([no_gt, no_det])
    
        for gg in range(no_gt):
            gt_box = gt_boxes[gg]
            for dd in range(no_det):
                dt_box = det_boxes[dd]
                iou_mtx[gg,dd] = IoU_box(gt_box, dt_box)
    
        # assume less than #MAX_ROIS boxes in each image
        if no_det > self.MAX_ROIS: print('MORE DETECTIONS THAN MAX ROIS!!')
        labels_np = np.zeros([self.MAX_ROIS, self.NUM_CLASSES], np.int32)
        rois_np = np.zeros([self.MAX_ROIS, 4], np.float32) # the 0th index will be used as the featmap index
    
        # TODO if no_gt or no_det is 0 this will give error
        # This is fixed within functions calling this
        if no_gt != 0 and no_det != 0:
            max_iou_for_each_det = np.max(iou_mtx, axis=0)
            index_for_each_det = np.argmax(iou_mtx, axis=0)
                

            for dd in range(no_det):
                cur_max_iou = max_iou_for_each_det[dd]

                
                top, left, bottom, right = detections[dd]['box']

                # # region of interest layer expects
                # # regions of interest as lists of:
                # # feature map index, upper left, bottom right coordinates
                rois_np[dd,:] = [top, left, bottom, right]
                if cur_max_iou < MATCHING_IOU:
                    labels_np[dd, -1] = 1 # bg class for softmax
                else:
                #matched_ann = annotations[index_for_each_det[dd]]
                    labels_np[dd, self.ACT_STR_TO_NO[action]] = 1

    
        return labels_np, rois_np, no_det

    def get_per_class_AP(self, results_list):
        '''
        results_list is a list where each
        result = ['path' [multilabel-binary labels] [probs vector]]

        returns per class_AP vector with class average precisions
        '''
        class_results = [{'truth':[], 'pred':[]} for _ in range(self.NUM_CLASSES)]

        for result in results_list:
            cur_key = result[0]
            cur_roi_id = result[1]
            cur_truths = result[2]
            cur_preds = result[3]
            
            # cur_preds = np.random.uniform(size=40)
            # cur_preds = [0 for _ in range(40)]

            for cc in range(self.NUM_CLASSES):
                class_results[cc]['truth'].append(cur_truths[cc])
                class_results[cc]['pred'].append(cur_preds[cc])

        ground_truth_count = []
        class_AP = []
        for cc in range(self.NUM_CLASSES):
            y_truth = class_results[cc]['truth']
            y_pred = class_results[cc]['pred']
            AP = average_precision_score(y_truth, y_pred)

            # print(AP)
            # plot_pr_curve(y_truth, y_pred)
            # import pdb; pdb.set_trace()

            if np.isnan(AP): AP = 0

            class_AP.append(AP)
            ground_truth_count.append(sum(y_truth))
            
        # import pdb; pdb.set_trace()
        return class_AP, ground_truth_count

    def get_class_AP_str(self, class_AP, cnt):
        ''' Returns a printable string'''
        ap_str = ''
        for cc in range(self.NUM_CLASSES):
            class_str = self.ACT_NO_TO_STR[cc][0:15] # just take the first 15 chars, some of them are too long
            class_cnt = cnt[cc]
            AP = class_AP[cc]
            # AP = AP if not np.isnan(AP) else 0
            # cur_row = '%s:    %i%% \n' %(class_str, AP*100)#class_str + ':    ' + str(tools.get_3_decimal_float(AP)) + '\n'
            cur_row = class_str + '(%i)' % class_cnt + ':'
            cur_row += (' ' * (25 -len(cur_row))) + '%.1f%%\n' % (AP*100.0)
            ap_str += cur_row
        class_avg = np.mean(class_AP)
        # class_avg = class_avg if not np.isnan(class_avg) else 0
        ap_str += '\n' + 'Average:' + (' '*17) + '%.1f%%\n' % (class_avg*100.0)
        return ap_str

    def get_AP_str(self, results_list):
        class_AP, cnt = self.get_per_class_AP(results_list)
        ap_str = self.get_class_AP_str(class_AP, cnt)
        return ap_str



def IoU_box(box1, box2):
    '''
    top1, left1, bottom1, right1 = box1
    top2, left2, bottom2, right2 = box2
 
    returns intersection over union
    '''
    # left1, top1, right1, bottom1 = box1
    # left2, top2, right2, bottom2 = box2
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
    


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    ava = Data_AVA()

    vallist = ava.get_val_list()

    # test basic functions
    segment_key = vallist[0]
    split = 'val'
    obj = ava.get_obj_detection_results(segment_key, split)
    frames = ava.get_video_frames(segment_key, split)
    np_sample = ava.get_data(segment_key, split)

    # test tfrecords
    sess = tf.Session()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    
    input_batch, labels, rois, no_dets, segment_keys = ava.setup_tfdatasets()
    
    ava.initialize_data_iterators(sess)

    feed_dict = {}
    ava.select_iterator(feed_dict, is_training=True)
    np_stuff = sess.run([input_batch, labels, rois, no_dets, segment_keys], feed_dict)

    feed_dict = {}
    ava.select_iterator(feed_dict, is_training=False)
    np_stuff = sess.run([input_batch, labels, rois, no_dets, segment_keys], feed_dict)

    # import pdb;pdb.set_trace()

    print('Tests Passed!')