import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
import json
import os
import argparse
import subprocess

import logging
from tqdm import tqdm

# import dataset_tf
import dataset_ava
import dataset_jhmdb
import process_results
import input_augmentation
import model_layers

import i3d

from tensorflow.contrib.framework.python.ops import variables
from tensorflow.python.ops import control_flow_ops
from sklearn.metrics import classification_report


import time
import cv2

DATE = time.strftime("%b-%d-time-%H-%M-%S") # Feb-10-time-7-36

NUM_EPOCHS = 10
TRAINING_ITERS = 5000
# TRAINING_ITERS = 1 # debug
VALIDATION_ITERS = 5000 # not used while balanced_validation
# BATCH_SIZE = 6
DEFAULT_BATCH_SIZE = 6


# NUM_CLASSES = 60

import socket
HOSTNAME = socket.gethostname()
# if HOSTNAME == 'skywalker':
#     PREPROCESS_CORES = 15
#     BUFFER_SIZE = 20
PREPROCESS_CORES = 5 # times number of gpus
BUFFER_SIZE = 10

ACAM_FOLDER = os.environ['ACAM_DIR']
# MAIN_FOLDER = os.environ['AVA_DIR']

# AVA_FOLDER = ACAM_FOLDER + '/data/AVA' 

# MODEL_SAVER_PATH = AVA_FOLDER + '/ckpts/model_ckpt'
# RESULT_SAVE_PATH = AVA_FOLDER + '/ActionResults/'

MODALITY = 'RGB'

# BOX_CROP_SIZE = [14, 14]
BOX_CROP_SIZE = [10, 10]
# BOX_CROP_SIZE = [7, 7]
USE_TFRECORD = True

TRAIN_FULL_MODEL = True
#TRAIN_FULL_MODEL = False

ONLY_INIT_I3D = False

GENERATE_ATTN_MAPS = False

TRACE_PERFORMANCE = False

DELAY = 0

def set_logger(model_id, evaluate, npy_seed, dataset_str):
    # npy seed is the same as the checkpoint number, so use it in logger name
    
    run_name = DATE

    if model_id: run_name = run_name + '_' + model_id
    if evaluate: run_name = run_name + '_' + 'evaluate'
    run_name = run_name + '_%.2i' % npy_seed

    log_file = ( ACAM_FOLDER +'/data/' + dataset_str.upper() + '/logs/' + run_name + '.txt')
    logging.getLogger('tensorflow').setLevel(logging.WARNING)

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s--%(funcName)s--%(message)s',
                        filename=log_file, datefmt='%I:%M:%S')
    logging.debug('\n\n-----NEW FILE RUN-----\n\n')


    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s--%(funcName)s--%(message)s', datefmt='%I:%M:%S'))
    logging.getLogger('').addHandler(console)

    logging.info('Logging to file: ' + log_file)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-l', type=float, default=1e-2, required=False)
    # parser.add_argument('-b', type=int, default=32, required=False)
    parser.add_argument('-g', type=str, required=True)
    parser.add_argument('-i', type=str, default='noid', required=False)
    parser.add_argument('-e', type=str, default='False', required=False)
    parser.add_argument('-c', '--ckpt_file', type=str, default='', required=False)
    parser.add_argument('-t', type=str, default='False', required=False)
    parser.add_argument('-s', '--seed_npy',type=int, default=0, required=False)
    parser.add_argument('-b', '--batch_size',type=int, default=DEFAULT_BATCH_SIZE, required=False)
    parser.add_argument('-a', '--architecture', type=str, default='i3d_tail', required=False)
    parser.add_argument('-d', '--dataset', type=str, default='ava', required=False) # 'ava' or 'jhmdb'

    args = parser.parse_args()

    # batch_size = args.b
    base_learning_rate = args.l
    model_id = args.i
    ckpt_file = args.ckpt_file
    evaluate = bool(args.e == 'True')
    run_test = bool(args.t == 'True')
    npy_seed = args.seed_npy
    batch_size = args.batch_size
    architecture = args.architecture
    dataset_str = args.dataset

    if run_test: # overwrite evaluate if we are testing
        evaluate = True

    if evaluate and ckpt_file == '':
        logging.error('EVALUATION MODE WITHOUT CHECKPOINT ARGUMENT!!')
        # import pdb;pdb.set_trace()


    gpu = args.g
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    ## cluster sets this up already
    # gpu = os.environ["CUDA_VISIBLE_DEVICES"]


    set_logger(model_id, evaluate, npy_seed, dataset_str)
    logging.info(   'Running ' + DATE +
                    ', on gpu ' + gpu +
                    ', for %i epochs' % NUM_EPOCHS +
                    ', with batch size %i' % batch_size + 
                    ', on numpy seed %i' % npy_seed)

    if run_test:
        logging.info('GENERATING TEST SET RESULTS!!!')

    

    # trainer.set_trainer()
    # setup cpu as default
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        trainer = Model_Trainer(base_learning_rate, ckpt_file, evaluate, run_test, model_id, gpu, npy_seed, batch_size, architecture, dataset_str)
        trainer.set_trainer()
        trainer.run_training()

class Model_Trainer():
    def __init__(self, base_learning_rate, ckpt_file, only_evaluate, run_test, model_id, gpu_str, npy_seed, batch_size, architecture_str, dataset_str):
        mygraph = tf.get_default_graph()
        tf.set_random_seed(npy_seed)
        np.random.seed(npy_seed)

        self.dataset_str = dataset_str
        if dataset_str == "ava":
            self.dataset_fcn = dataset_ava
        elif dataset_str == 'jhmdb':
            self.dataset_fcn = dataset_jhmdb

        self.base_learning_rate = base_learning_rate
        self.ckpt_file = ckpt_file
        self.evaluate = only_evaluate
        self.run_test = run_test
        self.model_id = model_id
        self.saver_path = self.dataset_fcn.MODEL_SAVER_PATH + '_%s_%s' % (model_id, dataset_str)
        self.batch_size = batch_size
        self.architecture_str = architecture_str
        

        self.no_gpus = len(gpu_str.split(','))

    def set_trainer(self):
        self.set_data_inputs()
        self.set_multi_gpu_trainer()
        self.set_session()


    def set_data_inputs(self):
        train_detection_segments = self.dataset_fcn.get_train_list()
        split = 'train'

        #               [sample, labels_np, rois_np, no_det, segment_key] 
        # output_types = [tf.uint8, tf.int32, tf.float32, tf.int64, tf.string]
        output_types = [tf.float32, tf.int32, tf.float32, tf.int64, tf.string]

        # shuffle the list outside tf so I know the order. 
        #np.random.seed(5)
        # np.random.seed(7)
        np.random.shuffle(train_detection_segments)


        if not USE_TFRECORD:
            dataset = tf.data.Dataset.from_tensor_slices((train_detection_segments,[split]*len(train_detection_segments)))
            dataset = dataset.repeat()# repeat infinitely
            dataset = dataset.map(lambda seg_key, c_split: 
                    tuple(tf.py_func(self.dataset_fcn.get_data, [seg_key,c_split], output_types)),
                    num_parallel_calls=PREPROCESS_CORES * self.no_gpus)
        else:
            tfrecord_list = dataset_ava.generate_tfrecord_list(train_detection_segments, 'train')
            dataset = tf.data.TFRecordDataset(tfrecord_list)
            dataset = dataset.repeat()# repeat infinitely
            dataset = dataset.map(dataset_ava.get_tfrecord_train, 
                    num_parallel_calls=PREPROCESS_CORES * self.no_gpus)
        # repeat infinitely and shuffle with seed
        # dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=1000, count=-1, seed=1))
        # dataset = dataset.flat_map(lambda x: tf.data.Dataset.from_tensors(x).repeat(2))
        # dataset = dataset.shuffle(BATCH_SIZE * self.no_gpus * 4)
        # import pdb;pdb.set_trace()
        # dataset = dataset.interleave(lambda x: dataset.from_tensors(x).repeat(2),
        #                                 cycle_length=10, block_length=1)
        # dataset = dataset.prefetch(buffer_size=BUFFER_SIZE)
        dataset = dataset.shuffle(self.batch_size * self.no_gpus * 20)
        dataset = dataset.batch(batch_size=self.batch_size*self.no_gpus)
        dataset = dataset.prefetch(buffer_size=BUFFER_SIZE)
        self.training_dataset = dataset
        self.train_detection_segments = train_detection_segments
        
        if not self.run_test:
            val_detection_segments = self.dataset_fcn.get_val_list()
            split = 'val'
                
        else:
            val_detection_segments = self.dataset_fcn.get_test_list()
            split = 'test'
        
        ## DEBUGGING
        # val_detection_segments = ['1j20qq1JyX4.1108']
        # val_detection_segments = [seg for seg in val_detection_segments if seg.startswith('Ov0za6Xb1LM')]
        # val_detection_segments = [seg for seg in val_detection_segments if seg.startswith('_eBah6c5kyA')]
        # val_detection_segments = [seg for seg in val_detection_segments if seg.startswith('om_83F5VwTQ.11')]
        # val_detection_segments = [seg for seg in val_detection_segments if seg.startswith('2PpxiG0WU18.091')] # 3 people scene talking
        # val_detection_segments = [seg for seg in val_detection_segments if seg.startswith('1j20qq1JyX4.112')]
        # val_detection_segments = [seg for seg in val_detection_segments if seg.startswith('55Ihr6uVIDA.092')] # Carry
        # val_detection_segments = [seg for seg in val_detection_segments if seg.startswith('9F2voT6QWvQ.102')] # Dance
        # val_detection_segments = [seg for seg in val_detection_segments if seg.startswith('Di1MG6auDYo.13')] # Answer Phone
        # val_detection_segments = [seg for seg in val_detection_segments if seg.startswith('Di1MG6auDYo.107')] # Eat
        # val_detection_segments = [seg for seg in val_detection_segments if seg.startswith('uNT6HrrnqPU.172')] # Kiss
        # val_detection_segments = [seg for seg in val_detection_segments if seg.startswith('IzvOYVMltkI.163')] # Ride
        # val_detection_segments = [seg for seg in val_detection_segments if seg.startswith('Di1MG6auDYo.142')] # Read
        # val_detection_segments = [seg for seg in val_detection_segments if seg.startswith('CMCPhm2L400.113')] # Run
        # val_detection_segments = [seg for seg in val_detection_segments if seg.startswith('er7eeiJB6dI.135')] # Bend
        # val_detection_segments = [seg for seg in val_detection_segments if seg.startswith('_eBah6c5kyA.118')] # Sleep
        # val_detection_segments = [seg for seg in val_detection_segments if seg.startswith('Hscyg0vLKc8.104')] # Fight
        # val_detection_segments = [seg for seg in val_detection_segments if seg.startswith('9Y_l9NsnYE0.133')] # Listen
        # val_detection_segments = [seg for seg in val_detection_segments if seg.startswith('rXFlJbXyZyc.154')] # Talk
        # val_detection_segments = CORRECT_DETECTIONS
        # val_detection_segments = MISSED_DETECTIONS
        ## DEBUGGING

        #               [sample, labels_np, rois_np, no_det, segment_key] 
        # output_types = [tf.uint8, tf.int32, tf.float32, tf.int64, tf.string]
        output_types = [tf.float32, tf.int32, tf.float32, tf.int64, tf.string]

        # shuffle with a known seed so that we always get the same samples while validating on like first 500 samples
        #if not self.evaluate:
        if True:
            np.random.seed(10)
            np.random.shuffle(val_detection_segments)

        if not USE_TFRECORD:
            dataset = tf.data.Dataset.from_tensor_slices((val_detection_segments,[split]*len(val_detection_segments)))
            dataset = dataset.repeat()# repeat infinitely
            dataset = dataset.map(lambda seg_key, c_split: 
                    tuple(tf.py_func(self.dataset_fcn.get_data, [seg_key,c_split], output_types)),
                    num_parallel_calls=PREPROCESS_CORES * self.no_gpus)
        else:
            if not self.run_test:
                tfrecord_list = dataset_ava.generate_tfrecord_list(val_detection_segments, 'val')
                dataset = tf.data.TFRecordDataset(tfrecord_list)
                dataset = dataset.repeat()# repeat infinitely
                dataset = dataset.map(dataset_ava.get_tfrecord_val, 
                        num_parallel_calls=PREPROCESS_CORES * self.no_gpus)
            else:
                tfrecord_list = dataset_ava.generate_tfrecord_list(val_detection_segments, 'test')
                dataset = tf.data.TFRecordDataset(tfrecord_list)
                dataset = dataset.repeat()# repeat infinitely
                dataset = dataset.map(dataset_ava.get_tfrecord_test,
                        num_parallel_calls=PREPROCESS_CORES * self.no_gpus)
        # dataset = dataset.prefetch(buffer_size=BUFFER_SIZE)
        dataset = dataset.batch(batch_size=self.batch_size*self.no_gpus)
        dataset = dataset.prefetch(buffer_size=BUFFER_SIZE)
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

        input_batch.set_shape([None, self.dataset_fcn.INPUT_T, self.dataset_fcn.INPUT_H, self.dataset_fcn.INPUT_W, 3])
        labels.set_shape([None, self.dataset_fcn.MAX_ROIS, self.dataset_fcn.NUM_CLASSES])
        rois.set_shape([None, self.dataset_fcn.MAX_ROIS, 4])
        no_dets.set_shape([None])
        segment_keys.set_shape([None])
        

        self.input_batch = input_batch
        self.labels =labels
        self.rois = rois
        self.no_dets = no_dets
        self.segment_keys = segment_keys
        
        pass

    def set_multi_gpu_trainer(self):
        input_batch = self.input_batch
        labels = self.labels
        rois = self.rois
        no_dets = self.no_dets
        self.is_training = tf.placeholder(tf.bool, [], name='TrainFlag')

        # TODO make i3d better
        # model = i3d_model.I3D_model(modality=MODALITY, num_classes=NUM_CLASSES)
        # self.model = model

        # Choose variables to train if we are not using the TRAIN_FULL_MODEL flag
        self.var_identifiers_for_training = [
            # 'Conv3d_1a_7x7',
            # # 'MaxPool3d_2a_3x3',
            # 'Conv3d_2b_1x1',
            # 'Conv3d_2c_3x3',
            # # 'MaxPool3d_3a_3x3',
            # 'Mixed_3b',
            # 'Mixed_3c',
            # # 'MaxPool3d_4a_3x3',
            # 'Mixed_4b',
            # 'Mixed_4c',
            # 'Mixed_4d',
            # 'Mixed_4e',
            # 'Mixed_4f',
            # 'MaxPool3d_5a_2x2',
            # 'Mixed_5b',
            # 'Mixed_5c',
            'CLS_Logits',
        ]

        # Initialize the optimizer
        with tf.variable_scope('Optimization'):
            global_step = tf.train.get_or_create_global_step()
            increment_global_step_op = tf.assign(global_step, global_step+1)

            self.global_step = global_step
            self.increment_global_step_op = increment_global_step_op

            # # exponential decay learning rate
            # decay_step = 5 # every decay_step epochs
            # decay_rate = 0.01
            # learning_rate = tf.train.exponential_decay(
            #     base_learning_rate,  # Base learning rate.
            #     global_step,         # Current index into the dataset.
            #     decay_step,          # Decay step.
            #     decay_rate,                 # Decay rate.
            #     staircase=True)
            # logging.info('Decreasing learning rate every ' + str(decay_step) + ' epochs with multiplier ' + str(decay_rate) + '. Base Learning Rate: ' + str(base_learning_rate))

            # placeholder learning rate to feed it using val loss
            self.learning_rate = tf.placeholder(tf.float32, shape=[], name='Learning_Rate')
            self.optimizer = tf.train.AdamOptimizer( learning_rate=self.learning_rate, epsilon=1e-1)
            # self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9)
            # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

        # Initialize lists to keep track of gpu outputs
        self.logits_list = []
        self.pred_probs_list = []
        self.loss_list = []
        self.gradients_list = []

        self.labels_list = []

        #### Generate inference graphs for each gpu
        with tf.variable_scope(tf.get_variable_scope()):
          for gg in range(self.no_gpus):
            # do these on cpu
            start_index = gg*self.batch_size
            end_index = (gg+1)*self.batch_size
            # slice the inputs
            cur_input_seq = input_batch[start_index:end_index]
            cur_labels = labels[start_index:end_index]
            cur_rois = rois[start_index:end_index]
            # cur_no_dets = no_dets[start_index:end_index]

            # setup mappings and rois for each gpu batch , nz:nonzero
            rois_nz, labels_nz, batch_indices_nz = model_layers.combine_batch_rois(cur_rois, cur_labels)

            with tf.device('/gpu:%d' % gg):
              with tf.name_scope('%s_%d' % ('GPU', gg)):
                # get the logits
                cur_logits = self.single_tower_inference(cur_input_seq, rois_nz, batch_indices_nz)
                self.logits_list.append(cur_logits)

                # get the loss
                # cur_loss, cur_pred_probs = self.single_tower_loss(labels_nz, cur_logits)
                cur_loss, cur_pred_probs = self.single_tower_loss(labels_nz, cur_logits, no_dets, batch_indices_nz)
                self.pred_probs_list.append(cur_pred_probs)
                self.loss_list.append(cur_loss)

                if gg == 0:
                    self.regularizers = self.get_regularization_val()
                    # self.regularizers = tf.constant(0.0)

                cur_combined_loss = cur_loss + self.regularizers
                # cur_combined_loss = cur_loss 
                # import pdb;pdb.set_trace()

                # Reuse variables for the next tower.
                tf.get_variable_scope().reuse_variables()

                cur_gradients = self.optimizer.compute_gradients(cur_combined_loss)

                self.gradients_list.append(cur_gradients)

                self.labels_list.append(labels_nz)

        # Calculate the average gradient across towers
        self.average_grads = self.average_gradients(self.gradients_list)

        # Filter gradients if we are not training on every layer
        if not TRAIN_FULL_MODEL:
            filtered_grads = self.filter_gradients(self.average_grads)
            self.average_grads = filtered_grads
            
            logging.info('Optimizing following weights:\n'+'\n'.join(var.name + ': ' + str(var.shape) for grad,var in self.average_grads))
        else:
            logging.info('Optimizing ALL weights: \n'+'\n'.join(var.name + ': ' + str(var.shape) for grad,var in self.average_grads))

        #with tf.device('/cpu:0'):
        self.optimization_op = self.optimizer.apply_gradients(self.average_grads)

        logging.info('Not Updating batchnorm')
        # bn_layers_to_update = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # if bn_layers_to_update: 
        #    logging.info('Updating batchnorm')
        #    with tf.control_dependencies(bn_layers_to_update):
        #        self.optimization_op = self.optimization_op
        

        # Generate the iteration stats
        self.pred_probs = tf.concat(self.pred_probs_list, axis=0)
        self.input_labels = tf.concat(self.labels_list, axis=0)
        

        self.loss_val = tf.add_n(self.loss_list)   # / float(self.no_gpus)

        pred_probs = self.pred_probs
        input_labels = self.input_labels

        preds = tf.cast(tf.greater_equal(pred_probs, 0.5), tf.int32)
        correct_preds = tf.logical_and(tf.equal(input_labels, preds), tf.cast(input_labels, tf.bool))
        false_positives = tf.logical_and(tf.equal(1-input_labels, preds), tf.cast(1-input_labels, tf.bool))
        no_false_positives = tf.reduce_sum(tf.cast(false_positives, tf.float32))
        no_correct_preds = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
        no_labels = tf.reduce_sum(input_labels)

        self.preds = preds
        self.correct_preds = correct_preds
        self.no_correct_preds = no_correct_preds
        self.no_false_positives = no_false_positives
        self.no_labels = no_labels

        # import pdb;pdb.set_trace()
        pass

        

    def single_tower_inference(self, cur_input_seq, cur_rois, cur_b_idx):
        # end_point = 'Mixed_4e'
        end_point = 'Mixed_4f'
        # end_point = 'Mixed_3c'
        # end_point = 'Mixed_5c'
        self.end_point = end_point

        logging.info('I3D end point is %s' % end_point)

        
        augmented_sequence, augmented_rois = input_augmentation.augment_input_sequences(cur_input_seq, cur_rois)
        self.original_seq = cur_input_seq
        self.augmented_seq = augmented_sequence
        cur_input_seq = tf.cond(self.is_training, lambda:augmented_sequence, lambda: cur_input_seq)
        cur_rois = tf.cond(self.is_training, lambda:augmented_rois, lambda: cur_rois)

        #### INITIAL FEATURES
        _, end_points = i3d.inference(cur_input_seq, self.is_training, self.dataset_fcn.NUM_CLASSES, end_point=end_point)
        features = end_points[end_point]
        # import pdb;pdb.set_trace()
        self.feature_temp_len = feature_temp_len = int(features.shape[1])


        augmented_rois = input_augmentation.augment_box_coords(cur_rois)
        # make sure they stay within boundaries
        # shifted_rois = tf.clip_by_value(shifted_rois, 0.0, 1.0)
        shifted_rois = tf.cond(self.is_training, lambda:augmented_rois, lambda: cur_rois)

        # debugging
        self.shifted_rois = shifted_rois
        self.cur_rois = cur_rois

        ## aug debug
        # self.regular_keyframe_rois = dataset_tf.temporal_roi_cropping(cur_input_seq, cur_rois, cur_b_idx, [100,100])[:,16,:]
        # self.shifted_keyframe_rois = dataset_tf.temporal_roi_cropping(cur_input_seq, shifted_rois, cur_b_idx, [100,100])[:,16,:]

        

        # box_features = dataset_tf.temporal_roi_cropping(features, cur_rois, cur_b_idx, BOX_CROP_SIZE)
        # box_features = dataset_tf.temporal_roi_cropping(features, shifted_rois, cur_b_idx, BOX_CROP_SIZE)

        ## Args model selection
        logging.info('Using model architecture: %s' % self.architecture_str )
        class_feats = model_layers.choose_roi_architecture(self.architecture_str, features, shifted_rois, cur_b_idx, self.is_training)
        


        logging.info('Using Dropout')
        class_feats_drop = tf.layers.dropout(inputs=class_feats, rate=0.4, training=self.is_training, name='CLS_DROP1')
        logits = tf.layers.dense(inputs=class_feats_drop, 
                                 units=self.dataset_fcn.NUM_CLASSES, 
                                 activation=None, 
                                 name='CLS_Logits', 
                                 kernel_initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.01))

        # logits = tf.layers.dense(inputs=class_feats, units=NUM_CLASSES, activation=None, name='CLS_Logits')

        return logits


    def single_tower_loss(self,cur_labels, cur_logits, no_dets, cur_b_idx):
        # Losses
        with tf.variable_scope('Losses'):
            input_labels = cur_labels
            logits = cur_logits

            fl_input_labels = tf.cast(input_labels, tf.float32)
            # loss_val = tf.losses.softmax_cross_entropy(labels_one_hot, logits)
            # pred_probs = tf.nn.softmax(logits)
            if self.dataset_str == 'jhmdb':
                pred_probs = tf.nn.softmax(logits)
                logging.info("Optimizing on Softmax Loss")
                softmax_loss = tf.nn.softmax_cross_entropy_with_logits(labels=fl_input_labels,
                                                                logits=logits)
                per_roi_loss = softmax_loss
            else:
            #if True:
            # In our case each logit is a probability itself
                pred_probs = tf.nn.sigmoid(logits)
                logging.info("Optimizing on Sigmoid X-Entropy loss! ")
                sigmoid_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=fl_input_labels,
                                                                logits=logits)
                per_roi_loss = tf.reduce_mean(sigmoid_loss, axis=1)
                #focus_on_classes = [dataset_ava.ANN2TRAIN[str(action)]['train_id'] for action in range(15,64) if str(action) in dataset_ava.ANN2TRAIN.keys()]#all object classes
                #focus_on_classes = [dataset_ava.ANN2TRAIN[str(action)]['train_id'] for action in range(62,63) if str(action) in dataset_ava.ANN2TRAIN.keys()]#work on computer
                #logging.info('ONLY FOCUSING ON FOLLOWING CLASSES')
                #logging.info([dataset_ava.TRAIN2ANN[(str(cc))]['class_str'] for cc in focus_on_classes])
                #class_filter = np.zeros([60])
                #class_filter[np.array(focus_on_classes)] = 1.
                #per_roi_loss = tf.reduce_sum(sigmoid_loss*class_filter, axis=1) / tf.cast(tf.reduce_sum(class_filter), tf.float32)
            # loss_val = tf.reduce_mean(-tf.reduce_sum(tf.cast(input_labels, tf.float32) * tf.log(tf.clip_by_value(pred_probs, 1e-10, 1e10)), reduction_indices=[1]))
            pred_probs = tf.clip_by_value(pred_probs, 1e-5, 1.0 - 1e-5)
            # pred_probs = tf.Print(pred_probs, [pred_probs], 'pred_probs:')
            # sigmoid_cross_entropy = fl_input_labels * tf.log(pred_probs)
            # sigmoid_cross_entropy += (1.0-fl_input_labels) * tf.log(1-pred_probs)
            # loss_val = tf.reduce_mean(tf.reduce_mean(-sigmoid_cross_entropy))

            # ## Sigmoid Xentropy Loss
            # logging.info("Optimizing on Sigmoid X-Entropy loss! ")
            # sigmoid_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=fl_input_labels,
            #                                                     logits=logits)
            # loss_val = tf.reduce_mean(sigmoid_loss)

            ## Sigmoid Xentropy averaged on samples, this allows having same gradients on multiple gpu training
            #logging.info("Optimizing on Sigmoid X-Entropy loss! ")
            #sigmoid_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=fl_input_labels,
            #                                                    logits=logits)

            #per_roi_sigmoid_loss = tf.reduce_mean(sigmoid_loss, axis=1)

            no_dets = tf.cast(no_dets, tf.int32)
            total_no_of_detections = tf.reduce_sum(no_dets) # this is no_dets across all gpus

            loss_val = tf.reduce_sum(per_roi_loss) / tf.cast(total_no_of_detections, tf.float32)

            # # calculate the per RoI weight. We are doing this becuase we average on RoIs not samples
            # # I want each sample to have same weight compared to each RoI. Beacuse each samples can generated multiple roi proposals
            # cur_no_dets = tf.cast(cur_no_dets, tf.float32)
            # weights = 1.0 / tf.cast(tf.gather(cur_no_dets, cur_b_idx, axis=0, name='loss_weights'), tf.float32)
            # B = tf.cast(tf.shape(cur_no_dets)[0], tf.float32)
            # weighted_sigmoid_loss = tf.reduce_sum(per_roi_sigmoid_loss * weights) / B

            # loss_val = weighted_sigmoid_loss

            # Per class sigmoid loss
            # recall_ce = fl_input_labels * tf.log(pred_probs)
            # precision_ce = (1.0-fl_input_labels) * tf.log(1-pred_probs)

            # ### Focal Loss
            # # since focal loss is - (1-pt)^g log(pt) and CE = og(pt)
            # # we can just calculate the modulating term (1-pt)^g and multiply it with sigmoid loss above
            # logging.info("Optimizing on Sigmoid X-Entropy loss! ")
            # sigmoid_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=fl_input_labels,
            #                                                     logits=logits)
            # pt = fl_input_labels * pred_probs + (1-fl_input_labels) * (1-pred_probs)
            # gamma = 0.5
            # focal_mod = (1 - pt)**gamma
            # focal_loss = focal_mod * sigmoid_loss
            # logging.info("Multiplying the loss with Focal Loss modulator")

            # loss_val = tf.reduce_mean(focal_loss)

            #logging.info("Optimizing on F1-loss! ")
            #eps = 1e-8
            #num = 2.0 * tf.reduce_sum(pred_probs * fl_input_labels) + eps
            #den = tf.reduce_sum(pred_probs) + tf.reduce_sum(fl_input_labels) + eps
            ##f1_loss = 1.0 - 2.0 * tf.reduce_sum(pred_probs * fl_input_labels) / (tf.reduce_sum(pred_probs) + tf.reduce_sum(fl_input_labels))
            #f1_loss = 1.0 - num/den
            #loss_val = f1_loss


            # dummy regu
            # regularizers = tf.constant(0, dtype=tf.float32)
             
            # feat_sel_vars = [var for var in tf.trainable_variables() if 'feat_sel' in var.name]
            # feat_sel_regu = sum(tf.nn.l2_loss(var) for var in feat_sel_vars)
            # feat_sel_lambda = 0.005

            total_loss = loss_val

        return total_loss, pred_probs

    def get_regularization_val(self):
        # Regularization
        vars_ids_for_NO_reg = ['CLS_Logits']
        # vars_ids_for_NO_reg = ['batch_norm']
        vars_to_reg = []

        for var in tf.trainable_variables():
            varname = var.name
            no_reg_flag_list = [var_id in varname for var_id in vars_ids_for_NO_reg]
            no_reg_flag = any(no_reg_flag_list)
            if not no_reg_flag:
                vars_to_reg.append(var)

        # if we are not training the whole model
        # only regularize the trainable ones # nevermind they are not in gradients so they wont be updated anyways
        # if not TRAIN_FULL_MODEL:
        #     trainable_vars = []
        #     for var in vars_to_reg:
        #         varname = var.name
        #         found = False
        #         for var_id in self.var_identifiers_for_training:
        #             if var_id in varname:
        #                 found=True
        #                 break
        #         if found:
        #             trainable_vars.append(var)
        #     
        #     vars_to_reg = trainable_vars

        logging.info('Regularizing following weights:\n'+'\n'.join(var.name for var in vars_to_reg))


        regularizers = sum(tf.nn.l2_loss(var) for var in vars_to_reg)

        # regu_constant = 1e-6
        if self.dataset_str == "ava":
            regu_constant = 1e-7
        elif self.dataset_str == "jhmdb":
            regu_constant = 1e-6
        # regu_constant = 1e-8
        # regu_constant = 0.
        regularizers = regu_constant * regularizers

        ## bypass regularizers
        # self.regularizers = tf.constant(0, dtype=tf.float32)
        # regularizers = tf.constant(0.0, dtype=tf.float32)

        return regularizers

    def filter_gradients(self, average_grads):
        ''' Called only when not TRAIN_FULL_MODEL
        '''
        filtered_gradients = []
        for grad, var in average_grads:
            found = False
            for var_id in self.var_identifiers_for_training:
                if var_id in var.name:
                    found=True
                    break
            if found:
                filtered_gradients.append((grad, var))

        return filtered_gradients

    def average_gradients(self, tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
            tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
            List of pairs of (gradient, variable) where the gradient has been averaged
            across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):

            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            variables = []
            for g, v in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

                variables.append(v)

            

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            # grad = tf.reduce_mean(grad, 0)
            grad = tf.reduce_sum(grad, 0) # loss is already divided by no of rois, so gradients are already averaged

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)

        
        return average_grads

    def set_session(self):

        # set seed
        # tf.set_random_seed(1234)
        # base_learning_rate = self.base_learning_rate
        ckpt_file = self.ckpt_file
        init_op = tf.global_variables_initializer()

        model_saver = tf.train.Saver(max_to_keep=15)
        self.model_saver = model_saver
        logging.info('Saving model to %s' % self.saver_path)

        hour = DELAY
        logging.info('Waiting %i hours' % hour)
        for tt in range(hour):
            time.sleep(60*60)
            logging.info('%i hour passed' % (tt+1))

        
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_options)#, log_device_placement=True)
        sess = tf.Session(config=config)
        # sess = tf.Session()
        self.sess = sess

        # You can use feedable iterators with a variety of different kinds of iterator
        # (such as one-shot and initializable iterators).
        self.training_iterator = self.training_dataset.make_one_shot_iterator()
        self.validation_iterator = self.validation_dataset.make_initializable_iterator()

        # The `Iterator.string_handle()` method returns a tensor that can be evaluated
        # and used to feed the `handle` placeholder.
        self.training_handle = sess.run(self.training_iterator.string_handle())
        self.validation_handle = sess.run(self.validation_iterator.string_handle())

        sess.run(init_op)

        i3d_ckpt = ACAM_FOLDER + '/model_training/weights/i3d_rgb_imagenet/model.ckpt'
        # Initialize weights
        i3d.initialize_weights(sess, i3d_ckpt)

        # Initialize the TAIL
        i3d.initialize_tail(sess, i3d_ckpt)

        # Initialize I3D feature extractor
        # import pdb;pdb.set_trace()
        # self.initialize_i3d_extractor()

        tfrandno = tf.random_uniform([1]) * 1000
        logging.info('This is a TF random number %i' % sess.run(tfrandno))

        # Load the checkpoint if the argument exists
        if ckpt_file:
            if ONLY_INIT_I3D == False:
                model_saver.restore(sess, ckpt_file)
                logging.info('Loading model checkpoint from: ' + ckpt_file)
                #custom_loader(sess,ckpt_file)
                #logging.info('Loading using CUSTOM saver and  model checkpoint from: ' + ckpt_file)

            else:
                i3d.initialize_all_i3d_from_ckpt(sess, ckpt_file)
            if not self.evaluate:
                sess.run(self.increment_global_step_op) # increment so that you dont override the ckpt

        g_step = sess.run(self.global_step)

        if not self.evaluate:
            logging.info('--------- Starting New Training Session ---------')
        else:
            logging.info('--------- Evaluating Checkpoint ---------')
        
        logging.info('Global Step: ' + str(g_step))
        # logging.info('Current learning rate is %.5f' % sess.run(self.learning_rate))

########## Model Running #############


    def run_training(self):
        val_losses = []
        tr_losses = []
        # self.current_learning_rate = self.base_learning_rate
        ## Loop for epochs
        for ee in range(NUM_EPOCHS):
            # self.current_learning_rate = self.base_learning_rate
            ### Cosine learning rate
            g_step = self.sess.run(self.global_step)
            lr_max = 0.02
            lr_min = 0.0001
            reset_interval = 10
            self.current_learning_rate = lr_min + 1/2. * (lr_max - lr_min) * (1 + np.cos(np.pi * g_step/reset_interval))
            logging.info('Current learning rate is %f' % self.current_learning_rate)
            if not self.evaluate:
                #### Training ####
                tr_loss, step = self.run_epoch(training_flag=True, epoch_no=ee)
                tr_losses.append(tr_loss/float(step))
                logging.info('Training Losses so far: ')
                logging.info(tr_losses)
                logging.info('\n')

                # report = '\n\n****** Training Report with loss %.3f ******\n\n' % (tr_loss/float(step))
                # report += '\n' + classification_report(cor_labels, predictions, target_names=class_names_list)
                # logging.info(report)
            if not self.evaluate:    
                # brain
                # Save model
                g_step = self.sess.run(self.global_step)
                self.model_saver.save(self.sess, self.saver_path, global_step=g_step)
                logging.info('Checkpoint Saved to ' + self.saver_path + '-' + str(g_step))


            #### Validation ####
            val_loss, step = self.run_epoch(training_flag=False, epoch_no=ee)
            val_losses.append(val_loss/float(step))
            logging.info('Validation Losses so far: ')
            logging.info(val_losses)
            logging.info('\n')

            # val loss checking for learning rate update
            val_loss_diff = np.diff(val_losses)
            # if last n steps dont improve val loss reduce learning rate
            # n = 5
            # if len(val_loss_diff) >= n and np.all(val_loss_diff[-n:] > 0):
            #     self.current_learning_rate *= 0.5
            #     logging.info('Reduced learning rate to %f' % self.current_learning_rate)
            logging.info('Current learning rate is %f' % self.current_learning_rate)

            # report = '\n\n****** Validation Report with loss %.3f ******\n\n' % (val_loss/float(step))
            # report += '\n' + classification_report(cor_labels, predictions, target_names=class_names_list)
            # logging.info(report)
            if self.model_id: logging.info('Model id is: ' + self.model_id)

            if True: # self.evaluate:
                ## process the final results and break
                g_step = self.sess.run(self.global_step)
                res_name = 'VALIDATION' + '_Results_'+ self.model_id +'_%.2i' % g_step
                # process_ava_style_results(res_name)
                self.dataset_fcn.process_evaluation_results(res_name)
            if self.evaluate:
                break

            else:
                logging.info('END OF EPOCH! \n\n')
                self.sess.run(self.increment_global_step_op)
                g_step = self.sess.run(self.global_step)
                logging.info('Global Step is %i' % g_step)
                # logging.info('Current learning rate is %.5f' % self.sess.run(self.learning_rate))



    def run_epoch(self, training_flag, epoch_no):
        # import pdb;pdb.set_trace()
        run_dict = {'loss_val': self.loss_val,
                    'regularization': self.regularizers,
                    'no_correct_preds': self.no_correct_preds,
                    'no_labels': self.no_labels,
                    'no_false_positives': self.no_false_positives,
                    'preds': self.preds,
                    'pred_probs': self.pred_probs,

                    'roi_labels': self.input_labels,
                    'segment_keys': self.segment_keys,
                    'no_dets': self.no_dets,
                    # 'correct_preds': self.correct_preds,
                    }
        feed_dict = {}

        ### aug debug
        # run_dict['shifted_rois'] = self.shifted_rois
        # run_dict['regular_rois'] = self.cur_rois
        # run_dict['original_seq'] = self.original_seq
        # run_dict['augmented_seq'] = self.augmented_seq
        if (self.architecture_str == 'soft_attn'  or self.architecture_str == 'single_attn') and GENERATE_ATTN_MAPS:
            run_dict['attention_map'] = tf.get_collection('attention_map')[0]
            run_dict['feature_activations'] = tf.get_collection('feature_activations')[0]
            run_dict['final_i3d_feats'] = tf.get_collection('final_i3d_feats')[0]
            run_dict['cls_weights'] = [var for var in tf.global_variables() if var.name == "CLS_Logits/kernel:0"][0]
            run_dict['input_batch'] = self.input_batch
            run_dict['rois'] = self.rois

        if training_flag: 
            run_dict['optimization_op'] = self.optimization_op
            feed_dict[self.input_handle] = self.training_handle
            logging.info('Training data')
        else:
            # init validation so we get same val data each time
            self.sess.run(self.validation_iterator.initializer)
            feed_dict[self.input_handle] = self.validation_handle
            logging.info('Validation data')


        # Initialize
        step = 0
        epoch_loss = 0.0
        
        no_correct_predictions = 0
        no_total_labels = 0
        no_total_false_positives = 0

        # predictions = []
        # cor_labels = []
        duration = 0.0

        all_results = [] # each term is a list itself, first term is the vid path

        # Every epoch go through every sample
        train_iters = len(self.train_detection_segments) // (self.batch_size * self.no_gpus)

        ## validate on everything
        # val_iters = len(self.val_detection_segments) // (BATCH_SIZE * self.no_gpus)
        # num_iters = TRAINING_ITERS if training_flag else val_iters

        # just on a few initial samples
        val_iters = VALIDATION_ITERS // (self.batch_size * self.no_gpus)
        # num_iters = TRAINING_ITERS if training_flag else val_iters
        num_iters = train_iters if training_flag else val_iters

        if not training_flag:
            num_iters = len(self.val_detection_segments) // (self.batch_size * self.no_gpus)
            logging.info('Evaluating on balanced val subset')
        if self.evaluate:
            num_iters = len(self.val_detection_segments) // (self.batch_size * self.no_gpus)
            # num_iters = 20
            logging.info('Evaluating on full validation set')

        # pbar = tqdm(total=len(data_prov.data_list)//BATCH_SIZE, ncols=100)
        pbar = tqdm(total=num_iters, ncols=100)

        # timeline
        if TRACE_PERFORMANCE:
            self.timeline_dict = None
        
        for ii in range(num_iters) :
            start_time = time.time()

            step += 1
            
            # feed_dict = {self.input_seq: feats_batch_np,
            #             self.input_labels: labels_batch_np,
            #             self.rois_tf: new_rois,
            #             self.mapping_tf: new_mapping,
            #             self.is_training: training_flag,
            #             self.learning_rate: self.current_learning_rate
            #             }
            feed_dict[self.is_training] = training_flag
            feed_dict[self.learning_rate] = self.current_learning_rate
            #import pdb;pdb.set_trace()
            
            # for oom cases
            if TRACE_PERFORMANCE:
                run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True, trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
            else:
                run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
                run_metadata = None

            # Run training
            out_dict = self.sess.run(run_dict, feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
            if GENERATE_ATTN_MAPS:
                roi_probs = out_dict['pred_probs']
                for nnn in range(out_dict['no_dets'][0]):
                    print('\n')
                    print( ', '.join(out_dict['segment_keys']))
                    print([(dataset_ava.TRAIN2ANN[str(ccc)]['class_str'], get_3_decimal_float(roi_probs[nnn][ccc])) for ccc in range(60) if
                           get_3_decimal_float(roi_probs[nnn][ccc]) > 0.1])
                    print("Labels: " + ",".join([dataset_ava.TRAIN2ANN[str(ccc)]['class_str'] for ccc in range(60) if out_dict['roi_labels'][nnn][ccc]==1]))
                    #img = generate_topk_variance_attention_maps(out_dict['attention_map'], out_dict['feature_activations'], out_dict['input_batch'], out_dict['rois'], nnn)
                    #img = generate_attention_visualization(out_dict['attention_map'], out_dict['feature_activations'], out_dict['input_batch'], out_dict['rois'], nnn)
                    #img = generate_attention_visualization(out_dict['attention_map'], out_dict['final_i3d_feats'], out_dict['input_batch'], out_dict['rois'], nnn)
                    #img = generate_class_activation_maps(out_dict['final_i3d_feats'], out_dict['cls_weights'], out_dict['input_batch'], out_dict['rois'], out_dict['pred_probs'], nnn)
                    img = generate_class_activation_maps(out_dict['final_i3d_feats'], out_dict['cls_weights'], out_dict['input_batch'], out_dict['rois'],
                                                         out_dict['pred_probs'], nnn, out_dict['roi_labels'])

                    cv2.imshow('Maps', img)
                    k = cv2.waitKey(0)
                    cv2.destroyWindow('Maps')
                    if k == ord("n"):
                        break
                    elif k == ord("q"):
                        os._exit(0)

                    # import pdb;pdb.set_trace()


            # Trace the timeline for debugging performance
            if TRACE_PERFORMANCE:
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                chrome_trace_dict = json.loads(chrome_trace)
                if not self.timeline_dict:
                    self.timeline_dict = chrome_trace_dict
                else:
                    for event in chrome_trace_dict['traceEvents']:
                        if 'ts' in event:
                            self.timeline_dict['traceEvents'].append(event)
                if ii % 20 == 0:
                    with open('timeline.json', 'w') as fp:
                        json.dump(self.timeline_dict, fp)

            # import pdb;pdb.set_trace()

            #### aug debug
            # shifted_rois = out_dict['shifted_rois']
            # regular_rois = out_dict['regular_rois']
            # for ii in range(shifted_rois.shape[0]):
            #     if ii == 0: import cv2
            #     shifted_image = shifted_rois[ii]
            #     regular_image = regular_rois[ii]
            #     cv2.imwrite('./rois/%.2i_shifted.jpg' % ii, shifted_image)
            #     cv2.imwrite('./rois/%.2i_regular.jpg' % ii, regular_image)
            #     print('%.2i written' % ii)

            # import pdb;pdb.set_trace()
            # print(info)
            # if len(info) > 4:
                # import pdb;pdb.set_trace()

            # to start where I left of, I can keep track of segment keys each iteration
            if ii == 0:
                seg_keys_str = ', '.join(out_dict['segment_keys'])
                logging.info('Starting the iteration with keys %s' % seg_keys_str)
            

            # keep track of out probs for each roi
            # if not training_flag:
            if True:
                roi_segment_keys = []
                skeys = out_dict['segment_keys']
                no_dets_np = out_dict['no_dets']
                for bb in range(no_dets_np.shape[0]): #there are #BATCH_SIZE no_dets
                    roi_segment_keys.extend([skeys[bb]] * no_dets_np[bb]) 

                roi_ids = []
                for no_det in no_dets_np:
                    roi_ids.extend(range(no_det))

                no_rois = np.sum(no_dets_np)

                #final size of roi_segment_mapping should be no_rois
                assert no_rois == len(roi_segment_keys)
                assert no_rois == len(roi_ids)
                
                for rr in range(no_rois):
                    tube_key = roi_segment_keys[rr]
                    tube_id = roi_ids[rr]

                    cur_labels = out_dict['roi_labels'][rr]
                    cur_probs = out_dict['pred_probs'][rr]
                    cur_probs = [get_3_decimal_float(prob) for prob in cur_probs]

                    result = [tube_key, tube_id, cur_labels.tolist(), cur_probs]
                    all_results.append(result)


            ######DEBUG
            #input_imgs = out_dict['inputs']
            #for bb in range(input_imgs.shape[0]):
            #    # for jj in range(input_imgs.shape[1]):
            #    cur_filename = 'test_images/%.3i.jpg' % (bb)
            #    cur_img = input_imgs[bb,:]
            #    #cur_img = cur_img[:,:,:]
            #    cv2.imwrite(cur_filename, cur_img)

            
            ######DEBUG

            cur_duration = time.time() - start_time
            duration += cur_duration

            # Update moving averages
            epoch_loss += out_dict['loss_val']

            no_correct_predictions += out_dict['no_correct_preds']
            no_total_labels += out_dict['no_labels']
            no_total_false_positives += out_dict['no_false_positives']
            # pdb.set_trace()
            epoch_recall = no_correct_predictions / float(no_total_labels)
            epoch_precision = no_correct_predictions / float(no_correct_predictions + no_total_false_positives)

            # predictions.append(out_dict['preds'])
            # cor_labels.append(labels)

            # Print information
            # pdb.set_trace()
            message = 'Train:' if training_flag else 'Valid:'
            # message += 'Epoch %i' % ee
            message += 'E_%i' % epoch_no
            #message += ', Step %i' % step
            #message += ', %.2f sec/batch' % (duration/float(step))
            message += ', Recall %.1f' % (100.0*epoch_recall)
            message += ', Prec %.1f' % (100.0*epoch_precision)
            message += ', Loss %.4f' % (epoch_loss/float(step))
            # message += '\n'
            message += ', Regu %.4f' % (out_dict['regularization'])
            # pdb.set_trace()

            # logging.info(message)
            # tqdm.write(message)
            pbar.set_description(message)
            pbar.update(1)

            # if ii % 2500 == 0 and ii != 0 and self.run_test:
            #     save_no = ii // 2500
            #     if self.run_test:
            #         split_name = 'TEST'
            #     else:
            #         split_name = 'TRAINING' if training_flag else 'VALIDATION'
            #     g_step = self.sess.run(self.global_step)
            #     save_path = RESULT_SAVE_PATH + split_name + '_Results_'+ self.model_id +('_%.2i.txt' % g_step) + 'part_%i' % save_no
            #     save_serialized_list(all_results, save_path)
            #     logging.info('Results saved to %s on step %i' %(save_path, ii))
                # all_results = []

            # if ii == 100 and training_flag: print('breaking'); break # for testing
            # if evaluate: pdb.set_trace()
        pbar.close()


        # Log the final result
        logging.info(message +'\n')
        
        # Sort and Save results in a file
        # pdb.set_trace()
        if self.run_test:
            split_name = 'TEST'
        else:
            split_name = 'TRAINING' if training_flag else 'VALIDATION'
        
        # do it for training also
        if not self.run_test:
            if self.dataset_str == 'ava':
                class_AP_str = process_results.get_AP_str(all_results)
            elif self.dataset_str == 'jhmdb':
                class_AP_str = self.dataset_fcn.get_AP_str(all_results)
            else:
                raise NotImplementedError
            logging.info('\n'+ split_name + '\n')
            logging.info('\nAverage Precision for each class \n' + class_AP_str)
            logging.info( split_name + '\n')

        if not training_flag:
            # all_results.sort(key=lambda x: x[0][0])
            #import pdb;pdb.set_trace()
            # class_AP_str = process_results.get_AP_str(all_results)
            # logging.info('\nAverage Precision for each class \n' + class_AP_str)
            # cor_labels = [np.argmax(res[1]) for res in all_results]
            # predictions = [np.argmax(res[2]) for res in all_results]

            # report = '\n\n****** Validation Report with loss %.3f ******\n\n' % (epoch_loss/float(step))
            # report += '\n' + classification_report(cor_labels, predictions, target_names=self.data_prov_tr.class_indices)
            # logging.info(report)

            g_step = self.sess.run(self.global_step)
            save_path = self.dataset_fcn.RESULT_SAVE_PATH + split_name + '_Results_'+ self.model_id +'_%.2i.txt' % g_step
            save_serialized_list(all_results, save_path)
            logging.info('Saved results to: ' + save_path)

        return epoch_loss, step




def no_of_params(var_list):
    total_parameters = 0
    # for variable in tf.trainable_variables():
    for variable in var_list:
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parametes = 1
        for dim in shape:
            # print(dim)
            variable_parametes *= dim.value
        # print(variable_parametes)
        total_parameters += variable_parametes
    # print('Total params: ' + str(total_parameters))
    return total_parameters

def read_serialized_results(file_path):
    with open(file_path) as fp:
        data = json.load(fp)
    return data

def save_serialized_list(input_list, file_path):
    with open(file_path, 'w') as fp:
        json.dump(input_list, fp)

def get_3_decimal_float(infloat):
    return float('%.3f' % infloat)

def custom_loader(sess, ckpt_file):
    global_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    var_map = {}
    for variable in global_vars:
        #if "Adam" not in variable.name and "moving" not in variable.name:
        if "CLS_Logits" not in variable.name: # for jhmdb
        #if "Adam" not in variable.name: # for jhmdb
            map_name = variable.name.replace(':0', '')
            #if "I3D_Model" in variable.name:
            #    map_name = map_name.replace('I3D_Model', 'RGB')
            var_map[map_name] = variable

    custom_saver = tf.train.Saver(var_list=var_map, reshape=True)
    custom_saver.restore(sess, ckpt_file)
    print('USED CUSTOM SAVER FOR LOADING CKPT!')

COLORS = np.random.randint(0, 255, [100, 3])
def draw_objects(frame, detections):
    H,W,C = frame.shape
    for bbid, bbox in enumerate(detections):
        color = COLORS[bbid,:]

        top, left, bottom, right = bbox
        left = int(W * left)
        right = int(W * right)

        top = int(H * top)
        bottom = int(H * bottom)

        conf = 1.0
        if conf < 0.20:
            continue
        label = 'person'
        message = label + ' %.2f' % conf

        cv2.rectangle(frame, (left,top), (right,bottom), color, 2)

        font_size =  max(0.5,(right - left)/50.0/float(len(message)))
        cv2.rectangle(frame, (left, top-int(font_size*40)), (right,top), color, -1)
        cv2.putText(frame, message, (left, top-12), 0, font_size, (255,255,255)-color, 1)


def generate_topk_variance_attention_maps(attention_map, feature_activations, input_batch, rois, roi_index, k=10):
    # img_to_show = cv2.resize(out_dict['attention_map'][0][4][:,:,index], (400,400), interpolation=0);cv2.imshow('map',cv2.applyColorMap(np.uint8(img_to_show*255),cv2.COLORMAP_JET))
    time_index = 4
    mask = np.float32(feature_activations != 0.)
    #masked_attention = attention_map * feature_activations
    masked_attention = attention_map * mask
    #masked_attention = feature_activations
    #masked_attention = mask
    #masked_attention = attention_map
    #var_list = np.argsort(np.var(np.reshape(masked_attention[roi_index][time_index], [-1,832]), axis=0))[::-1]
    #var_list = np.argsort(np.sum(np.reshape(masked_attention[roi_index][time_index], [-1,832]), axis=0))[::-1]
    #reshaped = np.reshape(masked_attention[roi_index][time_index], [-1,832])
    #total_n = np.sum(np.reshape(mask, [-1, 832]))
    #mean_val = np.sum(reshaped, axis=0) / total_n
    #var_list = np.argsort(np.sum((reshaped - mean_val)**2, axis=0) / total_n)[::-1]
    var_list = [0, 100, 200, 300, 400, 500, 600, 700, 800, 50, 150, 250, 350, 450]

    act_map = masked_attention[roi_index, :]
    avg_map = np.max(act_map, axis=0)

    top, left, bottom, right = rois[0, roi_index]
    input_frame = np.uint8(input_batch[0, 16])[:,:,::-1]
    img_to_show = input_frame.copy()
    H,W,C = img_to_show.shape
    left = int(W * left)
    right = int(W * right)
    top = int(H * top)
    bottom = int(H * bottom)
    cv2.rectangle(img_to_show, (left,top), (right,bottom), (0,255,0), 3)
    for ii in range(k):
        cur_index = var_list[ii]
        cur_attn_map = avg_map[:,:,cur_index]  #attention_map[roi_index][time_index][:,:,cur_index] * mask[roi_index][time_index][:,:,cur_index]
        rsz_attn_map = cv2.resize(cur_attn_map, (400,400), interpolation=0)
        
        min_val = np.min(masked_attention)
        max_val = np.max(masked_attention - min_val)
        normalized_image = np.uint8((rsz_attn_map-min_val) / max_val * 255.)

        colored_map = cv2.applyColorMap(normalized_image, cv2.COLORMAP_JET)
        overlay = input_frame.copy()
        overlay = cv2.addWeighted(overlay, 0.5, colored_map, 0.5, 0)
        img_to_show = np.concatenate([img_to_show, overlay], axis=1)

    #cv2.imshow('Maps', img_to_show)
    #cv2.waitKey(0)
    # import pdb;pdb.set_trace()
    #cv2.destroyWindow('Maps')
    return img_to_show

def generate_attention_visualization(attention_map, feature_activations, input_batch, rois, roi_index):
    time_index = 4
    feat_time_index = feature_activations.shape[1]//2
    mask = np.float32(feature_activations != 0.)
    #mask = mask[roi_index, time_index]
    #masked_attention = attention_map * feature_activations
    masked_attention = attention_map * mask
    #masked_attention = feature_activations
    #masked_attention = mask
    #masked_attention = attention_map * mask
    #masked_attention = masked_attention[roi_index,time_index]
    average_map = np.sum(masked_attention, axis=-1)
    #average_map = masked_attention[:,:,:,:,50]
    #average_map = np.sum(mask, axis=-1)

    #var_list = np.argsort(np.var(np.reshape(masked_attention[roi_index][time_index], [-1,832]), axis=0))[::-1]
    #var_list = np.argsort(np.sum(np.reshape(masked_attention[roi_index][time_index], [-1,832]), axis=0))[::-1]

    top, left, bottom, right = rois[0, roi_index]
    input_frame = np.uint8(input_batch[0, 16])[:,:,::-1]
    img_to_show = input_frame.copy()
    H,W,C = img_to_show.shape
    left = int(W * left)
    right = int(W * right)
    top = int(H * top)
    bottom = int(H * bottom)
    cv2.rectangle(img_to_show, (left,top), (right,bottom), (0,255,0), 3)

    # add the average_map
    #avg_map = average_map[roi_index, time_index]
    #if roi_index == 0:
    #    avg_map = average_map[0,4,:] - average_map[1,4,:]
    #elif roi_index == 1:
    #    avg_map = average_map[1,4,:] - average_map[0,4,:]
    #else:
    #    avg_map = average_map[roi_index, 4, :]
    #avg_map = average_map[roi_index, feat_time_index, :]
    normalizer = average_map
    #normalizer = np.sum(feature_activations, axis=-1)
    act_map = average_map[roi_index, :]
    avg_map = np.max(act_map, axis=0)
    rsz_avg_map = cv2.resize(avg_map, (400,400),)# interpolation=0)
    min_val = np.min(normalizer)
    max_val = np.max(normalizer - min_val)
    normalized_image = np.uint8((rsz_avg_map-min_val) / max_val * 255.)
    colored_map = cv2.applyColorMap(normalized_image, cv2.COLORMAP_JET)
    #colored_map = cv2.applyColorMap(normalized_image, cv2.COLORMAP_BONE)
    overlay = input_frame.copy()
    overlay = cv2.addWeighted(overlay, 0.5, colored_map, 0.5, 0)
    img_to_show = np.concatenate([img_to_show, overlay], axis=1)


    # for ii in range(k):
    #     cur_index = var_list[ii]
    #     cur_attn_map = attention_map[roi_index][time_index][:,:,cur_index] * mask[roi_index][time_index][:,:,cur_index]
    #     rsz_attn_map = cv2.resize(cur_attn_map, (400,400), interpolation=0)
    #     max_val = np.max(rsz_attn_map)
    #     normalized_image = np.uint8(rsz_attn_map / max_val * 255.)
    #     colored_map = cv2.applyColorMap(normalized_image, cv2.COLORMAP_JET)
    #     overlay = input_frame.copy()
    #     overlay = cv2.addWeighted(overlay, 0.5, colored_map, 0.5, 0)
    #     img_to_show = np.concatenate([img_to_show, overlay], axis=1)

    #cv2.imshow('Maps', img_to_show)
    #cv2.waitKey(0)
    # import pdb;pdb.set_trace()
    #cv2.destroyWindow('Maps')
    return img_to_show

def generate_class_activation_maps(feature_activations, cls_weights, input_batch, rois, pred_probs, roi_index, roi_labels):
    time_index = 4
    # feat_time_index = feature_activations.shape[1]//2
    feat_time_index = 0
    mask = np.float32(feature_activations != 0.)

    class_maps = np.matmul(feature_activations, cls_weights)

    top, left, bottom, right = rois[0, roi_index]
    input_frame = np.uint8(input_batch[0, 16])[:,:,::-1]
    img_to_show = input_frame.copy()
    H,W,C = img_to_show.shape
    left = int(W * left)
    right = int(W * right)
    top = int(H * top)
    bottom = int(H * bottom)
    cv2.rectangle(img_to_show, (left,top), (right,bottom), (0,255,0), 4)

    # add the average_map
    #avg_map = average_map[roi_index, time_index]
    #if roi_index == 0:
    #    avg_map = average_map[0,4,:] - average_map[1,4,:]
    #elif roi_index == 1:
    #    avg_map = average_map[1,4,:] - average_map[0,4,:]
    #else:
    #    avg_map = average_map[roi_index, 4, :]
    #print([(dataset_ava.TRAIN2ANN[str(ccc)]['class_str'], get_3_decimal_float(roi_probs[nnn][ccc])) for ccc in range(60) if
    #       get_3_decimal_float(roi_probs[nnn][ccc]) > 0.1])

    # visualize highest cams
    # action_classes = [cc for cc in range(dataset_ava.NUM_CLASSES) if pred_probs[roi_index, cc] > 0.1]

    # visualize specific cams
    #class_list = ["sit", "stand", 'touch', 'listen to', 'talk']
    #class_list = ["stand", "walk", "carry", "watch (a"]
    #class_list = ["talk", "listen to"]
    #class_list = ['talk']
    #action_classes = [cc for cc in range(dataset_ava.NUM_CLASSES) if any([cname in dataset_ava.TRAIN2ANN[str(cc)]['class_str'] for cname in class_list])]
    action_classes = np.where(roi_labels[roi_index])[0].tolist()
    for ii in action_classes:
        #avg_map = class_maps[roi_index, feat_time_index, :, :, ii]
        act_map = class_maps[roi_index, :, :, :, ii]
        avg_map = np.max(act_map, axis=0)
        rsz_avg_map = cv2.resize(avg_map, (400,400))#, interpolation=0)
        min_val = np.min(class_maps[:,:, :, :, :])
        max_val = np.max(class_maps[:,:, :, :, :] - min_val)
        normalized_image = np.uint8((rsz_avg_map-min_val) / max_val * 255.)
        colored_map = cv2.applyColorMap(normalized_image, cv2.COLORMAP_JET)
        overlay = input_frame.copy()
        overlay = cv2.addWeighted(overlay, 0.5, colored_map, 0.5, 0)
        img_to_show = np.concatenate([img_to_show, overlay], axis=1)
        print((dataset_ava.TRAIN2ANN[str(ii)]['class_str'], get_3_decimal_float(pred_probs[roi_index][ii])))


    # write the classes on image
    img_to_show = np.concatenate([img_to_show, np.zeros([50, 400*len(action_classes)+400, 3], np.uint8)])
    for cc in range(len(action_classes)):
        cls_no = action_classes[cc]
        left = 400 + 400*cc + 30
        top = 425
        message = dataset_ava.TRAIN2ANN[str(cls_no)]['class_str'][:26]
        cv2.putText(img_to_show, message, (left, top), 0, 1, (255,255,255), 1)


    # for ii in range(k):
    #     cur_index = var_list[ii]
    #     cur_attn_map = attention_map[roi_index][time_index][:,:,cur_index] * mask[roi_index][time_index][:,:,cur_index]
    #     rsz_attn_map = cv2.resize(cur_attn_map, (400,400), interpolation=0)
    #     max_val = np.max(rsz_attn_map)
    #     normalized_image = np.uint8(rsz_attn_map / max_val * 255.)
    #     colored_map = cv2.applyColorMap(normalized_image, cv2.COLORMAP_JET)
    #     overlay = input_frame.copy()
    #     overlay = cv2.addWeighted(overlay, 0.5, colored_map, 0.5, 0)
    #     img_to_show = np.concatenate([img_to_show, overlay], axis=1)

    #cv2.imshow('Maps', img_to_show)
    #cv2.waitKey(0)
    # import pdb;pdb.set_trace()
    #cv2.destroyWindow('Maps')
    return img_to_show



CORRECT_DETECTIONS     = [u'covMYDBa5dk.1360', u'Ekwy7wzLfjc.1191', u'b-YoBU0XT90.1281', u'er7eeiJB6dI.1350', u'KHHgQ_Pe4cI.1548', u'_eBah6c5kyA.1589',
                          u'CZ2NP8UsPuE.1673', u'CZ2NP8UsPuE.1674', u'b-YoBU0XT90.1500', u'CZ2NP8UsPuE.1672', u'9F2voT6QWvQ.1025', u'9F2voT6QWvQ.1026',
                          u'9F2voT6QWvQ.1064', u'9F2voT6QWvQ.1063', u'9F2voT6QWvQ.1078', u'OGNnUvJq9RI.1389', u'Hscyg0vLKc8.1696', u'Hscyg0vLKc8.1590',
                          u'OGNnUvJq9RI.1111', u'OGNnUvJq9RI.1125', u'u1ltv6r14KQ.1620', u'lDmLcWWBp1E.1066', u'UrsCy6qIGoo.1652', u'55Ihr6uVIDA.1600',
                          u'z-fsLpGHq6o.1730', u'Hscyg0vLKc8.1375', u'CMCPhm2L400.1148', u'er7eeiJB6dI.0921', u'er7eeiJB6dI.0920', u'CMCPhm2L400.1141',
                          u'_eBah6c5kyA.1181', u'_eBah6c5kyA.1183', u'BXCh3r-pPAM.1755', u'fpprSy6AzKk.1194', u'b-YoBU0XT90.1131', u'Hscyg0vLKc8.1523',
                          u'Hscyg0vLKc8.1186', u'Hscyg0vLKc8.1662', u'Hscyg0vLKc8.1660', u'Hscyg0vLKc8.1169', u'sNQJfYvhcPk.1348', u'sNQJfYvhcPk.1349',
                          u'CMCPhm2L400.1177', u'CMCPhm2L400.1180', u'CMCPhm2L400.1176', u'6d5u6FHvz7Q.1434', u'Di1MG6auDYo.1119', u'6d5u6FHvz7Q.1418',
                          u'6d5u6FHvz7Q.1436', u'6d5u6FHvz7Q.1413', u'9Y_l9NsnYE0.1156', u'Lg1jOu8cUBM.1172', u'55Ihr6uVIDA.0970', u'Lg1jOu8cUBM.1171',
                          u'55Ihr6uVIDA.0968', u'xeGWXqSvC-8.1030', u'wONG7Vh87B4.1158', u'xeGWXqSvC-8.1062', u'Ekwy7wzLfjc.1670', u'xeGWXqSvC-8.1061',
                          u'ZosVdkY76FU.1162', u'uNT6HrrnqPU.1683', u'om_83F5VwTQ.1468', u'55Ihr6uVIDA.1327', u'uNT6HrrnqPU.1682', u'O_NYCUhZ9zw.1479',
                          u'Di1MG6auDYo.1304', u'Di1MG6auDYo.1299', u'WSPvfxtqisg.0948', u'O_NYCUhZ9zw.1566', u'7T5G0CmwTPo.1739', u'7T5G0CmwTPo.1740',
                          u'j5jmjhGBW44.1612', u'QCLQYnt3aMo.1617', u'b-YoBU0XT90.1621', u'covMYDBa5dk.1789', u'b-YoBU0XT90.1311', u'covMYDBa5dk.1784',
                          u'b-YoBU0XT90.1312', u'jE0S8gYWftE.1417', u'l2XO3tQk8lI.1333', u'u1ltv6r14KQ.1417', u'l2XO3tQk8lI.1332', u'o4xQ-BEa3Ss.1602',
                          u'qx2vAO5ofmo.1307', u'6d5u6FHvz7Q.1175', u'7T5G0CmwTPo.1445', u'6d5u6FHvz7Q.1176', u'6d5u6FHvz7Q.1174', u'6d5u6FHvz7Q.1181',
                          u'oifTDWZvOhY.1094', u'KHHgQ_Pe4cI.1334', u'oifTDWZvOhY.1095', u'KHHgQ_Pe4cI.1333', u'9F2voT6QWvQ.1164', u'xeGWXqSvC-8.1576',
                          u'xeGWXqSvC-8.1726', u'xeGWXqSvC-8.1724', u'xeGWXqSvC-8.1725', u'xeGWXqSvC-8.1575', u'QTmwhrVal1g.1626', u'Ov0za6Xb1LM.0943',
                          u'QTmwhrVal1g.1450', u'QTmwhrVal1g.1608', u'QTmwhrVal1g.1627', u'bePts02nIY8.1020', u'bePts02nIY8.1022', u'bePts02nIY8.1021',
                          u'bePts02nIY8.1023', u'bePts02nIY8.1024', u'TEQ9sAj-DPo.1604', u'xeGWXqSvC-8.1783', u'sNQJfYvhcPk.1580', u'WSPvfxtqisg.0935',
                          u'sNQJfYvhcPk.1579', u'_7oWZq_s_Sk.1714', u'_7oWZq_s_Sk.1721', u'9F2voT6QWvQ.1317', u'_7oWZq_s_Sk.1716', u'oifTDWZvOhY.1646',
                          u'lDmLcWWBp1E.1314', u'b-YoBU0XT90.1358', u'fpprSy6AzKk.1128', u'OGNnUvJq9RI.1602', u'lDmLcWWBp1E.1328', u'9F2voT6QWvQ.1026',
                          u'9F2voT6QWvQ.0932', u'9F2voT6QWvQ.1058', u'9F2voT6QWvQ.0998', u'9F2voT6QWvQ.0902', u'WSPvfxtqisg.0936', u'qx2vAO5ofmo.0985',
                          u'u1ltv6r14KQ.1358', u'UOyyTUX5Vo4.1076', u'u1ltv6r14KQ.1547', u'fpprSy6AzKk.1215', u'fpprSy6AzKk.1102', u'fpprSy6AzKk.1216',
                          u'fpprSy6AzKk.1101', u'fpprSy6AzKk.1202', u'O_NYCUhZ9zw.1112', u'qx2vAO5ofmo.1039', u'fpprSy6AzKk.1649', u'qx2vAO5ofmo.1406',
                          u'9Y_l9NsnYE0.1798', u'u1ltv6r14KQ.1046', u'b-YoBU0XT90.1681', u'hHgg9WI8dTk.1796', u'KVq6If6ozMY.0994', u'_7oWZq_s_Sk.1642',
                          u'55Ihr6uVIDA.1502', u'Ov0za6Xb1LM.1759', u'6d5u6FHvz7Q.1018', u'6d5u6FHvz7Q.0991', u'Ov0za6Xb1LM.1700', u'QCLQYnt3aMo.1503',
                          u'QCLQYnt3aMo.1453', u'IzvOYVMltkI.1251', u'lDmLcWWBp1E.1322', u'QCLQYnt3aMo.1449', u'oifTDWZvOhY.0949', u'oifTDWZvOhY.0942',
                          u'O_NYCUhZ9zw.1421', u'Ekwy7wzLfjc.1490', u'KHHgQ_Pe4cI.1281', u'CZ2NP8UsPuE.1123', u'UrsCy6qIGoo.1131', u'UrsCy6qIGoo.1128',
                          u'oifTDWZvOhY.1319', u'UrsCy6qIGoo.1129', u'xeGWXqSvC-8.1045', u'xeGWXqSvC-8.1054', u'xeGWXqSvC-8.1016', u'xeGWXqSvC-8.1044',
                          u'xeGWXqSvC-8.0985', u'CMCPhm2L400.1115', u'Ov0za6Xb1LM.1486', u'Ov0za6Xb1LM.1517', u'Ov0za6Xb1LM.1497', u'Ov0za6Xb1LM.1510',
                          u'CZ2NP8UsPuE.1520', u'CZ2NP8UsPuE.1547', u'om_83F5VwTQ.1606', u'XpGRS72ghag.0971', u'IzvOYVMltkI.1462', u'ayAMdYfJJLk.0976',
                          u'UrsCy6qIGoo.0909', u'UrsCy6qIGoo.0908', u'5BDj0ow5hnA.1009', u'5BDj0ow5hnA.1699', u'Di1MG6auDYo.1600', u'bePts02nIY8.1656',
                          u'er7eeiJB6dI.0929', u'SCh-ZImnyyk.1772', u'KVq6If6ozMY.1770', u'wONG7Vh87B4.1388', u'_eBah6c5kyA.1129', u'wONG7Vh87B4.1384',
                          u'BXCh3r-pPAM.1782', u'yn9WN9lsHRE.1493', u'ZosVdkY76FU.1604', u'ZosVdkY76FU.1602', u'ZosVdkY76FU.1605', u'ZosVdkY76FU.1603',
                          u'ZosVdkY76FU.1606', u'nlinqZPgvVk.1740', u'KHHgQ_Pe4cI.1247', u'QTmwhrVal1g.1360', u'QTmwhrVal1g.1201', u'CMCPhm2L400.1387',
                          u'9Y_l9NsnYE0.1580', u'9Y_l9NsnYE0.1579', u'9Y_l9NsnYE0.1588', u'9Y_l9NsnYE0.1586', u'9Y_l9NsnYE0.1578', u'WSPvfxtqisg.1207',
                          u'Di1MG6auDYo.1435', u'WSPvfxtqisg.1206', u'Di1MG6auDYo.1434', u'Di1MG6auDYo.1436', u'KHHgQ_Pe4cI.1794', u'KHHgQ_Pe4cI.1793',
                          u'KHHgQ_Pe4cI.1796', u'KHHgQ_Pe4cI.1795', u'O_NYCUhZ9zw.1426', u'Hscyg0vLKc8.1523', u'Hscyg0vLKc8.1046', u'Hscyg0vLKc8.1668',
                          u'Hscyg0vLKc8.1041', u'Hscyg0vLKc8.1022', u'O_NYCUhZ9zw.1117', u'yn9WN9lsHRE.1353', u'O_NYCUhZ9zw.1118', u'IzvOYVMltkI.1258',
                          u'om_83F5VwTQ.1223', u'z-fsLpGHq6o.1229', u'NO2esmws190.1764', u'z-fsLpGHq6o.1227', u'nlinqZPgvVk.1111', u'_dBTTYDRdRQ.1559',
                          u'fpprSy6AzKk.1082', u'ZosVdkY76FU.1789', u'ZosVdkY76FU.1788', u'Hscyg0vLKc8.0935', u'fpprSy6AzKk.1081', u'zC5Fh2tTS1U.1584',
                          u'XpGRS72ghag.1577', u'7T5G0CmwTPo.1496', u'tghXjom3120.1247', u'tghXjom3120.1245', u'lDmLcWWBp1E.1712', u'_7oWZq_s_Sk.0923',
                          u'Gvp-cj3bmIY.1678', u'lDmLcWWBp1E.1711', u'UOyyTUX5Vo4.1585', u'Ov0za6Xb1LM.1208', u'Ov0za6Xb1LM.1207', u'uNT6HrrnqPU.1785',
                          u'2PpxiG0WU18.1570', u'uNT6HrrnqPU.1788', u'uNT6HrrnqPU.1729', u'uNT6HrrnqPU.1728', u'uNT6HrrnqPU.1726', u'uNT6HrrnqPU.1730',
                          u'uNT6HrrnqPU.1785', u'tghXjom3120.1690', u'covMYDBa5dk.1480', u'5BDj0ow5hnA.1043', u'covMYDBa5dk.1479', u'tghXjom3120.1692',
                          u'9Y_l9NsnYE0.1337', u'9Y_l9NsnYE0.1289', u'Lg1jOu8cUBM.1229', u'IzvOYVMltkI.1549', u'9Y_l9NsnYE0.1290', u'lDmLcWWBp1E.1214',
                          u'z-fsLpGHq6o.1732', u'z-fsLpGHq6o.1676', u'z-fsLpGHq6o.1224', u'l2XO3tQk8lI.1441', u'9F2voT6QWvQ.1054', u'9F2voT6QWvQ.1057',
                          u'9F2voT6QWvQ.1053', u'fpprSy6AzKk.1102', u'9F2voT6QWvQ.1058', u'O_NYCUhZ9zw.1118', u'jE0S8gYWftE.1362', u'Di1MG6auDYo.1418',
                          u'7T5G0CmwTPo.1678', u'O_NYCUhZ9zw.1612', u'z-fsLpGHq6o.1759', u'sNQJfYvhcPk.1247', u'z-fsLpGHq6o.1760', u'rXFlJbXyZyc.1549',
                          u'SCh-ZImnyyk.1058', u'Hscyg0vLKc8.1453', u'Hscyg0vLKc8.1454', u'ayAMdYfJJLk.1526', u'ayAMdYfJJLk.1525', u'ayAMdYfJJLk.1527']

MISSED_DETECTIONS = [u'QTmwhrVal1g.1154', u'Ekwy7wzLfjc.0983', u'NO2esmws190.1710', u'z-fsLpGHq6o.1586', u'lDmLcWWBp1E.1656', u'QTmwhrVal1g.1169',
                     u'UOyyTUX5Vo4.1403', u'jE0S8gYWftE.1779', u'UrsCy6qIGoo.1088', u'Hscyg0vLKc8.1714', u'qx2vAO5ofmo.1117', u'tghXjom3120.1179',
                     u'z-fsLpGHq6o.1212', u'tghXjom3120.1139', u'z-fsLpGHq6o.1146', u'KHHgQ_Pe4cI.1728', u'OGNnUvJq9RI.1190', u'Hscyg0vLKc8.0932',
                     u'WSPvfxtqisg.1710', u'BXCh3r-pPAM.1752', u'ZosVdkY76FU.1198', u'CMCPhm2L400.1238', u'QCLQYnt3aMo.1065', u'CMCPhm2L400.0988',
                     u'55Ihr6uVIDA.1145', u'z-fsLpGHq6o.1246', u'oifTDWZvOhY.1677', u'tghXjom3120.1635', u'CZ2NP8UsPuE.1190', u'tghXjom3120.1173',
                     u'nlinqZPgvVk.1295', u'NO2esmws190.1534', u'55Ihr6uVIDA.1403', u'Ekwy7wzLfjc.1086', u'NO2esmws190.1584', u'Hscyg0vLKc8.1504',
                     u'Hscyg0vLKc8.1784', u'Hscyg0vLKc8.1674', u'Hscyg0vLKc8.1618', u'Hscyg0vLKc8.1551', u'Ov0za6Xb1LM.1751', u'Hscyg0vLKc8.1225',
                     u'u1ltv6r14KQ.1107', u'lDmLcWWBp1E.1172', u'Ov0za6Xb1LM.1746', u'UrsCy6qIGoo.1529', u'_eBah6c5kyA.0987', u'jE0S8gYWftE.1251',
                     u'QTmwhrVal1g.1621', u'9Y_l9NsnYE0.1124', u'UrsCy6qIGoo.0913', u'QCLQYnt3aMo.1378', u'QCLQYnt3aMo.1384', u'Hscyg0vLKc8.1469',
                     u'er7eeiJB6dI.1059', u'Hscyg0vLKc8.1390', u'Ekwy7wzLfjc.1674', u'Ekwy7wzLfjc.1652', u'Hscyg0vLKc8.1388', u'Ekwy7wzLfjc.1661',
                     u'o4xQ-BEa3Ss.1441', u'yn9WN9lsHRE.1203', u'OGNnUvJq9RI.1228', u'TEQ9sAj-DPo.1493', u'TEQ9sAj-DPo.1548', u'SCh-ZImnyyk.1462',
                     u'ZosVdkY76FU.0969', u'QTmwhrVal1g.1496', u'IzvOYVMltkI.1764', u'ayAMdYfJJLk.1173', u'QCLQYnt3aMo.1384', u'BXCh3r-pPAM.1671',
                     u'ZosVdkY76FU.1676', u'ZosVdkY76FU.1607', u'QCLQYnt3aMo.1614', u'5BDj0ow5hnA.0965', u'wONG7Vh87B4.0929', u'uNT6HrrnqPU.1298',
                     u'l2XO3tQk8lI.1101', u'KHHgQ_Pe4cI.0903', u'QTmwhrVal1g.1711', u'XpGRS72ghag.1412', u'CMCPhm2L400.1465', u'CMCPhm2L400.1123',
                     u'CMCPhm2L400.1122', u'fpprSy6AzKk.1118', u'er7eeiJB6dI.1336', u'UOyyTUX5Vo4.0944', u'er7eeiJB6dI.1339', u'UOyyTUX5Vo4.0951',
                     u'IzvOYVMltkI.1446', u'qx2vAO5ofmo.1343', u'yn9WN9lsHRE.1127', u'o4xQ-BEa3Ss.1530', u'jE0S8gYWftE.0961', u'BXCh3r-pPAM.1684',
                     u'WSPvfxtqisg.1391', u'BXCh3r-pPAM.1240', u'uNT6HrrnqPU.1534', u'ayAMdYfJJLk.1322', u'2PpxiG0WU18.1153', u'jE0S8gYWftE.1251',
                     u'Gvp-cj3bmIY.1032', u'XpGRS72ghag.1170', u'G4qq1MRXCiY.1756', u'Gvp-cj3bmIY.1503', u'Gvp-cj3bmIY.1568', u'ayAMdYfJJLk.1212',
                     u'o4xQ-BEa3Ss.1162', u'z-fsLpGHq6o.0947', u'IzvOYVMltkI.1022', u'l2XO3tQk8lI.1214', u'fpprSy6AzKk.1634', u'NO2esmws190.0965',
                     u'QCLQYnt3aMo.0951', u'fpprSy6AzKk.1475', u'oifTDWZvOhY.1590', u'xeGWXqSvC-8.1341', u'wONG7Vh87B4.1651', u'_7oWZq_s_Sk.1798',
                     u'NO2esmws190.1607', u'o4xQ-BEa3Ss.1194', u'55Ihr6uVIDA.1151', u'WSPvfxtqisg.1790', u'55Ihr6uVIDA.1148', u'_dBTTYDRdRQ.1171',
                     u'_7oWZq_s_Sk.1078', u'_dBTTYDRdRQ.1352', u'_dBTTYDRdRQ.1170', u'_dBTTYDRdRQ.1342', u'lDmLcWWBp1E.1471', u'z-fsLpGHq6o.1719',
                     u'z-fsLpGHq6o.1720', u'O_NYCUhZ9zw.0925', u'55Ihr6uVIDA.1525', u'fpprSy6AzKk.0902', u'_dBTTYDRdRQ.1051', u'_7oWZq_s_Sk.1069',
                     u'lDmLcWWBp1E.1126', u'rXFlJbXyZyc.1395', u'2PpxiG0WU18.1483', u'qx2vAO5ofmo.1577', u'QTmwhrVal1g.1721', u'Di1MG6auDYo.1658',
                     u'CZ2NP8UsPuE.1649', u'KVq6If6ozMY.1756', u'Gvp-cj3bmIY.1276', u'UOyyTUX5Vo4.1314', u'9Y_l9NsnYE0.1738', u'5BDj0ow5hnA.1623',
                     u'Ov0za6Xb1LM.1754', u'IzvOYVMltkI.1449', u'5BDj0ow5hnA.1394', u'oifTDWZvOhY.1792', u'ZosVdkY76FU.1190', u'qx2vAO5ofmo.1199',
                     u'Di1MG6auDYo.1072', u'KHHgQ_Pe4cI.1699', u'G4qq1MRXCiY.1602', u'wONG7Vh87B4.1009', u'wONG7Vh87B4.1483', u'rXFlJbXyZyc.1469',
                     u'jE0S8gYWftE.1606', u'ayAMdYfJJLk.1360', u'QCLQYnt3aMo.1283', u'er7eeiJB6dI.0995', u'Gvp-cj3bmIY.0999', u'CZ2NP8UsPuE.1184',
                     u'er7eeiJB6dI.0974', u'CZ2NP8UsPuE.1187', u'rXFlJbXyZyc.1154', u'xeGWXqSvC-8.1133', u'rXFlJbXyZyc.1365', u'xeGWXqSvC-8.0942',
                     u'xeGWXqSvC-8.0903', u'Ov0za6Xb1LM.1514', u'QCLQYnt3aMo.1117', u'Z1YV6wB037M.1173', u'Z1YV6wB037M.1177', u'OGNnUvJq9RI.1037',
                     u'Lg1jOu8cUBM.1317', u'Lg1jOu8cUBM.1041', u'om_83F5VwTQ.1700', u'OGNnUvJq9RI.0948', u'om_83F5VwTQ.1684', u'ZosVdkY76FU.1545',
                     u'ZosVdkY76FU.1502', u'ZosVdkY76FU.1511', u'bePts02nIY8.1506', u'5BDj0ow5hnA.1701', u'5BDj0ow5hnA.1345', u'SCh-ZImnyyk.1747',
                     u'QTmwhrVal1g.1230', u'XpGRS72ghag.0943', u'bePts02nIY8.0969', u'yn9WN9lsHRE.1274', u'z-fsLpGHq6o.1253', u'rXFlJbXyZyc.0970',
                     u'1j20qq1JyX4.1414', u'om_83F5VwTQ.1725', u'G4qq1MRXCiY.1605', u'NO2esmws190.1551', u'wONG7Vh87B4.1196', u'jE0S8gYWftE.1768',
                     u'6d5u6FHvz7Q.1738', u'b-YoBU0XT90.1392', u'bePts02nIY8.1472', u'b-YoBU0XT90.1396', u'o4xQ-BEa3Ss.1442', u'bePts02nIY8.1480',
                     u'CZ2NP8UsPuE.1006', u'KVq6If6ozMY.1706', u'_dBTTYDRdRQ.1170', u'KHHgQ_Pe4cI.1758', u'XpGRS72ghag.1723', u'sNQJfYvhcPk.1567',
                     u'WSPvfxtqisg.1140', u'sNQJfYvhcPk.1570', u'Di1MG6auDYo.1242', u'Di1MG6auDYo.1224', u'ZosVdkY76FU.1125', u'ZosVdkY76FU.1150',
                     u'SCh-ZImnyyk.1792', u'ZosVdkY76FU.1129', u'9F2voT6QWvQ.1742', u'QCLQYnt3aMo.1050', u'j5jmjhGBW44.1195', u'QCLQYnt3aMo.1059',
                     u'KVq6If6ozMY.1134', u'Z1YV6wB037M.1246', u'2PpxiG0WU18.1129', u'914yZXz-iRs.1225', u'_7oWZq_s_Sk.1146', u'9F2voT6QWvQ.0941',
                     u'oifTDWZvOhY.1530', u'yn9WN9lsHRE.1146', u'om_83F5VwTQ.0903', u'Hscyg0vLKc8.1201', u'UOyyTUX5Vo4.1166', u'_dBTTYDRdRQ.1295',
                     u'G4qq1MRXCiY.1662', u'5BDj0ow5hnA.1572', u'5BDj0ow5hnA.0977', u'IzvOYVMltkI.1417', u'er7eeiJB6dI.0912', u'Z1YV6wB037M.1341',
                     u'z-fsLpGHq6o.1246', u'Z1YV6wB037M.1340', u'z-fsLpGHq6o.1652', u'QCLQYnt3aMo.1332', u'5BDj0ow5hnA.1420', u'bePts02nIY8.0967',
                     u'55Ihr6uVIDA.1672', u'NO2esmws190.1187', u'z-fsLpGHq6o.1418', u'covMYDBa5dk.1597', u'_dBTTYDRdRQ.1287', u'zC5Fh2tTS1U.1555',
                     u'UrsCy6qIGoo.1083', u'covMYDBa5dk.1463', u'wONG7Vh87B4.1057', u'QTmwhrVal1g.0908', u'wONG7Vh87B4.1053', u'2PpxiG0WU18.1134',
                     u'UrsCy6qIGoo.1051', u'qx2vAO5ofmo.1792', u'qx2vAO5ofmo.1795', u'UrsCy6qIGoo.1091', u'QCLQYnt3aMo.1066', u'j5jmjhGBW44.1008',
                     u'OGNnUvJq9RI.1101', u'l2XO3tQk8lI.1607', u'jE0S8gYWftE.1108', u'fpprSy6AzKk.1617', u'jE0S8gYWftE.1109', u'tghXjom3120.1710',
                     u'tghXjom3120.1722', u'uNT6HrrnqPU.1386', u'tghXjom3120.1718', u'QCLQYnt3aMo.1049', u'fpprSy6AzKk.1027', u'KVq6If6ozMY.1510',
                     u'fpprSy6AzKk.1456', u'zC5Fh2tTS1U.1658', u'rXFlJbXyZyc.1395', u'om_83F5VwTQ.1274', u'7T5G0CmwTPo.1486', u'z-fsLpGHq6o.1521',
                     u'_7oWZq_s_Sk.1643', u'_dBTTYDRdRQ.1214', u'7T5G0CmwTPo.1520', u'j5jmjhGBW44.1121', u'fpprSy6AzKk.1584', u'5BDj0ow5hnA.1578',
                     u'9F2voT6QWvQ.0921', u'_eBah6c5kyA.0987', u'qx2vAO5ofmo.1662', u'jE0S8gYWftE.1108', u'l2XO3tQk8lI.1506', u'G4qq1MRXCiY.0979']


if __name__ == '__main__':
    main()
