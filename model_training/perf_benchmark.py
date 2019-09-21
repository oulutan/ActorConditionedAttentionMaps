import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm


import dataset_ava
import i3d



PREPROCESS_CORES = 10 # times number of gpus
BUFFER_SIZE = 20

ACAM_FOLDER = os.environ['ACAM_DIR']
os.environ["CUDA_VISIBLE_DEVICES"] = '0'



def main():

    ########### set data
    val_detection_segments = dataset_ava.get_val_list()
    tfrecord_list = dataset_ava.generate_tfrecord_list(val_detection_segments)
    #tfrecord_list = tf.data.Dataset.list_files("/data/ulutan/AVA/tfrecords_combined/val/val_dataset.record-*-of-00010")
    #tfrecord_list = ["/data/ulutan/AVA/tfrecords_combined/val/val_dataset.record-00001-of-00010"]
    #dataset = tf.data.TFRecordDataset(tfrecord_list)
    dataset = tf.data.TFRecordDataset(tfrecord_list, num_parallel_reads=10)
    #dataset = tf.data.TFRecordDataset.list_files(tfrecord_list)
    #dataset = dataset.apply(tf.contrib.data.parallel_interleave(
    #        tf.data.TFRecordDataset, cycle_length=10))
    dataset = dataset.repeat()# repeat infinitely
    dataset = dataset.map(dataset_ava.get_tfrecord, 
            num_parallel_calls=10)
    # dataset = dataset.prefetch(buffer_size=BUFFER_SIZE)
    #dataset = dataset.filter(dataset_ava.filter_no_detections)
    dataset = dataset.batch(batch_size=2*4)
    #dataset = dataset.prefetch(buffer_size=1)
    dataset = dataset.apply(tf.contrib.data.prefetch_to_device('/gpu:0', buffer_size=1))


    input_handle = tf.placeholder(tf.string, shape=[])
    #iterator = tf.data.Iterator.from_string_handle( input_handle, dataset.output_types, dataset.output_shapes)
    iterator = dataset.make_one_shot_iterator()

    next_element = iterator.get_next()

    # Define shapes of the inputs coming from python functions
    input_batch, labels, rois, no_dets, segment_keys = next_element

    # input_batch = tf.cast(input_batch, tf.float32)

    input_batch.set_shape([None, dataset_ava.INPUT_T, dataset_ava.INPUT_H, dataset_ava.INPUT_W, 3])
    #labels.set_shape([None, self.dataset_fcn.MAX_ROIS, self.dataset_fcn.NUM_CLASSES])
    labels.set_shape([None, dataset_ava.MAX_ROIS_IN_TRAINING, dataset_ava.NUM_CLASSES])
    #rois.set_shape([None, self.dataset_fcn.MAX_ROIS, 4])
    rois.set_shape([None, dataset_ava.MAX_ROIS_IN_TRAINING, 4])
    no_dets.set_shape([None])
    segment_keys.set_shape([None])
    #input_batch = tf.zeros([8, 32, 400, 400, 3])


    ############## set model
    end_point = 'Mixed_4f'
    #end_point = 'Logits'
    is_training = tf.constant(True)
    num_classes = 60
    
    input_batch = tf.cast(input_batch, tf.float32)[:,:,:,::-1]
    _, end_points = i3d.inference(input_batch, is_training, num_classes, end_point=end_point)
    features = end_points[end_point]


    ##########3 set session
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)#, log_device_placement=True)
    sess = tf.Session(config=config)

    #training_iterator = self.training_dataset.make_one_shot_iterator()
    validation_iterator = dataset.make_initializable_iterator()

    # `Iterator.string_handle()` method returns a tensor that can be evaluated
    # used to feed the `handle` placeholder.
    #training_handle = sess.run(self.training_iterator.string_handle())
    #validation_handle = sess.run(validation_iterator.string_handle())
    validation_handle = 'test'

    init_op = tf.global_variables_initializer()
    sess.run(init_op)


    sess.run(validation_iterator.initializer)
    
    
    for ii in tqdm(range(1000)):
        #out = sess.run(input_batch, feed_dict={input_handle:validation_handle})
        #out = sess.run(labels)
        out = sess.run(features)
        #out = sess.run(features, feed_dict={input_handle:validation_handle})
        #print('Running')




if __name__ == '__main__':
    main()
