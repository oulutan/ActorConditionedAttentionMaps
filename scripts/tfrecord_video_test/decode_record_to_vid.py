import tensorflow as tf
import numpy as np


def decode(serialized_example):
  # Prepare feature list; read encoded JPG images as bytes
  features = dict()
  #features["class_label"] = tf.FixedLenFeature((), tf.int64)
  features["frames"] = tf.VarLenFeature(tf.string)
  features["num_frames"] = tf.FixedLenFeature((), tf.int64)
  features["filename"] = tf.FixedLenFeature((), tf.string)

  # Parse into tensors
  parsed_features = tf.parse_single_example(serialized_example, features)

  # Randomly sample offset from the valid range.
  #random_offset = tf.random_uniform(
  #    shape=(), minval=0,
  #    maxval=parsed_features["num_frames"] - SEQ_NUM_FRAMES, dtype=tf.int64)

  #offsets = tf.range(random_offset, random_offset + SEQ_NUM_FRAMES)

  # Decode the encoded JPG images
  #images = tf.map_fn(lambda i: tf.image.decode_jpeg(parsed_features["frames"].values[i]),        offsets)
  images = tf.map_fn(lambda i: tf.image.decode_jpeg(parsed_features["frames"].values[i]), tf.range(0, parsed_features['num_frames']), dtype=tf.uint8)

  #label  = tf.cast(parsed_features["class_label"], tf.int64)
  #label = parsed_features['filename']
  label = tf.py_func(get_labels, [parsed_features['filename']], [tf.string])[0]


  return images, label
  #return images

def get_labels(fname):
    return fname + ",".join([str(num) for num in np.arange(5).tolist()])

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer(),)
tfrecord_files = ["0902.tfrecord", "0903.tfrecord"]

dataset = tf.data.TFRecordDataset(tfrecord_files)

dataset = dataset.repeat(2000)
dataset = dataset.map(decode)
#dataset = dataset.map(preprocess_video)

BATCH_SIZE=2
# The parameter is the queue size
#dataset = dataset.shuffle(1000 + 3 * BATCH_SIZE)
dataset = dataset.batch(BATCH_SIZE)

iterator = dataset.make_one_shot_iterator()
next_batch = iterator.get_next()

sess.run(init_op)

while True:

    # Fetch a new batch from the dataset
    batch_videos, batch_labels = sess.run(next_batch)
    #batch_videos = sess.run(next_batch)
    import pdb;pdb.set_trace()

