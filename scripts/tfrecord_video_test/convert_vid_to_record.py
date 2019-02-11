import tensorflow as tf
import imageio
import numpy as np
import cv2

def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _bytes_list_feature(values):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

fname = "0903"
output_file = "%s.tfrecord" % fname
with tf.python_io.TFRecordWriter(output_file) as writer:

  # Read and resize all video frames, np.uint8 of size [N,H,W,3]
  vid_path = "./%s.mp4" % fname
  video = imageio.get_reader(vid_path)
  frame_gen = video.iter_data()
  frame_list = []
  for frame in frame_gen:
      frame_list.append(frame)
    
  frames = np.stack(frame_list, axis=0)

  features = {}
  features['num_frames']  = _int64_feature(frames.shape[0])
  features['height']      = _int64_feature(frames.shape[1])
  features['width']       = _int64_feature(frames.shape[2])
  features['channels']    = _int64_feature(frames.shape[3])
  #features['class_label'] = _int64_feature(example['class_id'])
  #features['class_text']  = _bytes_feature(tf.compat.as_bytes(example['class_label']))
  #features['filename']    = _bytes_feature(tf.compat.as_bytes(example['video_id']))
  features['filename']    = _bytes_feature(tf.compat.as_bytes(fname))

  # Compress the frames using JPG and store in as a list of strings in 'frames'
  encoded_frames = [tf.compat.as_bytes(cv2.imencode(".jpg", frame)[1].tobytes())
                    for frame in frames]
  features['frames'] = _bytes_list_feature(encoded_frames)

  tfrecord_example = tf.train.Example(features=tf.train.Features(feature=features))
  writer.write(tfrecord_example.SerializeToString())
