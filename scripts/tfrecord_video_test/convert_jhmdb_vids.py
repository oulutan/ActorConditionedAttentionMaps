import tensorflow as tf
import imageio
import numpy as np
import cv2
import os
#from tqdm import tqdm
from joblib import Parallel, delayed

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


#input_dir = "/home/ulutan/work/ActorConditionedAttentionMaps/data/AVA/segments/train/clips/"
#output_dir = "/home/ulutan/work/train_tfrecords/"
#input_dir = "/home/ulutan/work/ActorConditionedAttentionMaps/data/AVA/segments/val/clips/"
#input_dir = "/home/ulutan/work/ActorConditionedAttentionMaps/data/AVA/segments/test/clips/"
#output_dir = "/home/ulutan/work/val_tfrecords/"
#output_dir = "/data/ulutan/AVA/tfrecords/val_tfrecords/"
#output_dir = "/data/ulutan/AVA/tfrecords/test_tfrecords/"
input_dir = "/home/oytun/work/ActorConditionedAttentionMaps/data/JHMDB/ReCompress_Videos/"
output_dir = "/home/oytun/work/ActorConditionedAttentionMaps/data/JHMDB/tfrecords/"

movies = os.listdir(input_dir)
movies.sort()
#for movie in tqdm(movies):
def exception_wrapper(movie):
    try:
        generate_tfrecord(movie)
    except ValueError:
        print("####################Error with %s#########" % movie)


def generate_tfrecord(movie):
    segments = os.listdir(input_dir+movie)
    segments = [seg for seg in segments if seg != ".DS_Store"]
    segments.sort()
    os.mkdir(output_dir + movie)
    for segment in segments:
        segment_no_ext = segment.split(".")[0]
        output_file = os.path.join(output_dir, movie, "%s.tfrecord" % segment_no_ext)
        input_file = os.path.join(input_dir, movie, segment)
        with tf.python_io.TFRecordWriter(output_file) as writer:
             # Read and resize all video frames, np.uint8 of size [N,H,W,3]
             video = imageio.get_reader(input_file)
             #frame_gen = video.iter_data()
             #frame_list = []
             #for frame in frame_gen:
             #    resized_frame = cv2.resize(frame, (400,400))
             #    frame_list.append(resized_frame)
               
             #frames = np.stack(frame_list, axis=0)
             vidinfo = video.get_meta_data()
             # vidfps = vidinfo['fps']
             # vid_W, vid_H = vidinfo['size']
             no_frames = vidinfo['nframes'] # last frames seem to be bugged
 
             f_timesteps, f_H, f_W, f_C = [32, 400, 400, 3]
             f_timesteps = no_frames
 
             #slope = (no_frames-1) / float(f_timesteps - 1)
             #indices = (slope * np.arange(f_timesteps)).astype(np.int64)
 
             frames = np.zeros([f_timesteps, f_H, f_W, f_C], np.uint8)
 
             timestep = 0
             # for vid_idx in range(no_frames):
             #for vid_idx in indices.tolist():
             for frame in video:
                 #frame = video.get_data(vid_idx)
                 # frame = frame[:,:,::-1] #opencv reads bgr, i3d trained with rgb
                 reshaped = cv2.resize(frame, (f_W, f_H))
                 frames[timestep, :, :, :] = reshaped
                 timestep += 1
 
             video.close()
           
             features = {}
             features['num_frames']  = _int64_feature(frames.shape[0])
             features['height']      = _int64_feature(frames.shape[1])
             features['width']       = _int64_feature(frames.shape[2])
             features['channels']    = _int64_feature(frames.shape[3])
             #features['class_label'] = _int64_feature(example['class_id'])
             #features['class_text']  = _bytes_feature(tf.compat.as_bytes(example['class_label']))
             #features['filename']    = _bytes_feature(tf.compat.as_bytes(example['video_id']))
             #features['filename']    = _bytes_feature(tf.compat.as_bytes(fname))
             features['movie']    = _bytes_feature(tf.compat.as_bytes(movie))
             features['segment']    = _bytes_feature(tf.compat.as_bytes(segment_no_ext))
           
             # Compress the frames using JPG and store in as a list of strings in 'frames'
             encoded_frames = [tf.compat.as_bytes(cv2.imencode(".jpg", frame)[1].tobytes())
                               for frame in frames]
             features['frames'] = _bytes_list_feature(encoded_frames)
           
             tfrecord_example = tf.train.Example(features=tf.train.Features(feature=features))
             writer.write(tfrecord_example.SerializeToString())
             #tqdm.write("Output file %s written!" % output_file)
             #print("Output file %s written!" % output_file)

Parallel(n_jobs=10)(delayed(generate_tfrecord)(movie) for movie in movies)
