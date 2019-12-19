import os
import subprocess
import json

#from joblib import Parallel, delayed

# AVA_FOLDER = '/media/sidious_data/AVA'
# AVA_FOLDER = '/data/home/ulutan/sidious_data/AVA'
AVA_FOLDER = os.environ['ACAM_DIR'] + '/data/AVA'
#movies_folder = AVA_FOLDER + '/movies/'
#movies_folder_test = AVA_FOLDER + '/movies/test/'
#segments_folder = AVA_FOLDER + '/segments/'
annotations_folder = AVA_FOLDER + '/annotations/'
data_folder = AVA_FOLDER + '/data/'

####################################### THIS IS FOR UPDATING ANNOTATION FILES. YOU DONT HAVE TO RUN THIS IF YOU ARE RUNNING 02_crop##############################
print("THIS IS FOR UPDATING ANNOTATION FILES. YOU DONT HAVE TO RUN THIS IF YOU ARE RUNNING 02_crop ")

#split = 'train'
#split = 'val'
split = 'test'
#split_folder = os.path.join(segments_folder, split)

#clips_folder = os.path.join(split_folder, 'clips')
#midframes_folder = os.path.join(split_folder, 'midframes')

# movie2file = {}
# all_movies = os.listdir(movies_folder)
# for movie_file in all_movies:
#     movie_key = movie_file.split('.')[0]
#     movie2file[movie_key] = movie_file


def hou_min_sec(millis):
    millis = int(millis)
    seconds=(millis/1000)%60
    seconds = int(seconds)
    minutes=(millis/(1000*60))%60
    minutes = int(minutes)
    hours=(millis/(1000*60*60))
    return ("%d:%d:%d" % (hours, minutes, seconds))

# def crop_video_segment(segment_key):
#     movie_key, timestamp = segment_key.split('.')
#     movie_file = movie2file[movie_key]
#     movie_path = os.path.join(movies_folder, movie_file)
# 
#     midframe_path = os.path.join(midframes_folder, movie_key, '%s.jpg' % timestamp)
#     clip_path = os.path.join(clips_folder, movie_key, '%s.mp4' % timestamp)
# 
#     # extract midframe
#     ffmpeg_command = 'rm %(midframe_path)s; ffmpeg -loglevel error -ss %(timestamp)f -i %(movie_path)s -frames:v 1 %(midframe_path)s' % {
#         'midframe_path': midframe_path,
#         'movie_path': movie_path,
#         'timestamp': float(timestamp)
#         }
#     # print('\nFFMPEG_COMMAND: %s \n' %ffmpeg_command)
#     subprocess.call(ffmpeg_command, shell=True)
# 
#     # extract the segment
#     mid_timestamp = float(timestamp)
#     clip_start = mid_timestamp - 1
#     clip_length = 2.0
# 
#     start_stamp = hou_min_sec(clip_start*1000)
#     # import pdb;pdb.set_trace()
# 
#     ffmpeg_command = 'rm %(clip_path)s; ffmpeg -loglevel error -ss %(start_timestamp)s -i %(movie_path)s -g 1 -force_key_frames 0 -t %(clip_length)d -strict -2 %(clip_path)s' % {
# 		'start_timestamp': start_stamp,
# 		'clip_length': clip_length,
# 		'movie_path': movie_path,
# 		'clip_path': clip_path
# 		}
# 
#     # print('\nFFMPEG_COMMAND: %s \n' %ffmpeg_command)
#     subprocess.call(ffmpeg_command, shell=True)
#     print('\nDone with %s\n' %segment_key)
# 


    


def main():
    print("Working on %s" %split)
    # print('Generating output folders!')
    # if not os.path.exists(split_folder):
    #     os.mkdir(split_folder)
    # if not os.path.exists(clips_folder):
    #     os.mkdir(clips_folder)
    # if not os.path.exists(midframes_folder):
    #     os.mkdir(midframes_folder)

    #annotation_file = annotations_folder + 'ava_%s_v2.1.csv' % split
    annotation_file = annotations_folder + 'ava_%s_v2.2.csv' % split
    with open(annotation_file) as fp:
        file_name_list = fp.read().splitlines()

    #The format of a row is the following: video_id, middle_frame_timestamp, person_box, action_id
    #-5KQ66BBWC4,0904,0.217,0.008,0.982,0.966,12
    
    # for seg_info in file_name_list:
    #     movie_key = seg_info.split(',')[0]
    #     movie_clips = os.path.join(clips_folder, movie_key)
    #     movie_midframes = os.path.join(midframes_folder, movie_key)
    #     if not os.path.exists(movie_clips):
    #         os.mkdir(movie_clips)
    #     if not os.path.exists(movie_midframes):
    #         os.mkdir(movie_midframes)
        
    
    print('Finding the unique segments and save annotations on a dict')
    segment_keys = []
    annotations_dict = {}
    previous_key = ''
    for anno_index, seg_info in enumerate(file_name_list):
        info_parts = seg_info.split(',')
        movie_key = info_parts[0]
        timestamp = info_parts[1]
        segment_key = '%s.%s' % (movie_key, timestamp)
        if segment_key != previous_key:
            segment_keys.append(segment_key)
            annotations_dict[segment_key] = [ (anno_index, seg_info)]
        else:
            annotations_dict[segment_key].append((anno_index, seg_info))
        previous_key = segment_key

    print('Organizing annotations')
    combined_annotation_dict = {}
    for segment_key in segment_keys:
        current_anno_list = annotations_dict[segment_key]

        prev_bbox = (0.0, 0.0, 0.0, 0.0)
        boxes_list = []
        for anno in current_anno_list:
            anno_index, seg_info = anno
            movie_key, timestamp, left, top, right, bottom, action_id, person_id  = seg_info.split(',')

            left, top, right, bottom = [float(coord) for coord in [left, top, right, bottom]]
            current_bbox = (left, top, right, bottom)
            if prev_bbox != current_bbox:
                bbox_info = {}
                bbox_info['bbox'] = current_bbox
                bbox_info['actions'] = [action_id]
                bbox_info['annotation_index'] = [anno_index]
                boxes_list.append(bbox_info)
                prev_bbox = current_bbox
            else:
                boxes_list[-1]['actions'].append(action_id)
                boxes_list[-1]['annotation_index'].append(anno_index)
        
        combined_annotation_dict[segment_key] = boxes_list

    with open(data_folder + 'segment_annotations_v22_%s.json' % split, 'w') as fp:
        json.dump(combined_annotation_dict, fp)


    segment_keys.sort()
    ## stupid hard drive failure fixes
    # print("Working on failure cases only")
    # segment_keys = ["JNb4nWexD0I.0971", "JNb4nWexD0I.0970", "JNb4nWexD0I.0973", "JNb4nWexD0I.0974", "JNb4nWexD0I.0982", "JNb4nWexD0I.0981", "JNb4nWexD0I.0980", "JNb4nWexD0I.0983", "JNb4nWexD0I.0978", "JNb4nWexD0I.0968", "JNb4nWexD0I.0977", "JNb4nWexD0I.0975", "JNb4nWexD0I.0972", "JNb4nWexD0I.0984", "JNb4nWexD0I.0976", "JNb4nWexD0I.0979"]

    # print('Extracting segments!')
    # for segment_key in segment_keys:
    #     crop_video_segment(segment_key)
    # Parallel(n_jobs=10) (delayed(crop_video_segment)(segment_key, ) for segment_key in segment_keys)
    # Parallel(n_jobs=2) (delayed(crop_video_segment)(segment_key, ) for segment_key in segment_keys)

### For test split

# def crop_video_segment_test(segment_key, testkey2file):
#     movie_key, timestamp = segment_key.split('.')
#     movie_file = testkey2file[movie_key]
#     movie_path = os.path.join(movies_folder_test, movie_file)
# 
#     midframe_path = os.path.join(midframes_folder, movie_key, '%s.jpg' % timestamp)
#     clip_path = os.path.join(clips_folder, movie_key, '%s.mp4' % timestamp)
# 
#     # extract midframe
#     ffmpeg_command = 'rm %(midframe_path)s; ffmpeg -loglevel error -ss %(timestamp)f -i %(movie_path)s -frames:v 1 %(midframe_path)s' % {
#         'midframe_path': midframe_path,
#         'movie_path': movie_path,
#         'timestamp': float(timestamp)
#         }
#     # print('\nFFMPEG_COMMAND: %s \n' %ffmpeg_command)
#     subprocess.call(ffmpeg_command, shell=True)
# 
# 
#     # extract the segment
#     mid_timestamp = float(timestamp)
#     clip_start = mid_timestamp - 1
#     clip_length = 2.0
# 
#     start_stamp = hou_min_sec(clip_start*1000)
#     # import pdb;pdb.set_trace()
# 
#     ffmpeg_command = 'rm %(clip_path)s; ffmpeg -loglevel error -ss %(start_timestamp)s -i %(movie_path)s -g 1 -force_key_frames 0 -t %(clip_length)d -strict -2 %(clip_path)s' % {
# 		'start_timestamp': start_stamp,
# 		'clip_length': clip_length,
# 		'movie_path': movie_path,
# 		'clip_path': clip_path
# 		}
# 
#     # print('\nFFMPEG_COMMAND: %s \n' %ffmpeg_command)
#     subprocess.call(ffmpeg_command, shell=True)
#     print('\nDone with %s\n' %segment_key)

def main_test():
    print('Running on test samples')

    # print('Generating output folders!')
    # if not os.path.exists(split_folder):
    #     os.mkdir(split_folder)
    # if not os.path.exists(clips_folder):
    #     os.mkdir(clips_folder)
    # if not os.path.exists(midframes_folder):
    #     os.mkdir(midframes_folder)
    
    

    # movie_names_file = annotations_folder + 'ava_test_v2.2.txt'
    # with open(movie_names_file) as fp:
    #     movie_file_names = fp.readlines()
    # movie_file_names = [row.strip() for row in movie_file_names]

    # movie_ids = []
    # testkey2file = {}
    # for movie_file_name in movie_file_names:
    #     movie_id, file_ext = movie_file_name.split('.')
    #     movie_ids.append(movie_id)
    #     testkey2file[movie_id] = movie_file_name

    test_movie_names_file = annotations_folder + 'ava_test_v2.2.txt'
    with open(test_movie_names_file) as fp:
        movie_ids = fp.readlines()
    movie_ids = [row.strip() for row in movie_ids]

    ## DEBUG
    # movie_ids = ['W8TFzEy0gp0']
    
    # for movie_key in movie_ids:
    #     movie_clips = os.path.join(clips_folder, movie_key)
    #     movie_midframes = os.path.join(midframes_folder, movie_key)
    #     if not os.path.exists(movie_clips):
    #         os.mkdir(movie_clips)
    #     if not os.path.exists(movie_midframes):
    #         os.mkdir(movie_midframes)

    excluded_timestamps_file = annotations_folder + 'ava_test_excluded_timestamps_v2.2.csv'
    with open(excluded_timestamps_file) as fp:
        excluded_timestamps = fp.read().splitlines()
    # format for excluded: '-FLn0aeA6EU,0913'

    # go through this and make a hash mapped lists
    excluded_mapping = {}
    for excluded_timestamp in excluded_timestamps:
        movie_key, timestamp = excluded_timestamp.split(',')
        if movie_key in excluded_mapping.keys():
            excluded_mapping[movie_key].append(timestamp)
        else:
            excluded_mapping[movie_key] = [timestamp]

    all_timestamps = ['%.4i' % ii for ii in range(902, 1799)]


    segment_keys = []
    for movie_id in movie_ids:
        if movie_id in excluded_mapping.keys():
            mov_excluded = excluded_mapping[movie_id]
        else:
            mov_excluded = []
        for timestamp in all_timestamps:
            if timestamp in mov_excluded:
                continue
            else:
                segment_key = '%s.%s' % (movie_id, timestamp)
                segment_keys.append(segment_key)
    


    print('Creating empty annotations to keep same file structure as train and val')
    combined_annotation_dict = {}
    for segment_key in segment_keys:
        # current_anno_list = annotations_dict[segment_key]

        # prev_bbox = (0.0, 0.0, 0.0, 0.0)
        boxes_list = []
        # for anno in current_anno_list:
        #     anno_index, seg_info = anno
        #     movie_key, timestamp, left, top, right, bottom, action_id  = seg_info.split(',')

        #     left, top, right, bottom = [float(coord) for coord in [left, top, right, bottom]]
        #     current_bbox = (left, top, right, bottom)
        #     if prev_bbox != current_bbox:
        #         bbox_info = {}
        #         bbox_info['bbox'] = current_bbox
        #         bbox_info['actions'] = [action_id]
        #         bbox_info['annotation_index'] = [anno_index]
        #         boxes_list.append(bbox_info)
        #         prev_bbox = current_bbox
        #     else:
        #         boxes_list[-1]['actions'].append(action_id)
        #         boxes_list[-1]['annotation_index'].append(anno_index)
        
        combined_annotation_dict[segment_key] = boxes_list

    with open(data_folder + 'segment_annotations_v22_%s.json' % split, 'w') as fp:
        json.dump(combined_annotation_dict, fp)
    print('Saved segment annotations!')
    # import pdb;pdb.set_trace()

    # segment_keys.sort()
    # # stupid hard drive failure fixes
    # # segment_keys = ["WMFTBgYWJS8.0940", "WMFTBgYWJS8.0941", "WMFTBgYWJS8.0944", "WMFTBgYWJS8.0946", "WMFTBgYWJS8.0951", "WMFTBgYWJS8.0969"]
    # # segment_keys = ["WMFTBgYWJS8.0949"]
    # segment_keys = ["WMFTBgYWJS8.0937"]
    # print('Extracting segments!')
    # #debug
    # for segment_key in segment_keys:
    #     crop_video_segment_test(segment_key, testkey2file)
    # # Parallel(n_jobs=10) (delayed(crop_video_segment_test)(segment_key, testkey2file) for segment_key in segment_keys)




if __name__ == '__main__':
    print('Running on split %s' % split)
    if split != 'test':
        main()
    else:
        main_test()
