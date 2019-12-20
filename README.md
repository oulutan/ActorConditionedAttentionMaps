# ActorConditionedAttentionMaps
Work in Progress. Im updating the readme and preparing a tutorial for training on other datasets.

# Training the model on AVA

All of my python package versions are available in the requirements.txt file. Check there for version issues

Codebase is structured on multiple directories. 

`scripts`: The initial processing scripts. Downloading and processing ava

`model_training`: Includes files for training the ACAM model. Includes data loaders and training files

`data`: All dataset related files go here in their own separate subfolders

`obj_detection`: This is a codebase that focuses on running object detectors on AVA and finetuning it for AVA actors. Based on maskrcnn-benchmark from facebook. https://github.com/facebookresearch/maskrcnn-benchmark

`evaluation`: this includes the evaluation codebase for the AVA. It is a submodule from ACtivityNet and includes some custom files.

1. First of all run `source set_environment.sh` in the main folder `./ActorConditionedAttentionMaps`. This sets the environment which points to the full path of this directory in your system. This is essential as everything uses this env variable. Do this whenever you login or run any of my files. 

2. Go to `scripts` and run `01_download_videos.py`. This will download the movies for AVA trainval. If you want the test files change the `split` variable in the first lines of `01_download_videos.py` and run again. Most files use similar structure so you have to run them separately for splits.

3. Go to `scripts` and run `02_crop_video_segments.py`. This will crop the movies into 2-second video segments. As the data is annotated 1fps this makes an overlap between consecutive segments. This will also generate the more compact/easily accessible `segment_annotation` files. File `02b_get_segment_annotations.py` is optional and can be used to quickly update annotations without cropping. Don't run this

4. Go to the object_detector folder and follow the readme there. It shows you how to set up the object detector and run it for AVA. optionally you can fine tune the detector for AVA or download my fine-tuned weights. When everything is set you should be running `03_keyframe_detect_objects.py` script in `ActorConditionedAttentionMaps/obj_detection/maskrcnn-benchmark/AVA_scripts/` which will generate ACAM readable object detection results. 

5. At this point data is ready to be trained. However, I strongly recommend extracting tfrecord files for these videos. This helps with performance significantly especially if you are using multiple gpus (unefficient GPU loading could increase the training time to months!). At this point you should have ./data/AVA/segments which has 2 second video segments and ./data/AVA/objects_finetuned_mrcnn which has object detection results. Now run `04_convert_ava_vids_with_labels.py` in `scripts` to generate tfrecords files for each segment. Instead of a one big file, I used individual tfrecord files. This helps me to filter files on the run easily by just changing the input list during training. 

6. Now the data is ready. Go to `model_training` directory and run `train_multi_gpu.py` file. This script has various arguments which can be checked in between lines 105-115. Most of them have default options except GPU selection. -g argument sets the CUDA_VISIBLE_DEVICES and masks gpus. if you only want to use GPU 0 it should be `-g 0`, if you want to use GPUs 1,2,3 then `-g 1,2,3`. If you are using it in a system with a GPU scheduler, remove this part and handle it manually. 

7. Model will train for 1 epoch and then evaluate the results on the validation set. Periodically check this to see your performance on AVA. This will also generate AVA-style results which could be used for ActivityNet submissions. 

# Training on other Datasets
