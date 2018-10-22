#python -O $AVA_DIR/evaluation/ActivityNet/Evaluation/get_ava_performance.py \
export PYTHONPATH=$PYTHONPATH:$AVA_DIR/evaluation/ActivityNet/Evaluation
python -O $AVA_DIR/evaluation/get_ava_performance_custom.py \
  -l $AVA_DIR/AVA/annotations/ava_action_list_v2.1_for_activitynet_2018.pbtxt.txt \
  -g $AVA_DIR/AVA/annotations/ava_val_v2.1.csv \
  -e $AVA_DIR/AVA/annotations/ava_val_excluded_timestamps_v2.1.csv \
  -d $AVA_DIR/AVA/ava_style_results/$1.csv
#  -d $AVA_DIR/AVA/ava_style_results/VALIDATION_Results_v100sI3DTail_32t_14.csv
#  -d $AVA_DIR/AVA/ava_style_results/VALIDATION_Results_t32_10batch_v100_basic_25.csv

#for index in 02 05 09 10 13 14
#for index in 17 18 20 21 23 24 25
#do
#    python -O $AVA_DIR/evaluation/get_ava_performance_custom.py \
#      -l $AVA_DIR/AVA/annotations/ava_action_list_v2.1_for_activitynet_2018.pbtxt.txt \
#      -g $AVA_DIR/AVA/annotations/ava_val_v2.1.csv \
#      -e $AVA_DIR/AVA/annotations/ava_val_excluded_timestamps_v2.1.csv \
#      -d $AVA_DIR/AVA/ava_style_results/VALIDATION_Results_t32_10batch_v100_basic_$index.csv
##      -d $AVA_DIR/AVA/ava_style_results/VALIDATION_Results_v100sI3DTail_32t_$index.csv
#done
