import json
import os
import numpy as np



def read_serialized_results(file_path):
    with open(file_path) as fp:
        data = json.load(fp)
    return data

def save_serialized_list(input_list, file_path):
    with open(file_path, 'w') as fp:
        json.dump(input_list, fp)

#RES1 = "VALIDATION_Results_soft_attn_ava22_lowlr_finetuned_180"
#RES2 = "VALIDATION_Results_soft_attn_ava22_lowlr_finetuned_179"
#RES1 = "TEST_Results_soft_attn_ava22_lowlr_finetuned_180"
#RES1 = "VALIDATION_Results_soft_attn_ava22_lowlr_finetuned_180_VALIDATION_Results_soft_attn_less1000filtered_220"
#RES1 = "VALIDATION_Results_soft_attn_ava22_lowlr_finetuned_180_VALIDATION_Results_soft_attn_less1000filtered_220_VALIDATION_Results_soft_attn_1000_3000filtered_181"
#RES2 = "VALIDATION_Results_soft_attn_less1000filtered_220"
#RES2 = "VALIDATION_Results_soft_attn_1000_3000filtered_181"
#RES2 = "VALIDATION_Results_soft_attn_ava22filteredclasses_195"
#RES2 = "VALIDATION_Results_soft_attn_filteredclasses_196"
#RES2 = "VALIDATION_Results_soft_attn_filteredclasses_236"
#RES2 = "VALIDATION_Results_soft_attn_filtered_and_randomclasses_240"
#RES2 = "TEST_Results_soft_attn_filtered_and_randomclasses_240"
#RES2 = "VALIDATION_Results_soft_attn_filteredclasses_235"
#RES2 = "VALIDATION_Results_soft_attn_1000_3000filtered_205"
#RES2 = "VALIDATION_Results_soft_attn_less1000filtered_220"
RES1 = "VALIDATION_Results_soft_attn_ava22_lowlr_finetuned_194"
#RES2 = "VALIDATION_Results_soft_attn_ava22_300x300_val_194"
#RES2 = "VALIDATION_Results_soft_attn_ava22_200x200_val_194"
#RES2 = "VALIDATION_Results_soft_attn_ava22_500x500_val_194"
#RES2 = "VALIDATION_Results_soft_attn_ava22_centercrop_0_0001lr_195"
RES2 = "VALIDATION_Results_soft_attn_ava22_twotails_199"


ACAM_FOLDER = os.environ['ACAM_DIR']
AVA_FOLDER = ACAM_FOLDER + '/data/AVA'
RESULTS_FOLDER = AVA_FOLDER + '/ActionResults/'

results_path1 = RESULTS_FOLDER + '%s.txt' % RES1
results_path2 = RESULTS_FOLDER + '%s.txt' % RES2

allres1 = read_serialized_results(results_path1)
allres2 = read_serialized_results(results_path2)


minidx = np.argmin([len(allres1), len(allres2)])

longres = [allres1, allres2][1-minidx]
shortres = [allres1, allres2][minidx]

# make a dict out of shortres
shortdict = {}
print("Making shortdict")
for result in shortres:
    cur_key = result[0]
    cur_video_id, cur_timestamp = cur_key.split('.')
    cur_roi_id = result[1]
    cur_truths = result[2]
    cur_preds = result[3]

    dkey = "%s_%i" % (cur_key, cur_roi_id)

    shortdict[dkey] = cur_preds


# go thorugh longres and find if shortres has same datapoints
#combined_res = []
shared_cnt = 0
print("Iterating longres")
for ii,result in enumerate(longres):
    cur_key = result[0]
    cur_video_id, cur_timestamp = cur_key.split('.')
    cur_roi_id = result[1]
    cur_truths = result[2]
    cur_preds = result[3]

    dkey = "%s_%i" % (cur_key, cur_roi_id)

    if dkey in shortdict:
        shared_cnt += 1
        short_preds = shortdict[dkey]
        ## choose max
        longshort = np.stack([cur_preds, short_preds], axis=-1)
        final_res = np.max(longshort, -1).tolist()
        #final_res = np.mean(longshort, -1).tolist()
        result[3] = final_res
        ## per class based
        #for cid in [43, 22, 23, 28, 41, 5, 39, 15, 17, 37, 35]:
        #    result[3][cid] = max(short_preds[cid], result[3][cid])

print("%i / %i shared" % (shared_cnt, len(longres)))

final_name = RESULTS_FOLDER +  "%s_%s.txt" % (RES1, RES2)


save_serialized_list(longres, final_name)

print("%s written!" % final_name )



import dataset_ava

dataset_ava.process_evaluation_results("%s_%s" % (RES1, RES2))
