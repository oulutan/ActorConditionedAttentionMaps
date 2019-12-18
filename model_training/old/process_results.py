import numpy as np
import json
import os

from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score



NUM_CLASSES = 60

import socket
HOSTNAME = socket.gethostname()

ACAM_FOLDER = os.environ['ACAM_DIR']

AVA_FOLDER = ACAM_FOLDER + '/data/AVA' 

DATA_FOLDER = AVA_FOLDER + '/data/'

with open(DATA_FOLDER+'label_conversions.json') as fp:
    LABEL_CONVERSIONS = json.load(fp)

ANN2TRAIN = LABEL_CONVERSIONS['ann2train']
TRAIN2ANN = LABEL_CONVERSIONS['train2ann']
    

#from sklearn.metrics import precision_recall_curve
#import matplotlib.pyplot as plt
#def plot_pr_curve(y_truth, y_pred):
#    precision, recall, thresholds = precision_recall_curve(y_truth, y_pred)
#    plt.step(recall, precision, color='b', alpha=0.2,
#             where='post')
#    plt.fill_between(recall, precision, step='post', alpha=0.2,
#                     color='b')
#    
#    plt.xlabel('Recall')
#    plt.ylabel('Precision')
#    plt.ylim([0.0, 1.05])
#    plt.xlim([0.0, 1.0])
#    plt.show()



def get_per_class_AP(results_list):
    '''
    results_list is a list where each
    result = ['path' [multilabel-binary labels] [probs vector]]

    returns per class_AP vector with class average precisions
    '''
    class_results = [{'truth':[], 'pred':[]} for _ in range(NUM_CLASSES)]

    for result in results_list:
        cur_key = result[0]
        cur_roi_id = result[1]
        cur_truths = result[2]
        cur_preds = result[3]
        
        # cur_preds = np.random.uniform(size=40)
        # cur_preds = [0 for _ in range(40)]

        for cc in range(NUM_CLASSES):
            class_results[cc]['truth'].append(cur_truths[cc])
            class_results[cc]['pred'].append(cur_preds[cc])

    ground_truth_count = []
    class_AP = []
    class_recall = []
    class_prec = []
    for cc in range(NUM_CLASSES):
        y_truth = class_results[cc]['truth']
        y_pred = class_results[cc]['pred']
        AP = average_precision_score(y_truth, y_pred)
        recall_th = 0.50
        recall = recall_score(y_truth, np.array(y_pred) > recall_th) 
        precision = precision_score(y_truth, np.array(y_pred) > recall_th)
        class_recall.append(recall)
        class_prec.append(precision)

        # print(AP)
        # plot_pr_curve(y_truth, y_pred)
        # import pdb; pdb.set_trace()

        if np.isnan(AP): AP = 0

        class_AP.append(AP)
        ground_truth_count.append(sum(y_truth))
        
    # import pdb; pdb.set_trace()
    return class_AP, ground_truth_count, class_recall, class_prec

def get_class_AP_str(class_AP, cnt, class_recall, class_prec):
    ''' Returns a printable string'''
    ap_str = ''
    for cc in range(NUM_CLASSES):
        class_str = TRAIN2ANN[str(cc)]['class_str'][0:15] # just take the first 15 chars, some of them are too long
        class_cnt = cnt[cc]
        AP = class_AP[cc]
        rec = class_recall[cc]
        prec = class_prec[cc]
        # AP = AP if not np.isnan(AP) else 0
        # cur_row = '%s:    %i%% \n' %(class_str, AP*100)#class_str + ':    ' + str(tools.get_3_decimal_float(AP)) + '\n'
        cur_row = class_str + '(%i)' % class_cnt + ':'
        cur_row += (' ' * (25 -len(cur_row))) + '%.1f%%' % (AP*100.0) + "\t" + '%.1f%%' % (rec*100.0) + "\t" + '%.1f%%' % (prec*100.0) + '\n'
        ap_str += cur_row
    class_avg = np.mean(class_AP)
    avg_rec = np.mean(class_recall)
    avg_prec = np.mean(class_prec)
    # class_avg = class_avg if not np.isnan(class_avg) else 0
    ap_str += '\n' + 'Average:' + (' '*17) + '%.1f%%' % (class_avg*100.0) + "\t" + "%.1f%%" % (avg_rec*100.) + "\t" + "%.1f%%" % (avg_prec*100.) + '\n'
    return ap_str

def get_AP_str(results_list):
    class_AP, cnt, class_recall, class_prec = get_per_class_AP(results_list)
    ap_str = get_class_AP_str(class_AP, cnt, class_recall, class_prec)
    return ap_str

if __name__ == '__main__':
    pass
    # file_to_read = '/media/storage_brain/ulutan/AVA/ActionResults/goods/Test_Results_05.txt'
    # results_list = tools.read_serialized_results(file_to_read)
    # class_AP, cnt = get_per_class_AP(results_list)
    # print(get_class_AP_str(class_AP, cnt))
    # import pdb; pdb.set_trace()

    # test coco eval
    
    # process_results_coco(AVA_fake_result)


