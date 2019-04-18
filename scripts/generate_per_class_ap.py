import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


#res1_file = "VALIDATION_Results_soft_attn_23.json"
#res1_file = "VALIDATION_Results_soft_attn_pooled_51.json"
res1_file = "VALIDATION_Results_soft_attn_pooled_cosine_drop_170.json"
#res2_file = "VALIDATION_Results_soft_attn_23.json"
res2_file = "VALIDATION_Results_basic_model_33.json"

with open("../data/AVA/result_mAPs/%s" % res1_file) as fp:
    res1 = json.load(fp)


with open("../data/AVA/result_mAPs/%s" % res2_file) as fp:
    res2 = json.load(fp)

cls_strs = [r[0].split("PascalBoxes_PerformanceByCategory/AP@0.5IOU/")[-1] for r in res1]
#cls_strs[-1] = "mAP"
del cls_strs[-1]

ap_map1 = {cls_strs[ii]: int(100*res1[ii][1]) for ii in range(len(res1)-1)}
ap_map2 = {cls_strs[ii]: int(100*res2[ii][1]) for ii in range(len(res2)-1)}


data_dir = '../data/AVA/data/'
with open(data_dir + 'action_lists_train.json') as fp:
  tr_lists = json.load(fp)
with open(data_dir + 'action_lists_val.json') as fp:
  val_lists = json.load(fp)
with open(data_dir + 'label_conversions.json') as fp:
  label_conv = json.load(fp)


cls_nos = label_conv['ann2train'].keys()

tr_sizes = {label_conv['ann2train'][cls_no_str]['class_str']: len(tr_lists[cls_no_str]) for cls_no_str in cls_nos}
val_sizes = {label_conv['ann2train'][cls_no_str]['class_str']: len(val_lists[cls_no_str]) for cls_no_str in cls_nos}


# # sort by training size,ie index 1
# combined.sort(key=lambda x: x[1])
cls_strs.sort(key=lambda x: tr_sizes[x])
# draw first N
N = 60
classes_to_plot = cls_strs[-N:][::-1]

ap1 = [ap_map1[cc] for cc in classes_to_plot]
ap2 = [ap_map2[cc] for cc in classes_to_plot]

# # 14: walk (person movement), 15: answer_phone (object manipulation), ..., 63: write (object man), 64: fight/hit person (person interaction)
# movement_APs = []
# object_APs = []
# inter_APs = []
# for cls_no_str in cls_nos:
#   cls_str = label_conv['ann2train'][cls_no_str]['class_str']
#   class_AP = res_dict[cls_str]
#   if int(cls_no_str) <= 14:
#       movement_APs.append(class_AP)
#   elif int(cls_no_str) <= 63:
#       object_APs.append(class_AP)
#   elif int(cls_no_str) <= 80:
#       inter_APs.append(class_AP)
#   else:
#       print("Wrong class id error")
plt.figure(figsize=(15,5))

bar1_coords = np.arange(N)*4
bar2_coords = np.arange(N)*4+1
plt.bar(bar1_coords, ap1, width=1, color="darkblue")
plt.bar(bar2_coords, ap2, width=1, color="rosybrown")

#cls_reduced = ["{:<10}".format(cc[:10]) for cc in classes_to_plot]
#cls_reduced = [cc[:15] for cc in classes_to_plot]

cls_reduced = [ cc[:15] for cc in classes_to_plot]
plt.xticks(bar1_coords, cls_reduced, rotation='vertical')

for ii,cc in enumerate(classes_to_plot):
    ymax = max(ap_map1[cc], ap_map2[cc]) + 1.5
    curtext = ("+" if (ap_map1[cc]-ap_map2[cc])>=0 else "-") +  "%i " % abs(ap_map1[cc]-ap_map2[cc])
    plt.text(bar1_coords[ii]-2.5, ymax, curtext, fontsize=12)

plt.legend(["ACAM", "Base Model"])
plt.ylabel("Class AP")
plt.ylim([0,105])
plt.tight_layout()

plt.show()
plt.savefig("class_ap")
