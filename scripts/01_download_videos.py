import wget
import os

ACAM_FOLDER = os.environ['ACAM_DIR']
AVA_FOLDER = ACAM_FOLDER + '/data/AVA'

split = "trainval"
#split = "test"

file_source = "https://s3.amazonaws.com/ava-dataset/%s/" % split
#file_source = "https://s3.amazonaws.com/ava-dataset/trainval/"
#file_source = "https://s3.amazonaws.com/ava-dataset/test/"


#if split == "trainval":
#    files_txt = AVA_FOLDER + "/annotations/ava_file_names_trainval_v2.1.txt"
#else:
#    files_txt = AVA_FOLDER + "/annotations/ava_file_names_test_v2.1.txt"
files_txt = AVA_FOLDER + "/annotations/ava_file_names_%s_v2.1.txt" % split

with open(files_txt) as fp:
    file_name_list = fp.readlines()

file_name_list = [row.strip() for row in file_name_list]
file_name_list.sort()

# output_folder = '../movies/'
output_folder = AVA_FOLDER + "/movies/%s/" % split
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


for ff, file_name in enumerate(file_name_list):
    url = file_source + file_name
    file_out = wget.download(url, output_folder)
    print('\n %i/%i Downloaded %s' % (ff+1, len(file_name_list), file_name))
