import wget

# file_source = "https://s3.amazonaws.com/ava-dataset/trainval/"
file_source = "https://s3.amazonaws.com/ava-dataset/test/"


# with open('../annotations/ava_file_names_trainval.txt') as fp:
with open('../annotations/ava_file_names_test_v2.1.txt') as fp:
    file_name_list = fp.readlines()

file_name_list = [row.strip() for row in file_name_list]
file_name_list.sort()

# output_folder = '../movies/'
output_folder = '../movies/test/'


for ff, file_name in enumerate(file_name_list):
    url = file_source + file_name
    file_out = wget.download(url, output_folder)
    print('\n %i/%i Downloaded %s' % (ff+1, len(file_name_list), file_name))
