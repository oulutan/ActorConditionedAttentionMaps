import os

all_samples = []

#data_folder = '../HumanSegments/'
data_folder = '../SyntheticData/'
all_actors = os.listdir(data_folder)
all_actors.sort()

for actor in all_actors:
    actor_folder = data_folder + actor
    cur_actor_vids = os.listdir(actor_folder)
    cur_actor_vids = [v for v in cur_actor_vids if v.endswith('.mp4')] # for synth
    cur_actor_vids.sort()
    #cur_actor_vids = ["%s/%s" % (actor, vid) for vid in cur_actor_vids] # for human
    cur_actor_vids = ["%s-%s" % (actor, vid) for vid in cur_actor_vids] # for synth
    all_samples.append(cur_actor_vids)

import json
#fname = 'all_samples.json'
fname = 'all_synth_samples.json'
with open(fname, 'w') as fp:
    json.dump(all_samples, fp)
print('%s written' % fname)
import pdb;pdb.set_trace()
