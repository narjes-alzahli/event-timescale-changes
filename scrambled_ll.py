import os
import time
import numpy as np
import deepdish as dd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import pearsonr
from brainiak.eventseg.event import EventSegment
from sklearn.model_selection import train_test_split

# Directory containing .h5 files
directory_path = '../data/SL/'

# Get all .h5 files in the directory
h5_files = [file for file in os.listdir(directory_path) if file.endswith('.h5')]

test_size = 0.5
nEvents = np.arange(2,11)

n=0

areas_LLs = {}

for file in h5_files:

    # Construct the full file path
    ti = time.time()
    file_path = os.path.join(directory_path, file)
    area_name = file.split('.')[0]
    
    # Load the data from the .h5 file
    D_orig = dd.io.load(file_path)
    vox_N = D_orig[list(D_orig.keys())[0]]['Intact'].shape[2]
    D = np.zeros((30, 6, 60, vox_N))
    for i, s in enumerate(D_orig.keys()):
        D[i] = D_orig[s]['Intact']

    G = D.mean(0)

    N_subj = D.shape[0]

    LLs = np.empty((len(nEvents),6, 2), dtype=object)

    num_timepoints = G.shape[1]

    for ev_i, ev in enumerate(nEvents):
        for repeat in range(6):

            sum_LL = 0

            train_indices, test_indices = train_test_split(np.arange(N_subj),
                                                test_size=test_size, 
                                                random_state=42)

            train = D[train_indices].mean(0)
            test  = D[test_indices].mean(0)

            hmm = EventSegment(ev, split_merge=True)                
            hmm.fit(train[repeat])

            # meeting scramble = True, should fail, when the model doesn't work, 50 times or nulls
            _, LL = hmm.find_events(test[repeat])
            LLs[ev_i, repeat, 0] = LL

            n_ev = hmm.n_events
            orig_ev_pat = Evseg.event_pat_.copy()

            for p in range(50):
                rand_perm = np.random.permutation(n_ev)
                while np.all(rand_perm == np.arange(n_ev)):
                    rand_perm = np.random.permutation(n_ev)

                Evseg.set_event_patterns(orig_ev_pat[:, rand_perm])
                perm_LL[p] = Evseg.find_events(testing_data)

            scrambled = []
            for i in range(50):
                _, LL = hmm.find_events(test[repeat], scramble=True)
                scrambled.append(LL)

            LLs[ev_i, repeat, 1] = scrambled
            
    areas_LLs[area_name] = LLs

    n = n+1
    tf = time.time()
    print(n)
    print((tf-ti)/60)


# Save data to .npy file
np.save(f'./intact_maps/LL/scrambled_ll_0.npy', areas_LLs)
