import os
import sys
import time
import numpy as np
import deepdish as dd
from brainiak.eventseg.event import EventSegment
from sklearn.model_selection import train_test_split

# Directory containing .h5 files
directory_path = '../data/SL/'

# Get all .h5 files in the directory
if len(sys.argv) > 1:
    argument = sys.argv[1]
else:
    print("No argument provided.")

h5_files = np.load(f"{argument}.npy", allow_pickle=True)

h5_files = [file.split('.')[0] for file in h5_files]

done_path = './scrambled_maps/LL/timescalesRand_data'
done_files = [file.split('.')[0] for file in os.listdir(done_path) if file.endswith('.npy')]

left_files = set(h5_files) - set(done_files)

files  = [f'{file}.h5' for file in left_files]

test_size = 0.5
nEvents = np.arange(2, 11)

n = 0

print(len(h5_files))
print(len(done_files))
print(len(left_files))

for file in files:

    # Construct the full file path
    ti = time.time()
    file_path = os.path.join(directory_path, file)
    area_name = file.split('.')[0]

    # Load the data from the .h5 file (SRnd version)
    D_orig = dd.io.load(file_path)
    vox_N = D_orig[list(D_orig.keys())[0]]['SRnd'].shape[2]
    D = np.zeros((30, 6, 60, vox_N))
    for i, s in enumerate(D_orig.keys()):
        D[i] = D_orig[s]['SRnd']

    N_subj = D.shape[0]

    LLs = np.empty((len(nEvents), 6), dtype=object)

    num_timepoints = D.mean(0).shape[1]

    # Separate data for the two versions of the scrambled movie
    half_subj = int(N_subj/2)
    D_version1 = D[:half_subj]  # Data for the first half of the subjects
    D_version2 = D[half_subj:]  # Data for the second half of the subjects
    
    # Train-test split for version 1
    train_indices_v1, test_indices_v1 = train_test_split(np.arange(15),
                                                         test_size=test_size,
                                                         random_state=42)
    train_v1 = D_version1[train_indices_v1].mean(0)
    test_v1 = D_version1[test_indices_v1].mean(0)

    # Train-test split for version 2
    train_indices_v2, test_indices_v2 = train_test_split(np.arange(15),
                                                         test_size=test_size,
                                                         random_state=42)
    train_v2 = D_version2[train_indices_v2].mean(0)
    test_v2 = D_version2[test_indices_v2].mean(0)

    for ev_i, n_ev in enumerate(nEvents):

        for repeat in range(6):

            LLs[ev_i, repeat] = {
                'true' : {
                    'v1': None,
                    'v2': None
                }, 
                'nulls' : {
                    'v1': None,
                    'v2': None
                }
            }

            # HMM model for version 1
            hmm_v1 = EventSegment(n_ev, split_merge=True)
            hmm_v1.fit(train_v1[repeat])
            _, LL_v1 = hmm_v1.find_events(test_v1[repeat])
            LLs[ev_i, repeat]['true']['v1'] = LL_v1
            orig_ev_pat_v1 = hmm_v1.event_pat_.copy()

            # HMM model for version 2
            hmm_v2 = EventSegment(n_ev, split_merge=True)
            hmm_v2.fit(train_v2[repeat])
            _, LL_v2 = hmm_v2.find_events(test_v2[repeat])
            LLs[ev_i, repeat]['true']['v2'] = LL_v2
            orig_ev_pat_v2 = hmm_v2.event_pat_.copy()

            null_lls_v1 = []
            null_lls_v2 = []
            for p in range(50):
                rand_perm = np.random.permutation(n_ev)
                while np.all(rand_perm == np.arange(n_ev)):
                    rand_perm = np.random.permutation(n_ev)
                
                # Null event pattern for version 1
                hmm_v1.set_event_patterns(orig_ev_pat_v1[:, rand_perm])
                _, LL_v1 = hmm_v1.find_events(test_v1[repeat])
                null_lls_v1.append(LL_v1)

                # Null event pattern for version 2
                hmm_v2.set_event_patterns(orig_ev_pat_v2[:, rand_perm])
                _, LL_v2 = hmm_v2.find_events(test_v2[repeat])
                null_lls_v2.append(LL_v2)

            LLs[ev_i, repeat]['nulls']['v1'] = null_lls_v1
            LLs[ev_i, repeat]['nulls']['v2'] = null_lls_v2

    np.save(f'./scrambled_maps/LL/SRand_timescales_data/{area_name}.npy', LLs)

    n = n + 1
    tf = time.time()

    print(f"Areas: {file}, Area_n: {n}, Time: {(tf - ti) / 60}")
