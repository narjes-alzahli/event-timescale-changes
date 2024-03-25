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

done_path = './scrambled_maps/LL/SFix_timescales_data'
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

    print(f"Areas: {file}")

    # Construct the full file path
    ti = time.time()
    file_path = os.path.join(directory_path, file)
    area_name = file.split('.')[0]

    # Load SL data from the .h5 file
    D_orig = dd.io.load(file_path)
    vox_N = D_orig[list(D_orig.keys())[0]]['SFix'].shape[2]
    num_timepoints = D_orig[list(D_orig.keys())[0]]['SFix'].shape[1]
    N_subj = len(D_orig.keys())
    half_subj = int(N_subj/2)

    # Train-test split for version 1
    train_indices_v1, test_indices_v1 = train_test_split(np.arange(half_subj),
                                                         test_size=test_size,
                                                         random_state=40)

    # Train-test split for version 2
    train_indices_v2, test_indices_v2 = train_test_split(np.arange(half_subj),
                                                         test_size=test_size,
                                                         random_state=42)
    
    ##### SFix

    # Create SFix Data
    D_SFix = np.zeros((30, 6, 60, vox_N))
    for i, s in enumerate(D_orig.keys()):
        D_SFix[i] = D_orig[s]['SFix']
    LLs_SFix = np.empty((len(nEvents), 6), dtype=object)

    # Separate data for the two versions of the scrambled movie
    D_v1_SFix = D_SFix[:half_subj]  # Data for the first half of the subjects
    D_v2_SFix = D_SFix[half_subj:]  # Data for the second half of the subjects

    # Train-test data for version 1
    train_v1_SFix = D_v1_SFix[train_indices_v1].mean(0)
    test_v1_SFix = D_v1_SFix[test_indices_v1].mean(0)

    # Train-test data for version 2
    train_v2_SFix = D_v2_SFix[train_indices_v1].mean(0)
    test_v2_SFix = D_v2_SFix[test_indices_v1].mean(0)

    ##### SRnd
    
    # Create SRnd Data
    D_SRnd = np.zeros((30, 6, 60, vox_N))
    for i, s in enumerate(D_orig.keys()):
        D_SRnd[i] = D_orig[s]['SRnd']
    LLs_SRnd = np.empty((len(nEvents), 6), dtype=object)

    # Separate data for the two versions of the scrambled movie
    D_v1_SRnd = D_SRnd[:half_subj]  # Data for the first half of the subjects
    D_v2_SRnd = D_SRnd[half_subj:]  # Data for the second half of the subjects

    # Train-test data for version 1
    train_v1_SRnd = D_v1_SRnd[train_indices_v1].mean(0)
    test_v1_SRnd = D_v1_SRnd[test_indices_v1].mean(0)

    # Train-test data for version 2
    train_v2_SRnd = D_v2_SRnd[train_indices_v1].mean(0)
    test_v2_SRnd = D_v2_SRnd[test_indices_v1].mean(0)

    ##### Enter Big Loop

    try:

        for ev_i, n_ev in enumerate(nEvents):

            for repeat in range(6):

                ##### SFix
                LLs_SFix[ev_i, repeat] = {
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
                hmm_v1_SFix = EventSegment(n_ev, split_merge=True)
                hmm_v1_SFix.fit(train_v1_SFix[repeat])
                _, LL_v1_SFix = hmm_v1_SFix.find_events(test_v1_SFix[repeat])
                LLs_SFix[ev_i, repeat]['true']['v1'] = LL_v1_SFix
                orig_ev_pat_v1_SFix = hmm_v1_SFix.event_pat_.copy()

                # HMM model for version 2
                hmm_v2_SFix = EventSegment(n_ev, split_merge=True)
                hmm_v2_SFix.fit(train_v2_SFix[repeat])
                _, LL_v2_SFix = hmm_v2_SFix.find_events(test_v2_SFix[repeat])
                LLs_SFix[ev_i, repeat]['true']['v2'] = LL_v2_SFix
                orig_ev_pat_v2_SFix = hmm_v2_SFix.event_pat_.copy()

                null_lls_v1_SFix = []
                null_lls_v2_SFix = []

                ##### SRnd
                LLs_SRnd[ev_i, repeat] = {
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
                hmm_v1_SRnd = EventSegment(n_ev, split_merge=True)
                hmm_v1_SRnd.fit(train_v1_SRnd[repeat])
                _, LL_v1_SRnd = hmm_v1_SRnd.find_events(test_v1_SRnd[repeat])
                LLs_SRnd[ev_i, repeat]['true']['v1'] = LL_v1_SRnd
                orig_ev_pat_v1_SRnd = hmm_v1_SRnd.event_pat_.copy()

                # HMM model for version 2
                hmm_v2_SRnd = EventSegment(n_ev, split_merge=True)
                hmm_v2_SRnd.fit(train_v2_SRnd[repeat])
                _, LL_v2_SRnd = hmm_v2_SRnd.find_events(test_v2_SRnd[repeat])
                LLs_SRnd[ev_i, repeat]['true']['v2'] = LL_v2_SRnd
                orig_ev_pat_v2_SRnd = hmm_v2_SRnd.event_pat_.copy()

                null_lls_v1_SRnd = []
                null_lls_v2_SRnd = []

                for p in range(50):
                    rand_perm = np.random.permutation(n_ev)
                    while np.all(rand_perm == np.arange(n_ev)):
                        rand_perm = np.random.permutation(n_ev)
                    
                    ##### SFix 
                    # Null event pattern for version 1
                    hmm_v1_SFix.set_event_patterns(orig_ev_pat_v1_SFix[:, rand_perm])
                    _, LL_v1_SFix = hmm_v1_SFix.find_events(test_v1_SFix[repeat])
                    null_lls_v1_SFix.append(LL_v1_SFix)

                    # Null event pattern for version 2
                    hmm_v2_SFix.set_event_patterns(orig_ev_pat_v2_SFix[:, rand_perm])
                    _, LL_v2_SFix = hmm_v2_SFix.find_events(test_v2_SFix[repeat])
                    null_lls_v2_SFix.append(LL_v2_SFix)

                    ##### SRnd 
                    # Null event pattern for version 1
                    hmm_v1_SRnd.set_event_patterns(orig_ev_pat_v1_SRnd[:, rand_perm])
                    _, LL_v1_SRnd = hmm_v1_SRnd.find_events(test_v1_SRnd[repeat])
                    null_lls_v1_SRnd.append(LL_v1_SRnd)

                    # Null event pattern for version 2
                    hmm_v2_SRnd.set_event_patterns(orig_ev_pat_v2_SRnd[:, rand_perm])
                    _, LL_v2_SRnd = hmm_v2_SRnd.find_events(test_v2_SRnd[repeat])
                    null_lls_v2_SRnd.append(LL_v2_SRnd)

                ##### SFix 
                LLs_SFix[ev_i, repeat]['nulls']['v1'] = null_lls_v1_SFix
                LLs_SFix[ev_i, repeat]['nulls']['v2'] = null_lls_v2_SFix

                ##### SRnd
                LLs_SRnd[ev_i, repeat]['nulls']['v1'] = null_lls_v1_SRnd
                LLs_SRnd[ev_i, repeat]['nulls']['v2'] = null_lls_v2_SRnd

        np.save(f'./scrambled_maps/LL/SFix_timescales_data/{area_name}.npy', LLs_SFix)
        np.save(f'./scrambled_maps/LL/SRand_timescales_data/{area_name}.npy', LLs_SRnd)

    except Exception as e:
        print(f"An error occurred while processing {file}: {e}")
        continue 

    n = n + 1
    tf = time.time()

    print(f"Area_n: {n}, Time: {(tf - ti) / 60}")