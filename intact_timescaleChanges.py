import os
import sys
import time
import numpy as np
import deepdish as dd
from brainiak.eventseg.event import EventSegment
from sklearn.model_selection import train_test_split



# Get all .h5 files in the directory
if len(sys.argv) > 1:
    argument = sys.argv[1]
else:
    print("No argument provided.")

# Load SL names
h5_files = np.load(f"{argument}.npy", allow_pickle=True)

# which SLs
#existing_files = [file.split('.')[0] for file in os.listdir('./intact_maps/LL/misorder_view_lls/') if file.endswith('.npy')]
#left_files = list(set(h5_files) - set(existing_files))

# Directory containing .h5 files
directory_path = '../data/SL/'

# initialize some variables
test_size = 0.5
nEvents = np.arange(2,11)
PERMS = 20
NULLS = 50

# count how many SLs
n=0

print(len(h5_files))

for file in h5_files:

    # Construct the full file path
    ti = time.time()
    file_path = os.path.join(directory_path, f"{file}.h5")
    area_name = file.split('.')[0]
    
    # Load the data from the .h5 file
    D_orig = dd.io.load(file_path)

    # Get number of subj and vox
    N_subj = len(D_orig.keys())
    first_subject = list(D_orig.keys())[0]
    N_vox = D_orig[first_subject]['Intact'].shape[2]

    # Get Intact viewings of all subjects
    D = np.zeros((N_subj, 6, 60, N_vox))
    for i, s in enumerate(D_orig.keys()):
        D[i] = D_orig[s]['Intact']

    # Initialize data structures containing all LLs
    timescales = np.empty((6), dtype=object)
    inter_results = {}

    # split into train and test
    train_indices, test_indices = train_test_split(np.arange(N_subj),
                                                test_size=test_size, 
                                                random_state=42)
    
    for perm_i in range(PERMS + 1):

        scramble_event_order_nulls = np.empty((len(nEvents),6, 2), dtype=object)

        ######## case with real movie-repeat order #######
        if perm_i == 0:
            # average data amongst participants
            train = D[train_indices].mean(0)
            test  = D[test_indices].mean(0)

        ####### case with permuted movie-repeat order #######
        else:
            # permute movie-repeat order per subject
            permuted_D = np.copy(D)
            for subj in range(N_subj):
                np.random.shuffle(permuted_D[subj])
            # average data amongst participants
            train = permuted_D[train_indices].mean(0)
            test  = permuted_D[test_indices].mean(0)

        # Get LLs for each # of Events for each Repeat
        for repeat in range(6):

            events_lls = [] # store LLs for each event_segment for this repeat
            for ev_i, n_ev in enumerate(nEvents):

                # Fit HMM model
                hmm = EventSegment(n_ev, split_merge=True)                
                hmm.fit(train[repeat])
                # Get true LL value
                _, true_ll = hmm.find_events(test[repeat])

                # Get null LLs from Scrambled Event Order
                orig_ev_pat = hmm.event_pat_.copy()
                null_lls = []
                for p in range(NULLS):
                    rand_perm = np.random.permutation(n_ev)
                    while np.all(rand_perm == np.arange(n_ev)):
                        rand_perm = np.random.permutation(n_ev)

                    hmm.set_event_patterns(orig_ev_pat[:, rand_perm])
                    _, null_ll = hmm.find_events(test[repeat])
                    null_lls.append(null_ll)

                scramble_event_order_nulls[ev_i, repeat, 0] = true_ll
                scramble_event_order_nulls[ev_i, repeat, 1] = null_lls

                # Adjust true_ll by mean(null_lls)
                adjusted_true_ll = true_ll - np.mean(null_lls)
                events_lls.append(adjusted_true_ll)

            ####### compute timescale for this repeat #######

            # Exponentiate normalized LL values
            exp_lls = np.exp(events_lls)

            # Calculate weighted average for this repeat
            repeat_timescale = np.sum(exp_lls * np.arange(2,11)) / np.sum(exp_lls)

            #print(repeat, perm_i, repeat_timescale)

            # Put everything in respective dictionary
            if perm_i == 0:
                timescales[repeat] = {}
                timescales[repeat]['true'] = repeat_timescale
                timescales[repeat]['perms'] = []
                #timescales[repeat] = {"true": None, "perms": []}
            else:
                timescales[repeat]['perms'].append(repeat_timescale)

            #print(timescales)

        inter_results[perm_i] = scramble_event_order_nulls
    
    # save timescale file
    np.save(f'./intact_maps/LL/scrambleViews_timescales_nulls/{area_name}_{perm_i}.npy', scramble_event_order_nulls)
    np.save(f'./intact_maps/LL/scrambleViews_timescales/{area_name}.npy', timescales)

    n = n+1
    tf = time.time()

    print(f"Areas: {file}, Time: {(tf - ti) / 60}")
