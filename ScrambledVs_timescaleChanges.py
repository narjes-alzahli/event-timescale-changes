import os
import sys
import time
import numpy as np
import deepdish as dd
from brainiak.eventseg.event import EventSegment
from sklearn.model_selection import train_test_split

# Directory containing .h5 files
data_directory_path = '../data/SL/'

# Get all .h5 files in the directory
if len(sys.argv) > 1:
    version = sys.argv[1]
    v = sys.argv[2]
else:
    print("No argument provided.")

passing_SLs = np.load(f'./{version}_{v}/timescales/sig_timescales_SLs.npy', allow_pickle=True).item()
all_files = list(passing_SLs.keys())
done_files = [file.split('.')[0] for file in os.listdir(f'./{version}_{v}/timescaleChanges/sl_data') if file.endswith('.npy')]
left_files = set(all_files) - set(done_files)
files = [f'{file}.h5' for file in left_files]

print("SLs:", len(all_files), "-", len(done_files) ,"=", len(left_files))

# initialize some variables
test_size = 0.5
nEvents = np.arange(2, 11)
PERMS = 20
NULLS = 50
n = 0


for i in range(len(files)):

    ti = time.time()
    print(f"{version}_{v} area: {files[i].split('.')[0]}")

    # Construct the full file path
    file_path = os.path.join(data_directory_path, files[i])
    area = files[i].split('.')[0]
    # Load SL data from the .h5 file
    D_orig = dd.io.load(file_path)
    N_vox = D_orig[list(D_orig.keys())[0]][f'{version}'].shape[2]

    ######## Initialize data structures containing all LLs
    timescales = np.empty((6), dtype=object)

    # Other info
    N_subj = len(D_orig.keys())
    half_subj = int(N_subj/2)

    # Train-test split 
    train_indices, test_indices = train_test_split(np.arange(half_subj), test_size=test_size, random_state=42)

    # Create SRnd Data
    D = np.zeros((30, 6, 60, N_vox))
    for i, s in enumerate(D_orig.keys()):
        D[i] = D_orig[s][f'{version}']
    # Separate data for the two versions of the scrambled movie
    if v == "v1":
        D = D[:half_subj]  # Data for the first half of the subjects
    elif v == "v2":
        D = D[half_subj:]  # Data for the second half of the subjects
    else:
        print("wrong version provided")

    #try:
    ################ Enter permutations loop
    
    for perm_i in range(PERMS + 1):

        ######################## case with real movie-repeat order #######################
        if perm_i == 0:

            ################ Train-test data
            train = D[train_indices].mean(0)
            test = D[test_indices].mean(0)

        ####################### case with permuted movie-repeat order #######################
        else:

            ################ permute movie-repeat order per subject
            permuted_D = np.copy(D)
            for subj in range(half_subj):
                np.random.shuffle(permuted_D[subj])

            ################ Train-test data 
                
            train = permuted_D[train_indices].mean(0)
            test = permuted_D[test_indices].mean(0)

        ##################### Enter Our Fave Loop

        for repeat in range(6):

            events_lls = [] # store LLs for each event_segment for this repeat

            for ev_i, n_ev in enumerate(nEvents):

                # HMM model
                hmm = EventSegment(n_ev, split_merge=True)
                hmm.fit(train[repeat])
                _, true_ll = hmm.find_events(test[repeat])
                orig_ev_pat = hmm.event_pat_.copy()

                null_lls = []

                ###################### Timescales NULL loop ######################

                for p in range(NULLS):

                    ############# Permute event order

                    rand_perm = np.random.permutation(n_ev)
                    while np.all(rand_perm == np.arange(n_ev)):
                        rand_perm = np.random.permutation(n_ev)

                    ############# Timescale null event patterns

                    hmm.set_event_patterns(orig_ev_pat[:, rand_perm])
                    _, LL = hmm.find_events(test[repeat])
                    null_lls.append(LL)

                ############# Adjust true_ll by mean(null_lls)

                adjusted_true_ll = true_ll - np.mean(null_lls)
                events_lls.append(adjusted_true_ll)

            ###################### compute timescale for this repeat ######################

            ############### Exponentiate normalized LL values

            exp_lls = np.exp(events_lls)

            ############### Calculate weighted average for this repeat

            repeat_timescale = np.sum(exp_lls * np.arange(2,11)) / np.sum(exp_lls) 

            ############### Put everything in respective dictionary

            # containing true values with correct movie repeat order
            if perm_i == 0: 

                timescales[repeat] = {
                    'true' : None, 
                    'nulls' : []}
                timescales[repeat]['true'] = repeat_timescale

            # containing null values with permuted movie repeat order
            else:

                timescales[repeat]['nulls'].append(repeat_timescale)

    #np.save(f'./scrambled_maps/LL/SFix_timescaleChanges_data/{area_SFix}.npy', timescales_SFix)
    np.save(f'./{version}_{v}/timescaleChanges/sl_data/{area}.npy', timescales)

    n = n + 1
    tf = time.time()

    print(f"Area_n: {n}, Time: {(tf - ti) / 60}")

    #except Exception as e:
        #continue