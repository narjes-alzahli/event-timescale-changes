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
    run_no = sys.argv[1]
else:
    print("No argument provided.")

"""
####### SFix
h5_files_SFix = np.load(f"SFix_run_{run_no}.npy", allow_pickle=True)
all_files_SFix = [file.split('.')[0] for file in h5_files_SFix]
done_files_SFix = [file.split('.')[0] for file in os.listdir('./scrambled_maps/LL/SFix_timescaleChanges_data') if file.endswith('.npy')]
left_files_SFix = set(all_files_SFix) - set(done_files_SFix)
files_SFix = [f'{file}.h5' for file in left_files_SFix]

print("SFix:", len(h5_files_SFix), "-", len(done_files_SFix) ,"=", len(left_files_SFix))
"""

####### SRnd
h5_files_SRnd = np.load(f"SRand_run_{run_no}.npy", allow_pickle=True)
all_files_SRnd = [file.split('.')[0] for file in h5_files_SRnd]
done_files_SRnd = [file.split('.')[0] for file in os.listdir('./SRnd/timescaleChanges/sl_data') if file.endswith('.npy')]
left_files_SRnd = set(all_files_SRnd) - set(done_files_SRnd)
files_SRnd = [f'{file}.h5' for file in left_files_SRnd]

print("SRnd:", len(h5_files_SRnd), "-", len(done_files_SRnd) ,"=", len(left_files_SRnd))

# initialize some variables
test_size = 0.5
nEvents = np.arange(2, 11)
PERMS = 20
NULLS = 50
n = 0


for i in range(len(files_SRnd)):

    ti = time.time()
    #print(f"SFix areas: {files_SFix[i].split('.')[0]}")
    print(f"SRand areas: {files_SRnd[i].split('.')[0]}")

    """
    #### SFix

    # Construct the full file path
    file_path_SFix = os.path.join(data_directory_path, files_SFix[i])
    area_SFix = files_SFix[i].split('.')[0]
    # Load SL data from the .h5 file
    D_orig_SFix = dd.io.load(file_path_SFix)
    N_vox_SFix = D_orig_SFix[list(D_orig_SFix.keys())[0]]['SFix'].shape[2]
    """

    #### SRand

    # Construct the full file path
    file_path_SRnd = os.path.join(data_directory_path, files_SRnd[i])
    area_SRnd = files_SRnd[i].split('.')[0]
    # Load SL data from the .h5 file
    D_orig_SRnd = dd.io.load(file_path_SRnd)
    N_vox_SRnd = D_orig_SRnd[list(D_orig_SRnd.keys())[0]]['SRnd'].shape[2]

    ######## Initialize data structures containing all LLs
    #timescales_SFix = np.empty((6), dtype=object)
    timescales_SRnd = np.empty((6), dtype=object)

    # Other info
    N_subj = len(D_orig_SRnd.keys())
    half_subj = int(N_subj/2)

    # Train-test split for version 1
    train_indices_v1, test_indices_v1 = train_test_split(np.arange(half_subj), test_size=test_size, random_state=42)

    # Train-test split for version 2
    train_indices_v2, test_indices_v2 = train_test_split(np.arange(half_subj), test_size=test_size, random_state=42)
    
    """
    ##### SFix

    # Create SFix Data
    D_SFix = np.zeros((30, 6, 60, N_vox_SFix))
    for i, s in enumerate(D_orig_SFix.keys()):
        D_SFix[i] = D_orig_SFix[s]['SFix']
    # Separate data for the two versions of the scrambled movie
    D_v1_SFix = D_SFix[:half_subj]  # Data for the first half of the subjects
    D_v2_SFix = D_SFix[half_subj:]  # Data for the second half of the subjects
    """

    ##### SRnd

    # Create SRnd Data
    D_SRnd = np.zeros((30, 6, 60, N_vox_SRnd))
    for i, s in enumerate(D_orig_SRnd.keys()):
        D_SRnd[i] = D_orig_SRnd[s]['SRnd']
    # Separate data for the two versions of the scrambled movie
    D_v1_SRnd = D_SRnd[:half_subj]  # Data for the first half of the subjects
    D_v2_SRnd = D_SRnd[half_subj:]  # Data for the second half of the subjects


    #try:
    ################ Enter permutations loop
    
    for perm_i in range(PERMS + 1):

        ######################## case with real movie-repeat order #######################
        if perm_i == 0:

            ################ Train-test data
            """
            ##### SFix
            # v1
            train_v1_SFix = D_v1_SFix[train_indices_v1].mean(0)
            test_v1_SFix = D_v1_SFix[test_indices_v1].mean(0)
            # v2
            train_v2_SFix = D_v2_SFix[train_indices_v1].mean(0)
            test_v2_SFix = D_v2_SFix[test_indices_v1].mean(0)
            """

            ##### SRnd
            # v1
            train_v1_SRnd = D_v1_SRnd[train_indices_v1].mean(0)
            test_v1_SRnd = D_v1_SRnd[test_indices_v1].mean(0)
            # v2
            train_v2_SRnd = D_v2_SRnd[train_indices_v1].mean(0)
            test_v2_SRnd = D_v2_SRnd[test_indices_v1].mean(0)

        ####################### case with permuted movie-repeat order #######################
        else:

            ################ permute movie-repeat order per subject
            """permuted_D_v1_SFix = np.copy(D_v1_SFix) ##### SFix, v1
            permuted_D_v2_SFix = np.copy(D_v2_SFix) ##### SFix, v2 """
            permuted_D_v1_SRnd = np.copy(D_v1_SRnd) ##### SRnd, v1
            permuted_D_v2_SRnd = np.copy(D_v2_SRnd) ##### SRnd, v2
            for subj in range(half_subj):
                """np.random.shuffle(permuted_D_v1_SFix[subj])
                np.random.shuffle(permuted_D_v2_SFix[subj])"""
                np.random.shuffle(permuted_D_v1_SRnd[subj])
                np.random.shuffle(permuted_D_v2_SRnd[subj])

            ################ Train-test data for version 1
            """##### SFix
            # v1
            train_v1_SFix = permuted_D_v1_SFix[train_indices_v1].mean(0)
            test_v1_SFix = permuted_D_v1_SFix[test_indices_v1].mean(0)
            # v2
            train_v2_SFix = permuted_D_v2_SFix[train_indices_v1].mean(0)
            test_v2_SFix = permuted_D_v2_SFix[test_indices_v1].mean(0)"""
            ##### SRnd
            # v1
            train_v1_SRnd = permuted_D_v1_SRnd[train_indices_v1].mean(0)
            test_v1_SRnd = permuted_D_v1_SRnd[test_indices_v1].mean(0)
            # v2
            train_v2_SRnd = permuted_D_v2_SRnd[train_indices_v1].mean(0)
            test_v2_SRnd = permuted_D_v2_SRnd[test_indices_v1].mean(0)

        ##################### Enter Our Fave Loop

        for repeat in range(6):

            """
            ##### SFix
            timescales_SFix[repeat] = {
                'true' : {'v1': None,'v2': None}, 
                'nulls' : {'v1': None,'v2': None}}
            ##### SRnd
            timescales_SRnd[repeat] = {
                'true' : {'v1': None,'v2': None}, 
                'nulls' : {'v1': None,'v2': None}}
            """

            events_lls_v1_SRnd = [] # store LLs for each event_segment for this repeat
            events_lls_v2_SRnd = [] 

            for ev_i, n_ev in enumerate(nEvents):

                """
                ##### SFix

                # HMM model for version 1
                hmm_v1_SFix = EventSegment(n_ev, split_merge=True)
                hmm_v1_SFix.fit(train_v1_SFix[repeat])
                _, true_ll_v1_SFix = hmm_v1_SFix.find_events(test_v1_SFix[repeat])
                orig_ev_pat_v1_SFix = hmm_v1_SFix.event_pat_.copy()

                # HMM model for version 2
                hmm_v2_SFix = EventSegment(n_ev, split_merge=True)
                hmm_v2_SFix.fit(train_v2_SFix[repeat])
                _, true_ll_v2_SFix = hmm_v2_SFix.find_events(test_v2_SFix[repeat])
                orig_ev_pat_v2_SFix = hmm_v2_SFix.event_pat_.copy()

                null_lls_v1_SFix = []
                null_lls_v2_SFix = []
                events_lls_v1_SFix = [] # store LLs for each event_segment for this repeat
                events_lls_v2_SFix = [] 
                """

                ##### SRnd

                # HMM model for version 1
                hmm_v1_SRnd = EventSegment(n_ev, split_merge=True)
                hmm_v1_SRnd.fit(train_v1_SRnd[repeat])
                _, true_ll_v1_SRnd = hmm_v1_SRnd.find_events(test_v1_SRnd[repeat])
                orig_ev_pat_v1_SRnd = hmm_v1_SRnd.event_pat_.copy()

                # HMM model for version 2
                hmm_v2_SRnd = EventSegment(n_ev, split_merge=True)
                hmm_v2_SRnd.fit(train_v2_SRnd[repeat])
                _, true_ll_v2_SRnd = hmm_v2_SRnd.find_events(test_v2_SRnd[repeat])
                orig_ev_pat_v2_SRnd = hmm_v2_SRnd.event_pat_.copy()

                null_lls_v1_SRnd = []
                null_lls_v2_SRnd = []

                ###################### Timescales NULL loop ######################

                for p in range(NULLS):

                    ############# Permute event order

                    rand_perm = np.random.permutation(n_ev)
                    while np.all(rand_perm == np.arange(n_ev)):
                        rand_perm = np.random.permutation(n_ev)

                    ############# Timescale null event patterns
                    """
                    ##### SFix 
                    # v1
                    hmm_v1_SFix.set_event_patterns(orig_ev_pat_v1_SFix[:, rand_perm])
                    _, LL_v1_SFix = hmm_v1_SFix.find_events(test_v1_SFix[repeat])
                    null_lls_v1_SFix.append(LL_v1_SFix)
                    # v2
                    hmm_v2_SFix.set_event_patterns(orig_ev_pat_v2_SFix[:, rand_perm])
                    _, LL_v2_SFix = hmm_v2_SFix.find_events(test_v2_SFix[repeat])
                    null_lls_v2_SFix.append(LL_v2_SFix)
                    """

                    ##### SRnd 
                    # v1
                    hmm_v1_SRnd.set_event_patterns(orig_ev_pat_v1_SRnd[:, rand_perm])
                    _, LL_v1_SRnd = hmm_v1_SRnd.find_events(test_v1_SRnd[repeat])
                    null_lls_v1_SRnd.append(LL_v1_SRnd)
                    # v2
                    hmm_v2_SRnd.set_event_patterns(orig_ev_pat_v2_SRnd[:, rand_perm])
                    _, LL_v2_SRnd = hmm_v2_SRnd.find_events(test_v2_SRnd[repeat])
                    null_lls_v2_SRnd.append(LL_v2_SRnd)

                ############# Adjust true_ll by mean(null_lls)
                
                """
                ##### SFix 
                ## v1
                adjusted_true_ll_v1_SFix = true_ll_v1_SFix - np.mean(null_lls_v1_SFix)
                events_lls_v1_SFix.append(adjusted_true_ll_v1_SFix)
                ## v2
                adjusted_true_ll_v2_SFix = true_ll_v2_SFix - np.mean(null_lls_v2_SFix)
                events_lls_v2_SFix.append(adjusted_true_ll_v2_SFix)
                """

                ##### SRnd
                ## v1
                adjusted_true_ll_v1_SRnd = true_ll_v1_SRnd - np.mean(null_lls_v1_SRnd)
                events_lls_v1_SRnd.append(adjusted_true_ll_v1_SRnd)
                ## v2
                adjusted_true_ll_v2_SRnd = true_ll_v2_SRnd - np.mean(null_lls_v2_SRnd)
                events_lls_v2_SRnd.append(adjusted_true_ll_v2_SRnd)
            
            ###################### compute timescale for this repeat ######################

            ############### Exponentiate normalized LL values
            """
            ##### SFix 
            exp_lls_v1_SFix = np.exp(events_lls_v1_SFix) ## v1
            exp_lls_v2_SFix = np.exp(events_lls_v2_SFix) ## v2
            """

            ##### SRnd
            exp_lls_v1_SRnd = np.exp(events_lls_v1_SRnd) ## v1
            exp_lls_v2_SRnd = np.exp(events_lls_v2_SRnd) ## v2

            ############### Calculate weighted average for this repeat
            """
            ##### SFix 
            repeat_timescale_v1_SFix = np.sum(exp_lls_v1_SFix * np.arange(2,11)) / np.sum(exp_lls_v1_SFix) ## v1
            repeat_timescale_v2_SFix = np.sum(exp_lls_v2_SFix * np.arange(2,11)) / np.sum(exp_lls_v2_SFix) ## v2
            """

            ##### SRnd
            repeat_timescale_v1_SRnd = np.sum(exp_lls_v1_SRnd * np.arange(2,11)) / np.sum(exp_lls_v1_SRnd) ## v1
            repeat_timescale_v2_SRnd = np.sum(exp_lls_v2_SRnd * np.arange(2,11)) / np.sum(exp_lls_v2_SRnd) ## v2

            ############### Put everything in respective dictionary

            # containing true values with correct movie repeat order
            if perm_i == 0: 
                """
                ##### SFix 
                timescales_SFix[repeat] = {
                    'true' : {'v1': None,'v2': None}, 
                    'nulls' : {'v1': [],'v2': []}}
                timescales_SFix[repeat]['true']['v1'] = repeat_timescale_v1_SFix
                timescales_SFix[repeat]['true']['v2'] = repeat_timescale_v2_SFix
                """
                ##### SRnd
                timescales_SRnd[repeat] = {
                    'true' : {'v1': None,'v2': None}, 
                    'nulls' : {'v1': [],'v2': []}}
                timescales_SRnd[repeat]['true']['v1'] = repeat_timescale_v1_SRnd
                timescales_SRnd[repeat]['true']['v2'] = repeat_timescale_v2_SRnd

            # containing null values with permuted movie repeat order
            else:
                """
                ##### SFix
                timescales_SFix[repeat]['nulls']['v1'].append(repeat_timescale_v1_SFix)
                timescales_SFix[repeat]['nulls']['v2'].append(repeat_timescale_v2_SFix)
                """
                ##### SRnd
                timescales_SRnd[repeat]['nulls']['v1'].append(repeat_timescale_v1_SRnd)
                timescales_SRnd[repeat]['nulls']['v2'].append(repeat_timescale_v2_SRnd)

    #np.save(f'./scrambled_maps/LL/SFix_timescaleChanges_data/{area_SFix}.npy', timescales_SFix)
    np.save(f'./SRnd/timescaleChanges/sl_data/{area_SRnd}.npy', timescales_SRnd)

    n = n + 1
    tf = time.time()

    print(f"Area_n: {n}, Time: {(tf - ti) / 60}")

    #except Exception as e:
        #continue