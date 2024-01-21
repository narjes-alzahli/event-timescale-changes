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

eLL = {}
eWBC = {}

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

    LLs = np.empty((len(nEvents),6), dtype=object)
    corr_diffs = np.empty((len(nEvents),6), dtype=object)
    timescales = {}

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

            #################################################################### LL begin
            # meeting scramble = True, should fail, when the model doesn't work, 50 times or nulls
            _, LL = hmm.find_events(test[repeat])
            LLs[ev_i, repeat] = LL

            for i in range(50):
                _, LL = hmm.find_events(test[repeat], scramble=True)
                

            ################################################################### WBC begin
            events = hmm.segments_[0]
            most_likely_event = np.argmax(events, axis=1)

            within_event_corrs = []
            between_event_corrs = []

            for i in range(num_timepoints - 5):

                j = i+5  # Consider timepoints ~5 TRs apart
                corr, _ = pearsonr(test[repeat, i], test[repeat, j])
                if most_likely_event[i] == most_likely_event[j]:
                    within_event_corrs.append(corr)
                else:
                    between_event_corrs.append(corr)

            # Compute average correlations
            avg_within_event_corr = np.mean(within_event_corrs)
            avg_between_event_corr = np.mean(between_event_corrs)

            # Calculate the difference
            corr_diff = avg_within_event_corr - avg_between_event_corr
            corr_diffs[ev_i, repeat] = corr_diff


    ############################################################################ LL begin again

    # Normalize LL values to avoid underflow when exponentiating
    max_negative_ll = np.max(LLs)
    normalized_lls = LLs + np.abs(max_negative_ll)

    # Exponentiate the normalized LL values
    exp_lls = np.empty((len(nEvents), 6), dtype=object)
    for i in range(len(nEvents)):
        for j in range (6):
            exp_lls[i,j] = np.exp(normalized_lls[i,j])

    # Initialize an array to store the weighted averages for each run
    ll_weighted_avg_per_run = np.empty(6)

    # Calculate the weighted average for each run
    for run in range(6):
        weights = exp_lls[:, run]
        ll_weighted_avg_per_run[run] = np.sum(weights * np.arange(2,11)) / np.sum(weights)

    eLL[area_name] = ll_weighted_avg_per_run


    ############################################################################ WBC begin again

    # Exponentiate the normalized WBC values
    exp_wbc = np.empty((len(nEvents), 6), dtype=object)
    for i in range(len(nEvents)):
        for j in range (6):
            exp_wbc[i,j] = np.exp(corr_diffs[i,j])

    # Initialize an array to store the weighted averages for each run
    wbc_weighted_avg_per_run = np.empty(corr_diffs.shape[1])

    # Calculate the weighted average for each run
    for run in range(corr_diffs.shape[1]):
        weights = exp_wbc[:, run]
        wbc_weighted_avg_per_run[run] = np.sum(weights * np.arange(2,11)) / np.sum(weights)

    eWBC[area_name] = wbc_weighted_avg_per_run

    ############################################################################
    
    timescales['eLL'] = ll_weighted_avg_per_run
    timescales['eWBC'] = wbc_weighted_avg_per_run

    np.save(f'{area_name}.npy', timescales)

    n = n+1
    tf = time.time()
    print(n)
    print((tf-ti)/60)


# Save data to .npy file
np.save('./intact_maps/LL/LL_scrambled.npy', eLL)
np.save('./intact_maps/WBC/WBC_scrambled.npy', eWBC)