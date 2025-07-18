import mne
import os
import numpy as np
import pandas as pd
import gc
import mne_connectivity

import helper_functions as tlbx
import AttenVis_connectivity_config as cfg
import matplotlib.pyplot as plt   
import matplotlib as mpl 
from mne.parallel import parallel_func

participants_df, participants_to_study = tlbx.load_participants()

print("Saving as " + cfg.data_savename)

def get_connectivity(sub_id,overwrite_data=True):
    """    Compute functional connectivity for a given subject."""
    participant_data = []
    diagnosis, study,visit_dir,subjID_date = tlbx.read_participant_details_from_dataframe(participants_df,sub_id)
    participant_data_savename = os.path.join(visit_dir,cfg.data_fname.replace('.pkl','_' + sub_id + '.pkl'))

    if not os.path.exists(participant_data_savename) or overwrite_data:
        inverse_operator        = tlbx.load_inverse_operator('_prestim_baseline_inv.fif',visit_dir)
        src = inverse_operator["src"]
    
        #load seed label
        seed_label_hemis = []
        target_label_hemis = []

        for hemi in cfg.hemisphere:
            # load seed and target labels for FC analyses
            seed_label   = tlbx.load_drawn_labels(cfg.seed_label,hemi,subjID_date,visit_dir,grown=True)
            seed_label_hemis.append(seed_label)
            target_label = tlbx.load_drawn_labels(cfg.target_label,hemi,subjID_date,visit_dir,grown=True)
            target_label_hemis.append(target_label)
        
        #load condition specific epochs and extract time series
        for condition in cfg.condition:
            file_tag_cond  = '_AttenVis_nobaseline_nofilter_metadata_' + condition
            epo_load_fname = tlbx.find_files(file_tag_cond + '_behaviour_cleaned_epo.fif', visit_dir)[0]
            epochs         = mne.read_epochs(epo_load_fname)
            epochs.crop(tmin=-0.5, tmax=1.5)
            n_epochs       = len(epochs)
            sfreq          = epochs.info['sfreq']  # the sampling frequency
            stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inverse_operator, cfg.lambda2, cfg.con_method, pick_ori="normal", verbose=False)
            vertices = stcs[0].vertices
            del epochs  # free memory
            gc.collect()  # collect garbage to free memory
            # this is some stuff we need to specify how FC is computed by the mne-connectivity toolbox
            indices        = mne_connectivity.seed_target_indices([0], [1])
            cwt_freqs      = np.arange(cfg.freq_min, cfg.freq_max+1, 1)
            cwt_n_cycles   = 3 #cwt_freqs / 2.0  # number of cycles for the CWT
            seed_stcs = []
            target_stcs = []
            for hemi_idx, hemi in enumerate(cfg.hemisphere):
                seed_stc = mne.extract_label_time_course(stcs, seed_label_hemis[hemi_idx], src, mode='mean', verbose=False)
                seed_stcs.append(seed_stc)
                target_stc = mne.extract_label_time_course(stcs, target_label_hemis[hemi_idx], src, mode='mean', verbose=False)
                target_stcs.append(target_stc)
            del stcs  # free memory
            gc.collect()  # collect garbage to free memory
            for hemi_idx, hemi in enumerate(cfg.hemisphere):
                for target_hemi_idx, target_hemi in enumerate(cfg.hemisphere):
                    comb_ts = list(zip(seed_stcs[hemi_idx], target_stcs[target_hemi_idx]))  # combine time series for each hemisphere
                    print(f'\n>>> Computing FC for {hemi.upper()} and {target_hemi.upper()}\n')
                    con = mne_connectivity.spectral_connectivity_epochs(  # get functional connectivity
                        comb_ts, method=cfg.fc_method, mode=cfg.fc_mode, indices=indices, sfreq=sfreq,
                        cwt_freqs=cwt_freqs, cwt_n_cycles=cwt_n_cycles, tmin=cfg.time_windows[0], tmax=cfg.time_windows[1])

                    participant_data_hemi = [sub_id,subjID_date,diagnosis,condition,hemi,target_hemi,np.squeeze(con.get_data()),vertices,con.times,n_epochs]
                    participant_data.append(participant_data_hemi)

        df= pd.DataFrame(participant_data, columns = ['Participant','SubjID_Date','Diagnosis','Condition','hemisphere','target_hemi','connectivity_data','vertices','time','n_epochs'])
        df.to_pickle(participant_data_savename)

    return sub_id

n_jobs = 2

debug = False
if debug:
    n_jobs = 1

parallel, run_func, _ = parallel_func(get_connectivity, n_jobs=n_jobs)
results = parallel(run_func(subject) for subject in participants_to_study)

