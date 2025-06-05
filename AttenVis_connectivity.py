import mne
import os
import numpy as np
import pandas as pd
import mne_connectivity

import helper_functions as tlbx
import AttenVis_config as cfg

participants_df, participants_to_study = tlbx.load_participants()

report = tlbx.generate_report()

print("Saving as " + cfg.data_savename)

for sub_id in participants_to_study:

    participant_data = []
    visit_dir        = participants_df[participants_df['Participant'] == sub_id]['Visit_Dir'].values[0]
    participant_data_savename = os.path.join(visit_dir,cfg.data_fname.replace('.pkl','_' + sub_id + '.pkl'))

    if not os.path.exists(participant_data_savename) or cfg.overwrite_data:

        diagnosis   = participants_df[participants_df['Participant'] == sub_id]['Diagnosis'].values[0]
        study       = participants_df[participants_df['Participant'] == sub_id]['Study'].values[0]
        subjID_date = participants_df[participants_df['Participant'] == sub_id]['SubjID_Date'].values[0]

        load_fname, epochs      = tlbx.load_epochs('_nobaseline_nofilter_all_conditions_metadata_epo.fif',visit_dir,resample=True)
        cleaned_epochs, summary = tlbx.clean_epochs_by_behaviour(epochs, rt_based=(0.1,1.2), percent=None, correct_answers_only=True)
        inverse_operator        = tlbx.load_inverse_operator('_prestim_baseline_inv.fif',visit_dir)
        src = inverse_operator["src"]
    
        #load seed label
        seed_label_hemis = []
        target_label_hemis = []
        for hemi in cfg.hemisphere:

            # load seed and target labels for FC analyses
            seed_label   = tlbx.load_drawn_labels(cfg.seed_label,hemi,subjID_date,visit_dir,grown=True)
            seed_label_hemis.append(seed_label)
            target_label = tlbx.load_annot_labels(cfg.target_label,subjID_date,'aparc.a2009s',hemi,cfg.subj_dir)
            target_label_hemis.append(target_label)
        
        target_label = target_label_hemis[0] + target_label_hemis[1]

        #load condition specific epochs and extract time series
        for condition in cfg.condition:
            for difficulty in cfg.difficulty:

                condition_tag = f'(Condition == "{condition}") & (difficulty == "{difficulty}")'
                out_fname     = load_fname.replace('all_conditions_metadata_epo.fif','_'.join(['metadata',condition,difficulty,'clean','epo.fif']))
                if not os.path.exists(out_fname) or cfg.overwrite_epochs:
                    epochs_clean = tlbx.get_condition_epochs(epochs,condition_tag)
                    epochs_clean.save(out_fname) #always saves epochs with bad epochs removed but without any filtering or baseline correction
                else:
                    epochs_clean = mne.read_epochs(out_fname)
                stcs = mne.minimum_norm.apply_inverse_epochs(epochs_clean, inverse_operator, cfg.lambda2, cfg.con_method, pick_ori="normal", verbose=False)

                #extract time series
                stcs_label_lh = mne.extract_label_time_course(stcs, seed_label_hemis[0], src, mode='mean', verbose=False)
                stcs_label_rh = mne.extract_label_time_course(stcs, seed_label_hemis[1], src, mode='mean', verbose=False)
                
                verts2include = [i.in_label(target_label) for i in stcs]

                #combine time series for each hemisphere
                comb_ts_lh = list(zip(stcs_label_lh, verts2include))
                comb_ts_rh = list(zip(stcs_label_rh, verts2include))

                # this is some stuff we need to specify how FC is computed by the mne-connectivity toolbox 
                vertices      = [verts2include[0].vertices[i] for i in range(2)]
                n_signals_tot = 1 + len(vertices[0]) + len(vertices[1])
                indices       = mne_connectivity.seed_target_indices([0], np.arange(1, n_signals_tot)) 
                sfreq = epochs.info['sfreq']  # the sampling frequency
                cwt_freqs = np.arange(cfg.freq_min, cfg.freq_max+1, 1)
                cwt_n_cycles = cfg.con_n_cycles

                # FC: left auditory seed to ROI
                print('\n>>> Computing FC for lh\n')
                con_lh = mne_connectivity.spectral_connectivity_epochs(
                    comb_ts_lh, method=cfg.fc_method, mode=cfg.fc_mode, indices=indices, sfreq=sfreq,
                    cwt_freqs=cwt_freqs, cwt_n_cycles=cwt_n_cycles, tmin=cfg.time_windows[0], tmax=cfg.time_windows[1])   
                
                # FC: right auditory seed to ROI
                print('\n>>> Computing FC for rh\n')
                con_rh = mne_connectivity.spectral_connectivity_epochs(
                    comb_ts_rh, method=cfg.fc_method, mode=cfg.fc_mode, indices=indices, sfreq=sfreq,
                    cwt_freqs=cwt_freqs, cwt_n_cycles=cwt_n_cycles, tmin=cfg.time_windows[0], tmax=cfg.time_windows[1])

                participant_data_lh = [sub_id,subjID_date,diagnosis,condition,difficulty,'lh',con_lh.get_data(),verts2include[0].vertices,con_lh.times,len(epochs_clean)]
                participant_data_rh = [sub_id,subjID_date,diagnosis,condition,difficulty,'rh',con_rh.get_data(),verts2include[0].vertices,con_rh.times,len(epochs_clean)]
                participant_data.append(participant_data_lh)
                participant_data.append(participant_data_rh)

        df= pd.DataFrame(participant_data, columns = ['Participant','SubjID_Date','Diagnosis','Condition','Difficulty','hemisphere','connectivity_data','vertices','time','n_epochs'])
        df.to_pickle(participant_data_savename)
        