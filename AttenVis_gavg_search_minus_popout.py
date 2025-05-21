import mne
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import helper_functions as tlbx
import AttenVis_config as cfg

participants_df, participants_to_study = tlbx.load_participants()
# participants_to_study = ['114001','114501','032901','042203','106501','108201','133101','132901']

src_to = mne.read_source_spaces('/local_mount/space/hypatia/2/users/Jasmine/MNE-sample-data/subjects/fsaverage/bem/fsaverage-ico-5-src.fif')
data_savename = '/local_mount/space/hypatia/2/users/Jasmine/AttenVis/analyses/activations/pop_out_minus_search/subtracted_stcs.pkl'
# morph_files = tlbx.find_files('_morph.h5',cfg.data_dir)
# Subtract pop-out from search for each participant
subj_stcs = {}
subtracted_stcs = []
if not os.path.exists(data_savename): # or cfg.add_participants  or cfg.overwrite_data
    for sub_id in participants_to_study:
        participant_data = []
        diagnosis = participants_df[participants_df['Participant'] == sub_id]['Diagnosis'].values[0]
        study = participants_df[participants_df['Participant'] == sub_id]['Study'].values[0]
        subjID_date = participants_df[participants_df['Participant'] == sub_id]['SubjID_Date'].values[0]
        visit_dir = participants_df[participants_df['Participant'] == sub_id]['Visit_Dir'].values[0]
        #load inverse operator
        inv_path = tlbx.find_files('_prestim_baseline_inv.fif',visit_dir)[0]
        inverse_operator = mne.minimum_norm.read_inverse_operator(inv_path)
        src = inverse_operator["src"]
        morph_fname = inv_path.replace('_prestim_baseline_inv.fif','_morph.h5')
        if not os.path.exists(morph_fname):
            morph = mne.compute_source_morph(src, subject_from=subjID_date, subject_to='fsaverage', src_to=src_to, subjects_dir=cfg.subj_dir)
            morph.save(morph_fname)
        else:
            morph = mne.read_source_morph(morph_fname)

        for condition in cfg.brain_selected_conditions:
            epochs_fname = tlbx.find_files('_AttenVis_nobaseline_nofilter_'+ condition +'_clean_epo.fif',visit_dir)
            epochs_clean = mne.read_epochs(epochs_fname[0],preload=True)
            baseline_evoked = tlbx.get_evoked(epochs_clean,filter=(1,30),baseline=cfg.baseline)
            stc = mne.minimum_norm.apply_inverse(baseline_evoked, inverse_operator, cfg.lambda2, method=cfg.con_method, pick_ori=None, verbose=True)
            subj_stcs[condition] = stc

        subtracted_stc = subj_stcs['pop-out'] - subj_stcs['search']
        morphed_stc = morph.apply(subtracted_stc)
        data = [sub_id,diagnosis,study,'pop-out-search',subtracted_stc,morphed_stc]
        subtracted_stcs.append(data)

    # Save it in a dataframe
    df = pd.DataFrame(subtracted_stcs,columns=['Participant','Diagnosis','Study','Condition','Subtracted_stc','Morphed_stc'])
    df.to_pickle(data_savename)
else: 
    df = pd.read_pickle(data_savename)
# Average within each group
for diagnosis in cfg.diagnoses:
    group_average = df[df['Diagnosis'] == diagnosis]['Morphed_stc'].values.mean(axis=0)
    brain = group_average.plot(
        subjects_dir='/autofs/space/transcend/MRI/WMA/recons/',
        hemi='both',
        initial_time=1.00,
        clim=dict(kind="values", lims=[-0.4,0.05,0.6]), #99.5, 99.7, 99.9
        colormap='bwr',
        smoothing_steps=7,
        views = cfg.brain_view,
        time_viewer = True
        )

# Plot the resulting stc onto the fsaverage
group_average.plot(subject='fsaverage')
