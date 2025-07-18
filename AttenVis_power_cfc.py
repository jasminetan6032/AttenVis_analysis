#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 12:31:59 2024

@author: jwt30
"""

import mne
import os
import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pactools
from mne.parallel import parallel_func

import helper_functions as tlbx
import AttenVis_power_config as cfg

participants_df, participants_to_study = tlbx.load_participants()
report = tlbx.generate_report()

def get_pac_in_label(sub_id,overwrite_data=False):
    participant_data = []
    diagnosis, study,visit_dir,subjID_date = tlbx.read_participant_details_from_dataframe(participants_df,sub_id)
    participant_data_savename = os.path.join(visit_dir,cfg.data_fname.replace('.pkl','_' + sub_id + '.pkl'))
    if not os.path.exists(participant_data_savename) or overwrite_data:
        inverse_operator = tlbx.load_inverse_operator('_prestim_baseline_inv.fif',visit_dir)
        src = inverse_operator['src']
        
        for condition in cfg.condition:
            file_tag_cond = '_AttenVis_nobaseline_nofilter_metadata_' + condition.replace('/','_') 
            epo_load_fname = tlbx.find_files(file_tag_cond + '_behaviour_cleaned_epo.fif',visit_dir)[0]
            print(epo_load_fname)
            epochs = mne.read_epochs(epo_load_fname)
            epochs   = epochs.resample(cfg.sfreq)
            baseline_evoked = tlbx.get_evoked(epochs,filter=None,baseline=None)
            stc = mne.minimum_norm.apply_inverse(baseline_evoked, inverse_operator, cfg.lambda2, method='MNE', pick_ori=None, verbose=True)
            stc = stc.crop(tmin=cfg.time_windows[0], tmax=cfg.time_windows[1])  # Crop to the desired time window

            for hemi in cfg.hemisphere:
                #load_labels 
                annot_label = tlbx.load_drawn_labels(cfg.labels_of_interest,hemi,subjID_date,visit_dir,grown=True)
                label_restricted = annot_label.restrict(src)
                stc_label= mne.extract_label_time_course(
                        stc, label_restricted, src, mode ="mean", verbose="error"
                    )
                #get pac
                low_fq_range = np.linspace(4, 12, 20)
                estimator = pactools.Comodulogram(fs=cfg.sfreq, low_fq_range=low_fq_range,
                                        low_fq_width=1, method=cfg.pac_method,
                                        progress_bar=False)        
                estimator.fit(stc_label)
                data = [diagnosis,sub_id,study,condition,hemi,cfg.labels_of_interest[0],estimator.comod_,estimator.low_fq_range,estimator.high_fq_range,annot_label,epochs.times]
                participant_data.append(data)
        df_participant = pd.DataFrame(participant_data,columns=['Diagnosis','Participant','Study','Condition','hemisphere','label','pac','low_freqs','high_freqs','grown_label','time'])
        df_participant.to_pickle(participant_data_savename)
        pics = tlbx.plot_participant_pacs(df_participant,sub_id)
    else:
        df_participant = pd.read_pickle(participant_data_savename)
        pics = tlbx.plot_participant_pacs(df_participant,sub_id)
    return sub_id,pics

n_jobs = 8

debug = False
if debug:
    n_jobs = 1

parallel, run_func, _ = parallel_func(get_pac_in_label, n_jobs=n_jobs)
results = parallel(run_func(participant) for participant in participants_to_study)

report = tlbx.generate_report()
for sub_id, pics in results:
    for pic, title, condition in pics:
        report.add_figure(fig=pic, title=title, section=sub_id, tags=[condition,'pac'], replace=True)
        plt.close(pic)

report.save(cfg.report_savename_hdf5, verbose=False, overwrite=True)
df = tlbx.collate_participants_data(participants_df,participants_to_study)
# participants_to_study = tlbx.update_participants_n(df,cfg.excluded_participants,cfg.paradigm)

tlbx.add_pacs_to_report(df,report,'gavg')
tlbx.add_pacs_comparison_to_report(df,report,'gavg',analysis_type = 'within_group')
tlbx.add_pacs_comparison_to_report(df,report,'gavg',analysis_type = 'between_group')

tlbx.add_interaction_plot_to_report(df,report,'gavg')

tlbx.show_report(cfg.report_savename_hdf5)
