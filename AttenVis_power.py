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
import matplotlib as mpl
import matplotlib
# matplotlib.use('Agg') 
import matplotlib.pyplot as plt    
from mne.parallel import parallel_func

import helper_functions as tlbx
import AttenVis_power_config as cfg

participants_df, participants_to_study = tlbx.load_participants()

def get_power_in_label(sub_id,overwrite_data=True):
    participant_data = []
    diagnosis, study,visit_dir,subjID_date = tlbx.read_participant_details_from_dataframe(participants_df,sub_id)
    participant_data_savename = os.path.join(visit_dir,cfg.data_fname.replace('.pkl','_' + sub_id + '.pkl'))

    if not os.path.exists(participant_data_savename) or overwrite_data:
        inverse_operator = tlbx.load_inverse_operator('_prestim_baseline_inv.fif',visit_dir)
        
        for condition in cfg.brain_selected_conditions:
            file_tag_cond = '_AttenVis_nobaseline_nofilter_metadata_' + condition.replace('/','_') 
            epo_load_fname = tlbx.find_files(file_tag_cond + '_behaviour_cleaned_epo.fif',visit_dir)[0]
            print(epo_load_fname)
            epochs = mne.read_epochs(epo_load_fname)
            epochs   = epochs.resample(cfg.sfreq)

            for hemi in cfg.hemisphere:
                #load_labels 
                annot_label = tlbx.load_drawn_labels(cfg.labels_of_interest,hemi,subjID_date,visit_dir,grown=True)
                #get power and itc
                freqs = np.arange(cfg.freq_min,cfg.freq_max,1)
                n_cycles = freqs/3
                power, itc = mne.minimum_norm.source_induced_power(
                    epochs,
                    inverse_operator,
                    method= cfg.con_method,
                    freqs = freqs,
                    label = annot_label,
                    baseline = cfg.baseline,
                    baseline_mode = 'logratio',
                    n_cycles = n_cycles,
                    n_jobs = None,
                )

                data = [diagnosis,sub_id,study,condition,hemi,cfg.labels_of_interest[0],np.mean(power,axis=0),itc,annot_label,epochs.times]
                participant_data.append(data)
        df_participant = pd.DataFrame(participant_data,columns=['Diagnosis','Participant','Study','Condition','hemisphere','label','power','itc','grown_label','time'])
        df_participant.to_pickle(participant_data_savename)
        pics = tlbx.plot_participant_tfrs(df_participant,sub_id)
    else:
        df_participant = pd.read_pickle(participant_data_savename)
        pics = tlbx.plot_participant_tfrs(df_participant,sub_id)
    return sub_id, pics
n_jobs = 8

debug = False
# participants_to_study = ['008301','009901','011201','011301','011302']
if debug:
    n_jobs = 1

parallel, run_func, _ = parallel_func(get_power_in_label, n_jobs=n_jobs)
results = parallel(run_func(subject) for subject in participants_to_study)

report = tlbx.generate_report()

for sub_id, pics in results:
    for pic, title, condition in pics:
        report.add_figure(fig=pic, title=title, section=sub_id, tags=[condition,'power'], replace=True)
        plt.close(pic)

report.save(cfg.report_savename_hdf5, verbose=False, overwrite=True)

df = tlbx.collate_participants_data(participants_df,participants_to_study)

mpl.rcParams["svg.fonttype"] = "none"

tlbx.add_tfrs_to_report(df,report,'gavg')
tlbx.add_tfrs_comparison_to_report(df,report,'gavg')
tlbx.add_gavg_power_over_time_to_report(df,report,'gavg','Diagnosis','Theta-Alpha',(6,12),ci=False)
tlbx.add_gavg_power_over_time_to_report(df,report,'gavg','Condition','Theta-Alpha',(6,12),ci=False)
tlbx.add_gavg_power_over_time_to_report(df,report,'gavg','Diagnosis','Theta',(4,12),ci=False)
tlbx.add_gavg_power_over_time_to_report(df,report,'gavg','Condition','Theta',(4,12),ci=False)
tlbx.add_gavg_power_over_time_to_report(df,report,'gavg','Diagnosis','Alpha',(8,12),ci=False)
tlbx.add_gavg_power_over_time_to_report(df,report,'gavg','Condition','Alpha',(8,12),ci=False)
tlbx.add_gavg_power_over_time_to_report(df,report,'gavg','Diagnosis','Beta',(13,30),ci=False)
tlbx.add_gavg_power_over_time_to_report(df,report,'gavg','Condition','Beta',(13,30),ci=False)

# tlbx.add_fsaverage_to_report(report,df,cfg.labels_of_interest[0] + '_grown')

tlbx.show_report(cfg.report_savename_hdf5)


