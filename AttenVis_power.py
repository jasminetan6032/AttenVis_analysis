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
import matplotlib.pyplot as plt
import matplotlib as mpl
import pactools
from matplotlib.gridspec import GridSpec

import helper_functions as tlbx
import AttenVis_config as cfg

participants_df, participants_to_study = tlbx.load_participants(cfg.participants_csv)

participants_to_study = ['113301','111401','114001','114501','032901','042203','106201','104101']
# if cfg.add_participants:
#     participants_to_study = tlbx.update_participants(cfg.data_savename,participants_to_study)
# # participants_to_study = ['148501']
all_participants = []
if not os.path.exists(cfg.data_savename) or cfg.overwrite_data or cfg.add_participants:
    for sub_id in participants_to_study:
        diagnosis, study,visit_dir,subjID_date = tlbx.get_participant_details(participants_df,sub_id)
        #load inverse operator
        inv_path = tlbx.find_files('_inv.fif',visit_dir)[0]
        inverse_operator = mne.minimum_norm.read_inverse_operator(inv_path)
        src = inverse_operator["src"] 
        
        for condition in cfg.brain_selected_conditions:
            file_tag_cond = '_nobaseline_nofilter_' + condition.replace('/','_') 
            epo_load_fname = tlbx.find_files(file_tag_cond + '_clean_epo.fif',visit_dir)[0]
            print(epo_load_fname)
            epochs = mne.read_epochs(epo_load_fname)
            epochs   = epochs.resample(cfg.sfreq)

            for hemi in cfg.hemisphere:
                #load_labels 
                annot_label = tlbx.load_drawn_labels(cfg.labels_of_interest,hemi,subjID_date,cfg.subj_dir,grown=True)
                #get power and itc
                power, itc = mne.minimum_norm.source_induced_power(
                    epochs,
                    inverse_operator,
                    method= cfg.con_method,
                    freqs = np.arange(cfg.freq_min,cfg.freq_max,1),
                    label = annot_label,
                    baseline = (-0.2, 0.0),
                    baseline_mode = 'logratio',
                    n_cycles = cfg.con_n_cycles,
                    n_jobs = None,
                )
                #get pac
                # low_fq_range = np.linspace(8, 12, 20)
                # estimator = pactools.Comodulogram(fs=cfg.sfreq, low_fq_range=low_fq_range,
                #                         low_fq_width=1, method='canolty',
                #                         progress_bar=False)        
                # estimator.fit(stc_label)
                data = [diagnosis,sub_id,study,condition,hemi,cfg.labels_of_interest[0],np.mean(power,axis=0),itc,annot_label,epochs.times]
                all_participants.append(data)
    df= pd.DataFrame(all_participants, columns = ['Diagnosis','Participant','Study','Condition','hemisphere','label','power','itc','grown_label','time']) 
    if cfg.add_participants:
        old_df = pd.read_pickle(cfg.data_savename)
        add_to_df = pd.concat([old_df,df])
        add_to_df = add_to_df.sort_values(by='Participant')
        df = add_to_df

    df.to_pickle(cfg.data_savename)
else: 
    df = pd.read_pickle(cfg.data_savename)

#open report if it exists 
if os.path.exists(cfg.report_savename_hdf5):
    report = mne.open_report(cfg.report_savename_hdf5)
else:
    report = mne.Report(title=cfg.report_title)

exclude_participants = [] 
tlbx.update_participants_n(participants_df,exclude_participants,'all')
for sub_id in participants_to_study:
    df_participant = df[df["Participant"]==sub_id]
    tlbx.add_participant_tfrs_to_report(df_participant,report,sub_id)

# for study in cfg.study:
#     df_study = df[df["Study"]==study]
#     tlbx.add_tfrs_to_report(df_study,report,study,cfg.diagnoses['misophonia'][study],cfg.diagnoses['td'][study])
mpl.rcParams["svg.fonttype"] = "none"

tlbx.add_tfrs_to_report(df,report,'gavg',cfg.diagnoses['asd']['label_n'],cfg.diagnoses['td']['label_n'])
tlbx.add_gavg_power_over_time_to_report(df,report,'gavg','Diagnosis',(8,13),ci=False)
tlbx.add_fsaverage_to_report(report,df,cfg.labels_of_interest[0] + '_grown')
# exclude_participants = ['147401','146201','151101','150901'] #'147401','146201',
# removed_participants = df['Participant'].isin(exclude_participants)
# df = df[~removed_participants]
# participants_used = tlbx.update_participants_n(participants_df,exclude_participants,'all')

report.save(cfg.report_savename_html, overwrite=True)
report.save(cfg.report_savename_hdf5, overwrite=True)
