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
matplotlib.use('Agg') 
import helper_functions as tlbx
import AttenVis_power_config as cfg
# import time

participants_df, participants_to_study = tlbx.load_participants()
report = tlbx.generate_report()

# for sub_id in participants_to_study:
#     participant_data = []
#     diagnosis, study,visit_dir,subjID_date = tlbx.read_participant_details_from_dataframe(participants_df,sub_id)
#     participant_data_savename = os.path.join(visit_dir,cfg.data_fname.replace('.pkl','_' + sub_id + '.pkl'))
#     if not os.path.exists(participant_data_savename) or cfg.overwrite_data:
#         inverse_operator = tlbx.load_inverse_operator('_prestim_baseline_inv.fif',visit_dir)
        
#         for condition in cfg.brain_selected_conditions:
#             file_tag_cond = '_AttenVis_nobaseline_nofilter_' + condition.replace('/','_') 
#             epo_load_fname = tlbx.find_files(file_tag_cond + '_clean_epo.fif',visit_dir)[0]
#             print(epo_load_fname)
#             epochs = mne.read_epochs(epo_load_fname)
#             epochs   = epochs.resample(cfg.sfreq)

#             for hemi in cfg.hemisphere:
#                 #load_labels 
#                 annot_label = tlbx.load_drawn_labels(cfg.labels_of_interest,hemi,subjID_date,visit_dir,grown=True)
#                 #get power and itc
#                 power, itc = mne.minimum_norm.source_induced_power(
#                     epochs,
#                     inverse_operator,
#                     method= cfg.con_method,
#                     freqs = np.arange(cfg.freq_min,cfg.freq_max,1),
#                     label = annot_label,
#                     baseline = cfg.baseline,
#                     baseline_mode = 'logratio',
#                     n_cycles = cfg.con_n_cycles,
#                     n_jobs = None,
#                 )

#                 data = [diagnosis,sub_id,study,condition,hemi,cfg.labels_of_interest[0],np.mean(power,axis=0),itc,annot_label,epochs.times]
#                 participant_data.append(data)
#         df_participant = pd.DataFrame(participant_data,columns=['Diagnosis','Participant','Study','Condition','hemisphere','label','power','itc','grown_label','time'])
#         df_participant.to_pickle(participant_data_savename)
#         tlbx.add_participant_tfrs_to_report(df_participant,report,sub_id)
#         # tlbx.show_report(cfg.report_savename_hdf5)

#     else:
#         df_participant = pd.read_pickle(participant_data_savename)
#         tlbx.add_participant_tfrs_to_report(df_participant,report,sub_id)
#         # tlbx.show_report(cfg.report_savename_hdf5)
df = tlbx.collate_participants_data(participants_df,participants_to_study)

mpl.rcParams["svg.fonttype"] = "none"

tlbx.add_tfrs_to_report(df,report,'gavg',cfg.diagnoses['asd']['label_n'],cfg.diagnoses['td']['label_n'])
tlbx.add_gavg_power_over_time_to_report(df,report,'gavg','Diagnosis',(40,80),ci=False)
tlbx.add_gavg_power_over_time_to_report(df,report,'gavg','Condition',(40,80),ci=False)

# tlbx.add_fsaverage_to_report(report,df,cfg.labels_of_interest[0] + '_grown')

tlbx.show_report(cfg.report_savename_hdf5)

