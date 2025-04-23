#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17:00:00 2025

@author: jwt30
"""

import mne
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import helper_functions as tlbx
import AttenVis_config as cfg

participants_df, participants_to_study = tlbx.load_participants(cfg.participants_csv)

# participants_to_study = ['008301']
participants_to_study = ['133101','135201','135501','136701','136901','137101','137501','139501','139801','140101','140401','140601','142001','143701','143901','149701','150701']
# for part in participants_to_rename:
#     tlbx.rename_files(part,'_V1_grown-lh.label','_vis_lh.label')
#     tlbx.rename_files(part,'_V1_grown-rh.label','_vis_rh.label')
if cfg.add_participants:
    participants_to_study = tlbx.update_participants(cfg.data_savename,participants_to_study)
    if participants_to_study == []:
        raise ValueError('No new participants to add')
# for sub_id in participants_to_study:
#     tlbx.rename_files(sub_id,'_nobaseline_all_epo.fif','_nobaseline_nofilter_all_epo.fif')
report = mne.Report(title=cfg.report_title)
# participants_to_study = ['149401']
all_participants = []
if not os.path.exists(cfg.data_savename) or cfg.overwrite_data or cfg.add_participants:
    for sub_id in participants_to_study:
        participant_data = []
        diagnosis,study,visit_dir,subjID_date = tlbx.get_participant_details(participants_df,sub_id)
        #load inverse operator
        inv_path = tlbx.find_files('_inv.fif',visit_dir)[0]
        inverse_operator = mne.minimum_norm.read_inverse_operator(inv_path)
        src = inverse_operator["src"]
        #load epochs 
        file_tag = '_nobaseline_nofilter_all_epo.fif'
        load_fname = tlbx.find_files(file_tag,visit_dir)[0]
        epochs = mne.read_epochs(load_fname)
        epochs   = epochs.resample(cfg.sfreq)
        
        for condition in cfg.plot_selected_conditions:
            out_fname = load_fname.replace('all_epo.fif','_'.join([condition.replace('/','_'),'clean','epo.fif']))
            if not os.path.exists(out_fname) or cfg.overwrite_epochs:
                epochs_clean = tlbx.get_condition_epochs(epochs,condition)
                epochs_clean.save(out_fname) #always saves epochs with bad epochs removed but without any filtering or baseline correction
            else:
                epochs_clean = mne.read_epochs(out_fname)
            baseline_evoked = tlbx.get_evoked(epochs_clean,filter=(1,30),baseline=cfg.baseline)
            stc = mne.minimum_norm.apply_inverse(baseline_evoked, inverse_operator, cfg.lambda2, method=cfg.con_method, pick_ori=None, verbose=True)
            for hemi in cfg.hemisphere:
                #load_labels 
                if condition == 'search':
                    parc = 'aparc.a2009s'
                    # annot_label = tlbx.load_annot_labels(cfg.labels_list,subjID_date,parc,hemi,cfg.subj_dir)
                    annot_label = tlbx.morph_fslabel(cfg.labels_list[0],subjID_date,hemi)[0]
                    stc_from_annot_label = stc.in_label(annot_label)
                    grown_label,morphed_label,label_fname,peak_time = tlbx.find_peak_grow_label(stc_from_annot_label,hemi,cfg.peak_time_window[0],cfg.peak_time_window[1],5,subjID_date,'pow',visit_dir)
                    cfg.peak_labels_hemis.update({hemi:grown_label})
                    cfg.peak_times_hemis.update({hemi:peak_time})
                    condition_name = condition
                    difficulty = np.nan
                    brain_image_name = tlbx.plot_stc(stc,hemi,cfg.peak_times_hemis[hemi],grown_label,condition,label_color = 'blue')

                else:
                    grown_label = cfg.peak_labels_hemis[hemi]
                    morphed_label= []
                    condition_name = condition
                    difficulty = np.nan
                    # condition_name = condition.split('/')[0]
                    # difficulty = condition.split('/')[1]
                #get stcs in label
                stc_label= mne.extract_label_time_course(
                    stc, grown_label, src, mode ="mean", verbose="error"
                )

                data = [diagnosis,sub_id,study,condition_name,difficulty,condition,hemi,cfg.labels_of_interest,stc_label,morphed_label,stc.times]
                columns = ['Diagnosis','Participant','Study','Condition','Difficulty','Combined','hemisphere','label','stc','morphed_label','time'] 
                participant_data.append(data)
                all_participants.append(data)
        #get brain pics
        for condition in cfg.brain_selected_conditions:
            out_fname = load_fname.replace('all_epo.fif','_'.join([condition.replace('/','_'),'clean','epo.fif']))
            if not os.path.exists(out_fname) or cfg.overwrite_epochs:
                epochs_clean = tlbx.get_condition_epochs(epochs,condition)
                epochs_clean.save(out_fname) #always saves epochs with bad epochs removed but without any filtering or baseline correction
            else:
                epochs_clean = mne.read_epochs(out_fname)
            baseline_evoked = tlbx.get_evoked(epochs_clean,filter=(1,30),baseline=cfg.baseline)
            stc = mne.minimum_norm.apply_inverse(baseline_evoked, inverse_operator, cfg.lambda2, method=cfg.con_method, pick_ori=None, verbose=True)
            for hemi in cfg.hemisphere:
                brain_image_name = tlbx.plot_stc(stc,hemi,cfg.peak_times_hemis[hemi],cfg.peak_labels_hemis[hemi],condition,label_color = 'blue')

        df_participant = pd.DataFrame(participant_data,columns=columns)
        tlbx.add_participant_activations_to_report(df_participant,report,sub_id)
report.save(cfg.report_savename_html, overwrite=True)
#     df= pd.DataFrame(all_participants, columns = ['Diagnosis','Participant','Study','Condition','Difficulty','Combined','hemisphere','label','stc','morphed_label','time'] )
#     if cfg.add_participants:
#         old_df = pd.read_pickle(cfg.data_savename)
#         add_to_df = pd.concat([old_df,df])
#         add_to_df = add_to_df.sort_values(by='Participant')
#         df = add_to_df 
                    
#     df.to_pickle(cfg.data_savename)

# else: 
#     df = pd.read_pickle(cfg.data_savename)
#     report = mne.open_report(cfg.report_savename_hdf5)
# tlbx.add_gavg_activations_to_report(df,report,'gavg', cfg.diagnoses,'Diagnosis')
# tlbx.add_fsaverage_to_report(report,df,cfg.labels_of_interest[0] + '_grown')

# # tlbx.show_report(report,cfg.report_savename_hdf5)
# report.save(cfg.report_savename_html, overwrite=True)
# report.save(cfg.report_savename_hdf5, overwrite=True)