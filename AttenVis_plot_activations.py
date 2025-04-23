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

# participants_to_study = ['113301','111401','114001','114501','032901','042203','106201','104101']
# participants_to_study = ['133101','135201','135501','136701','136901','137101','137501','139501','139801','140101','140401','140601','142001','143701','143901','149701','150701']
# for part in participants_to_rename:
#     tlbx.rename_files(part,'_V1_grown-lh.label','_vis_lh.label')
#     tlbx.rename_files(part,'_V1_grown-rh.label','_vis_rh.label')
participants_to_study = ['148501']
if cfg.add_participants:
    participants_to_study = tlbx.update_participants(cfg.data_savename,participants_to_study)
    if participants_to_study == []:
        raise ValueError('No new participants to add')

report = mne.Report(title=cfg.report_title)
report.save(cfg.report_savename_hdf5, overwrite=True)

all_participants = []
if not os.path.exists(cfg.data_savename) or cfg.overwrite_data or cfg.add_participants:
    for sub_id in participants_to_study:
        participant_data = []
        diagnosis,study,visit_dir,subjID_date = tlbx.get_participant_details(participants_df,sub_id)
        #load inverse operator
        inv_path = tlbx.find_files('_inv.fif',visit_dir)[0] # erm_test_prestim_baseline_inv.fif
        inverse_operator = mne.minimum_norm.read_inverse_operator(inv_path)
        src = inverse_operator["src"]
        file_tag = '_nobaseline_nofilter_all_epo.fif'
        load_fname = tlbx.find_files(file_tag,visit_dir)[0]

        for condition in cfg.plot_selected_conditions:
            out_fname = load_fname.replace('all_epo.fif','_'.join([condition.replace('/','_'),'clean','epo.fif']))
            epochs_clean = mne.read_epochs(out_fname)
            baseline_evoked = tlbx.get_evoked(epochs_clean,filter=(1,30),baseline=cfg.baseline)
            stc = mne.minimum_norm.apply_inverse(baseline_evoked, inverse_operator, cfg.lambda2, method=cfg.con_method, pick_ori=None, verbose=True)
            for hemi in cfg.hemisphere:
                #load_labels 
                annot_label = tlbx.load_drawn_labels(cfg.labels_list,hemi,subjID_date,visit_dir,grown=False)
                morphed_label= mne.morph_labels([annot_label], subject_to='fsaverage', subject_from=subjID_date, subjects_dir=cfg.subj_dir, surf_name='inflated')

                stc_from_annot_label = stc.in_label(annot_label)
                if condition == 'search':
                    peak_vertex,peak_time = stc.get_peak(hemi = hemi, tmin = cfg.peak_time_window[0],tmax = cfg.peak_time_window[1])
                    cfg.peak_labels_hemis.update({hemi:annot_label})
                    cfg.peak_times_hemis.update({hemi:peak_time})
                condition_name = condition
                difficulty = np.nan
                brain_image_name = tlbx.plot_stc(stc,hemi,cfg.peak_times_hemis[hemi],annot_label,condition,label_color = 'blue')
                #get stcs in label
                stc_label= mne.extract_label_time_course(
                    stc, annot_label, src, mode ="mean", verbose="error"
                )

                data = [diagnosis,sub_id,study,condition_name,difficulty,condition,hemi,cfg.labels_of_interest,stc_label,morphed_label,stc.times]
                columns = ['Diagnosis','Participant','Study','Condition','Difficulty','Combined','hemisphere','label','stc','morphed_label','time'] 
                participant_data.append(data)
                all_participants.append(data)

        df_participant = pd.DataFrame(participant_data,columns=columns)
        participant_data_savename = os.path.join(visit_dir,cfg.data_fname.replace('.pkl','_' + sub_id + '.pkl'))
        df_participant.to_pickle(participant_data_savename)
        tlbx.add_participant_activations_to_report(df_participant,report,sub_id)
        tlbx.show_report(report,cfg.report_savename_hdf5)

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
# tlbx.add_fsaverage_to_report(report,df,cfg.labels_of_interest[0] + '_drawn')

# # tlbx.show_report(report,cfg.report_savename_hdf5)
# report.save(cfg.report_savename_html, overwrite=True)
# report.save(cfg.report_savename_hdf5, overwrite=True)