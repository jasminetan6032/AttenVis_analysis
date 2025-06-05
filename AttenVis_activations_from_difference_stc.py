#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs May 29

@author: jwt30
"""

import mne
import os
import pandas as pd

import helper_functions as tlbx
import AttenVis_config as cfg

participants_df, participants_to_study = tlbx.load_participants()
report = tlbx.generate_report()

if cfg.overwrite_data:
    for sub_id in participants_to_study:
        participant_data = []
        visit_dir = participants_df[participants_df['Participant'] == sub_id]['Visit_Dir'].values[0]
        participant_data_savename = os.path.join(visit_dir,cfg.data_fname.replace('.pkl','_' + sub_id + '.pkl'))
        if not os.path.exists(participant_data_savename) or cfg.overwrite_data:
            diagnosis = participants_df[participants_df['Participant'] == sub_id]['Diagnosis'].values[0]
            study = participants_df[participants_df['Participant'] == sub_id]['Study'].values[0]
            subjID_date = participants_df[participants_df['Participant'] == sub_id]['SubjID_Date'].values[0]
            inverse_operator = tlbx.load_inverse_operator('_prestim_baseline_inv.fif',visit_dir)
            load_fname, stc = tlbx.load_stc('_search_minus_pop-out_stc.fif-lh.stc',visit_dir,filter=(1,30))
            tlbx.draw_label_from_stc(visit_dir,subjID_date,stc,label_to_draw_from = 'fs_drawn')
            #get brain pics
            for condition in cfg.brain_selected_conditions:
                condition_tag = '_nobaseline_nofilter_' + condition.replace('/','_') + '_response_clean_epo.fif'
                load_fname, epochs_clean = tlbx.load_epochs(condition_tag,visit_dir,resample=True)
                baseline_evoked = tlbx.get_evoked(epochs_clean,filter=(1,30),baseline=cfg.baseline)
                stc = mne.minimum_norm.apply_inverse(baseline_evoked, inverse_operator, cfg.lambda2, method=cfg.con_method, pick_ori=None, verbose=True)
                for hemi in cfg.hemisphere:
                    brain_image_name = tlbx.plot_stc(stc,hemi,cfg.peak_times_hemis[hemi],cfg.peak_labels_hemis[hemi],condition,label_color = 'blue')
                    #get stcs in label
                    stc_label= mne.extract_label_time_course(
                        stc, cfg.peak_labels_hemis[hemi], inverse_operator["src"], mode ="mean", verbose="error"
                    )

                    data = [diagnosis,sub_id,study,condition,hemi,cfg.labels_of_interest,stc_label,cfg.peak_morphed_labels_hemis[hemi],stc.times]
                    columns = ['Diagnosis','Participant','Study','Condition','hemisphere','label','stc','morphed_label','time'] 
                    participant_data.append(data)

            df_participant = pd.DataFrame(participant_data,columns=columns)
            df_participant.to_pickle(participant_data_savename)
            tlbx.add_participant_activations_to_report(df_participant,report,sub_id)

df = tlbx.collate_participants_data(participants_df,participants_to_study)
tlbx.add_gavg_activations_to_report(df,report,'gavg', cfg.diagnoses,'Diagnosis')
tlbx.add_gavg_activations_to_report(df,report,'gavg', cfg.plot_selected_conditions,'Condition')

tlbx.add_fsaverage_to_report(report,df,cfg.labels_of_interest[0] + '_grown')

tlbx.show_report(cfg.report_savename_hdf5)

