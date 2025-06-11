#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17:00:00 2025

@author: jwt30
"""

import mne
import os
import pandas as pd

import helper_functions as tlbx
import AttenVis_config as cfg

participants_df, participants_to_study = tlbx.load_participants()
report = tlbx.generate_report()
peak_times = {}
for sub_id in participants_to_study:
    participant_data = []
    diagnosis, study,visit_dir,subjID_date = tlbx.read_participant_details_from_dataframe(participants_df,sub_id)
    participant_data_savename = os.path.join(visit_dir,cfg.data_fname.replace('.pkl','_' + sub_id + '.pkl'))
    if not os.path.exists(participant_data_savename) or cfg.overwrite_data:
        load_fname, epochs = tlbx.load_epochs(cfg.epochs_to_use,visit_dir,resample=True)
        inverse_operator = tlbx.load_inverse_operator('_prestim_baseline_inv.fif',visit_dir)
        tlbx.draw_label_from_epochs(visit_dir,subjID_date,epochs,inverse_operator,label_to_draw_from = 'fs_drawn',filter=(1,30))
        peak_times[sub_id] = cfg.peak_times_hemis
        #get brain pics
        for condition in cfg.brain_selected_conditions:
            condition_tag = "(Condition == '" + condition + "')"
            out_fname = load_fname.replace(cfg.epochs_to_use,'_'.join(['_nobaseline','nofilter','metadata',condition.replace('/','_'),'clean','epo.fif']))
            # if not os.path.exists(out_fname) or cfg.overwrite_epochs:
            #     epochs_clean = tlbx.get_condition_epochs(epochs,condition_tag)
            #     epochs_clean.save(out_fname) #always saves epochs with bad epochs removed but without any filtering or baseline correction
            # else:
            epochs_clean = mne.read_epochs(out_fname)
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

    else:
        df_participant = pd.read_pickle(participant_data_savename)
        tlbx.add_participant_activations_to_report(df_participant,report,sub_id)

tlbx.save_peak_times(peak_times)

df = tlbx.collate_participants_data(participants_df,participants_to_study)
tlbx.add_gavg_activations_to_report(df,report,'gavg', cfg.diagnoses,'Diagnosis')
tlbx.add_gavg_activations_to_report(df,report,'gavg', cfg.plot_selected_conditions,'Condition')

tlbx.add_fsaverage_to_report(report,df,cfg.labels_of_interest[0] + '_grown')

tlbx.show_report(cfg.report_savename_hdf5)
