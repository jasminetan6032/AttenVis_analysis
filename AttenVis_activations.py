#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17:00:00 2025

@author: jwt30
"""

import mne
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt    
from mne.parallel import parallel_func

import helper_functions as tlbx
import AttenVis_config as cfg

participants_df, participants_to_study = tlbx.load_participants()

def get_activations_in_label(sub_id,overwrite_data=False):
    participant_data = []
    diagnosis, study,visit_dir,subjID_date = tlbx.read_participant_details_from_dataframe(participants_df,sub_id)
    participant_data_savename = os.path.join(visit_dir,cfg.data_fname.replace('.pkl','_' + sub_id + '.pkl'))
    if not os.path.exists(participant_data_savename) or overwrite_data:
        load_fname, epochs = tlbx.load_epochs(cfg.epochs_to_use,visit_dir,resample=True)
        cleaned_epochs, summary = tlbx.clean_epochs_by_behaviour(epochs, rt_based=(0.1,1.2), percent=None, correct_answers_only=True)
        inverse_operator = tlbx.load_inverse_operator('_prestim_baseline_inv.fif',visit_dir)
        peak_info = tlbx.draw_label_from_epochs(visit_dir,subjID_date,cleaned_epochs,inverse_operator,label_to_draw_from = 'annot',filter=(1,30))
        #get brain pics
        for condition in cfg.brain_selected_conditions:
            condition_tag = "(Condition == '" + condition + "')"
            out_fname = load_fname.replace(cfg.epochs_to_use,'_'.join(['_nobaseline','nofilter','metadata',condition.replace('/','_'),'behaviour_cleaned','epo.fif']))
            if not os.path.exists(out_fname) or overwrite_data:
                epochs_clean = tlbx.get_condition_epochs(cleaned_epochs,condition_tag)
                epochs_clean.save(out_fname,overwrite = True) #always saves epochs with bad epochs removed but without any filtering or baseline correction
            else:
                epochs_clean = mne.read_epochs(out_fname)
            # difficulty_mask = (epochs_clean.metadata['difficulty'] =='8') | (epochs_clean.metadata['difficulty']=='10')
            # select_epochs = epochs_clean[difficulty_mask]
            baseline_evoked = tlbx.get_evoked(epochs_clean,filter=(1,30),baseline=cfg.baseline)
            stc = mne.minimum_norm.apply_inverse(baseline_evoked, inverse_operator, cfg.lambda2, method=cfg.con_method, pick_ori=None, verbose=True)
            for hemi in cfg.hemisphere:
                brain_image_name = tlbx.plot_stc(sub_id,stc,hemi,peak_info[hemi]['time'],peak_info[hemi]['label'],condition,label_color = 'blue')
                #get stcs in label
                stc_label= mne.extract_label_time_course(
                    stc, peak_info[hemi]['label'], inverse_operator["src"], mode ="mean", verbose="error"
                )

                data = [diagnosis,sub_id,study,condition,hemi,cfg.labels_of_interest,stc_label,peak_info[hemi]['morphed_label'],stc.times,len(epochs_clean),brain_image_name]
                columns = ['Diagnosis','Participant','Study','Condition','hemisphere','label','stc','morphed_label','time','n_epochs','brain_image'] 
                participant_data.append(data)

        df_participant = pd.DataFrame(participant_data,columns=columns)
        df_participant.to_pickle(participant_data_savename)
        pics = tlbx.plot_participant_activations(df_participant)

    else:
        df_participant = pd.read_pickle(participant_data_savename)
        peak_info = []
        pics = tlbx.plot_participant_activations(df_participant)
    return sub_id,peak_info,pics
n_jobs = 8

debug = True
if debug:
    n_jobs = 1

parallel, run_func, _ = parallel_func(get_activations_in_label, n_jobs=n_jobs)
results = parallel(run_func(subject) for subject in participants_to_study)

# tlbx.save_peak_info(results)

report = tlbx.generate_report()

for sub_id, peak_info, pics in results:
    diagnosis = participants_df[participants_df['Participant'] == sub_id]['Diagnosis'].values[0]
    for fig, hemi in pics:
        title = '_'.join([sub_id, hemi])
        report.add_figure(fig=fig, title=title, section=sub_id, tags=[hemi,'activations',diagnosis], replace=True)
        plt.close(fig)

report.save(cfg.report_savename_hdf5, verbose=False, overwrite=True)

df = tlbx.collate_participants_data(participants_df,participants_to_study)
tlbx.add_gavg_activations_to_report(df,report,'gavg', cfg.diagnoses,'Diagnosis')
tlbx.add_gavg_activations_to_report(df,report,'gavg', cfg.plot_selected_conditions,'Condition')

tlbx.add_fsaverage_to_report(report,df,cfg.labels_of_interest[0] + '_grown')

tlbx.show_report(cfg.report_savename_hdf5)
