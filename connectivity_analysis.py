#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 19:25:10 2024

@author: jwt30
"""

import mne
import os
import pandas as pd
from autoreject import get_rejection_threshold
from mne_connectivity import spectral_connectivity_epochs, seed_target_indices, seed_target_multivariate_indices
from mne.minimum_norm import apply_inverse_epochs
import numpy as np

def find_files(search_string,data_dir):
    files = []
    for path, directory_names, filenames in os.walk(data_dir):
        for filename in filenames:
            if search_string in filename:
                file = os.path.join(path,filename)
                files.append(file)
                
    return files  

def get_condition_epochs(epochs,stimuli):
    condition_epochs = epochs[stimuli]
    reject = get_rejection_threshold(condition_epochs, ch_types=['mag','grad'], decim=2)
    condition_epochs.drop_bad(reject=reject)
    #evoked = condition_epochs.average()
    return condition_epochs

def find_mri_recons(subj_dir,filename):
    participant = os.path.split(filename)[1].split('_')[0]
    possible_directories = []
    for path, directory_names, filenames in os.walk(subj_dir):
        for dir in directory_names:
            if participant + '_' in dir:
                possible_directories.append(dir)
                
    valid_directories = [i for i in range(0, len(possible_directories)) if len(possible_directories[i].split('_')) == 2 and len(possible_directories[i].split('_')[1])==8]
    
    
    meg_date = int(os.path.split(os.path.split(filename)[0])[1].split('_')[1])
    
    if len(valid_directories) == 1:
        subjID_date = possible_directories[valid_directories[0]]
    else:
        date_differences = []
        for i in range(0, len(valid_directories)):
            date=int(possible_directories[valid_directories[i]].split('_')[1])
            date_difference = meg_date-date
            date_differences.append(abs(date_difference))
        correct_file = valid_directories[date_differences.index(min(date_differences))]
        subjID_date = possible_directories[correct_file]
    return subjID_date

def get_diagnosis(csvfile,subject):
    """
    Get relevant behavioral data from redcap csv generated label file. This function requires as csv (labels) file generated from redcap and located in
    the paradigm folder in local_mount (cfg.paradigm_dir). The file should be saved as redcap_info.csv

    Parameters
    ----------
    subject : str
        subject ID
    """

        # diagnosis
    diagnosis = []
    for i in list(csvfile.columns[5:8]):
        this_val = csvfile[i][csvfile['Subject ID:'] == subject].dropna()
        if not this_val.empty:
            diagnosis.append(this_val.values)
    diagnosis = 'asd' if 'Yes' in diagnosis else 'td'
    
    return diagnosis


local_dir = '/local_mount/space/hypatia/2/users/Jasmine/'
paradigm = 'AttenVis'
data_dir = os.path.join(local_dir, paradigm)
subj_dir = '/autofs/space/transcend/MRI/WMA/recons/'

#get stcs for each epoch
method = "dSPM"
snr = 1.0  # use lower SNR for single epochs
lambda2 = 1.0 / snr**2

#connectivity settings                    
fmin = (8.0,35.0,60.0)
fmax = (12.0,80.0,80.0)
tmin_con = 0.0

condition = ['Search','Pop-Outs']
hemisphere = ['lh','rh']
difficulty = ['4','6','8','10']

#load participant list
fname = '/local_mount/space/hypatia/2/users/Jasmine/AttenVis/updated_meg_mri_alignment_20240522.csv'
csvfile = pd.read_csv(fname, sep=',')
csvfile['Subject'] = csvfile['Subject'].astype('string').str.zfill(6)
participants_list = list(set(csvfile['Subject']))
exclude_participants = ['000000','007501','073801','125401','126801','110401','073701','102001','103101','900005','132901','133101','135201','135501','136901','137101','137501']

#load fixation_basic for diagnosis
diagnosis_fname   = os.path.join(data_dir,'behavioral_and_demographics.csv')
diagnosis_file = pd.read_csv(diagnosis_fname, sep=',')
diagnosis_file['Subject ID:'] = diagnosis_file['Subject ID:'].astype('string').str.zfill(6)

all_participants = []

for sub_id in participants_list:
    if sub_id not in exclude_participants:
        if sub_id == '111501':
            participant = sub_id
            participant_dir = os.path.join(local_dir,paradigm,str(participant),'visit_20230811')
        else:
            participant = sub_id
            participant_dir = os.path.join(local_dir,paradigm,str(participant))
            participant_condition_con = []
            
            #find diagnosis
            diagnosis = get_diagnosis(diagnosis_file,sub_id)

            #load epochs
            load_fname = find_files('Search_epo.fif',participant_dir)[0]
            subjID_date = find_mri_recons(subj_dir, load_fname)
            info = mne.io.read_info(load_fname)
            sfreq = info["sfreq"]  # the sampling frequency)
            
            #load inverse operator
            fwd_path = find_files('_fwd.fif',participant_dir)[0]
            inv_fname = fwd_path.replace('_fwd.fif', '_inv.fif')
            if os.path.isfile(inv_fname):
                inverse_operator = mne.minimum_norm.read_inverse_operator(inv_fname)
            else:
                fwd   = mne.read_forward_solution(fwd_path, verbose=False)
                covfname = find_files('_cov.fif',participant_dir)[0]
                noise_cov = mne.read_cov(covfname)
                inverse_operator = mne.minimum_norm.make_inverse_operator(
                    info, fwd, noise_cov, loose=0.2, depth=0.8
                )
                mne.minimum_norm.write_inverse_operator(inv_fname,inverse_operator)
            
            #select epochs
            for condition_type in condition:
                condition_name = condition_type + '_epo.fif'
                epoch_name = find_files(condition_name,participant_dir)[0]
                epochs = mne.read_epochs(epoch_name)
                for hemi in hemisphere:
                    label_name = 'vis_' + hemi +'.label'
                    fname_label = find_files(label_name,participant_dir)[0]
                    vis_label = mne.read_label(fname_label)
                    src = inverse_operator["src"]  # the source space used
                    try:
                        IPL_label = mne.read_labels_from_annot(subjID_date, parc = 'aparc.a2009s',hemi = hemi, surf_name = 'white', regexp = 'S_intrapariet&P_trans', subjects_dir=subj_dir)
                    except:
                        IPL_label = mne.read_labels_from_annot(subjID_date, parc = 'aparc.a2009s',hemi = hemi, surf_name = 'white', regexp = 'S_intrapariet_and_P_trans', subjects_dir=subj_dir)

                    for difficulty_level in difficulty:
                        condition_epochs=get_condition_epochs(epochs,difficulty_level)
                        #extract stc from relevant epochs
                        stcs = apply_inverse_epochs(
                            condition_epochs, inverse_operator, lambda2, method, pick_ori="normal", return_generator=True
                        )
                        
                        #get label time series for visual labels
                        seed_ts = mne.extract_label_time_course(
                            stcs, vis_label, src, mode="mean_flip", verbose="error"
                        )
                        
                        #get label time series for intraparietal labels
                        stcs = apply_inverse_epochs(
                            condition_epochs, inverse_operator, lambda2, method, pick_ori="normal", return_generator=True
                        )
                        target_ts = mne.extract_label_time_course(
                            stcs, IPL_label, src, mode="mean_flip", verbose="error"
                        )
                        
                        #combine stcs
                        comb_ts = list(zip(seed_ts,target_ts))
                        
                        #set up indices
                        indices = seed_target_indices([0],[1])
                        
                        #calculate connectivity (coherence)                    
                        con = spectral_connectivity_epochs(
                            comb_ts,
                            indices=indices,
                            method="coh",
                            fmin = fmin,
                            fmax = fmax,
                            faverage = True, 
                            sfreq=sfreq,
                            n_jobs=1,
                            gc_n_lags = 40
                        )
                        
                        participant_data = [participant,diagnosis,condition_type,hemi,difficulty_level,con.get_data()[0][0],con.get_data()[0][1],con.get_data()[0][2]]
                        participant_condition_con.append(participant_data)

        all_participants.append(participant_condition_con)

data = pd.DataFrame()
for i in range(len(all_participants)): 
    data = data.append(pd.DataFrame(all_participants[i]))
    
data.columns = ['Participant','Diagnosis','Condition','hemisphere','difficulty','alpha','broad_gamma','high_gamma']
data.to_csv('alpha_gamma_AttenVis_coh.csv',index=False)
