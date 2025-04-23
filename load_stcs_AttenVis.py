#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:44:23 2024

@author: jwt30
"""

import os
import mne

import helper_functions as tlbx
import AttenVis_config as cfg

def find_file(search_string,data_dir):
    for path, directory_names, filenames in os.walk(data_dir):
        for filename in filenames:
            if search_string in filename:
                file = os.path.join(path,filename)
                
    return file  

participant = '011201'
paradigm = 'AttenVis'
subj_dir='/autofs/space/transcend/MRI/WMA/recons/'

possible_directories = []
for path, directory_names, filenames in os.walk(subj_dir):
    for dir in directory_names:
        if participant + '_' in dir:
            possible_directories.append(dir)
  
valid_directories = [i for i in range(0, len(possible_directories)) if len(possible_directories[i].split('_')) == 2 and len(possible_directories[i].split('_')[1])==8]

fsaverageDir = '/local_mount/space/hypatia/2/users/Jasmine/MNE-sample-data/subjects/'

local_dir = '/local_mount/space/hypatia/2/users/Jasmine'
data_dir = os.path.join(local_dir,paradigm,participant) 

if participant != "fsaverage":

    #load inverse operator
    inv_path = tlbx.find_files('_inv.fif',data_dir)[0]
    inverse_operator = mne.minimum_norm.read_inverse_operator(inv_path)
    src = inverse_operator["src"]
    load_fname1_lh = find_file('_nobaseline_nofilter_Pop-Outs_epo.fif', data_dir) #'  _20190515_Search_epo.fif 20190515_Search-lh.stc
    epochs = mne.read_epochs(load_fname1_lh)
    baseline_evoked = tlbx.get_evoked(epochs,filter=(0.5,20),baseline=(-0.2,0))
    stc = mne.minimum_norm.apply_inverse(baseline_evoked, inverse_operator, cfg.lambda2, method='dSPM', pick_ori=None, verbose=True)
    meg_date = int(os.path.split(load_fname1_lh)[1].split('_')[2])

    # if len(valid_directories) == 1:
    #     subjID_date = possible_directories[valid_directories[0]]
    # else:
    #     date_differences = []
    #     for i in range(0, len(valid_directories)):
    #         date=int(possible_directories[valid_directories[i]].split('_')[1])
    #         date_difference = meg_date-date
    #         date_differences.append(abs(date_difference))
    #     correct_file = valid_directories[date_differences.index(min(date_differences))]
    #     subjID_date = possible_directories[correct_file]

    # # print(load_fname1_lh)
    # stc = mne.read_source_estimate(load_fname1_lh,subject=subjID_date)
# else:
#     load_fname1_lh = find_file('miso_minus_novel_fsaverage-lh.stc', data_dir)
#     stc_popout = mne.read_source_estimate(load_fname1_lh,subject = "fsaverage")

mne.viz.set_browser_backend('matplotlib', verbose=None)
initial_time = 0.0
#.savgol_filter(30)
brain = stc.plot(
    subjects_dir='/autofs/space/transcend/MRI/WMA/recons/',
    hemi='both',
    initial_time=initial_time,
    clim=dict(kind="percent", lims=[97, 98, 99]),
    smoothing_steps=7
)



