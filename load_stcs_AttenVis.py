#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:44:23 2024

@author: jwt30
"""

import os
import mne


def find_file(search_string,data_dir):
    for path, directory_names, filenames in os.walk(data_dir):
        for filename in filenames:
            if search_string in filename:
                file = os.path.join(path,filename)
                
    return file  

participant = '121201'
subj_dir='/autofs/space/transcend/MRI/WMA/recons/'
paradigm = 'AttenVis'

possible_directories = []
for path, directory_names, filenames in os.walk(subj_dir):
    for dir in directory_names:
        if participant + '_' in dir:
            possible_directories.append(dir)
            
valid_directories = [i for i in range(0, len(possible_directories)) if len(possible_directories[i].split('_')) == 2 and len(possible_directories[i].split('_')[1])==8]


local_dir = '/local_mount/space/hypatia/2/users/Jasmine'
data_dir = os.path.join(local_dir,paradigm,participant) 

load_fname1_lh = find_file('Pop-Outs-lh.stc', data_dir)
meg_date = int(os.path.split(load_fname1_lh)[1].split('_')[2])

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


# load_fname1_rh = find_file('_Pop-Outs-rh.stc', data_dir)

# load_fname2_lh = find_file('_Search-lh.stc', data_dir)
# load_fname2_rh = find_file('_Search-rh.stc', data_dir)

# stc_popout_lh = mne.read_source_estimate(load_fname1_lh,subject=subjID_date)
# stc_popout_rh = mne.read_source_estimate(load_fname1_rh,subject=subjID_date)

# stc_search_lh = mne.read_source_estimate(load_fname2_lh,subject=subjID_date)
# stc_search_rh = mne.read_source_estimate(load_fname2_rh,subject=subjID_date)

stc_popout = mne.read_source_estimate(load_fname1_lh,subject=subjID_date)
# stc_search = mne.read_source_estimate(load_fname2_lh,subject=subjID_date)

initial_time = 0.0
brain = stc_popout.plot(
    subjects_dir='/autofs/space/transcend/MRI/WMA/recons/',
    hemi='both',
    initial_time=initial_time,
    clim=dict(kind="percent", lims=[97, 98, 99]),
    smoothing_steps=7,
)

# initial_time = 0.0
# brain = stc_search.plot(
#     subjects_dir='/autofs/space/transcend/MRI/WMA/recons/',
#     hemi='both',
#     initial_time=initial_time,
#     clim=dict(kind="percent", lims=[97, 98, 99]),
#     smoothing_steps=7,
# )



