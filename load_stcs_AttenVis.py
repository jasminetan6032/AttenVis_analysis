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

participant = '127601'
subj_dir='/autofs/space/transcend/MRI/WMA/recons/'
paradigm = 'AttenVis'

for path, directory_names, filenames in os.walk(subj_dir):
    for dir in directory_names:
        if participant + '_' in dir:
            subjID_date = dir

local_dir = '/local_mount/space/hypatia/2/users/Jasmine'
data_dir = os.path.join(local_dir,paradigm,participant)

load_fname1_lh = find_file('Pop-Outs-lh.stc', data_dir)
# load_fname1_rh = find_file('_Pop-Outs-rh.stc', data_dir)

load_fname2_lh = find_file('_Search-lh.stc', data_dir)
# load_fname2_rh = find_file('_Search-rh.stc', data_dir)

# stc_popout_lh = mne.read_source_estimate(load_fname1_lh,subject=subjID_date)
# stc_popout_rh = mne.read_source_estimate(load_fname1_rh,subject=subjID_date)

# stc_search_lh = mne.read_source_estimate(load_fname2_lh,subject=subjID_date)
# stc_search_rh = mne.read_source_estimate(load_fname2_rh,subject=subjID_date)

stc_popout = mne.read_source_estimate(load_fname1_lh,subject=subjID_date)
stc_search = mne.read_source_estimate(load_fname2_lh,subject=subjID_date)

initial_time = 0.0
brain = stc_popout.plot(
    subjects_dir='/autofs/space/transcend/MRI/WMA/recons/',
    hemi='both',
    initial_time=initial_time,
    clim=dict(kind="percent", lims=[97, 98, 99]),
    smoothing_steps=7,
)

initial_time = 0.0
brain = stc_search.plot(
    subjects_dir='/autofs/space/transcend/MRI/WMA/recons/',
    hemi='both',
    initial_time=initial_time,
    clim=dict(kind="percent", lims=[97, 98, 99]),
    smoothing_steps=7,
)



