#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 16:05:39 2024

@author: jwt30
"""

import os
import mne
import numpy

def find_file(search_string,data_dir):
    for path, directory_names, filenames in os.walk(data_dir):
        for filename in filenames:
            if search_string in filename:
                file = os.path.join(path,filename)
                
    return file  

subj_dir='/autofs/space/transcend/MRI/WMA/recons/'
paradigm = 'AttenVis'
local_dir = '/local_mount/space/hypatia/2/users/Jasmine'

#get participant list
participants = []

for stc_path, stc_directory_names, stc_filenames in os.walk(os.path.join(local_dir,paradigm)):
    for filename in stc_filenames:
        if 'Pop-Outs-lh.stc' in filename:

            participant = filename.split('_')[0]
            participants.append(participant)

subs2reject = ['082901','057101','008301','009901','030801']

pop_out_filenames = []
stcs = []
for participant in participants:
    if participant not in subs2reject:
        for mri_path, mri_directory_names, mri_filenames in os.walk(subj_dir):
            for dir in mri_directory_names:
                if participant + '_' in dir:
                    subjID_date = dir


        data_dir = os.path.join(local_dir,paradigm,participant)

        load_fname1_lh = find_file('Pop-Outs-lh.stc', data_dir)

        pop_out_filenames.append(load_fname1_lh)
                
        stc_popout = mne.read_source_estimate(load_fname1_lh,subject=subjID_date)
        stc_fsaverage = mne.compute_source_morph(stc_popout, subjects_dir=subj_dir).apply(stc_popout)
        stcs.append(stc_fsaverage)

gavg_stc_popout = numpy.mean(stcs)