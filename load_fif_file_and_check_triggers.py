#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 16:19:19 2023

@author: jwt30
"""
import os
import quick_analyse_config as cfg
import mne

def find_file(search_string,data_dir):
    for path, directory_names, filenames in os.walk(data_dir):
        for filename in filenames:
            if search_string in filename:
                file = os.path.join(path,filename)
                
    return file    


participant = '130901'

transcend_dir = '/autofs/space/transcend/MEG/'

local_dir = '/local_mount/space/hypatia/2/users/Jasmine/'

data_dir = os.path.join(transcend_dir,'AttenVis',participant)

run = 'run03_raw'

#load raw file
raw_file = find_file(run, data_dir) 
raw = mne.io.read_raw_fif(raw_file, preload=True, verbose=False)
try:
    events = mne.find_events(raw, stim_channel='STI101',uint_cast=True)
    print('no input error')
except:
    events = mne.find_events(raw, stim_channel='STI101',shortest_event=1, uint_cast=True)
    print('ValueError:shortest_event=1')

stimuli_list = events[:, 2].tolist()
stimuli_triggers = numpy.array([i for i in stimuli_list if i < 32])
target_triggers = numpy.array([i for i in stimuli_list if i == 32])
response_triggers = numpy.array([i for i in stimuli_list if i > 255])
stim_targets = numpy.array([i for i in stimuli_list if i < 255])

mat_file   = raw_file.replace('_raw.fif','_behaviour.mat')
mat = scipy.io.loadmat(mat_file)
mat_triggers = numpy.transpose(mat['triggers'][0])

warning_triggers = numpy.array_equal(mat_triggers,stimuli_triggers)

stimuli_list = fixed_events[:, 2].tolist()
uncorrected_events = set(stimuli_list) - set(list(cfg.event_dict.values()) + cfg.all_responses)

missing_triggers = [i for i in range(0, len(fixed_events)-1) if numpy.any(fixed_events[i,2]== list(cfg.event_dict.values())[0:8]) and (numpy.all(fixed_events[i+1,2]!= list(cfg.event_dict.values())[9:11]))]

if uncorrected_events:
    for event in uncorrected_events:
        index_of_uncorrected_events = [index for index in range(len(fixed_events)) if numpy.any(fixed_events[index,2] == event)]

check_missing_triggers=fixed_events[fixed_events[:,2]==128,:]        

new_event = [287037,0,32]
fixed_events = numpy.insert(fixed_events,343,new_event,axis=0)
fixed_events = numpy.delete(fixed_events,[344,345],axis=0)

events_fname = raw_file.replace('_raw.fif','_fixed_eve.fif')
mne.write_events(events_fname,fixed_events,overwrite=True)

events_check=mne.read_events('/autofs/cluster/transcend/MEG/AttenVis/106201/visit_20210614/106201_AttenVis_run02_fixed_eve.fif')