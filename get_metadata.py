#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 19:59:47 2023

@author: jwt30
"""

import mne
import pandas
import os
import scipy
import numpy
import pickle
from autoreject import get_rejection_threshold
import quick_analyse_config as cfg

def find_file(search_string,data_dir):
    for path, directory_names, filenames in os.walk(data_dir):
        for filename in filenames:
            if search_string in filename:
                file = os.path.join(path,filename)
                
    return file    


def get_evoked(epochs,stimuli):
    condition_epochs = epochs[stimuli]
    reject = get_rejection_threshold(condition_epochs, ch_types=['mag','grad'], decim=2)
    condition_epochs.drop_bad(reject=reject)
    evoked = condition_epochs.average()
    return evoked

#create metadata
def get_metadata_and_behaviour(events):
    row_events = [event_name for event_name in list(cfg.event_dict.keys()) if not 'response' in event_name]
    keep_first = ["response"]
    metadata, events, event_id = mne.epochs.make_metadata(
        events=events, event_id=cfg.event_dict,
        tmin=cfg.metadata_tmin, tmax=cfg.metadata_tmax, sfreq=raw.info['sfreq'],
        row_events=row_events,
        keep_first=keep_first)
    
    if cfg.paradigm in ('AttenAud','Misophonia'):
        
        metadata[['attention_side','stimuli_type', 'pitch','stimulus_side']] = metadata.event_name.str.split('/',expand = True)
        metadata.loc[(metadata['first_response'] == metadata['stimulus_side']) & (metadata['stimuli_type'].str.contains('target')),
                     'response_correct'] = 'Hit'
        metadata.loc[(metadata['first_response'] != metadata['stimulus_side']) & (metadata['stimuli_type'].str.contains('target')),
                     'response_correct'] = 'Incorrect Response'
        metadata.loc[~(metadata['first_response'].isnull()) & ~(metadata['stimuli_type'].str.contains('target')),
                     'response_correct'] = 'False Alarm'
        metadata.loc[(metadata['first_response'].isnull()) & (metadata['stimuli_type'].str.contains('target')),
                     'response_correct'] = 'Missed Target'
        metadata.loc[(metadata['first_response'].isnull()) & ~(metadata['stimuli_type'].str.contains('target')),
                     'response_correct'] = 'Correct Rejection'
        
        stimuli_counts = metadata.value_counts(['stimuli_type','stimulus_side','attention_side'], sort = False)
        
        if not set(cfg.stimuli_count_reference).issubset(stimuli_counts.index):
            stimuli_counts= stimuli_counts.reindex(cfg.stimuli_count_reference,fill_value=0)
        
        stimuli_counts = stimuli_counts.sort_index()
        
        response_counts = metadata['response_correct'].value_counts()
        
        if not set(cfg.response_count_reference).issubset(response_counts.index):
            response_counts= response_counts.reindex(cfg.response_count_reference,fill_value=0)
    
        
        average_performance_in_run = pandas.DataFrame([{"%correct": round(response_counts['Hit']/stimuli_counts['target'].sum(),2), "reaction_time":metadata['response'].median()}])
        
        if cfg.paradigm == 'AttenAud':
            run_info = pandas.DataFrame([{"Stds_on_R_Attend_R":stimuli_counts['standard','right','attendRight'],"Stds_on_L_Attend_L":stimuli_counts['standard','left','attendLeft'],
                        "Beeps_on_R_Attend_L":stimuli_counts['beep','right','attendLeft'],"Beeps_on_L_Attend_R":stimuli_counts['beep','left','attendRight'],
                        "Targets_on_R_Attend_R":stimuli_counts['target','right'].sum(),"Targets_on_L_Attend_L":stimuli_counts['target','left'].sum(),
                        "Novels_on_R_Attend_L":stimuli_counts['novel','right'].sum(),"Novels_on_L_Attend_R":stimuli_counts['novel','left'].sum(),
                        "Distractors_on_R_Attend_L":stimuli_counts['dev','right'].sum(),"Distractors_on_L_Attend_R":stimuli_counts['dev','left'].sum(),
                        "Hits": response_counts['Hit'], "FalseAlarms": response_counts['False Alarm'], "IncorrectResponses":response_counts['Incorrect Response'],
                        "Misses":response_counts['Missed Target'],"CorrectRejections":response_counts['Correct Rejection'],
                        "nStimuli":len(metadata),"nResponses":metadata['response'].notna().sum()}])
        
        elif cfg.paradigm == 'Misophonia':
            run_info = pandas.DataFrame([{"Stds_on_R_Attend_R":stimuli_counts['standard','right','attendRight'],"Stds_on_L_Attend_L":stimuli_counts['standard','left','attendLeft'],
                        "Beeps_on_R_Attend_L":stimuli_counts['beep','right','attendLeft'],"Beeps_on_L_Attend_R":stimuli_counts['beep','left','attendRight'],
                        "Targets_on_R_Attend_R":stimuli_counts['target','right'].sum(),"Targets_on_L_Attend_L":stimuli_counts['target','left'].sum(),
                        "Novels_on_R_Attend_L":stimuli_counts['novel','right'].sum(),"Novels_on_L_Attend_R":stimuli_counts['novel','left'].sum(),
                        "Distractors_on_R_Attend_L":stimuli_counts['dev','right'].sum(),"Distractors_on_L_Attend_R":stimuli_counts['dev','left'].sum(),
                        "Misophones_on_R_Attend_L":stimuli_counts['misophone','right'].sum(),"Misophones_on_L_Attend_R":stimuli_counts['misophone','left'].sum(),
                        "Hits": response_counts['Hit'], "FalseAlarms": response_counts['False Alarm'], "IncorrectResponses":response_counts['Incorrect Response'],
                        "Misses":response_counts['Missed Target'],"CorrectRejections":response_counts['Correct Rejection'],
                        "nStimuli":len(metadata),"nResponses":metadata['response'].notna().sum()}])
        
    elif cfg.paradigm == 'AttenVis':
        metadata[['condition','difficulty']] = metadata.event_name.str.split('/',expand = True)
        metadata.loc[metadata['event_name'] == 'target',
                     'response'] = float("nan")
        metadata.loc[metadata['event_name'] == 'target',
                     'first_response'] = 'None'
        stimuli_counts = metadata.value_counts(['condition','difficulty'], sort = False)
        response_counts = metadata['first_response'].value_counts()
        run_info = pandas.DataFrame([{"pop_out4": stimuli_counts['pop-out','4'],"pop_out6": stimuli_counts['pop-out','6'], 
                    "pop_out8": stimuli_counts['pop-out','8'], "pop_out10": stimuli_counts['pop-out','10'],
                    "search4": stimuli_counts['search','4'],"search6": stimuli_counts['search','6'], 
                    "search8": stimuli_counts['search','8'], "search10": stimuli_counts['search','10'],
                    "total_pop-outs": stimuli_counts['pop-out'].sum(),"total_search": stimuli_counts['search'].sum(), 
                    "right_responses": response_counts['right'], "left_responses": response_counts['left'],
                    "nStimuli": stimuli_counts.sum(),"nResponses":metadata['response'].notna().sum()}])
        average_performance_in_run = metadata['response'].groupby(by=[metadata['condition'],metadata["difficulty"]]).median()
   
    return run_info, average_performance_in_run, metadata, events, event_id


participant = '106201'

transcend_dir = '/autofs/space/transcend/MEG/'

data_dir = os.path.join(transcend_dir,cfg.paradigm,participant)

run = 'run02_raw.fif'

#load raw file
raw_file = find_file(run, data_dir) 

raw = mne.io.read_raw_fif(raw_file, preload=True, verbose=False)

events_file = raw_file.replace('_raw.fif','_fixed_eve.fif')
fixed_events = mne.read_events(events_file)

#make metadata
run_info,average_performance_in_run, metadata, events, event_id = get_metadata_and_behaviour(fixed_events)

#load matlab file and attach results
mat_file   = raw_file.replace('_raw.fif','_behaviour.mat')
mat = scipy.io.loadmat(mat_file)

#for AttenVis
mat_triggers = numpy.transpose(mat['triggers'][0])

#check against .fif file events
stimuli_list = fixed_events[:, 2].tolist()
stimuli_triggers = numpy.array([i for i in stimuli_list if i < 32])

warning_triggers = numpy.array_equal(mat_triggers,stimuli_triggers)

#add correctTrials to metadata
correctTrials = mat['correctTrials'][0]
metadata.loc[pandas.isnull(metadata['response']) == False,
                     'correct'] = correctTrials
metadata["correct"] = metadata["correct"].replace({1:True,0:False})
RT=numpy.transpose(mat['reactionTime'][0])
metadata.loc[pandas.isnull(metadata['response']) == False,
                     'reactionTime'] = RT

#save clean metadata
metadata_fname = raw_file.replace('_raw.fif','_metadata.pkl')
with open(metadata_fname, 'wb')as f:  
    pickle.dump([metadata, events, event_id],f)

#load cleaned data from local dir
local_dir = '/local_mount/space/hypatia/2/users/Jasmine'
data_dir = os.path.join(local_dir,cfg.paradigm,participant)
run_tsss = run.replace('.fif','_tsss.fif')

tsss_filename = find_file(run_tsss,data_dir)

raw_sss = mne.io.read_raw_fif(tsss_filename, preload=True, verbose=False)
            
icafile = tsss_filename.replace('_raw_tsss.fif','_ica.fif')
ica     = mne.preprocessing.read_ica(icafile)
ica.apply(raw_sss)

#reload metadata and epoch
with open(metadata_fname,'rb') as f:  # Python 3: open(..., 'rb')
    metadata, events, event_id = pickle.load(f)

epochs = mne.Epochs(
    raw=raw_sss,
    tmin=-0.2,
    tmax=0.8,
    events=events,
    event_id=event_id,
    metadata=metadata,
    preload=True,
)

#calculate average performance from metadata
total_perf=[]
total_perf.append(average_performance_in_run)
average_performance = pandas.concat(total_perf).groupby(by=["condition","difficulty"]).mean()