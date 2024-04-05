#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 16:19:10 2023

@author: jwt30
"""
import mne 
import os
from autoreject import get_rejection_threshold

# event_dict = {'search/4': 1,
#                      'search/6': 2,
#                      'search/8': 3,
#                      'search/10': 4,
#                      'pop-out/4': 5,
#                      'pop-out/6': 6,
#                      'pop-out/8': 7,
#                      'pop-out/10': 8,
#                      'target': 32,
#                      'response/right': 2048,
#                      'response/left': 32768
#                      }

# file = '/local_mount/space/hypatia/2/users/Jasmine/AttenVis/082901/visit_20210313/082901_AttenVis_run03_raw_tsss.fif'

# #file = '/autofs/space/transcend/MEG/AttenVis//082901/visit_20210313/082901_AttenVis_run01_raw.fif'
# raw = mne.io.read_raw_fif(file)
# if raw.info['nchan'] != 312:
#     raw.drop_channels(['CHPI001','CHPI002','CHPI003','CHPI004','CHPI005','CHPI006','CHPI007','CHPI008','CHPI009'])
# events = mne.find_events(raw, stim_channel='STI101', uint_cast= True)
# epochs = mne.Epochs(raw,events,tmin = -0.2, tmax = 1.0,event_id = event_dict, preload=True)

# all_epochs = []
# all_epochs.append(epochs)

# all_epochs_save = mne.concatenate_epochs(all_epochs,on_mismatch = 'ignore')


paradigm = 'AttenVis'
ASD = ['089201','097201','104101','104901','106201',
       '007501','008301','030801','057101','085601','085702','086401'] #'085701' replace after cleaning triggers
TD = ['101901','106501','108201','116201',
            '011201','011301','011302','013703','032901','073801','075801','089901'] 

central = ['MEG0711','MEG1821','MEG1831','MEG2241','MEG2211','MEG0721','MEG0741','MEG0731']
frontal = ['MEG0811', 'MEG0521','MEG0511','MEG0311','MEG0541','MEG0611','MEG0531','MEG1011','MEG0821','MEG0941','MEG1021','MEG0931', 'MEG0921','MEG1211','MEG0911']
occipital = ['MEG1741','MEG1931','MEG2111','MEG2121','MEG2331','MEG2541','MEG2131','MEG2121','MEG2141']
def get_evoked(epochs,stimuli):
    condition_epochs = epochs[stimuli]
    reject = get_rejection_threshold(condition_epochs, ch_types=['mag','grad'], decim=2)
    condition_epochs.drop_bad(reject=reject)
    evoked = condition_epochs.average()
    return evoked

#MEG_dir = '/autofs/space/megraid_research/MEG/tal'
transcend_dir = '/local_mount/space/hypatia/2/users/Jasmine'

#load and epoch search and pop-outs, clean for each type of stimuli, save
for participant in ASD + TD:
    
    data_dir = os.path.join(transcend_dir,paradigm,participant + '/')
    
    for path, directory_names, filenames in os.walk(data_dir):
        for filename in filenames:
            if '_epo.fif' in filename: 
                epo_file = os.path.join(path,filename)
                date = path[-8:]
                epochs = mne.read_epochs(epo_file)
                stimuli = 'search'
                evoked1 = get_evoked(epochs,stimuli)
                evoked1.savgol_filter(30).plot_joint()
                search_ave = evoked1.savgol_filter(10)
                out_fname = participant + '_' + paradigm + '_' + str(date) + '_' + stimuli +'_ave.fif'
                evoked1.save(os.path.join(path,out_fname))
                stimuli = 'pop-out'
                evoked2 = get_evoked(epochs,stimuli)
                evoked2.savgol_filter(30).plot_joint()
                pop_out_ave = evoked2.savgol_filter(10)
                out_fname = participant + '_' + paradigm + '_' + str(date) + '_' + stimuli +'_ave.fif'
                evoked2.save(os.path.join(path,out_fname))
                
                
                plot_epochs = {'search':search_ave,
                               'pop-out': pop_out_ave}

                mne.viz.plot_compare_evokeds(plot_epochs, picks= central, combine = 'mean',show_sensors=True,
                                             title = 'Evoked responses from central areas \n Participant ' + participant)
                mne.viz.plot_compare_evokeds(plot_epochs, picks= frontal, combine = 'mean',show_sensors=True,
                                             title = 'Evoked responses from frontal areas \n Participant ' + participant)



#load for ASD/TD and get grand average
search_condition = []
for participant in ASD:
    
    data_dir = os.path.join(transcend_dir,paradigm,participant + '/')
    
    for path, directory_names, filenames in os.walk(data_dir):
        for filename in filenames:
            if 'search_ave.fif' in filename: 
                evo_file = os.path.join(path,filename)
                evoked = mne.read_evokeds(evo_file)
                search_condition.append(evoked[0])
                
search = mne.grand_average(search_condition)

popout_condition = []
for participant in ASD:
    
    data_dir = os.path.join(transcend_dir,paradigm,participant + '/')
    
    for path, directory_names, filenames in os.walk(data_dir):
        for filename in filenames:
            if 'pop-out_ave.fif' in filename: 
                evo_file = os.path.join(path,filename)
                evoked = mne.read_evokeds(evo_file)
                popout_condition.append(evoked[0])
                
popout = mne.grand_average(popout_condition)

plot_epochs = {'search':search,
               'pop-out': popout}


mne.viz.plot_compare_evokeds(plot_epochs, picks= occipital, combine = 'mean',show_sensors=True)
mne.viz.plot_compare_evokeds(plot_epochs, picks= frontal, combine = 'mean',show_sensors=True)
mne.viz.plot_compare_evokeds(plot_epochs, picks= central, combine = 'mean',show_sensors=True)

#load for ASD/TD and get grand average
search_condition = []
for participant in TD:
    
    data_dir = os.path.join(transcend_dir,paradigm,participant + '/')
    
    for path, directory_names, filenames in os.walk(data_dir):
        for filename in filenames:
            if 'search_ave.fif' in filename: 
                evo_file = os.path.join(path,filename)
                evoked = mne.read_evokeds(evo_file)
                search_condition.append(evoked[0])
                
search = mne.grand_average(search_condition)

popout_condition = []
for participant in TD:
    
    data_dir = os.path.join(transcend_dir,paradigm,participant + '/')
    
    for path, directory_names, filenames in os.walk(data_dir):
        for filename in filenames:
            if 'pop-out_ave.fif' in filename: 
                evo_file = os.path.join(path,filename)
                evoked = mne.read_evokeds(evo_file)
                popout_condition.append(evoked[0])
                
popout = mne.grand_average(popout_condition)

plot_epochs = {'search':search,
               'pop-out': popout}


mne.viz.plot_compare_evokeds(plot_epochs, picks= occipital, combine = 'mean',show_sensors=True)
mne.viz.plot_compare_evokeds(plot_epochs, picks= frontal, combine = 'mean',show_sensors=True)
mne.viz.plot_compare_evokeds(plot_epochs, picks= central, combine = 'mean',show_sensors=True)