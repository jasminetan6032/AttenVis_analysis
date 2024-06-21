#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 18:04:34 2024

@author: jwt30
"""
import mne
import os
import numpy as np
import seaborn as sns, matplotlib.pyplot as plt


def find_files(search_string,data_dir):
    files = []
    for path, directory_names, filenames in os.walk(data_dir):
        for filename in filenames:
            if search_string in filename:
                file = os.path.join(path,filename)
                files.append(file)
                
    return files  

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


local_dir = '/local_mount/space/hypatia/2/users/Jasmine/'
paradigm = 'AttenVis'
data_dir = os.path.join(local_dir, paradigm)

#find and load labels in data_dir
label_fnames=[]
for path, directory_names, filenames in os.walk(data_dir):
    for filename in filenames:
        if '.label' in filename:
            label_fname = os.path.join(path,filename)
            label_fnames.append(label_fname)


label_name = '_atten_'
labels_of_interest=[]
for label_fname in label_fnames:
    if label_name in label_fname:
        labels_of_interest.append(label_fname)

    
        
#plot labels
# from matplotlib.pyplot import cm
# color = cm.rainbow(np.linspace(0, 1, 7))
# for i, c in enumerate(color):
#     color[i,3] = 0.5

# participants_list = participants['miso']+participants['TD']
# participants_color = dict.fromkeys(participants_list)

# for i in range(0,7):
#     participants_color[participants_list[i]]=color[i]


fsaverageDir = '/local_mount/space/hypatia/2/users/Jasmine/MNE-sample-data/subjects/'
subj_dir = '/autofs/space/transcend/MRI/WMA/recons/'

participants_cond1 = []
participants_cond2 = []

brain = mne.viz.Brain(subject = 'fsaverage',hemi = 'both',views = 'lateral',subjects_dir = fsaverageDir,surf='inflated',background='white')

for label_fname in labels_of_interest:
    
    subjID_date = find_mri_recons(subj_dir,label_fname)

    label = [mne.read_label(label_fname,subject= subjID_date)]

    morphed_label= mne.morph_labels(label, subject_to='fsaverage', subject_from=subjID_date, subjects_dir=subj_dir, surf_name='inflated')
    hemi = os.path.split(label_fname)[1].split('_')[2].split('.')[0]
    print('Plotting participant ' + subjID_date)
    brain.add_label(morphed_label[0], hemi = hemi, alpha=1)
    print('Finished plotting participant ' + subjID_date)
 

#plot activations
    participant = subjID_date.split('_')[0]
    participant_dir = os.path.join(local_dir,paradigm,participant)
    load_fname = find_files('_Pop-Outs-lh.stc',participant_dir)[0]
    stc_cond1 = mne.read_source_estimate(load_fname,subject=subjID_date)

    load_fname = find_files('_Search-lh.stc',participant_dir)[0]
    stc_cond2 = mne.read_source_estimate(load_fname,subject=subjID_date)


    stc_label_cond1_lh   = stc_cond1.in_label(label[0])
    stc_label_cond2_lh   = stc_cond2.in_label(label[0])

    participants_cond1.append(stc_label_cond1_lh.data[0])
    participants_cond2.append(stc_label_cond2_lh.data[0])

participants_cond1_stc_ave = np.mean(participants_cond1,axis=0)
participants_cond2_stc_ave = np.mean(participants_cond2,axis=0)

fontsize = 20
sns.set(style="white")
sub_fig,sub_ax1 = plt.subplots(figsize=(10,4), layout='constrained')

sub_ax1.plot(stc_label_cond1_lh.times,participants_cond1_stc_ave, label='pop-outs')
sub_ax1.plot(stc_label_cond2_lh.times,participants_cond2_stc_ave, label='search')
sub_ax1.set_xlim([-0.2,0.6])
sub_ax1.set_ylim([0,50])
sub_ax1.tick_params(labelsize=fontsize)
sub_ax1.set_xlabel('Time (s)',fontsize=fontsize)
sub_ax1.set_ylabel('dSPM activation (AU)',fontsize=fontsize)
sub_ax1.axvline(x=0, ls='--', color='k')
sub_ax1.set_title('Grand-averaged activation in visual ROIs \n AttenVis',fontsize=24)
sub_ax1.legend(fontsize=fontsize)
sub_ax1.axvspan(0.38, 0.48, color='black', alpha=.15)