import mne
import os
import numpy as np
import seaborn as sns, matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
import pprint
import shutil
import scipy.stats as st
from collections import Counter
from autoreject import get_rejection_threshold, AutoReject
import scipy
import pickle
import csv
from scipy.stats import ttest_ind, f_oneway
from scipy.stats import ttest_rel, ttest_ind
from scipy.ndimage import label
import matplotlib.patches as mpatches
from statsmodels.stats.multitest import fdrcorrection

import AttenVis_config as cfg

#plotting parameters
SMALL_SIZE = 22
plt.rcParams["font.family"] = 'DejaVu Sans'
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=SMALL_SIZE)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rcParams['figure.constrained_layout.use'] = True

def find_files(search_string,data_dir):
    files = []
    for path, directory_names, filenames in os.walk(data_dir):
        for filename in filenames:
            if search_string in filename:
                file = os.path.join(path,filename)
                files.append(file)
                
    return files  

def rename_files(participant,original,new_name,copy=False):
    participant_dir = os.path.join(cfg.data_dir,participant)
    print(participant_dir)
    files_to_rename = find_files(original,participant_dir)
    files_to_rename.sort()
    if not files_to_rename:
        print('no files found')
    else:            
        for file in files_to_rename:
            print(file)
            new_filename = file.replace(original,new_name)
            print(new_filename)
            if copy:
                shutil.copy(file, new_filename)
            else:
                os.rename(file,new_filename)
        check_files = find_files(original,participant_dir)
        check_files.sort()
        if check_files:
            print('Files have been successfully renamed!')
            pprint.pprint(check_files)

def find_directories(path):
    """Finds all directories in the given path."""

    directories = []
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            directories.append(full_path)
    return directories

def find_mri_recons(subj_dir,sub_id,visit_date):
    possible_directories = []
    for path, directory_names, filenames in os.walk(subj_dir):
        for dir in directory_names:
            if sub_id + '_' in dir:
                possible_directories.append(dir)
                
    valid_directories = [i for i in range(0, len(possible_directories)) if len(possible_directories[i].split('_')) == 2 and len(possible_directories[i].split('_')[1])==8]
        
    meg_date = int(visit_date)
    
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

def get_participant_details(participants_df,sub_id):
    diagnosis = participants_df[participants_df['Participant']==sub_id]["Diagnosis"].values[0]
    if cfg.paradigm == 'AttenVis':
        study = 'AttenVis'
    else:
        study = participants_df[participants_df['Participant']==sub_id]["Study"].values[0]
    participant_dir = os.path.join(cfg.local_dir,study,str(sub_id))
    visit_dir = find_directories(participant_dir)[0]
    visit_date = os.path.split(visit_dir)[1].split('_')[1]
    if sub_id == '148501':
        subjID_date = '148501_20250224'
    else:
        subjID_date = find_mri_recons(cfg.subj_dir,sub_id,visit_date)
    return diagnosis,study,visit_dir,subjID_date

def read_participant_details_from_dataframe(participants_df,sub_id):
    visit_dir = participants_df[participants_df['Participant'] == sub_id]['Visit_Dir'].values[0]
    diagnosis = participants_df[participants_df['Participant'] == sub_id]['Diagnosis'].values[0]
    study = participants_df[participants_df['Participant'] == sub_id]['Study'].values[0]
    subjID_date = participants_df[participants_df['Participant'] == sub_id]['SubjID_Date'].values[0]
    return diagnosis, study,visit_dir,subjID_date

def update_participants(csv,participants_already_in_dataset):
    df = pd.read_csv(csv, sep=',')
    df['Participant'] = df['Participant'].astype('string').str.zfill(6)
    participants_in_csv = np.unique(df['Participant'].values)
    participants_to_add = [i for i in participants_in_csv if i not in participants_already_in_dataset]
    return participants_to_add

def update_meg_mri_csv():
    participants_df = pd.read_csv(cfg.participants_csv.replace('.csv','_mri_meg.csv'), sep=',')
    participants_df['Participant'] = participants_df['Participant'].astype('string').str.zfill(6)
    participants_to_add = update_participants(cfg.participants_csv,np.unique(participants_df['Participant'].values))
    if participants_to_add:
        new_participants = participants_df[participants_df['Participant'].isin(participants_to_add)]
        new_participants_added_df = pd.concat([participants_df, new_participants], ignore_index=True)
        df_sorted = new_participants_added_df.sort_values(by='Participant', ascending=True)
    else: 
        print('No new participants to add. Checking for empty fields...')
        nan_rows = participants_df[participants_df[['Visit_Dir','SubjID_Date']].isna().any(axis=1)]
        if not nan_rows.empty:
            print('Updating empty fields...')
            for sub_id in nan_rows['Participant'].values:
                diagnosis, study,visit_dir,subjID_date = get_participant_details(participants_df,sub_id)
                participants_df.loc[participants_df['Participant'] == sub_id,'Visit_Dir'] = visit_dir
                participants_df.loc[participants_df['Participant'] == sub_id,'SubjID_Date'] = subjID_date
        else:
            print('No empty fields to update.')
        df_sorted = participants_df.sort_values(by='Participant', ascending=True)        
    if 'Study' not in df_sorted.columns:
        df_sorted['Study'] = cfg.paradigm
    df_sorted.to_csv(cfg.participants_csv.replace('.csv','_mri_meg.csv'),index=False)
    return df_sorted

def load_misonat_participants(study):
    #load participant list
    participants = [['td', '114001','MisoNat'],
                    ['misophonia', '138201','MisoNat'],
                    ['misophonia', '138701','MisoNat'],
                    ['td', '140101','MisoNat'],
                    ['td', '142001','MisoNat'],
                    ['misophonia', '144601','MisoNat'],
                    ['misophonia', '145801','MisoNat'],
                    ['td', '143701','MisoNat'],
                    ['misophonia','135401','MisoNat2'],
                    ['misophonia','146201','MisoNat2'],
                    ['misophonia','147001','MisoNat2'],
                    ['misophonia','147401','MisoNat2'],
                    ['misophonia','148201','MisoNat2'],
                    ['misophonia','148301','MisoNat2'],
                    ['misophonia','148501','MisoNat2'],
                    ['misophonia','148901','MisoNat2'],
                    ['misophonia','149401','MisoNat2'],      
                    ['td','150801','MisoNat2'],
                    ['td','150901','MisoNat2'],
                    ['td','151001','MisoNat2'],
                    ['td','151101','MisoNat2'],
                    ['td','999901','MisoNat2']]
                    # ['td','999902','MisoNat2']]
    participants_df = pd.DataFrame(participants,columns = ['Diagnosis','Participant','Study'])
    participants_to_study_exclude = update_participants_n(participants_df,cfg.excluded_participants,study)

    return participants_df, participants_to_study_exclude

def update_participants_n(participants_df,exclude_participants,study):
    #updates number of participants currently only by Diagnosis
    if study == 'all':
        participants_to_study = participants_df['Participant'].to_list()
    elif study == 'MisoNat':
        participants_to_study = participants_df[participants_df['Study']=='MisoNat']['Participant'].to_list() 
    elif study == 'MisoNat2':
        participants_to_study = participants_df[participants_df['Study']=='MisoNat2']['Participant'].to_list()
    elif study == 'miso_only':
        participants_to_study = participants_df[participants_df['Diagnosis']=='misophonia']['Participant'].to_list()
    else:
        participants_to_study = np.unique(participants_df['Participant'].to_list())
    cols_to_keep = ['Participant','Diagnosis']
    df_unique = participants_df[cols_to_keep].drop_duplicates(subset='Participant')
    participants_to_study_exclude = [x for x in participants_to_study if x not in exclude_participants]
    df_analysed_participants = df_unique[df_unique['Participant'].isin(participants_to_study_exclude)]
    n_participants = df_analysed_participants.value_counts(['Diagnosis'],sort=False)
    for diagnosis in cfg.diagnoses:
        try:
            cfg.diagnoses[diagnosis].update({'group_n':n_participants[diagnosis]})
            cfg.diagnoses[diagnosis].update({'label_n': cfg.diagnoses[diagnosis]['label'] + ' (n=' + str(n_participants[diagnosis]) + ')'})
        except:
            cfg.diagnoses[diagnosis].update({'group_n':0})
            cfg.diagnoses[diagnosis].update({'label_n': cfg.diagnoses[diagnosis]['label'] + ' (n=0)'})

    if study == 'all':
        n_participants = df_analysed_participants.value_counts(['Study','Diagnosis'],sort=False)
        for diagnosis in cfg.diagnoses:
            cfg.diagnoses[diagnosis].update({'MisoNat':cfg.diagnoses[diagnosis]['label'] + ' (n=' + str(n_participants['MisoNat'][diagnosis]) + ')'})
            cfg.diagnoses[diagnosis].update({'MisoNat2': cfg.diagnoses[diagnosis]['label'] + ' (n=' + str(n_participants['MisoNat2'][diagnosis]) + ')'})
    # elif study == 'miso_only':
    #     cfg.diagnoses[diagnosis].update({'group_n':n_participants[diagnosis]})
    #     cfg.diagnoses[diagnosis].update({'label_n': cfg.diagnoses[diagnosis]['label'] + ' (n=' + str(n_participants[diagnosis]) + ')'})
    return participants_to_study_exclude

def load_participants():
    """"
    This function takes the output of analyse_miso_participants, either analysed_participants_demographics.csv or collected_participants_demographics.csv or a manually set up csv for miso_asd_td comparisons.

    """
    participants_df = update_meg_mri_csv()

    participants_to_study = list(set(participants_df['Participant']))
    participants_to_study.sort()

    df_analysed_participants = participants_df[participants_df['Participant'].isin(participants_to_study)]
    participants_to_study_exclude = update_participants_n(df_analysed_participants,cfg.excluded_participants,'na')

    return participants_df,participants_to_study_exclude

def get_condition_epochs(epochs,condition):
    if condition == None:
        cond_epochs = epochs
    else:
        cond_epochs = epochs[condition]
    reject = get_rejection_threshold(cond_epochs, ch_types=['mag','grad'], decim=2)
    epochs_clean = cond_epochs.drop_bad(reject=reject)
    return epochs_clean

def get_evoked(epochs_clean, filter = None, baseline = None):
    if filter == None:
        filtered_epochs = epochs_clean
    else:
        filtered_epochs = epochs_clean.filter(filter[0],filter[1])
    if baseline == None:
        baselined_epochs = filtered_epochs
    else:
        baselined_epochs = filtered_epochs.apply_baseline(baseline=baseline)
    baseline_evoked = baselined_epochs.average()
    return baseline_evoked

def flip_peaks(time_series):
    time_series_abs = np.abs(time_series)
    max_index = np.argmax(time_series_abs)
    if time_series[max_index] <0:
        time_series_output = time_series * -1
    else:
        time_series_output = time_series
    return time_series_output

def load_drawn_labels(labels_list,hemi,subjID_date,participant_dir,grown=False):
    """"
    Takes a list of drawn labels. If they are to be combined, give the label names in a list. 
    If you want them in separate labels, run in a loop through. 
    """
    labels_to_combine = []
    for label_name in labels_list:
        if grown == True:
            label_name = label_name + '_grown'+ '-' + hemi +'.label'#labels grown from seeds follow the mne convention of -lh,-rh
        else:
            label_name = label_name + '_' + hemi +'.label'
        fname_label = find_files(label_name,participant_dir)[0]
        single_drawn_label = [mne.read_label(fname_label,subject=subjID_date)]
        labels_to_combine = labels_to_combine.__add__(single_drawn_label)
    drawn_label = labels_to_combine[0]
    if len(labels_to_combine) > 1:
        for i in labels_to_combine[1:]: 
            drawn_label+=i 

    return drawn_label

def load_annot_labels(labels_list,subjID_date,parc,hemi,subj_dir):
    """"
    Takes a list of labels from annotations. If they are to be combined, give the label names in a list. 
    If you want them in separate labels, run in a loop through. 
    """
    aparc_differences = ['G_Ins_lg&S_cent_ins','G_Ins_lg_and_S_cent_ins',
                        'S_intrapariet&P_trans','S_intrapariet_and_P_trans',
                        'G&S_cingul-Ant', 'G&S_cingul-Mid-Ant','G&S_cingul-Mid-Post',
                        'G_and_S_cingul-Ant', 'G_and_S_cingul-Mid-Ant','G_and_S_cingul-Mid-Post'
                        ]
    
    def check_other_naming_convention(label_name):
        if '_and_' in label_name:
            label_name_v2 = label_name.replace('_and_','&')
        elif '&' in label_name:
            label_name_v2 = label_name.replace('&','_and_')
        
        return label_name_v2

    labels_to_combine = []
    for label_name in labels_list:
        if label_name in aparc_differences:
            try:
                next_label = mne.read_labels_from_annot(subjID_date, parc = parc,hemi = hemi, surf_name = 'white', regexp = label_name, subjects_dir=subj_dir)
            except:
                other_naming_convention = check_other_naming_convention(label_name)
                next_label = mne.read_labels_from_annot(subjID_date, parc = parc,hemi = hemi, surf_name = 'white', regexp = other_naming_convention, subjects_dir=subj_dir)
        else:
            next_label = mne.read_labels_from_annot(subjID_date, parc = parc,hemi = hemi, surf_name = 'white', regexp = label_name, subjects_dir=subj_dir)

        labels_to_combine = labels_to_combine.__add__(next_label)
    annot_label = labels_to_combine[0]
    if len(labels_to_combine) > 1:
        for i in labels_to_combine[1:]: 
            annot_label+=i 
    return annot_label

def morph_fslabel(label,subjID_date,hemi):
    label_name = label + '-' + hemi + '.label'
    fname_label = find_files(label_name,cfg.data_dir)[0]
    drawn_label = [mne.read_label(fname_label,subject='fsaverage')]
    morphed_label= mne.morph_labels(drawn_label, subject_to=subjID_date, subject_from='fsaverage', subjects_dir=cfg.subj_dir, surf_name='inflated')
    brain = mne.viz.Brain(subject = subjID_date,hemi = hemi ,views = cfg.brain_view,subjects_dir = cfg.subj_dir,surf='inflated',background='white',show = True)
    brain.add_label(morphed_label[0], hemi = hemi, alpha=0.75)
    brain_fig_name = 'brain.tiff'
    brain_image_name = os.path.join(cfg.output_dir,brain_fig_name)
    brain.save_image(filename=brain_image_name, mode='rgb')
    fig = plt.figure(figsize=(6,6), layout='constrained')
    gs  = GridSpec(1, 1, figure=fig) 
    ax1 = fig.add_subplot(gs[0,0])
    ax1.imshow(plt.imread(brain_image_name))
    ax1.axis('off')
    # report  = mne.Report(title=cfg.morph_report_savename) if not os.path.exists(os.path.join(cfg.savedir,cfg.morph_report_savename_hdf5)) else mne.open_report(os.path.join(cfg.savedir,cfg.morph_report_savename_hdf5))
    # report.add_figure(fig = fig,title = hemi, section = subjID_date)
    # report.save(os.path.join(cfg.savedir,cfg.morph_report_savename_hdf5),verbose=False,overwrite=True)

    brain.close()

    return morphed_label[0],fig

def show_report(report_name):
    """
    Show html repot for paradigm
    """
    print('\n>>> TranscendTLBX: Creating .html report\n')
    if os.path.exists(report_name):
        report  = mne.open_report(report_name)
        report.save(os.path.join(report_name.replace('.hdf5','.html')),verbose=False,overwrite=True)    
    else:
        raise TypeError('ERROR: No report found in paradgm directory')

def find_peak_grow_label(stc,hemi,tmin,tmax,label_size,subjID_date,peak_type,file_location,mode = 'abs'):
    peak_vertex,peak_time = stc.get_peak(hemi = hemi, tmin = tmin,tmax = tmax,mode = mode)
    if hemi == 'lh':
        hemis = 0
    elif hemi == 'rh':
        hemis = 1
    if peak_type == 'coh':
        annot_label = mne.grow_labels(subjID_date,peak_vertex,label_size,hemis,subjects_dir = cfg.subj_dir,names=[cfg.labels_of_interest[0] + '_coh_' + hemi + '_grown'])[0]
    elif peak_type == 'diff':
        annot_label = mne.grow_labels(subjID_date,peak_vertex,label_size,hemis,subjects_dir = cfg.subj_dir,names=[cfg.labels_of_interest[0] + '_diff_' + hemi + '_grown'])[0]  
    else:
        annot_label = mne.grow_labels(subjID_date,peak_vertex,label_size,hemis,subjects_dir = cfg.subj_dir,names=[cfg.labels_of_interest[0] + '_grown'])[0]
    label_fname = os.path.join(file_location,'_'.join([subjID_date.split('_')[0],annot_label.name + '.label']))
    mne.write_label(label_fname,annot_label)
    morphed_label= mne.morph_labels([annot_label], subject_to='fsaverage', subject_from=subjID_date, subjects_dir=cfg.subj_dir, surf_name='inflated')
    
    return annot_label,morphed_label,label_fname,peak_time


def plot_time_frequency(times, freqs, condition, output_dir, datasets, titles, hemi=None, add_vlines=None):
    SMALL_SIZE = 22
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=SMALL_SIZE)
    plt.rcParams['figure.constrained_layout.use'] = True

    x_lims = [cfg.tmin_plot, cfg.tmax_plot]
    levels = np.linspace(cfg.power_plot_lims[0], cfg.power_plot_lims[1], cfg.power_plot_lims[2])

    n = len(datasets)
    fig = plt.figure(figsize=(6 * n, 5))
    gs = fig.add_gridspec(nrows=2, ncols=3 * n)

    for i, (data, title) in enumerate(zip(datasets, titles)):
        ax = fig.add_subplot(gs[0:2, 3 * i:3 * (i + 1)])
        cbh = ax.contourf(times, freqs, data, levels=levels, extend='both')
        if add_vlines:
            for line in cfg.vlines:
                ax.axvline(x=line, color='black', linestyle='--', linewidth=1)
        ax.set_xlim(x_lims)
        if i == 0:
            ax.set_ylabel('Frequency (Hz)')
        else:
            ax.set_yticklabels('')
        ax.set_title(title)

    fig.supxlabel('Time (s)', fontsize=22)
    fig.colorbar(cbh, ax=fig.axes, label='Power (dB)', ticks=np.linspace(cfg.power_plot_lims[0], cfg.power_plot_lims[1], cfg.power_plot_lims[3]))

    # Title based on condition
    condition_titles = {
        'miso': 'Evoked response to Trigger Sounds',
        'sound2': 'Evoked response to Neutral Sounds',
        'white_noise': 'Evoked response to white noise',
        'amp_mod': 'Evoked response to amplitude modulated white noise',
        'search': 'Evoked response to stimuli during Search',
        'pop-out': 'Evoked response to stimuli during Pop-Out',
    }
    full_title = condition_titles.get(condition, 'Evoked response')
    if hemi:
        full_title += f" ({hemi.upper()})"
    fig.suptitle(full_title, fontsize=22)

    # Save figure
    if hemi is None:
        savetitle = '_'.join([condition, 'power', 'plot'])
    else:
        savetitle = '_'.join([hemi, condition, 'power', 'plot'])

    savename = os.path.join(output_dir, savetitle + ".tiff")
    fig.savefig(savename, dpi=300)
    plt.close()
    return fig, savename


def plot_pac(low_fq_range, high_fq_range, condition, output_dir, data_list, titles, hemi=None):
    SMALL_SIZE = 22
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=SMALL_SIZE)
    plt.rcParams['figure.constrained_layout.use'] = True

    levels = np.linspace(cfg.crossfreq_plot_lims[0], cfg.crossfreq_plot_lims[1], cfg.crossfreq_plot_lims[2])
    n_panels = len(data_list)

    fig, ax = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))

    # Ensure ax is iterable even for one panel
    if n_panels == 1:
        ax = [ax]

    for i, (data, title) in enumerate(zip(data_list, titles)):
        cf = ax[i].contourf(low_fq_range, high_fq_range, data.T, levels=levels, extend='both')
        ax[i].set_xlabel('Driver Frequency (Hz)')
        ax[i].set_title(title)
        if i == 0:
            ax[i].set_ylabel('Frequency (Hz)')
        else:
            ax[i].set_yticklabels('')

    # Shared colorbar
    fig.colorbar(cf, ax=ax, label='PAC', ticks=np.linspace(cfg.crossfreq_plot_lims[0], cfg.crossfreq_plot_lims[1], cfg.crossfreq_plot_lims[3]))

    # Title based on condition
    condition_titles = {
        'miso': 'Evoked response to Trigger Sounds',
        'sound2': 'Evoked response to Neutral Sounds',
        'white_noise': 'Evoked response to white noise',
        'amp_mod': 'Evoked response to amplitude modulated white noise',
        'search': 'Evoked response to stimuli during Search',
        'pop-out': 'Evoked response to stimuli during Pop-Out',
    }
    full_title = condition_titles.get(condition, 'Evoked response')
    if hemi:
        full_title += f" ({hemi.upper()})"
    fig.suptitle(full_title, fontsize=22)

    # Save fig
    savetitle = '_'.join(filter(None, [hemi, condition, 'pac', 'plot']))
    savename = os.path.join(output_dir, savetitle + ".tiff")
    fig.savefig(savename, dpi=300)
    plt.close(fig)
    return fig, savename
       
def add_tfrs_to_report(df,report,id):
    times = df["time"].values[0]
    time_to_plot = [find_nearest(times,cfg.tmin_plot),find_nearest(times,cfg.tmax_plot)]
    time_for_plot = times[time_to_plot[0]:time_to_plot[1]]
    freqs = np.arange(cfg.freq_min,cfg.freq_max+1,1)
    freq_to_plot = [np.where(freqs==cfg.freq_min_plot)[0][0],np.where(freqs==cfg.freq_max_plot)[0][0]]
    for condition in cfg.brain_selected_conditions:
        image_names = []
        for hemi in cfg.hemisphere:
            df_hemi = df[df["hemisphere"]==hemi]
            datasets = []
            for diagnosis in cfg.diagnoses:
                df_to_plot = df_hemi[(df_hemi['Diagnosis'] == diagnosis) & (df_hemi['Condition']==condition)]
                if df_to_plot.empty:
                    print(f"No data for {diagnosis} in {condition} for {hemi}. Skipping...")
                    continue
                avg_data = np.stack(df_to_plot['power'].values).mean(axis=0)
                sliced_data = avg_data[freq_to_plot[0]:freq_to_plot[1],time_to_plot[0]:time_to_plot[1]]
                datasets.append(sliced_data)
            titles = [cfg.diagnoses[diag]['label_n'] for diag in cfg.diagnoses]

            fig1, name = plot_time_frequency(time_for_plot,freqs[freq_to_plot[0]:freq_to_plot[1]],condition,cfg.output_dir,datasets,titles, hemi=hemi,add_vlines=cfg.vlines)
            #save fig
            fig_to_save = fig1.get_figure()
            fig_to_save.savefig(name.replace('.tiff','.svg'),format="svg")
            fig_to_save.savefig(name,dpi=300)
            image_names.append(name)
            plt.close()
        fig = plt.figure(figsize=(18,6), layout='constrained')
        gs  = GridSpec(1, 2, figure=fig) 
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[0,1])
        ax1.imshow(plt.imread(image_names[0]))
        ax1.axis('off')
        ax2.imshow(plt.imread(image_names[1]))
        ax2.axis('off')
        title = '_'.join([id,condition,'power'])
        
        report.add_figure(fig=fig, title=title, section=id, tags=[condition,'power'],replace=True)
        report.save(cfg.report_savename_hdf5, verbose=False, overwrite=True)
        plt.close('all')

def plot_participant_tfrs(df,id):
    pics = []
    time = df['time'].values[0]
    freqs = np.arange(cfg.freq_min,cfg.freq_max+1,1)
    time_to_plot = [find_nearest(time,cfg.tmin_plot),find_nearest(time,cfg.tmax_plot)]
    time_for_plot = time[time_to_plot[0]:time_to_plot[1]]
    freq_to_plot = [np.where(freqs==cfg.freq_min_plot)[0][0],np.where(freqs==cfg.freq_max_plot)[0][0]]
    for condition in cfg.condition:
        datasets = []
        for hemi in cfg.hemisphere:
            df_hemi = df[df["hemisphere"]==hemi]
            #plot non-normalised power
            df_to_plot = df_hemi[(df_hemi['Condition']==condition)]
            if df_to_plot.empty:
                print(f"No data for {condition} in {hemi}. Skipping...")
                continue
            avg_data = np.stack(df_to_plot['power'].values).mean(axis=0)
            sliced_data = avg_data[freq_to_plot[0]:freq_to_plot[1],time_to_plot[0]:time_to_plot[1]]
            datasets.append(sliced_data)
        titles = ['Left Hemisphere', 'Right Hemisphere']

        fig1, name = plot_time_frequency(time_for_plot,freqs[freq_to_plot[0]:freq_to_plot[1]],condition,cfg.output_dir,datasets,titles,add_vlines= cfg.vlines)
        title = '_'.join([id,condition,'power'])
        pics.append([fig1,title,condition])
    return pics


def plot_participant_pacs(df,id):
    pics = []
    low_fq_range = df["low_freqs"].values[0]
    high_fq_range = df["high_freqs"].values[0]
    for condition in cfg.condition:
        datasets = []
        for hemi in cfg.hemisphere:
            df_hemi = df[df["hemisphere"]==hemi]
            #plot non-normalised power
            df_to_plot = df_hemi[(df_hemi['Condition']==condition)]
            if df_to_plot.empty:
                print(f"No data for {condition} in {hemi}. Skipping...")
                continue
            avg_data = np.stack(df_to_plot['pac'].values).mean(axis=0)
            datasets.append(avg_data)
        titles = ['Left Hemisphere', 'Right Hemisphere']
        
        fig1, name = plot_pac(low_fq_range,high_fq_range,condition,cfg.output_dir,datasets,titles,hemi=hemi)
        title = '_'.join([id,condition,'pac'])
        pics.append([fig1,title,condition])
    return pics

def add_pacs_to_report(df,report,id):
    low_fq_range = df["low_freqs"].values[0]
    high_fq_range = df["high_freqs"].values[0]
    for condition in cfg.condition:
        image_names = []
        for hemi in cfg.hemisphere:
            df_hemi = df[df["hemisphere"]==hemi]
            datasets = []
            for diagnosis in cfg.diagnoses:
                df_to_plot = df_hemi[(df_hemi['Diagnosis'] == diagnosis) & (df_hemi['Condition']==condition)]
                if df_to_plot.empty:
                    print(f"No data for {diagnosis} in {condition} for {hemi}. Skipping...")
                    continue
                avg_data = np.stack(df_to_plot['pac'].values).mean(axis=0)
                datasets.append(avg_data)
            titles = [cfg.diagnoses[diag]['label_n'] for diag in cfg.diagnoses]

            fig1, name = plot_pac(low_fq_range,high_fq_range,condition,cfg.output_dir,datasets,titles,hemi=hemi)
            title = '_'.join([id,condition,'pac'])
            #save fig
            fig_to_save = fig1.get_figure()
            fig_to_save.savefig(name.replace('.tiff','.svg'),format="svg")
            fig_to_save.savefig(name,dpi=300)
            image_names.append(name)
            plt.close()
        fig = plt.figure(figsize=(18,6), layout='constrained')
        gs  = GridSpec(1, 2, figure=fig) 
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[0,1])
        ax1.imshow(plt.imread(image_names[0]))
        ax1.axis('off')
        ax2.imshow(plt.imread(image_names[1]))
        ax2.axis('off')
        title = '_'.join([id,condition,'power'])
        report.add_figure(fig=fig, title=title, section=id, tags=[condition,'pac'],replace=True)
        report.save(cfg.report_savename_hdf5, verbose=False, overwrite=True)
        plt.close('all')

def plot_participant_labels_on_fsaverage(df,hemi):
    brain = mne.viz.Brain(subject = 'fsaverage',hemi = hemi ,views = cfg.brain_view,subjects_dir = cfg.fsaverageDir,surf='inflated',background='white')
    if cfg.analysis_type == 'connectivity':
        df_hemi = df[df["target_hemi"]==hemi]
    else:
        df_hemi = df[df["hemisphere"]==hemi]

    drawn_labels = df_hemi["morphed_label"].dropna().values
    for info in drawn_labels:
        if info:
            brain.add_label(info[0], hemi = hemi, alpha=0.75)
    return brain
def plot_participant_labels_on_brain(subjID_date,hemi):
    brain = mne.viz.Brain(subject = subjID_date,hemi = hemi ,views = cfg.brain_view,subjects_dir = cfg.subj_dir,surf='inflated',background='white')
    brain.add_label(cfg.peak_labels_hemis[hemi], hemi = hemi, alpha=0.75)
    return brain

def add_fsaverage_to_report(report,df,label_name,seed_hemi=None):
    fig = plt.figure(figsize=(12,6), layout='constrained')
    gs  = GridSpec(1, 2, figure=fig) 
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    brain1 = plot_participant_labels_on_fsaverage(df,'lh')
    brain_fig_name = 'lh_brain.tiff'
    brain_image_name = os.path.join(cfg.output_dir,brain_fig_name)
    brain1.save_image(filename=brain_image_name, mode='rgb')
    ax1.imshow(plt.imread(brain_image_name))
    ax1.set_title('Labels in left hemisphere')
    ax1.axis('off')
    brain2 = plot_participant_labels_on_fsaverage(df,'rh')
    brain_fig_name = 'rh_brain.tiff'
    brain_image_name = os.path.join(cfg.output_dir,brain_fig_name)
    brain2.save_image(filename=brain_image_name, mode='rgb')
    ax2.imshow(plt.imread(brain_image_name))
    ax2.set_title('Labels in right hemisphere')
    ax2.axis('off')

    #save fig
    fig_to_save = fig.get_figure()
    title_filtered = [x for x in [label_name,seed_hemi] if x is not None]
    savetitle = '_'.join(title_filtered)
    savename = os.path.join(cfg.output_dir, savetitle + ".tiff")
    fig_to_save.savefig(savename,dpi=300)
    plt.close()
    report.add_figure(fig=fig, title=savetitle, section='gavg', tags=label_name,replace=True)
    report.save(cfg.report_savename_hdf5, verbose=False, overwrite=True)



def plot_stc(id,stc,hemi,initial_time,annot_label,condition,label_color,other_hemi=None):
    brain = stc.plot(
    subjects_dir='/autofs/space/transcend/MRI/WMA/recons/',
    hemi=hemi,
    initial_time=initial_time,
    clim=dict(kind="percent", lims=[99.5, 99.7, 99.9]),
    smoothing_steps=7,
    views = cfg.brain_view,
    time_viewer = False
    )
    brain.add_label(annot_label, hemi = hemi, alpha=0.75,color = label_color)
    brain.add_text(0.1, 0.9, condition, "title", font_size=16)
    if other_hemi is None:
        other_hemi = hemi
    brain_image_name = os.path.join(cfg.output_dir,'_'.join([id,condition,hemi, other_hemi, "brain.tiff"]))
    brain.save_image(filename=brain_image_name, mode='rgb')
    brain.close()

    return brain_image_name

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def plot_line(df,variable,ax,label,color,freqs = None,ci=False,group=False):
    time = df['time'].values[0]
    time_to_plot = [find_nearest(time,cfg.tmin_plot),find_nearest(time,cfg.tmax_plot)]
    time_for_plot = time[time_to_plot[0]:time_to_plot[1]]
    if freqs is not None:
        freq_to_plot = [np.where(np.arange(cfg.freq_min,cfg.freq_max+1,1)==freqs[0])[0][0],np.where(np.arange(cfg.freq_min,cfg.freq_max+1,1)==freqs[1])[0][0]]

    if not group:
        data_to_plot = df[variable].mean().squeeze()
        if freqs is not None:
            data = np.mean(data_to_plot[freq_to_plot[0]:freq_to_plot[1],time_to_plot[0]:time_to_plot[1]],axis=0)
        else:  
            data = data_to_plot[time_to_plot[0]:time_to_plot[1]]
        ax.plot(time_for_plot,data, color=color, label=label)

    else:
        data_to_plot = np.stack(df[variable].values).squeeze()
        if freqs is not None:
            data = np.mean(data_to_plot[:,freq_to_plot[0]:freq_to_plot[1],time_to_plot[0]:time_to_plot[1]],axis=1)
        else: 
            data = data_to_plot[:,time_to_plot[0]:time_to_plot[1]]
        ax.plot(time_for_plot,np.mean(data,axis=0), color=color, label=label)

    if ci:
        cis = st.norm.interval(confidence=cfg.confidence,
                            loc=np.mean(data, axis=0),
                            scale=st.sem(data, axis=0))
        ax.fill_between(time_for_plot, cis[0], cis[1], color=color, alpha=.1)

    return ax

def compute_sig_mask(df, factor_name, variable='stc', alpha=0.05, paired=False):
    """
    Computes a significance mask across time points between levels of a categorical variable.

    For 2 levels: uses independent (or paired) t-test.
    For >2 levels: uses one-way ANOVA.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the time series data.
    factor_name : str
        Column name for the categorical variable (e.g., 'Condition').
    levels : list of str
        Levels to compare (e.g., ['Search', 'Pop-out'] or ['ASD', 'TD', 'Other']).
    variable : str
        Column name containing 1D time series arrays (e.g., 'stc').
    alpha : float
        Significance threshold.
    paired : bool
        If True and len(levels) == 2, performs a paired t-test (requires matching subjects in order).

    Returns:
    --------
    sig_mask : np.ndarray of bool
        Boolean array indicating where the comparison is significant (p < alpha).
    p_vals : np.ndarray of float
        P-values at each time point.
    """
    data_by_level = []

    for level in cfg.df_varnames[factor_name]:
        series_list = df[df[factor_name] == level][variable].to_list()
        arr = np.vstack(series_list)  # shape: (n_subjects, n_times)
        data_by_level.append(arr)

    n_times = data_by_level[0].shape[1]

    if len(levels) == 2:
        data1, data2 = data_by_level
        if paired:
            from scipy.stats import ttest_rel
            t_stats, p_vals = ttest_rel(data1, data2, axis=0)
        else:
            t_stats, p_vals = ttest_ind(data1, data2, axis=0, equal_var=False)
    else:
        # ANOVA for each time point
        p_vals = np.array([
            f_oneway(*(group[:, t] for group in data_by_level))[1]
            for t in range(n_times)
        ])
    rej, p_vals_corr = fdrcorrection(p_vals, alpha=alpha, method='indep', is_sorted=False)
    sig_mask = rej #p_vals < alpha
    time = df['time'].values[0]
    time_to_plot = [find_nearest(time,cfg.tmin_plot),find_nearest(time,cfg.tmax_plot)]
    time_for_plot_sig_mask = sig_mask[time_to_plot[0]:time_to_plot[1]]
    return time_for_plot_sig_mask, p_vals_corr

def add_significance_bar(ax, sig_mask, times, y_offset=0.02, bar_height=8, color='black'):
    """
    Adds a horizontal bar just above the x-axis to indicate significant time windows.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis to draw on.
    sig_mask : array-like of bool
        Boolean array indicating significant time points.
    times : array-like
        Time points corresponding to sig_mask.
    y_offset : float
        Offset (in axis units) above the bottom y-limit to place the bar.
    bar_height : float
        Line width of the significance bar.
    color : str
        Color of the significance bar.
    """
    if sig_mask is None or times is None:
        return

    y_min, y_max = ax.get_ylim()
    bar_y = y_min + y_offset * (y_max - y_min)

    for i in range(len(times) - 1):
        if sig_mask[i]:
            ax.hlines(y=bar_y, xmin=times[i], xmax=times[i + 1],
                      color=color, linewidth=bar_height)
            
def plot_activations(df,plot_title,tag,factor,factor_name_in_df,group=False,ci=False,add_vlines = None,sig_mask=None,paired = True):
    #plot activations in time window
    SMALL_SIZE = 22
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=SMALL_SIZE)
    time = df['time'].values[0]
    time_to_plot = [find_nearest(time,cfg.tmin_plot),find_nearest(time,cfg.tmax_plot)]
    time_for_plot = time[time_to_plot[0]:time_to_plot[1]]
    sub_fig,sub_ax1 = plt.subplots(figsize=(6.4,4.8), layout='constrained')
    for level in factor:       
        df_to_plot = df[(df[factor_name_in_df]==level)]
        sub_ax1 = plot_line(df_to_plot,'stc',sub_ax1,level,cfg.color_dict[level],ci=ci,group=group)
    sub_ax1.legend(fontsize = 16)
    if group:
        sub_ax1.set_ylim(cfg.ylims)
    sub_ax1.set_xlabel('Time (s)',fontsize=cfg.fontsize)
    sub_ax1.set_ylabel('dSPM activation (AU)',fontsize=cfg.fontsize)
    sub_ax1.axvline(x=0, ls='--', color='k')
    if add_vlines is not None:
        for vline in add_vlines:
            sub_ax1.axvline(x=vline, ls='--', color='k')
    if sig_mask is not None:
        sig_mask, p_vals = compute_sig_mask(df, factor_name_in_df,variable='stc', alpha=cfg.alpha, paired=paired)
    add_significance_bar(sub_ax1, sig_mask, time_for_plot)

    title = plot_title + tag + ' \n '
    sub_ax1.set_title(title,fontsize=16)
    
    savetitle = 'activations_plot' + tag
    savename = os.path.join(cfg.output_dir, savetitle + ".tiff")
    sub_fig.savefig(savename,dpi=300)
        
    return sub_fig,savename

def plot_participant_activations(df):
    pics = []
    for hemi in cfg.hemisphere:
        df_hemi = df[df["hemisphere"]==hemi]
        hemi_tag = ' (' + hemi.upper() + ')'
        #plot time series activations   
        fig1,filename= plot_activations(df_hemi,'Activations from ',df['Participant'].values[0] + hemi_tag,cfg.plot_selected_conditions,'Condition', 
                                        group=False,add_vlines=cfg.vlines,sig_mask=None,paired=None)
        fig = plt.figure(figsize=(18,6), layout='constrained')
        gs  = GridSpec(1, 3, figure=fig) 
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[0,1])
        ax3 = fig.add_subplot(gs[0,2])
        brain1_image_name = df_hemi[df_hemi['Condition'] == cfg.brain_selected_conditions[0]]['brain_image'].values[0]
        brain2_image_name = df_hemi[df_hemi['Condition'] == cfg.brain_selected_conditions[1]]['brain_image'].values[0]
        ax1.imshow(plt.imread(brain1_image_name))
        ax1.axis('off')
        ax2.imshow(plt.imread(brain2_image_name))
        ax2.axis('off')
        ax3.imshow(plt.imread(filename))
        ax3.axis('off')
        pics.append([fig,hemi])
    return pics

def add_gavg_activations_to_report(df,report,id,grouping_factor,factor_name_in_df):
    for factor_level in grouping_factor:
        filenames_hemis = {}
        for hemi in cfg.hemisphere:
            hemi_tag = ' (' + hemi.upper() + ')'
            df_to_plot = df[(df[factor_name_in_df] == factor_level) & (df['hemisphere'] == hemi)]
            if factor_name_in_df == 'Diagnosis':
                fig1,filename= plot_activations(df_to_plot,'Activations from ',factor_level.upper() + hemi_tag,cfg.plot_selected_conditions,'Condition',
                                                group=True,ci = True,add_vlines=cfg.vlines,sig_mask=True,paired=True)
            elif factor_name_in_df == 'Condition':
                fig1,filename= plot_activations(df_to_plot,'Activations from ',factor_level.capitalize() + hemi_tag,cfg.diagnoses,'Diagnosis',
                                                group=True,ci = True,add_vlines=cfg.vlines,sig_mask=True,paired=False)
            filenames_hemis.update({hemi:filename})
        fig = plt.figure(figsize=(18,6), layout='constrained')
        gs  = GridSpec(1, 2, figure=fig) 
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[0,1])
        image_name1 = filenames_hemis['lh']
        ax1.imshow(plt.imread(image_name1))
        ax1.axis('off')
        image_name2 = filenames_hemis['rh']
        ax2.imshow(plt.imread(image_name2))
        ax2.axis('off')
        title = '_'.join([id,factor_level,'activations'])

        report.add_figure(fig=fig, title=title, section='_'.join([id,factor_level]), tags=[hemi,'activations'],replace=True)
        plt.close('all')
        report.save(cfg.report_savename_hdf5, verbose=False, overwrite=True)

def find_strongest_sensor(sensor_data,filt):
    strongest_sensors = {'mag':{},
                         'grad':{}}
    filt_sensors = sensor_data.filter(1,filt)
    for ch_type in [*strongest_sensors]:
        for hemi in cfg.sensor_hemis:
            epochs_data = filt_sensors[hemi]
            max_values = []
            for ep_i in range(0,len(epochs_data)):
                    evoked = epochs_data[ep_i].average()
                    ch, lat, amp = evoked.get_peak(
                        ch_type=ch_type, tmin=-0.5, tmax=0, mode="abs", return_amplitude=True
                        )
                    max_values.append([ch,amp])
            chs = list(list(zip(*max_values))[0])
            data = Counter(chs)
            strongest_sensor_in_hemisphere = data.most_common(1)[0][0]
            strongest_sensors[ch_type].update({hemi:strongest_sensor_in_hemisphere})

    return strongest_sensors

def erpimage_by_peak_latency(epochs,sensor):
    max_values_lat=[]
    for i in range(0,len(epochs)):
        evoked = epochs[i].pick(sensor).average()
        ch, lat, amp = evoked.get_peak(
            ch_type=None, tmin=-0.5, tmax=0, mode="abs", return_amplitude=True
            )
    max_values_lat.append(lat)
    order = np.argsort(max_values_lat,axis=0)
    epochs.plot_image(picks=[sensor],order=order)

def get_tri_from_SFive(csvfile,subject):
    """
    Get diagnosis from the Misophonia report (ID:140542) from RedCap. Columns have been specified to this report.

    Parameters
    ----------
    subject : str
        subject ID
    """

    # diagnosis
    subject = int(subject)
    self_report_diagnosis = []
    trigger_relative_intensity = csvfile["Trigger Relative Intensity"][csvfile['Subject ID:'] == subject].dropna()
    diagnosis = []
    for i in list(csvfile.columns[4:6]):
        this_val = csvfile[i][csvfile['Subject ID:'] == subject].dropna()
        if not this_val.empty:
            diagnosis.append(this_val.values)
    if not diagnosis:
        self_report_diagnosis = np.nan
    else:
        self_report_diagnosis = 'misophonia' if 'Yes' in diagnosis else 'td'

    if not trigger_relative_intensity.empty: 

        trigger_relative_intensity = csvfile["Trigger Relative Intensity"][csvfile['Subject ID:'] == subject].dropna().values[0]
        triggers_greater_than_0 = csvfile["Number of triggers greater than 0"][csvfile['Subject ID:'] == subject].dropna().values[0]
        triggers_greater_than_8 = csvfile["Number of triggers greater than or equal to 8"][csvfile['Subject ID:'] == subject].dropna().values[0]
        if trigger_relative_intensity > 6 or (trigger_relative_intensity < 6 and triggers_greater_than_0 > 25 and triggers_greater_than_8 >=5):
            sfive_diagnosis = 'misophonia'
        else:
            sfive_diagnosis = 'td'
    else:
        sfive_diagnosis = np.nan
        trigger_relative_intensity  = np.nan

    return self_report_diagnosis,sfive_diagnosis,trigger_relative_intensity

def get_average_freq_power(power,times,freqs,time_window):
    freq_range = np.arange(cfg.freq_min,cfg.freq_max,1)
    freqs2plot = (freq_range >= freqs[0]) & (freq_range <= freqs[1])
    times2plot = (times >= time_window[0]) & (times <= time_window[1])
    tw_of_interest = np.mean(power[:, times2plot],axis=1)
    freq_of_interest = np.mean(tw_of_interest[freqs2plot])

def get_coh_stc(df,plot_type):
    con_res = df['connectivity_data'].values[0]
    con_res_times = df['time'].values[0]
    con_res_vertices = df['vertices'].values[0]
    con_subjID_date = df['SubjID_Date'].values[0]
    cwt_freqs = np.arange(cfg.freq_min, cfg.freq_max+1, 1)

    if plot_type == 'time':
        tmin  = con_res_times[0] 
        tstep = con_res_times[1] - tmin
        coh_to_plot = np.mean(con_res[:,(cwt_freqs >= 10) & (cwt_freqs <= 13),:], axis=1) #(cwt_freqs >= 10) & (cwt_freqs <= 13)
    elif plot_type == 'freq':
        tmin  = np.mean(cwt_freqs[0])
        tstep = np.mean(cwt_freqs[1]) - tmin
        times2plot = (np.array(con_res_times) >= 0.5) & (np.array(con_res_times) <= 1.0)
        coh_to_plot = np.mean(con_res[:,:,times2plot], axis=2)
    coh_stc = mne.SourceEstimate(
        coh_to_plot,
        vertices=con_res_vertices,
        tmin=tmin,
        tstep=tstep,
        subject=con_subjID_date
    )
    
    return coh_stc, con_subjID_date
        

def plot_coh_conditions(ax,df,tag,factor,factor_name_in_df,ylims,ci=False,group=False,plot_title=None,highlight_area=False,area_time_window=None,zcoh=False):

    for level in factor:       
        df_to_plot = df[(df[factor_name_in_df]==level)]
        ax = plot_line(df_to_plot,'stc',ax,cfg.plot_labels[level]['label'],cfg.color_dict[level],ci=ci,group=group)
    ax.legend(fontsize = 16)
    ax.set_ylim(ylims)
    ax.set_xlim(left=0)
    ax.set_xlabel('Time (s)',fontsize=cfg.fontsize)
    if zcoh:
        ax.set_ylabel('Coherence (z)',fontsize=cfg.fontsize)
    else:
        ax.set_ylabel('Coherence',fontsize=cfg.fontsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.axvline(x=0, ls='--', color='k')
    if highlight_area:
        ax.axvline(x=area_time_window[0], ls='--', color='k')
        ax.axvline(x=area_time_window[1], ls='--', color='k')
    if plot_title:
        title = plot_title + tag + ' \n '
        ax.set_title(title,fontsize=16)

    return ax

def plot_power_over_time(ax,df,tag,factor_name_in_df,freqs,group=False,highlight_area=False,area_time_window=None,plot_title=None,ylims=None,ci=False,sig_mask=None,paired=True):
    time = df['time'].values[0]
    time_to_plot = [find_nearest(time,cfg.tmin_plot),find_nearest(time,cfg.tmax_plot)]
    time_for_plot = time[time_to_plot[0]:time_to_plot[1]]
    for level in cfg.df_varnames[factor_name_in_df]:       
        df_to_plot = df[(df[factor_name_in_df]==level)]
        ax = plot_line(df_to_plot,'power',ax,cfg.plot_labels[level]['label'],cfg.color_dict[level],freqs=freqs,ci=True,group=group)
        ax.legend(fontsize = 16)
    ax.set_ylim(ylims)
    # ax.set_xlim(left=0)
    ax.set_xlabel('Time (s)',fontsize=cfg.fontsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axvline(x=0, ls='--', color='k')
    ax.axvline(x=0.8, ls='--', color='k')
    if highlight_area:
        ax.axvline(x=area_time_window[0], ls='--', color='k')
        ax.axvline(x=area_time_window[1], ls='--', color='k')
    if plot_title:
        title = plot_title + tag + ' \n '
        ax.set_title(title,fontsize=16)
    if sig_mask is not None:
        sig_mask, p_vals = compute_sig_mask(df, factor_name_in_df, variable='stc', alpha=cfg.alpha, paired=paired)
    add_significance_bar(ax, sig_mask, time_for_plot)

    return ax

def add_gavg_power_over_time_to_report(df,report,id,factor_name_in_df,freq_label,freqs,ci=False):

    plot_title = freq_label + ' power in ' 
    ylims = cfg.power_line_plot_ylims
    report_tag = freq_label + '_power'
    grouping_factor = cfg.df_varnames[factor_name_in_df]
    for factor_level in grouping_factor:
        fig = plt.figure(figsize=(12,6), layout='constrained')
        gs  = GridSpec(1, 2, figure=fig) 
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[0,1])
        hemi_axes={'lh':ax1,
                    'rh':ax2}
        for hemi in cfg.hemisphere:
            hemi_tag = ' (' + hemi.upper() + ')'
            df_to_plot = df[(df[factor_name_in_df] == factor_level) & (df['hemisphere'] == hemi)]
            if factor_name_in_df == 'Condition':
                plot_power_over_time(hemi_axes[hemi],df_to_plot,factor_level + hemi_tag,'Diagnosis',freqs, ylims=ylims,group=True,plot_title=plot_title,ci=ci)
            elif factor_name_in_df == 'Diagnosis':
                plot_power_over_time(hemi_axes[hemi],df_to_plot,factor_level + hemi_tag,'Condition',freqs, ylims=ylims,group=True,plot_title=plot_title, ci=ci)

        title = '_'.join([id,factor_level,report_tag])

        report.add_figure(fig=fig, title=title, section='_'.join([id,factor_level]), tags=[hemi,report_tag],replace=True)
        report.save(cfg.report_savename_hdf5, verbose=False, overwrite=True)

        plt.close('all')

def add_participant_coh_to_report(df,report,id):
    diagnosis = df["Diagnosis"].values[0]
    for hemi in cfg.hemisphere:
        hemi_tag = ' (' + hemi.upper() + ')'
        #plot time series activations   
        fig = plt.figure(figsize=(18,6), layout='constrained')
        gs  = GridSpec(1, 4, figure=fig) 
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[0,1])
        ax3 = fig.add_subplot(gs[0,2])
        ax4 = fig.add_subplot(gs[0,3])
        brain1_image_name = os.path.join(cfg.output_dir,'_'.join([cfg.selected_conditions[0],hemi,hemi, "brain.tiff"]))
        ax1.imshow(plt.imread(brain1_image_name))
        ax1.axis('off')
        other_hemi = [x for x in cfg.hemisphere if x not in hemi][0]
        other_hemi_tag = ' (' + other_hemi.upper() + ')'
        brain2_image_name = os.path.join(cfg.output_dir,'_'.join([cfg.selected_conditions[0],other_hemi, hemi, "brain.tiff"]))
        ax2.imshow(plt.imread(brain2_image_name))
        ax2.axis('off')
        df_to_plot = df[(df["hemisphere"]==hemi) & (df["target_hemi"]==hemi)]
        ax3 = plot_coh_conditions(ax3,df_to_plot,df['Participant'].values[0] + hemi_tag + ' to ' + hemi_tag,cfg.selected_conditions,'Condition', cfg.ylims, group=False,plot_title='Coherence from ')
        df_to_plot = df[(df["hemisphere"]==hemi) & (df["target_hemi"]==other_hemi)]
        ax4 = plot_coh_conditions(ax4,df_to_plot,df['Participant'].values[0] + hemi_tag+ ' to' + other_hemi_tag,cfg.selected_conditions,'Condition', cfg.ylims, group=False,plot_title='Coherence from ')
        title = '_'.join([id,hemi,'coherence'])
        report.add_figure(fig=fig, title=title, section=id, tags=[hemi,'coherence',diagnosis],replace=True)
        plt.close('all')
    
def add_gavg_coh_to_report(df,report,id,factor_name_in_df,zcoh=False,ci=False):
    if zcoh:
        plot_title = 'Normalised coherence from '
        ylims = cfg.zcoh_ylims
        report_tag = 'z_coherence'
        grouping_factor = cfg.df_varnames[factor_name_in_df][:-1] #assumes variable to normalise against is always last
    else:
        plot_title = 'Coherence from '
        ylims = cfg.ylims
        report_tag = 'coherence'
        grouping_factor = cfg.df_varnames[factor_name_in_df]
    for factor_level in grouping_factor:
        fig = plt.figure(figsize=(12,12), layout='constrained')
        gs  = GridSpec(2, 2, figure=fig) 
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[0,1])
        ax3 = fig.add_subplot(gs[1,0])
        ax4 = fig.add_subplot(gs[1,1])
        hemi_axes={'lh':{'lh':ax1,
                         'rh':ax2},  
                    'rh':{'rh':ax3,
                         'lh':ax4}}
        for hemi in cfg.hemisphere:
            for target_hemi in cfg.hemisphere:
                hemi_tag = ' (' + hemi.upper() + ')'
                target_hemi_tag = ' (' + target_hemi.upper() + ')'
                df_to_plot = df[(df[factor_name_in_df] == factor_level) & (df['hemisphere'] == hemi) & (df['target_hemi'] == target_hemi)]
                if factor_name_in_df == 'Condition':
                    plot_coh_conditions(hemi_axes[hemi][target_hemi],df_to_plot,factor_level + hemi_tag + ' to' + target_hemi_tag,cfg.diagnoses,'Diagnosis',ylims,group=True,plot_title=plot_title,ci=ci)
                elif factor_name_in_df == 'Diagnosis':
                    plot_coh_conditions(hemi_axes[hemi][target_hemi],df_to_plot,factor_level + hemi_tag + ' to' + target_hemi_tag,cfg.selected_conditions,'Condition',ylims,group=True,plot_title=plot_title, ci=ci)

        title = '_'.join([id,factor_level,report_tag])

        report.add_figure(fig=fig, title=title, section='_'.join([id,factor_level]), tags=[hemi,report_tag],replace=True)
        plt.close('all')
        
def get_zcoh(df_hemi,condition_of_interest,normalising_condition):
    df_condition_of_interest = df_hemi[(df_hemi['Condition']==condition_of_interest)]
    condition_of_interest_data = df_condition_of_interest['stc'].values[0] #get data of interest
    df_normalising_condition = df_hemi[(df_hemi['Condition']==normalising_condition)] #get data to normalise against
    normalising_condition_data = df_normalising_condition['stc'].values[0] #get data of interest

    condition_of_interest_epoch_count = df_condition_of_interest['n_epochs'].values[0]
    normalising_condition_epoch_count = df_normalising_condition['n_epochs'].values[0]

    z_coh  = ( (np.emath.arctanh(np.abs(condition_of_interest_data))      - (1 / (condition_of_interest_epoch_count-2)))  -
               (np.emath.arctanh(np.abs(normalising_condition_data)) - (1 / (normalising_condition_epoch_count-2))))  / np.sqrt((1/(condition_of_interest_epoch_count-2))+(1/(normalising_condition_epoch_count-2)))
    return z_coh

def barplot_with_swarmplot(ax,df,x_var,y_var):

    if np.unique(df[x_var].values)[0] in cfg.diagnoses:
        palette = [cfg.color_dict["misophonia"],cfg.color_dict["td"]]
        order = ['misophonia','td']
        hue_order = ['misophonia','td']
        labels = ['Misophonia','Controls']
    elif np.unique(df[x_var].values)[0] in cfg.selected_conditions:
        palette = [cfg.color_dict["miso"],cfg.color_dict["sound2"],cfg.color_dict["white_noise"]]
        order = cfg.selected_conditions
        hue_order = cfg.selected_conditions
        labels=['Trigger','Neutral','White Noise']
    with sns.axes_style('ticks'):
        ax = sns.barplot(x=x_var, y=y_var, data=df, capsize=.1, errorbar='se', hue = x_var, ax = ax,
                         palette = palette,width = 0.6,order= order,hue_order= hue_order)
        ax = sns.swarmplot(x=x_var, y=y_var, data=df, color="0", alpha=.35,size = 9,order = order,ax=ax)
        ax.set_title('',fontsize = cfg.fontsize, fontweight="normal")
        ax.set_xlabel('',fontsize = cfg.fontsize, fontweight="normal")
        ax.set_ylabel('Coherence (z)',fontsize = cfg.fontsize, fontweight="normal")
        plt.yticks(fontsize = 16, weight = "normal")
        ax.set_xticklabels(labels=labels)
        plt.xticks(fontsize = 14, weight = "normal",rotation = 45,ha = 'right')
        sns.despine()
    
    return ax

def add_connectivity_plot(df,report,timings,hemi,target_hemi,factor_name_in_df,id,zcoh=False):
    if zcoh:
        plot_title = 'Normalised coherence from '
        ylims = cfg.zcoh_ylims
        y_name = 'Coherence (z)'
        report_tag = 'z_coherence_combined'
        grouping_factor = cfg.df_varnames[factor_name_in_df][:-1] #assumes variable to normalise against is always last
    else:
        plot_title = 'Coherence from '
        ylims = cfg.ylims
        y_name = 'Coherence'
        report_tag = 'coherence_combined'
        grouping_factor = cfg.df_varnames[factor_name_in_df]

    times = np.array(df['time'].values[0])
    times2plot = (times >= timings[0]) & (times <= timings[1])
    result = [np.mean(x[times2plot]) for x in df['stc']]
    df[y_name] = result

    for factor_level in grouping_factor:
        fig = plt.figure(figsize=(10,4.8), layout='constrained')
        gs  = GridSpec(1, 4, figure=fig) 
        ax1 = fig.add_subplot(gs[0,0:3])
        ax2 = fig.add_subplot(gs[0,3])

        df_to_plot = df[(df[factor_name_in_df] == factor_level) & (df['hemisphere'] == hemi) & (df['target_hemi'] == target_hemi)]
        hemi_tag = ' (' + hemi.upper() + ')'
        target_hemi_tag = ' (' + target_hemi.upper() + ')'
        if factor_name_in_df == 'Condition':
            ax1= plot_coh_conditions(ax1,df_to_plot,cfg.plot_labels[factor_level]['label'] + hemi_tag + ' to' + target_hemi_tag,cfg.diagnoses,'Diagnosis', (-1.2,1.2), 
                        plot_title=plot_title, ci = False,group=True,highlight_area=True,area_time_window=timings,zcoh=zcoh)
            ax2 = barplot_with_swarmplot(ax2,df_to_plot,'Diagnosis',y_name)

        elif factor_name_in_df == 'Diagnosis':
            ax1= plot_coh_conditions(ax1,df_to_plot,cfg.plot_labels[factor_level]['label'] + hemi_tag + ' to' + target_hemi_tag,cfg.selected_conditions,'Condition', (0.1,0.7), 
                        plot_title=plot_title, ci = False,group=True,highlight_area=True,area_time_window=timings,zcoh=zcoh)
            ax2 = barplot_with_swarmplot(ax2,df_to_plot,'Condition',y_name)

        title = '_'.join([id,report_tag,factor_level,hemi,target_hemi,str(timings[0]),str(timings[1])])
        savetitle = title + report_tag
        savename = os.path.join(cfg.output_dir, savetitle + ".tiff")
        fig.savefig(savename,dpi=300)
        report.add_figure(fig=fig, title=title, section=id, tags=[hemi,target_hemi,report_tag],replace=True)

def initialize_error_log(filepath='error_log.csv'):
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'error_message'])

def attenvis_metadata(events,sfreq,mat_file,participant,locked_to = 'stimuli'):
    find_response_triggers(events)
    error_log = []
    if locked_to == 'response':
        row_events = [ "response/left","response/right"]
        keep_last = ["condition"]
        metadata_tmax = ["condition/search/4", "condition/search/6", "condition/search/8", "condition/search/10","condition/pop-out/4", "condition/pop-out/6", "condition/pop-out/8", "condition/pop-out/10"]
        metadata, events_meta, event_id_meta = mne.epochs.make_metadata(
            events=events,
            event_id=cfg.event_dict,
            tmin=metadata_tmax,
            tmax= row_events,
            sfreq=sfreq,
            row_events=row_events,
            keep_last=keep_last,
        )  
        metadata[['Condition','difficulty']] = metadata['last_condition'].str.split('/',expand = True)
        metadata['RT'] = metadata['condition'] #this will get RT as a negative number because it is counting backwards from the response, but you shouldn't rely on this metadata for calculating RTs. Only calculate RTs with stimuli_locked epochs.

    else:
        row_events = ["target"]
        keep_first = ["condition","response"]
        metadata_tmax = ["response/left", "response/right"]

        metadata, events_meta, event_id_meta = mne.epochs.make_metadata(
            events=events,
            event_id=cfg.event_dict,
            tmin=None,
            tmax=metadata_tmax,
            sfreq=sfreq,
            row_events=row_events,
            keep_first=keep_first,
        )  
        metadata[['Condition','difficulty']] = metadata['first_condition'].str.split('/',expand = True)
        metadata['RT'] = metadata['response'] - metadata['condition']
    metadata.reset_index(drop=True,inplace=True)
    metadata = metadata.drop(columns=['condition/search/4','condition/search/6','condition/search/8','condition/search/10','condition/pop-out/4','condition/pop-out/6','condition/pop-out/8','condition/pop-out/10','target','response/right','response/left'])
    try: 
        mat = scipy.io.loadmat(mat_file)
        msg = 'Loading ' + mat_file
        print(msg)
        error_log.append(msg)
    except:
        msg = 'No behaviour mat file found'
        print(msg)
        error_log.append(msg)
        mat = None
    if len(mat['correctTrials'][0]) == len(metadata):
        metadata['correct'] = mat['correctTrials'][0]
    elif os.path.split(mat_file)[1] == '057101_AttenVis_run03_behaviour.mat':
        metadata['correct'] = mat['correctTrials'][0][8:]
    elif os.path.split(mat_file)[1] == '086901_AttenVis_run03_behaviour.mat':
        metadata['correct'] = mat['correctTrials'][0][5:]
    elif os.path.split(mat_file)[1] == '098101_AttenVis_run01_behaviour.mat' and locked_to == 'stimuli':
        metadata['correct'] = mat['correctTrials'][0][:-2]
    elif os.path.split(mat_file)[1] == '098101_AttenVis_run01_behaviour.mat' and locked_to == 'response':
        metadata['correct'] = mat['correctTrials'][0][:-3]
    elif os.path.split(mat_file)[1] == '101901_AttenVis_run01_behaviour.mat':
        metadata['correct'] = mat['correctTrials'][0][4:]  
    elif os.path.split(mat_file)[1] == '109101_AttenVis_run03_behaviour.mat':
        metadata['correct'] = mat['correctTrials'][0][3:]        
    else:
        msg = 'Incorrect number of trials. Adding empty correct column'
        print(msg)
        error_log.append(msg)
        metadata['correct'] = np.nan
    if metadata[metadata[['RT']].isna().any(axis=1)].empty:
        msg = 'No missing values in RT'
        print(msg)
        error_log.append(msg)
    else:
        msg = metadata[metadata[['RT']].isna().any(axis=1)]
        print(msg)
        error_log.append(msg)

    with open('error_log.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        for msg in error_log:
            writer.writerow([participant,msg])
    return metadata, events_meta, event_id_meta

def clean_metadata(dataset,rt_based = None, percent = None,correct_answers_only = False):
    difficulty_order = ['4','6','8','10']
    dataset['difficulty'] = pd.Categorical(dataset['difficulty'], categories=difficulty_order, ordered=True)
    condition_order = ['pop-out','search']
    dataset['Condition'] = pd.Categorical(dataset['Condition'], categories=condition_order, ordered=True)
    if correct_answers_only:
        if dataset['correct'].isnull().all():
            correct_answers = dataset
        else:
            correct_answers = dataset.loc[dataset['correct']==1,:]
    else:
        correct_answers = dataset
    if rt_based:
        correct_answers_lower_bound = correct_answers.loc[(correct_answers["RT"] < rt_based[1]) & (correct_answers["RT"]> rt_based[0])] #usually 100ms, so 0.1 
    else: 
        correct_answers_lower_bound = correct_answers.loc[correct_answers["RT"]>0.1] # will always remove improbable responses, usually 100ms, so 0.1 
    if percent:  
        convert_percent = 1-percent
        all_conditions_difficulties = []
        for level in difficulty_order:
            for condition in condition_order:
                correct_answers_condition_difficulty = correct_answers_lower_bound.loc[(correct_answers_lower_bound['difficulty'] == level) & (correct_answers_lower_bound['Condition'] == condition)]
                percent_cutoff = correct_answers_lower_bound["RT"].quantile(convert_percent)
                correct_answers_cleaned = correct_answers_lower_bound.loc[correct_answers_lower_bound["RT"]<percent_cutoff]
                all_conditions_difficulties.append(correct_answers_condition_difficulty)
        correct_answers_cleaned = pd.concat(all_conditions_difficulties).sort_index()
    else:
        correct_answers_cleaned = correct_answers_lower_bound
    summary = correct_answers_cleaned.groupby(['Condition', 'difficulty']).size().reset_index(name='count')

    return correct_answers_cleaned,summary

def clean_epochs_by_behaviour(epochs,rt_based=None, percent=None, correct_answers_only=False):
    difficulty_order = ['4','6','8','10']
    epochs.metadata['difficulty'] = pd.Categorical(epochs.metadata['difficulty'], categories=difficulty_order, ordered=True)
    condition_order = ['pop-out','search']
    epochs.metadata['Condition'] = pd.Categorical(epochs.metadata['Condition'], categories=condition_order, ordered=True)
    if correct_answers_only:
        if epochs.metadata['correct'].isnull().all():
            print('no column for correct answers in metadata')
        else:
            correct_answers = epochs[('correct == 1 ')]
    else:
        correct_answers = epochs
    if rt_based:
        rt_mask = (correct_answers.metadata['RT'] > rt_based[0]) & (correct_answers.metadata['RT'] < rt_based[1])
        correct_answers_lower_bound = correct_answers[rt_mask.to_numpy()]
    else: 
        correct_answers_lower_bound = correct_answers[('RT>0.1')] # will always remove improbable responses, usually 100ms, so 0.1 
    if percent:  
        convert_percent = 1-percent
        all_conditions_difficulties = []
        for level in difficulty_order:
            for condition in condition_order:
                correct_answers_condition_difficulty = correct_answers_lower_bound.loc[(correct_answers_lower_bound['difficulty'] == level) & (correct_answers_lower_bound['Condition'] == condition)]
                percent_cutoff = correct_answers_lower_bound["RT"].quantile(convert_percent)
                correct_answers_cleaned = correct_answers_lower_bound.loc[correct_answers_lower_bound["RT"]<percent_cutoff]
                all_conditions_difficulties.append(correct_answers_condition_difficulty)
        correct_answers_cleaned = pd.concat(all_conditions_difficulties).sort_index()
    else:
        correct_answers_cleaned = correct_answers_lower_bound
    summary = correct_answers_cleaned.metadata.groupby(['Condition', 'difficulty']).size().reset_index(name='count')

    return correct_answers_cleaned, summary

def plot_participant_RT_hist(data, data_cleaned):
    fig = plt.figure(figsize=(10,4.8), layout='constrained')
    gs  = GridSpec(1, 2, figure=fig) 
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax1.hist(data["RT"],bins=30)
    ax1.set_title('Reaction Time - Before Cleaning')
    ax1.set_xlabel('RT (ms)')
    ax1.set_ylabel('Frequency')

    ax2.hist(data_cleaned["RT"],bins=30)
    ax2.set_title('Reaction Time - After Cleaning')
    ax2.set_xlabel('RT (ms)')
    ax2.set_ylabel('Frequency')

    return fig

def plot_RT(data, group = False):
# Create a figure handle
    fig, ax = plt.subplots(figsize=(10, 6))

    if group:
        # Plot with median and bootstrapped 95% confidence intervals
        sns.lineplot(
            data=data,
            x='difficulty',
            y='RT',
            hue='Diagnosis',
            marker='o',
            style='Condition',
            estimator=np.median,
            errorbar=('ci', 95),         # Bootstrapped CI
            n_boot=1000,       # Number of bootstrap samples (adjust as needed)
            ax=ax              # Use figure handle
        )
    else:
        # Plot with median and bootstrapped 95% confidence intervals
        sns.lineplot(
            data=data,
            x='difficulty',
            y='RT',
            hue='Condition',
            marker='o',
            estimator=np.median,
            errorbar=('ci', 95),         # Bootstrapped CI
            n_boot=1000,       # Number of bootstrap samples (adjust as needed)
            ax=ax              # Use figure handle
        )
    ax.set_title('Median Reaction Time by Difficulty and Condition')
    ax.set_ylabel('Median RT (ms)')
    ax.set_xlabel('Difficulty')
    # Move legend outside
    ax.legend(
        bbox_to_anchor=(1.02, 1),  # move outside right
        loc='upper left',
        borderaxespad=0
    )

    plt.tight_layout()  # ensure everything fits
    return fig

def add_table_to_report(df,report,id):
    text_block = df.to_html()
    report.add_html(text_block, title='RT Summary Table', section=id, tags=['RT_summary_table'], replace=True)

def find_response_triggers(events):
    events_data = pd.DataFrame(events)
    events_data.columns = ['sample', 'initialState','trigger']
    trigger_counts = events_data['trigger'].value_counts()
    #check for largest number of triggers (leave case for those that are missing one side - eg. participant 075801)
    likely_response_triggers = trigger_counts[trigger_counts.index > 255].nlargest(2).index
    
    if len(likely_response_triggers) == 2:
        if np.any(likely_response_triggers[0] == cfg.right_responses):
            cfg.event_dict.update({'response/right':likely_response_triggers[0]})
            cfg.event_dict.update({'response/left':likely_response_triggers[1]})
        elif np.any(likely_response_triggers[0] == cfg.left_responses):
            cfg.event_dict.update({'response/right':likely_response_triggers[1]})
            cfg.event_dict.update({'response/left':likely_response_triggers[0]})

    elif len(likely_response_triggers) == 1:
        if np.any(likely_response_triggers[0] == cfg.right_responses):
            cfg.event_dict.update({'response/right':likely_response_triggers[0]})
            cfg.event_dict.pop('response/left')
            print("Warning: Missing left response")

        elif np.any(likely_response_triggers[0] == cfg.left_responses):
            cfg.event_dict.update({'response/left':likely_response_triggers[0]})
            cfg.event_dict.pop('response/right')
            print("Warning: Missing right response")
            
    elif len(likely_response_triggers) == 0:
        print("Warning: Missing all response triggers")

def plot_RTs_attenvis(metadata):
    metadata['difficulty'] = pd.Categorical(metadata['difficulty'],categories = ['4','6','8','10'], ordered=True)
    metadata['Condition'] = pd.Categorical(metadata['Condition'],categories = ['pop-out','search'], ordered=True)
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=metadata, x='difficulty', y='RT', hue='Condition', marker='o',estimator = np.median,errorbar='se')


def epochs_metadata(participant,visit_dir,locked_to ='stimuli', overwrite=False):
    transcend_data_dir = os.path.join(cfg.transcend_data_dir,'AttenVis',participant,os.path.split(visit_dir)[1])
    raw_sss_file = find_files('_raw_tsss.fif',visit_dir)
    raw_sss_file.sort()
    epochs_list = []

    for file in raw_sss_file:

        raw_sss = mne.io.read_raw_fif(file,preload=True,verbose=False)
        ica_file = file.replace('_raw_tsss.fif','_ica.fif')
        ica = mne.preprocessing.read_ica(ica_file)
        ica.apply(raw_sss)
        events_fname_tag = os.path.split(file)[1].replace('_raw_tsss.fif','_fixed_eve.fif')
        events_fname = find_files(events_fname_tag,transcend_data_dir)
        events = mne.read_events(events_fname[0])

        mat_file = events_fname[0].replace('_fixed_eve.fif','_behaviour.mat')
        metadata,events_meta,event_id_meta = attenvis_metadata(events,raw_sss.info['sfreq'],mat_file,participant,locked_to=locked_to)
        epochs = mne.Epochs(
            raw=raw_sss,
            events=events_meta,
            tmin=cfg.time_windows[0],
            tmax=cfg.time_windows[1],
            event_id=event_id_meta,
            reject = None,
            reject_by_annotation = False,
            picks="meg",
            baseline = None,
            on_missing="ignore",
            metadata=metadata,
        ).load_data()
        epochs_list.append(epochs)

    all_epochs = mne.concatenate_epochs(epochs_list)
    if locked_to == 'response':
        all_epochs.save(os.path.join(visit_dir,'_'.join([participant,'AttenVis','nobaseline_nofilter_all_conditions_metadata_response_epo.fif'])),overwrite=overwrite)
    else:
        all_epochs.save(os.path.join(visit_dir,'_'.join([participant,'AttenVis','nobaseline_nofilter_all_conditions_metadata_epo.fif'])),overwrite=overwrite)

    return all_epochs

def inverse_from_prestimulus_baseline(all_epochs,visit_dir,overwrite=False):
    #load fwd for inverse operator
    fwd_fname = find_files('_fwd.fif',visit_dir)[0]
    fwd = mne.read_forward_solution(fwd_fname)

    #compute covariance from baseline
    cov_fname = fwd_fname.replace('_run01_fwd.fif','_prestim_baseline_cov.fif')
    if not os.path.exists(cov_fname) or overwrite:
        noise_cov = mne.compute_covariance(all_epochs.apply_baseline(cfg.prestimulus_baseline), tmax=0, method = "auto",rank = None)
        mne.write_cov(cov_fname, noise_cov, overwrite=overwrite)
    else:
        noise_cov = mne.read_cov(cov_fname)
        msg = 'Covariance already exists. Using existing covariance.'
        print(msg)

    #compute inverse operator
    inv_fname = fwd_fname.replace('_run01_fwd.fif','_prestim_baseline_inv.fif')
    if not os.path.exists(inv_fname) or overwrite:
        inv_operator  = mne.minimum_norm.make_inverse_operator(all_epochs.info, fwd, noise_cov, loose=0.2, depth=0.8, rank='info')
        mne.minimum_norm.write_inverse_operator(inv_fname,inv_operator,overwrite=overwrite)
    return cov_fname

def add_whitened_evoked_prestim_baseline(participant,cov_fname,all_epochs,evoked,report):
    report.add_covariance(cov = cov_fname,info = all_epochs.info,title = participant + ' Covariance from prestimulus baseline',replace = True)
    noise_cov = mne.read_cov(cov_fname)
    report.add_evokeds(evoked,titles = participant + ' Evoked from prestimulus baseline',noise_cov = noise_cov,replace = True)
    report.save(cfg.inv_report_savename_hdf5, verbose=False, overwrite=True)

def add_whitened_evoked_erm(participant,visit_dir,evoked,report):
    erm_cov_fname = find_files(participant + '_erm_cov.fif', visit_dir)
    erm_cov = mne.read_cov(erm_cov_fname[0])
    fig = evoked.plot_white(erm_cov, time_unit = 's',show=False)
    report.add_figure(fig=fig, title=participant + ' Evoked from erm',  tags='whitened_erm',replace=True)
    # report.add_evokeds(evoked,titles = participant + ' Evoked from erm',noise_cov = erm_cov_fname,replace = True)
    report.save(cfg.inv_report_savename_hdf5, verbose=False, overwrite=True)

def collate_participants_data(participants_df,participants_to_study):
    all_participants = []
    for participant in participants_to_study:
        visit_dir = participants_df[participants_df['Participant'] == participant]['Visit_Dir'].values[0]
        filename = find_files(cfg.data_fname.replace('.pkl','_' + participant + '.pkl'),visit_dir)
        try:
            participant_df = pd.read_pickle(filename[0])
        except:
            print(f"Error loading data for participant {participant}. File not found or corrupted.")
            continue
        all_participants.append(participant_df)
    all_participants_df = pd.concat(all_participants,ignore_index=True).sort_values(by = 'Participant')
    return all_participants_df

def draw_label_from_epochs(visit_dir,subjID_date,epochs,inverse_operator,label_to_draw_from = 'fs_drawn',filter = None):
    peak_info = {}
    #load epochs and get evoked
    epochs_clean = get_condition_epochs(epochs.copy(),condition = None)
    baseline_evoked = get_evoked(epochs_clean,filter=filter,baseline=cfg.baseline)
    stc = mne.minimum_norm.apply_inverse(baseline_evoked, inverse_operator, cfg.lambda2, method=cfg.con_method, pick_ori=None, verbose=False)
    
    for hemi in cfg.hemisphere:
        #load_labels
        if label_to_draw_from == 'fs_drawn': 
            annot_label, fig = morph_fslabel(cfg.labels_list[0],subjID_date,hemi)
        elif label_to_draw_from == 'annot':
            parc = 'aparc.a2009s'
            annot_label = load_annot_labels(cfg.labels_list,subjID_date,parc,hemi,cfg.subj_dir)
        stc_from_annot_label = stc.in_label(annot_label)
        grown_label,morphed_label,label_fname,peak_time = find_peak_grow_label(stc_from_annot_label,hemi,cfg.peak_time_window[0],cfg.peak_time_window[1],5,subjID_date,'pow',visit_dir)
        peak_info[hemi] = {
            "label": grown_label,
            "morphed_label": morphed_label,
            "time": peak_time,
            "label_picture": fig
        }
    return peak_info

def draw_label_from_stc(visit_dir,subjID_date,stc,label_to_draw_from = 'fs_drawn',mode = 'abs'):
    peak_info={}
    for hemi in cfg.hemisphere:
        #load_labels
        if label_to_draw_from == 'fs_drawn': 
            annot_label, fig = morph_fslabel(cfg.labels_list[0],subjID_date,hemi)
        elif label_to_draw_from == 'annot':
            parc = 'aparc.a2009s'
            annot_label = load_annot_labels(cfg.labels_list,subjID_date,parc,hemi,cfg.subj_dir)
        stc_from_annot_label = stc.in_label(annot_label)
        grown_label,morphed_label,label_fname,peak_time = find_peak_grow_label(stc_from_annot_label,hemi,cfg.peak_time_window[0],cfg.peak_time_window[1],5,subjID_date,'diff',visit_dir,mode=mode)
        peak_info[hemi] = {
            "label": grown_label,
            "morphed_label": morphed_label,
            "time": peak_time
        }
    return peak_info

def save_peak_info(data):
    data_to_save = [
        {subject_id: peak_info}
        for subject_id, peak_info, _ in data
    ]
    with open(cfg.peak_times_savename, "wb") as f:
        pickle.dump(data_to_save, f)

def load_epochs(file_tag,visit_dir,resample=False):
    load_fname = find_files(file_tag,visit_dir)[0]
    epochs = mne.read_epochs(load_fname)
    if resample:
        epochs   = epochs.resample(cfg.sfreq)
    return load_fname, epochs
def load_stc(file_tag,visit_dir,filter=None):
    #load stc
    stc_path = find_files(file_tag,visit_dir)[0] 
    stc = mne.read_source_estimate(stc_path)
    stc._data = stc._data.astype('float64') #convert to float32 to save memory
    if filter:
        stc = stc.filter(filter[0],filter[1],verbose=True)
    return stc_path,stc
def load_inverse_operator(file_tag,visit_dir):
    #load inverse operator
    inv_path = find_files(file_tag,visit_dir)[0] 
    inverse_operator = mne.minimum_norm.read_inverse_operator(inv_path)
    return inverse_operator

def generate_report(inv = False):
    if inv:
        report_savename = cfg.inv_report_savename_hdf5
        report_title = cfg.inv_report_title
    else:
        report_savename = cfg.report_savename_hdf5
        report_title = cfg.report_title
    if os.path.exists(report_savename):
        report = mne.open_report(report_savename)
    else:
        report = mne.Report(title=report_title)
        report.save(report_savename, overwrite=True)

    return report

def plot_tf_comparison(data_by_hemi, x_axis, y_axis, titles,
                        diagnosis, output_dir, alpha=0.05, paired=True,
                        cmap='RdBu_r', vmin=None, vmax=None,
                        return_masks=False):
    """
    Compare TF plots between two conditions and draw contours around significant differences.

    Parameters
    ----------
    data_by_hemi : list of [data1, data2] pairs
        Each item is a list or tuple: [cond1_data, cond2_data], shape = (n_subjects, freqs, times).
        So the input is [[lh_search, lh_popout], [rh_search, rh_popout]].
    x_axis : array-like
        Driver frequency axis (e.g., 4–40 Hz) or time axis.
    y_axis : array-like
        Frequency axis.
    titles : list of str
        Titles for each panel (e.g., ['Left Hemisphere', 'Right Hemisphere']).
    condition : str
        Name of the condition for figure title.
    output_dir : str
        Where to save the figure.
    alpha : float
        Significance threshold.
    paired : bool
        Whether to use paired t-test.
    hemi : str or None
        Hemisphere label to include in file name.
    cmap : str
        Colormap for plotting.
    vmin, vmax : float
        Manual color limits.
    return_masks : bool
        Whether to return sig_masks and p_vals for further use.

    Returns
    -------
    fig : matplotlib.figure.Figure
    savename : str
    (optional) sig_masks : list of bool arrays
    (optional) p_vals : list of float arrays
    """
    SMALL_SIZE = 22
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=SMALL_SIZE)
    plt.rcParams['figure.constrained_layout.use'] = True
    levels = np.linspace(cfg.crossfreq_plot_lims[0], cfg.crossfreq_plot_lims[1], cfg.crossfreq_plot_lims[2])

    n_panels = len(data_by_hemi)
    fig, ax = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        ax = [ax]

    sig_masks = []
    pval_maps = []

    for i, (data1, data2) in enumerate(data_by_hemi):
        assert data1.shape == data2.shape, f"Shape mismatch in panel {i}"

        if paired:
            t_vals, p_vals = ttest_rel(data1, data2, axis=0)
        else:
            t_vals, p_vals = ttest_ind(data1, data2, axis=0)

        diff = np.mean(data1 - data2, axis=0)
        sig_mask = p_vals < alpha
        sig_masks.append(sig_mask)
        pval_maps.append(p_vals)

        if vmin is None or vmax is None:
            vmax_plot = np.max(np.abs(diff))
            vmin = -vmax_plot
            vmax = vmax_plot
            levels = np.linspace(vmin, vmax, cfg.crossfreq_plot_lims[2])


        cf = ax[i].contourf(x_axis, y_axis, diff.T, levels=levels,
                            cmap=cmap, vmin=vmin, vmax=vmax, extend='both')
        ax[i].set_title(titles[i])
        if cfg.analysis_type == 'cross_freq':
            ax[i].set_xlabel('Driving Frequency (Hz)')
        else:
            ax[i].set_xlabel('Time (s)')
        if i == 0:
            ax[i].set_ylabel('Frequency (Hz)')
        else:
            ax[i].set_yticklabels('')

        # Draw significance contours
        labeled, n_clusters = label(sig_mask.T)
        for c in range(1, n_clusters + 1):
            cluster = labeled == c
            ax[i].contour(x_axis, y_axis, cluster, colors='k', linewidths=1.5)

    # Shared colorbar
    cbar = fig.colorbar(cf, ax=ax, label='Mean Difference', orientation='vertical', ticks=np.linspace(vmin, vmax, cfg.crossfreq_plot_lims[2]))

    # Set figure title
    full_title = "Comparing " + '-'.join([key.capitalize() for key in cfg.condition.keys()]) + f" ({diagnosis.upper()})"
    fig.suptitle(full_title, fontsize=22)

    # Save figure
    fname_parts = list(filter(None, [diagnosis, 'tf', 'cluster', 'plot']))
    savename = os.path.join(output_dir, '_'.join(fname_parts) + ".tiff")
    fig.savefig(savename, dpi=300)
    plt.close(fig)

    if return_masks:
        return fig, savename, sig_masks, pval_maps
    return fig, savename

def add_pacs_comparison_to_report(df,report,id):
    low_fq_range = df["low_freqs"].values[0]
    high_fq_range = df["high_freqs"].values[0]
    image_names = []
    for diagnosis in cfg.diagnoses:
        df_diagnosis = df[df["Diagnosis"]==diagnosis]
        datasets = []
        for hemi in cfg.hemisphere:
            hemi_dataset = []
            for condition in cfg.condition:
                df_to_plot = df_diagnosis[(df_diagnosis["hemisphere"]==hemi) & (df_diagnosis['Condition']==condition)]
                if df_to_plot.empty:
                    print(f"No data for {diagnosis} in {condition} for {hemi}. Skipping...")
                    continue
                all_data = np.stack(df_to_plot['pac'].values)
                hemi_dataset.append(all_data)
            datasets.append(hemi_dataset)
        titles = ['Left Hemisphere', 'Right Hemisphere']

        fig1, name = plot_tf_comparison(
                    datasets, low_fq_range, high_fq_range, titles,
                    diagnosis, cfg.output_dir, alpha=0.05, paired=True,
                    cmap='RdBu_r', vmin=None, vmax=None,
                    return_masks=False)
        title = '_'.join([id,condition,'pac'])
        #save fig
        fig_to_save = fig1.get_figure()
        fig_to_save.savefig(name.replace('.tiff','.svg'),format="svg")
        fig_to_save.savefig(name,dpi=300)
        image_names.append(name)
        plt.close()
    fig = plt.figure(figsize=(18,6), layout='constrained')
    gs  = GridSpec(1, 2, figure=fig) 
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax1.imshow(plt.imread(image_names[0]))
    ax1.axis('off')
    ax2.imshow(plt.imread(image_names[1]))
    ax2.axis('off')
    title = '_'.join([id,condition,'pac_comparison'])
    report.add_figure(fig=fig, title=title, section=id, tags=[condition,'pac'],replace=True)
    report.save(cfg.report_savename_hdf5, verbose=False, overwrite=True)
    plt.close('all')