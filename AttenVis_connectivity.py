import mne
import os
import numpy as np
import pandas as pd
import mne_connectivity

import helper_functions as tlbx
import miso_config as cfg

if cfg.paradigm == 'MisoNat':
    participants_df, participants_to_study = tlbx.load_misonat_participants('all')
elif cfg.paradigm == 'MisoNat2':
    participants_df, participants_to_study = tlbx.load_misonat_participants('MisoNat2')
else:    
    participants_df, participants_to_study = tlbx.load_participants(cfg.participants_csv)

if cfg.add_participants:
    participants_to_study = tlbx.update_participants(cfg.data_savename,participants_to_study)

report = mne.Report(title=cfg.report_title)
print("Saving as " + cfg.data_savename)

all_participants = []
if not os.path.exists(cfg.data_savename) or cfg.overwrite_data or cfg.add_participants:
    for sub_id in participants_to_study:
        #get participant details
        diagnosis, study,visit_dir,subjID_date = tlbx.get_participant_details(participants_df,sub_id)
        #load inverse operator
        inv_path = tlbx.find_files('_inv.fif',visit_dir)[0]
        inverse_operator = mne.minimum_norm.read_inverse_operator(inv_path)
        src = inverse_operator["src"] 
        
        #load seed label
        seed_label_hemis = []
        target_label_hemis = []
        for hemi in cfg.hemisphere:
            seed_label = tlbx.load_drawn_labels(cfg.seed_label,hemi,subjID_date,visit_dir,grown=True)
            seed_label_hemis.append(seed_label)
            target_label = tlbx.morph_fslabel(cfg.target_label[0],subjID_date,hemi)[0]
            target_label_hemis.append(target_label)
        
        #load target labels
        # target_label = tlbx.load_annot_labels(cfg.target_label,subjID_date,'aparc.a2009s','both',cfg.subj_dir)
        target_label = target_label_hemis[0] + target_label_hemis[1] 

        #load condition specific epochs and extract time series
        for condition in cfg.condition:
            file_tag_cond = '_nobaseline_nofilter_' + condition 
            epo_load_fname = tlbx.find_files(file_tag_cond + '_epo.fif',visit_dir)[0]
            print(epo_load_fname)
            epochs = mne.read_epochs(epo_load_fname)
            epochs   = epochs.resample(cfg.sfreq)
            stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inverse_operator, cfg.lambda2, cfg.con_method, pick_ori="normal", verbose=False)

            #extract time series
            stcs_label_lh = mne.extract_label_time_course(stcs, seed_label_hemis[0], src, mode='mean', verbose=False)
            stcs_label_rh = mne.extract_label_time_course(stcs, seed_label_hemis[1], src, mode='mean', verbose=False)

            verts2include = [i.in_label(target_label) for i in stcs]

            #combine time series for each hemisphere
            comb_ts_lh = list(zip(stcs_label_lh, verts2include))
            comb_ts_rh = list(zip(stcs_label_rh, verts2include))

            # this is some stuff we need to specify how FC is computed by the mne-connectivity toolbox 
            vertices      = [verts2include[0].vertices[i] for i in range(2)]
            n_signals_tot = 1 + len(vertices[0]) + len(vertices[1])
            indices       = mne_connectivity.seed_target_indices([0], np.arange(1, n_signals_tot)) 
            sfreq = epochs.info['sfreq']  # the sampling frequency
            cwt_freqs = np.arange(cfg.freq_min, cfg.freq_max+1, 1)
            cwt_n_cycles = cfg.con_n_cycles

            # FC: left auditory seed to ROI
            print('\n>>> Computing FC for lh\n')
            con_lh = mne_connectivity.spectral_connectivity_epochs(
                comb_ts_lh, method=cfg.fc_method, mode=cfg.fc_mode, indices=indices, sfreq=sfreq,
                cwt_freqs=cwt_freqs, cwt_n_cycles=cwt_n_cycles, tmin=cfg.time_windows[0], tmax=cfg.time_windows[1])   
            
            # FC: right auditory seed to ROI
            print('\n>>> Computing FC for rh\n')
            con_rh = mne_connectivity.spectral_connectivity_epochs(
                comb_ts_rh, method=cfg.fc_method, mode=cfg.fc_mode, indices=indices, sfreq=sfreq,
                cwt_freqs=cwt_freqs, cwt_n_cycles=cwt_n_cycles, tmin=cfg.time_windows[0], tmax=cfg.time_windows[1])

            participant_data_lh = [sub_id,subjID_date,diagnosis,condition,'lh',con_lh.get_data(),verts2include[0].vertices,con_lh.times,len(epochs)]
            participant_data_rh = [sub_id,subjID_date,diagnosis,condition,'rh',con_rh.get_data(),verts2include[0].vertices,con_rh.times,len(epochs)]
            all_participants.append(participant_data_lh)
            all_participants.append(participant_data_rh)
                

    df= pd.DataFrame(all_participants, columns = ['Participant','SubjID_Date','Diagnosis','Condition','hemisphere','connectivity_data','vertices','time','n_epochs']) 
    if cfg.add_participants:
        old_df = pd.read_pickle(cfg.data_savename)
        add_to_df = pd.concat([old_df,df])
        add_to_df = add_to_df.sort_values(by='Participant')
        df = add_to_df

    df.to_pickle(cfg.data_savename)
else:
    #load existing dataframe
    df = pd.read_pickle(cfg.data_savename)
    if os.path.exists(cfg.con_report_savename_hdf5):
        report = mne.open_report(cfg.con_report_savename_hdf5)

# tlbx.show_report(cfg.morph_report_savename_hdf5)

print("Getting peak coherence labels and z coherence, saving as " + cfg.connectivity_compare_data_fname)
if not os.path.exists(cfg.connectivity_compare_data_savename.replace(".pkl","_coh_peak.pkl")):
    zcoh_all=[]
    all_participants_output = []
    for sub_id in participants_to_study:
        participant_data=[]
        df_participant = df[df["Participant"]==sub_id]
        diagnosis, study,visit_dir,subjID_date = tlbx.get_participant_details(participants_df,sub_id)
        for condition in cfg.selected_conditions:
            for hemi in cfg.hemisphere:
                df_res = df_participant[(df_participant["Condition"]==condition) & (df_participant['hemisphere']== hemi)]
                n_epochs = df_res['n_epochs'].values[0]
                coh_stc,con_subjID_date = tlbx.get_coh_stc(df_res,'time')
                for target_hemi in cfg.hemisphere:
                    if condition == 'miso':
                        coh_peak_label,morphed_coh_peak_label,coh_peak_label_fname, peak_time = tlbx.find_peak_grow_label(coh_stc,target_hemi,0,0.5,5,con_subjID_date,'coh',visit_dir)
                        cfg.peak_labels_hemis[hemi].update({target_hemi:coh_peak_label})
                        cfg.peak_times_hemis[hemi].update({target_hemi:peak_time})
                    else:
                        coh_peak_label = cfg.peak_labels_hemis[hemi][target_hemi] 
                        peak_time = cfg.peak_times_hemis[hemi][target_hemi]
                        morphed_coh_peak_label= []
                    coh_stc_label = np.mean(coh_stc.in_label(coh_peak_label).data,axis=0)
                    brain_image_name = tlbx.plot_stc(coh_stc,target_hemi,peak_time,coh_peak_label,condition,'blue',other_hemi = hemi)
                    data = [sub_id,subjID_date,diagnosis,condition,hemi,target_hemi,coh_stc_label,morphed_coh_peak_label,df_res['time'].values[0],n_epochs]
                    participant_data.append(data)
        
        output_df = pd.DataFrame(participant_data,columns = ['Participant','SubjID_Date','Diagnosis','Condition','hemisphere','target_hemi','stc','morphed_label','time','n_epochs'])
        all_participants_output.append(output_df)
        tlbx.add_participant_coh_to_report(output_df,report,sub_id)
        for hemi in cfg.hemisphere:
            for target_hemi in cfg.hemisphere:
                df_to_analyse = output_df[(output_df["hemisphere"]== hemi) & (output_df['target_hemi']== target_hemi)]
                zcoh_hemi = tlbx.get_zcoh(df_to_analyse,'miso','white_noise')
                data = [sub_id,subjID_date,diagnosis,hemi,target_hemi,zcoh_hemi,'miso',df_res['time'].values[0]]
                zcoh_all.append(data)
                zcoh_hemi = tlbx.get_zcoh(df_to_analyse,'sound2','white_noise')
                data = [sub_id,subjID_date,diagnosis,hemi,target_hemi,zcoh_hemi,'sound2',df_res['time'].values[0]]
                zcoh_all.append(data)

    all_participants_df = pd.concat(all_participants_output)
    all_participants_df.to_pickle(cfg.connectivity_compare_data_savename.replace(".pkl","_coh_peak.pkl"))
    zcoh_df = pd.DataFrame(zcoh_all,columns = ['Participant','SubjID_Date','Diagnosis','hemisphere','target_hemi','stc','Condition','time'])
    zcoh_df.to_pickle(cfg.connectivity_compare_data_savename.replace(".pkl","_zcoh.pkl"))
else:    
    all_participants_df = pd.read_pickle(cfg.connectivity_compare_data_savename.replace(".pkl","_coh_peak.pkl"))
    zcoh_df = pd.read_pickle(cfg.connectivity_compare_data_savename.replace(".pkl","_zcoh.pkl"))

twi = [(1.1,1.7)]

exclude_participants = []
excluded_participants = tlbx.update_participants_n(participants_df,exclude_participants,'MisoNat')
removed_participants = zcoh_df['Participant'].isin(excluded_participants)
zcoh_df_misonat2 = zcoh_df[~removed_participants]
removed_participants = all_participants_df['Participant'].isin(excluded_participants)
all_participants_df_misonat2 = all_participants_df[~removed_participants]
exclude_participants = []
excluded_participants = tlbx.update_participants_n(participants_df,exclude_participants,'MisoNat2')
removed_participants = zcoh_df['Participant'].isin(excluded_participants)
zcoh_df_misonat1 = zcoh_df[~removed_participants]
removed_participants = all_participants_df['Participant'].isin(excluded_participants)
all_participants_df_misonat1 = all_participants_df[~removed_participants]
exclude_participants = ['147401','146201','151101','150901']
removed_participants = zcoh_df['Participant'].isin(exclude_participants)
zcoh_df_subgroup = zcoh_df[~removed_participants]
removed_participants = all_participants_df['Participant'].isin(exclude_participants)
all_participants_df_subgroup = all_participants_df[~removed_participants]

participant_groupings = {'gavg':all_participants_df,
                         'gavg_misonat2':all_participants_df_misonat2,
                         'gavg_misonat1':all_participants_df_misonat1,
                         'gavg_subgroup':all_participants_df_subgroup}

participant_groupings_zcoh = {'gavg':zcoh_df,
                         'gavg_misonat2':zcoh_df_misonat2,
                         'gavg_misonat1':zcoh_df_misonat1,
                         'gavg_subgroup':zcoh_df_subgroup}


tlbx.add_gavg_coh_to_report(all_participants_df,report,'gavg','Condition')
tlbx.add_gavg_coh_to_report(all_participants_df,report,'gavg','Diagnosis')

tlbx.add_gavg_coh_to_report(zcoh_df,report,'gavg','Condition',zcoh=True,ci=False)

for group in participant_groupings_zcoh:
    for time_window in twi:
        for hemi in cfg.hemisphere:
            for target_hemi in cfg.hemisphere:
                tlbx.add_connectivity_plot(participant_groupings_zcoh[group],report,time_window,hemi,target_hemi,'Condition',group,zcoh=True)

for group in participant_groupings:
    for time_window in twi:
        for hemi in cfg.hemisphere:
            for target_hemi in cfg.hemisphere:
                tlbx.add_connectivity_plot(participant_groupings[group],report,time_window,hemi,target_hemi,'Diagnosis',group,zcoh=False)

seed_lh_df = all_participants_df[all_participants_df['hemisphere']=='lh']
tlbx.add_fsaverage_to_report(report,seed_lh_df,cfg.labels_of_interest[0] + '_coh_grown','lh')
seed_rh_df = all_participants_df[all_participants_df['hemisphere']=='rh']
tlbx.add_fsaverage_to_report(report,seed_rh_df,cfg.labels_of_interest[0] + '_coh_grown','rh')

report.save(cfg.con_report_savename_html, overwrite=True)
report.save(cfg.con_report_savename_hdf5, overwrite=True)