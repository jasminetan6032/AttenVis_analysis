import mne
import os
import numpy as np
import pandas as pd
import mne_connectivity

import helper_functions as tlbx
import AttenVis_config as cfg

participants_df, participants_to_study = tlbx.load_participants()

report = tlbx.generate_report()

for sub_id in participants_to_study:

    participant_data = []
    diagnosis, study,visit_dir,subjID_date = tlbx.read_participant_details_from_dataframe(participants_df,sub_id)
    participant_data_savename = os.path.join(visit_dir,cfg.data_fname.replace('.pkl','_' + sub_id + '.pkl'))
    df_participant = pd.read_pickle(participant_data_savename)

    coh_stc,con_subjID_date = tlbx.get_coh_stc(df_res,'time')
    peak_info = tlbx.draw_label_from_stc(coh_stc,target_hemi,0,0.5,5,con_subjID_date,'coh',visit_dir)
    
    for condition in cfg.condition:
        for difficulty_level in cfg.difficulty:
            for hemi in cfg.hemisphere:
                df_res = df_participant[(df_participant["Condition"]==condition) & (df_participant['hemisphere']== hemi) & (df_participant['Difficulty']==difficulty_level)]
                n_epochs = df_res['n_epochs'].values[0]
                for target_hemi in cfg.hemisphere:
                    coh_peak_label = peak_info[hemi][target_hemi] 
                    peak_time = peak_info[hemi][target_hemi]
                    morphed_coh_peak_label= peak_info[hemi][target_hemi]
                    coh_stc_label = np.mean(coh_stc.in_label(coh_peak_label).data,axis=0)
                    brain_image_name = tlbx.plot_stc(coh_stc,target_hemi,peak_time,coh_peak_label,condition,'blue',other_hemi = hemi)
                    data = [sub_id,subjID_date,diagnosis,condition,hemi,target_hemi,coh_stc_label,morphed_coh_peak_label,df_res['time'].values[0],n_epochs]
                    participant_data.append(data)
        
        output_df = pd.DataFrame(participant_data,columns = ['Participant','SubjID_Date','Diagnosis','Condition','hemisphere','target_hemi','stc','morphed_label','time','n_epochs'])
#         all_participants_output.append(output_df)
#         tlbx.add_participant_coh_to_report(output_df,report,sub_id)
#         for hemi in cfg.hemisphere:
#             for target_hemi in cfg.hemisphere:
#                 df_to_analyse = output_df[(output_df["hemisphere"]== hemi) & (output_df['target_hemi']== target_hemi)]
#                 zcoh_hemi = tlbx.get_zcoh(df_to_analyse,'miso','white_noise')
#                 data = [sub_id,subjID_date,diagnosis,hemi,target_hemi,zcoh_hemi,'miso',df_res['time'].values[0]]
#                 zcoh_all.append(data)
#                 zcoh_hemi = tlbx.get_zcoh(df_to_analyse,'sound2','white_noise')
#                 data = [sub_id,subjID_date,diagnosis,hemi,target_hemi,zcoh_hemi,'sound2',df_res['time'].values[0]]
#                 zcoh_all.append(data)

#     all_participants_df = pd.concat(all_participants_output)
#     all_participants_df.to_pickle(cfg.connectivity_compare_data_savename.replace(".pkl","_coh_peak.pkl"))
#     zcoh_df = pd.DataFrame(zcoh_all,columns = ['Participant','SubjID_Date','Diagnosis','hemisphere','target_hemi','stc','Condition','time'])
#     zcoh_df.to_pickle(cfg.connectivity_compare_data_savename.replace(".pkl","_zcoh.pkl"))
# else:    
#     all_participants_df = pd.read_pickle(cfg.connectivity_compare_data_savename.replace(".pkl","_coh_peak.pkl"))
#     zcoh_df = pd.read_pickle(cfg.connectivity_compare_data_savename.replace(".pkl","_zcoh.pkl"))

# twi = [(1.1,1.7)]

# exclude_participants = []
# excluded_participants = tlbx.update_participants_n(participants_df,exclude_participants,'MisoNat')
# removed_participants = zcoh_df['Participant'].isin(excluded_participants)
# zcoh_df_misonat2 = zcoh_df[~removed_participants]
# removed_participants = all_participants_df['Participant'].isin(excluded_participants)
# all_participants_df_misonat2 = all_participants_df[~removed_participants]
# exclude_participants = []
# excluded_participants = tlbx.update_participants_n(participants_df,exclude_participants,'MisoNat2')
# removed_participants = zcoh_df['Participant'].isin(excluded_participants)
# zcoh_df_misonat1 = zcoh_df[~removed_participants]
# removed_participants = all_participants_df['Participant'].isin(excluded_participants)
# all_participants_df_misonat1 = all_participants_df[~removed_participants]
# exclude_participants = ['147401','146201','151101','150901']
# removed_participants = zcoh_df['Participant'].isin(exclude_participants)
# zcoh_df_subgroup = zcoh_df[~removed_participants]
# removed_participants = all_participants_df['Participant'].isin(exclude_participants)
# all_participants_df_subgroup = all_participants_df[~removed_participants]

# participant_groupings = {'gavg':all_participants_df,
#                          'gavg_misonat2':all_participants_df_misonat2,
#                          'gavg_misonat1':all_participants_df_misonat1,
#                          'gavg_subgroup':all_participants_df_subgroup}

# participant_groupings_zcoh = {'gavg':zcoh_df,
#                          'gavg_misonat2':zcoh_df_misonat2,
#                          'gavg_misonat1':zcoh_df_misonat1,
#                          'gavg_subgroup':zcoh_df_subgroup}


# tlbx.add_gavg_coh_to_report(all_participants_df,report,'gavg','Condition')
# tlbx.add_gavg_coh_to_report(all_participants_df,report,'gavg','Diagnosis')

# tlbx.add_gavg_coh_to_report(zcoh_df,report,'gavg','Condition',zcoh=True,ci=False)

# for group in participant_groupings_zcoh:
#     for time_window in twi:
#         for hemi in cfg.hemisphere:
#             for target_hemi in cfg.hemisphere:
#                 tlbx.add_connectivity_plot(participant_groupings_zcoh[group],report,time_window,hemi,target_hemi,'Condition',group,zcoh=True)

# for group in participant_groupings:
#     for time_window in twi:
#         for hemi in cfg.hemisphere:
#             for target_hemi in cfg.hemisphere:
#                 tlbx.add_connectivity_plot(participant_groupings[group],report,time_window,hemi,target_hemi,'Diagnosis',group,zcoh=False)

# seed_lh_df = all_participants_df[all_participants_df['hemisphere']=='lh']
# tlbx.add_fsaverage_to_report(report,seed_lh_df,cfg.labels_of_interest[0] + '_coh_grown','lh')
# seed_rh_df = all_participants_df[all_participants_df['hemisphere']=='rh']
# tlbx.add_fsaverage_to_report(report,seed_rh_df,cfg.labels_of_interest[0] + '_coh_grown','rh')

# report.save(cfg.con_report_savename_html, overwrite=True)
# report.save(cfg.con_report_savename_hdf5, overwrite=True)