import mne
import os
import numpy as np
import pandas as pd
import mne_connectivity

import helper_functions as tlbx
import AttenVis_connectivity_config as cfg
import matplotlib.pyplot as plt   
import matplotlib as mpl 
from mne.parallel import parallel_func

participants_df, participants_to_study = tlbx.load_participants()

report = tlbx.generate_report()

print("Saving as " + cfg.data_savename)

def get_connectivity(sub_id,overwrite_data=False):
    """    Compute functional connectivity for a given subject."""
    participant_data = []
    diagnosis, study,visit_dir,subjID_date = tlbx.read_participant_details_from_dataframe(participants_df,sub_id)
    participant_data_savename = os.path.join(visit_dir,cfg.data_fname.replace('.pkl','_' + sub_id + '.pkl'))

    if not os.path.exists(participant_data_savename) or overwrite_data:
        inverse_operator        = tlbx.load_inverse_operator('_prestim_baseline_inv.fif',visit_dir)
        src = inverse_operator["src"]
    
        #load seed label
        seed_label_hemis = []
        target_label_hemis = []

        for hemi in cfg.hemisphere:
            # load seed and target labels for FC analyses
            seed_label   = tlbx.load_drawn_labels(cfg.seed_label,hemi,subjID_date,visit_dir,grown=True)
            seed_label_hemis.append(seed_label)
            target_label = tlbx.load_drawn_labels(cfg.target_label,hemi,subjID_date,visit_dir,grown=True)
            target_label_hemis.append(target_label)
        
        #load condition specific epochs and extract time series
        for condition in cfg.condition:
            file_tag_cond  = '_AttenVis_nobaseline_nofilter_metadata_' + condition
            epo_load_fname = tlbx.find_files(file_tag_cond + '_behaviour_cleaned_epo.fif', visit_dir)[0]
            epochs         = mne.read_epochs(epo_load_fname)
            cleaned_epochs, summary = tlbx.clean_epochs_by_behaviour(epochs, rt_based=(0.1,1.2), percent=None, correct_answers_only=True)
            stcs = mne.minimum_norm.apply_inverse_epochs(cleaned_epochs, inverse_operator, cfg.lambda2, cfg.con_method, pick_ori="normal", verbose=False)
            # this is some stuff we need to specify how FC is computed by the mne-connectivity toolbox
            indices        = mne_connectivity.seed_target_indices([0], [1])
            sfreq          = cleaned_epochs.info['sfreq']  # the sampling frequency
            cwt_freqs      = np.arange(cfg.freq_min, cfg.freq_max+1, 1)
            cwt_n_cycles   = 3 #cwt_freqs / 2.0  # number of cycles for the CWT

            for hemi in cfg.hemisphere:
                if hemi == 'lh':
                    hemi_idx = 0
                else:
                    hemi_idx = 1
                seed_stc = mne.extract_label_time_course(stcs, seed_label_hemis[hemi_idx], src, mode='mean', verbose=False)
                for target_hemi in cfg.hemisphere:
                    if target_hemi == 'lh':
                        target_hemi_idx = 0
                    else:
                        target_hemi_idx = 1
                    target_stc = mne.extract_label_time_course(stcs, target_label_hemis[target_hemi_idx], src, mode='mean', verbose=False)
                    comb_ts = list(zip(seed_stc, target_stc))                                                                #combine time series for each hemisphere
                    print(f'\n>>> Computing FC for {hemi.upper()} and {target_hemi.upper()}\n')
                    con = mne_connectivity.spectral_connectivity_epochs(                                                       #get functional connectivity
                        comb_ts, method=cfg.fc_method, mode=cfg.fc_mode, indices=indices, sfreq=sfreq,
                        cwt_freqs=cwt_freqs, cwt_n_cycles=cwt_n_cycles, tmin=cfg.time_windows[0], tmax=cfg.time_windows[1])

                    participant_data_hemi = [sub_id,subjID_date,diagnosis,condition,hemi,target_hemi,np.squeeze(con.get_data()),stcs[0].vertices,con.times,len(cleaned_epochs)]
                    participant_data.append(participant_data_hemi)

        df= pd.DataFrame(participant_data, columns = ['Participant','SubjID_Date','Diagnosis','Condition','hemisphere','target_hemi','connectivity_data','vertices','time','n_epochs'])
        df.to_pickle(participant_data_savename)
        pics = tlbx.plot_participant_coh_line(df,(6,12), skip_brain_images=True)
        tfr_pics_bihemi = []
        for hemi in cfg.hemisphere:
            df_hemi = df[df["hemisphere"]==hemi]
            tfr_pics = tlbx.plot_participant_tfrs(df_hemi,sub_id,label_hemi=hemi)
            tfr_pics_bihemi.append(tfr_pics)

        # for hemi in cfg.hemisphere:
        #     for target_hemi in cfg.hemisphere:
        #         df_to_analyse = df[(df["hemisphere"]== hemi) & (df['target_hemi']== target_hemi)]
        #         zcoh_hemi = tlbx.get_zcoh(df_to_analyse,'search','pop-out')
        #         zcoh_data = [sub_id,subjID_date,diagnosis,hemi,target_hemi,zcoh_hemi,df['time'].values[0]]

    else:
        df_participant = pd.read_pickle(participant_data_savename)
        pics = tlbx.plot_participant_coh_line(df_participant,(6,12),skip_brain_images=True)
        tfr_pics_bihemi = []
        for hemi in cfg.hemisphere:
            df_hemi = df_participant[df_participant["hemisphere"]==hemi]
            tfr_pics = tlbx.plot_participant_tfrs(df_hemi,sub_id,label_hemi=hemi)
            tfr_pics_bihemi.append(tfr_pics)
    return sub_id, pics, tfr_pics_bihemi

n_jobs = 3

debug = False
if debug:
    n_jobs = 1

parallel, run_func, _ = parallel_func(get_connectivity, n_jobs=n_jobs)
results = parallel(run_func(subject) for subject in participants_to_study)

report = tlbx.generate_report()

for sub_id, pics, tfr_pics_bihemi in results:
    for pic,title in pics:
        report.add_figure(fig=pic, title=sub_id + '_' + title, section=sub_id, tags=['coherence'], replace=True)
        plt.close(pic)

    for tfr_pics in tfr_pics_bihemi:
        for pic,title,_,_ in tfr_pics:
            report.add_figure(fig=pic, title=title, section=sub_id, tags=['tfr'], replace=True)
            plt.close(pic)

report.save(cfg.report_savename_hdf5, verbose=False, overwrite=True)
# overwrite_zcoh_data = False
# if not os.path.exists(cfg.connectivity_compare_data_savename.replace(".pkl","_zcoh.pkl")) or overwrite_zcoh_data:
#     zcoh_rows = [res[-1] for res in results]  # assuming zcoh_data is the last item in each tuple
#     zcoh_df = pd.DataFrame(zcoh_rows, columns=['Participant','SubjID_Date','Diagnosis','hemisphere','target_hemi','zcoh','time'])
#     zcoh_df.to_pickle(cfg.connectivity_compare_data_savename.replace(".pkl","_zcoh.pkl"))

df = tlbx.collate_participants_data(participants_df,participants_to_study)

mpl.rcParams["svg.fonttype"] = "none"

tlbx.add_gavg_coh_to_report(df,report,'gavg','Condition',freqs = (6,12),ci = True)
tlbx.add_gavg_coh_to_report(df,report,'gavg','Diagnosis',freqs = (6,12),ci = True)

for hemi in cfg.hemisphere:
    df_hemi = df[df["hemisphere"]==hemi]
    tlbx.add_tfrs_to_report(df_hemi,report,'gavg',hemi_label=hemi)
    tlbx.add_tfrs_comparison_to_report(df_hemi,report,'gavg',analysis_type = 'within_group',hemi_label=hemi)
    tlbx.add_tfrs_comparison_to_report(df_hemi,report,'gavg',analysis_type = 'between_group',hemi_label=hemi)

report.save(cfg.report_savename_hdf5, verbose=False, overwrite=True)
tlbx.show_report(cfg.report_savename_hdf5)    