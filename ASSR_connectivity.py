import mne
import pandas as pd
import numpy as np
import ASSR_config as cfg
import matplotlib.pyplot as plt
import scipy.stats as st
import mne_connectivity
from os.path import join, split, exists
from glob import glob

# load alignment file
ASSR_df       = pd.read_csv(join(cfg.paradigm_dir,'alignment_file.csv'), index_col=0, dtype=str)
subject_info  = pd.read_csv(join(cfg.paradigm_dir,'ASSR_subject_info.csv'), dtype=str)

# get subject list
subjects      = list(ASSR_df['subject']) 
restrict2rois = True
overwrite     = True
subs2exclude  = []
source_method = 'dSPM'
fc_method     = 'pli'
freq_band     = 'alpha'
con_method    = 'cwt_morlet'
varcycles     = False

# create dictionary for different conditions
data_jump_asd  = {}
data_stay_asd  = {}
data_jump_td   = {}
data_stay_td   = {}

# dict wit frequency band info
bands_dict = {'delta': (1,4), 'theta' : (4,8), 'alpha' : (8,12), 'beta' : (15,30), 'low_gamma' : (30,50), 'high_gamma' : (70,90)}
fmin  = bands_dict[freq_band][0]
fmax  = bands_dict[freq_band][1]

# open a report if it already exists or create one if it doesn't
report  = mne.Report(title=source_method+'_'+cfg.report_name) if not exists(join(cfg.paradigm_dir,source_method+'_'+cfg.report_name)) else mne.open_report(join(cfg.paradigm_dir,source_method+'_'+cfg.report_name))

for counter,sub_i in enumerate(subjects):

    if sub_i not in subs2exclude:

        # print counter
        print('\n\n >>>>> Estimating FC for subject %d / %d <<<<< \n\n' %(counter+1,len(subjects)))

        # set subject paths
        sss_path    = list(ASSR_df['sss path'][ASSR_df['subject'] == sub_i])[0]
        info        = mne.io.read_info(sss_path)
        this_visit  = [i for i in sss_path.split('/') if 'visit' in i][0]
        recons_path = list(ASSR_df['recons path'][ASSR_df['subject'] == sub_i])[0]
        output_dir  = join(cfg.paradigm_dir,sub_i,this_visit)
        subject     = split(recons_path)[-1]
        subjects_dir = cfg.recons_dir

        # some subjects don't have good MRIs
        if sub_i in ['082802','082601','082501']:
            subject      = 'fsaverage'
            subjects_dir = '/local_mount/space/tapputi/1/users/sergio/MNE-sample-data/subjects'

        # get diagnosis
        diagnosis   = list(subject_info[subject_info['Subj_ID'] == sub_i]['diagnosis'])[0]

        # load label
        try:
            func_label_rh = mne.read_label(join(output_dir,'corrected_labels',source_method+'_rh.label'))
        except:
            func_label_rh = mne.read_label(join(output_dir,source_method+'_rh.label'))
            
        try:
            func_label_lh = mne.read_label(join(output_dir,'corrected_labels',source_method+'_lh.label'))
        except:    
            func_label_lh = mne.read_label(join(output_dir,source_method+'_lh.label'))   

        # check if the seed-based connectivity data already exists        
        fname = '%s_FC_*_1-30hz_%s_%s.npz' %(sub_i,fc_method,con_method.split('_')[1])       

        if len(glob(join(cfg.paradigm_dir,sub_i,this_visit,fname))) == 0 or overwrite:

            # load src
            src_fname = glob(join(output_dir,'*src.fif'))[0]
            src       = mne.read_source_spaces(src_fname)
            
            # load stc data and pre-defined labels
            stc_fname  = split(sss_path)[1].split('0hp')[0]+'01_120hz.stc'

            # read epochs
            epochs_fname = stc_fname.replace('.stc','_epo.fif')   
            all_epochs   = mne.read_epochs(join(output_dir,epochs_fname),  preload=True)
            all_epochs   = all_epochs.resample(250)

            # downsampled data is 1 sample longer. Trim those files
            # if len(all_epochs.times) == 2001: all_epochs = all_epochs.crop(tmax=all_epochs.times[-2]) 
            jump_epochs = all_epochs['jump'] 
            stay_epochs = all_epochs['stay']

            # get the inverse operator
            cov_path  = glob(join(cfg.transcend_dir,sub_i,this_visit,'epoched','*cov*'))[0]
            noise_cov = mne.read_cov(cov_path)

            # compute the inverse operator
            fwd_fname     = stc_fname.split('_01')[0]+'_fwd.fif'
            fwd           = mne.read_forward_solution(join(output_dir,fwd_fname), verbose=False)
            inv_operator  = mne.minimum_norm.make_inverse_operator(info, fwd, noise_cov, loose=0.2, depth=0.8, rank='info')

            # get source time courses
            snr     = 3
            lambda2 = 1.0 / snr**2

            # apply inverse to epochs
            stcs_jump = mne.minimum_norm.apply_inverse_epochs(jump_epochs, inv_operator, lambda2, source_method, pick_ori="normal", verbose=False)
            stcs_stay = mne.minimum_norm.apply_inverse_epochs(stay_epochs, inv_operator, lambda2, source_method, pick_ori="normal", verbose=False)

            # extract signals from labels
            stc_jump_lh = mne.extract_label_time_course(stcs_jump, func_label_lh, src, mode='mean_flip', verbose=False)
            stc_jump_rh = mne.extract_label_time_course(stcs_jump, func_label_rh, src, mode='mean_flip', verbose=False)
            stc_stay_lh = mne.extract_label_time_course(stcs_stay, func_label_lh, src, mode='mean_flip', verbose=False)
            stc_stay_rh = mne.extract_label_time_course(stcs_stay, func_label_rh, src, mode='mean_flip', verbose=False)
            
            if restrict2rois:

                # define our search area
                prefrontal1  = mne.read_labels_from_annot(subject=subject, parc="aparc", subjects_dir=subjects_dir, regexp='rostralmiddlefrontal')
                prefrontal2  = mne.read_labels_from_annot(subject=subject, parc="aparc", subjects_dir=subjects_dir, regexp='caudalmiddlefrontal')
                prefrontal3  = mne.read_labels_from_annot(subject=subject, parc="aparc", subjects_dir=subjects_dir, regexp='parsopercularis')
                prefrontal4  = mne.read_labels_from_annot(subject=subject, parc="aparc", subjects_dir=subjects_dir, regexp='parstriangularis')
                prefrontal5  = mne.read_labels_from_annot(subject=subject, parc="aparc", subjects_dir=subjects_dir, regexp='parsorbitalis')
                precentral   = mne.read_labels_from_annot(subject=subject, parc="aparc", subjects_dir=subjects_dir, regexp='precentral')
                inf_parietal = mne.read_labels_from_annot(subject=subject, parc="aparc", subjects_dir=subjects_dir, regexp='inferiorparietal')

                # combine them per hemisphere           
                all_labels_lh = prefrontal1[0] + prefrontal2[0] + prefrontal3[0] + prefrontal4[0] + prefrontal5[0] + precentral[0] + inf_parietal[0]
                all_labels_rh = prefrontal1[1] + prefrontal2[1] + prefrontal3[1] + prefrontal4[1] + prefrontal5[1] + precentral[1] + inf_parietal[1]

                # point to all vertices in src corresponding to these labels (to restrict FC computation to these areas only)
                verts2include = [i.in_label(mne.BiHemiLabel(all_labels_lh,all_labels_rh)) for i in stcs_jump]
                
                # combine seed STCs with all other stcs (from ROIs ONLY) for connectivity analysis
                jump_comb_ts_lh = list(zip(stc_jump_lh, verts2include))
                jump_comb_ts_rh = list(zip(stc_jump_rh, verts2include))
                stay_comb_ts_lh = list(zip(stc_stay_lh, verts2include))
                stay_comb_ts_rh = list(zip(stc_stay_rh, verts2include))

                # this is some stuff we need to specify how FC is computed by the mne-connectivity toolbox 
                vertices      = [verts2include[0].vertices[i] for i in range(2)]
                n_signals_tot = 1 + len(vertices[0]) + len(vertices[1])
                indices       = mne_connectivity.seed_target_indices([0], np.arange(1, n_signals_tot)) 
            else:
                # combine seed STCs with all other stcs in the entire cortex for connectivity analysis
                jump_comb_ts_lh = list(zip(stc_jump_lh, stcs_jump))
                jump_comb_ts_rh = list(zip(stc_jump_rh, stcs_jump))
                stay_comb_ts_lh = list(zip(stc_stay_lh, stcs_stay))
                stay_comb_ts_rh = list(zip(stc_stay_rh, stcs_stay))

                # this is some stuff we need to specify how FC is computed by the mne-connectivity toolbox 
                vertices      = [src[i]['vertno'] for i in range(2)]
                n_signals_tot = 1 + len(vertices[0]) + len(vertices[1])
                indices       = mne_connectivity.seed_target_indices([0], np.arange(1, n_signals_tot)) 

            # Some parameters for the FC computation 
            tmin  = -.4
            tmax  = 1.4
            fmin  = 4
            fmax  = 30
            sfreq = all_epochs.info['sfreq']  # the sampling frequency
            cwt_freqs = np.arange(fmin, fmax+1, 1)
            cwt_n_cycles = (cwt_freqs / 7) if varcycles else 4

            # FC: left/jump seed to entire cortex
            print('\n>>> Computing FC for jump events - lh\n')
            jump_con_lh = mne_connectivity.spectral_connectivity_epochs(
                jump_comb_ts_lh, method=fc_method, mode=con_method, indices=indices, sfreq=sfreq,
                cwt_freqs=cwt_freqs, cwt_n_cycles=cwt_n_cycles, tmin=tmin, tmax=tmax)       
            # FC: left/stay seed to entire cortex
            print('\n>>> Computing FC for stay events - lh\n')
            stay_con_lh = mne_connectivity.spectral_connectivity_epochs(
                stay_comb_ts_lh, method=fc_method, mode=con_method, indices=indices, sfreq=sfreq,
                cwt_freqs=cwt_freqs, cwt_n_cycles=cwt_n_cycles, tmin=tmin, tmax=tmax) 
            
            # FC: right/jump seed to entire cortex
            print('\n>>> Computing FC for jump events - rh\n')
            jump_con_rh = mne_connectivity.spectral_connectivity_epochs(
                jump_comb_ts_rh, method=fc_method, mode=con_method, indices=indices, sfreq=sfreq,
                cwt_freqs=cwt_freqs, cwt_n_cycles=cwt_n_cycles, tmin=tmin, tmax=tmax)  
            # FC: right/stay seed to entire cortex 
            print('\n>>> Computing FC for stay events - rh\n')
            stay_con_rh = mne_connectivity.spectral_connectivity_epochs(
                stay_comb_ts_rh, method=fc_method, mode=con_method, indices=indices, sfreq=sfreq,
                cwt_freqs=cwt_freqs, cwt_n_cycles=cwt_n_cycles, tmin=tmin, tmax=tmax) 

            # save data for this subject
            if varcycles:
                np.savez(join(cfg.paradigm_dir,sub_i,this_visit,fname.replace('*','jump_varcycles')), data_lh=jump_con_lh.get_data(), 
                        data_rh=jump_con_rh.get_data(), vertices_lh=vertices[0], vertices_rh=vertices[1], times=jump_con_lh.times, freqs=jump_con_lh.freqs)
                np.savez(join(cfg.paradigm_dir,sub_i,this_visit,fname.replace('*','stay_varcycles')), data_lh=stay_con_lh.get_data(), 
                        data_rh=stay_con_rh.get_data(), vertices_lh=vertices[0], vertices_rh=vertices[1], times=stay_con_rh.times, freqs=stay_con_rh.freqs)
            else:
                np.savez(join(cfg.paradigm_dir,sub_i,this_visit,fname.replace('*','jump')), data_lh=jump_con_lh.get_data(), 
                        data_rh=jump_con_rh.get_data(), vertices_lh=vertices[0], vertices_rh=vertices[1], times=jump_con_lh.times, freqs=jump_con_lh.freqs)
                np.savez(join(cfg.paradigm_dir,sub_i,this_visit,fname.replace('*','stay')), data_lh=stay_con_lh.get_data(), 
                        data_rh=stay_con_rh.get_data(), vertices_lh=vertices[0], vertices_rh=vertices[1], times=stay_con_rh.times, freqs=stay_con_rh.freqs)

            # data from the frequency band and time window of interest that we want to plot (for plotting only) 
            times2plot = (np.array(jump_con_lh.times) >= .55) & (np.array(jump_con_lh.times) <= .75)

            # left seed to cortex
            data2plot_jump_lh  = np.mean(jump_con_lh.get_data()[:,(np.array(jump_con_lh.freqs) >= fmin) & (np.array(jump_con_lh.freqs) <= fmax),:], axis=1)  # average across freqs
            data2plot_jump_lh  = np.mean(data2plot_jump_lh[:,times2plot], axis=1)                                                                            # average across time
            data2plot_stay_lh  = np.mean(stay_con_lh.get_data()[:,(np.array(stay_con_lh.freqs) >= fmin) & (np.array(stay_con_lh.freqs) <= fmax),:], axis=1)
            data2plot_stay_lh  = np.mean(data2plot_stay_lh[:,times2plot], axis=1)

            data2plot_jump_rh  = np.mean(jump_con_rh.get_data()[:,(np.array(jump_con_rh.freqs) >= fmin) & (np.array(jump_con_rh.freqs) <= fmax),:], axis=1) # average across freqs
            data2plot_jump_rh  = np.mean(data2plot_jump_rh[:,times2plot], axis=1)                                                                           # average across time
            data2plot_stay_rh  = np.mean(stay_con_rh.get_data()[:,(np.array(stay_con_rh.freqs) >= fmin) & (np.array(stay_con_rh.freqs) <= fmax),:], axis=1)
            data2plot_stay_rh  = np.mean(data2plot_stay_rh[:,times2plot], axis=1)
            
        else:
            if varcycles:
                # if data already exists, load it and get the info we need
                FC_jump_data = np.load(join(cfg.paradigm_dir,sub_i,this_visit,fname.replace('*','jump_varcycles')))
                FC_stay_data = np.load(join(cfg.paradigm_dir,sub_i,this_visit,fname.replace('*','stay_varcycles')))
            else:
                # if data already exists, load it and get the info we need
                FC_jump_data = np.load(join(cfg.paradigm_dir,sub_i,this_visit,fname.replace('*','jump')))
                FC_stay_data = np.load(join(cfg.paradigm_dir,sub_i,this_visit,fname.replace('*','stay')))

            # data from the frequency band and time window of interest that we want to plot (for plotting only) 
            times2plot = (np.array(FC_jump_data['times']) >= .55) & (np.array(FC_jump_data['times']) <= .75)

            # left seed to cortex
            data2plot_jump_lh  = np.mean(FC_jump_data['data_lh'][:,(FC_jump_data['freqs'] >= fmin) & (FC_jump_data['freqs'] <= fmax),:], axis=1)  # average across freqs
            data2plot_jump_lh  = np.mean(data2plot_jump_lh[:,times2plot], axis=1)                                                                 # average across time
            data2plot_stay_lh  = np.mean(FC_stay_data['data_lh'][:,(FC_stay_data['freqs'] >= fmin) & (FC_stay_data['freqs'] <= fmax),:], axis=1)  
            data2plot_stay_lh  = np.mean(data2plot_stay_lh[:,times2plot], axis=1)

            data2plot_jump_rh  = np.mean(FC_jump_data['data_rh'][:,(FC_jump_data['freqs'] >= fmin) & (FC_jump_data['freqs'] <= fmax),:], axis=1)  # average across freqs
            data2plot_jump_rh  = np.mean(data2plot_jump_rh[:,times2plot], axis=1)                                                                 # average across time
            data2plot_stay_rh  = np.mean(FC_stay_data['data_rh'][:,(FC_stay_data['freqs'] >= fmin) & (FC_stay_data['freqs'] <= fmax),:], axis=1)  
            data2plot_stay_rh  = np.mean(data2plot_stay_rh[:,times2plot], axis=1)

            # put together the list of vertices again
            vertices = [FC_jump_data['vertices_lh'],FC_jump_data['vertices_rh']]

        # create and STC with the data to plot for the MNE report
        jump_con_stc_lh = mne.SourceEstimate( data2plot_jump_lh, vertices=vertices, tmin=0, tstep=1, subject=subject)
        stay_con_stc_lh = mne.SourceEstimate( data2plot_stay_lh, vertices=vertices, tmin=0, tstep=1, subject=subject)
        
        # create and STC with the data to plot for the MNE report
        jump_con_stc_rh = mne.SourceEstimate(data2plot_jump_rh, vertices=vertices, tmin=0, tstep=1, subject=subject)
        stay_con_stc_rh = mne.SourceEstimate(data2plot_stay_rh, vertices=vertices, tmin=0, tstep=1, subject=subject)

        # now we do the same for the difference
        brain_fig, axes = plt.subplots(2,4, figsize=(15,9))
        views    = ['lateral','lateral','dorsal','ventral']            
        kind     = 'percent'
        pos_lims = (90, 92, 95)

        brain = (jump_con_stc_lh-stay_con_stc_lh).plot(surface='inflated', hemi='both',
                            time_label=fc_method,
                            subjects_dir=subjects_dir,
                            clim=dict(kind=kind, pos_lims=pos_lims))
        brain.add_label(func_label_lh, color='black')

        for val,view in enumerate(views):
            hemi = 'rh' if val == 1 else 'lh'  
            brain.show_view(view=view,hemi=hemi)
            img = brain.screenshot()
            if val < 2:
                axes[0][val].imshow(img)
                axes[0][val].axis('off')
            else:
                axes[1][val-2].imshow(img)
                axes[1][val-2].axis('off')       
        brain.close()   
        brain_fig.text(0.28,.9, 'left_seed', fontsize=14)

        # plot connectivity for right label
        brain = (jump_con_stc_rh-stay_con_stc_rh).plot(surface='inflated', hemi='both',
                            time_label=fc_method,
                            subjects_dir=subjects_dir,
                            show_traces=False,
                            clim=dict(kind=kind, pos_lims=pos_lims))
        brain.add_label(func_label_rh, color='black')

        for val,view in enumerate(views):
            hemi = 'rh' if val == 1 else 'lh'  
            brain.show_view(view=view,hemi=hemi)
            img = brain.screenshot()
            if val < 2:
                axes[0][val+2].imshow(img)
                axes[0][val+2].axis('off')
            else:
                axes[1][val].imshow(img)
                axes[1][val].axis('off')       
        brain.close()
        brain_fig.text(0.68,.9, 'right_seed', fontsize=14)
        brain_fig.suptitle('Jump - Stay\ntime = 550-750ms, freq = %s' %(freq_band), fontsize=14)

        # add figure to 
        report.add_figure(fig=brain_fig, title='FC_Jump-Stay_'+freq_band+'_'+fc_method, section=sub_i, tags=(['FC_' + freq_band,diagnosis]), replace=True)
        report.save(join(cfg.paradigm_dir,source_method+'_'+cfg.report_name), verbose=False, overwrite=True)
        plt.close('all')

report.save(join(cfg.paradigm_dir,source_method+'_'+cfg.report_name.replace('.hdf5','.html')),verbose=False,overwrite=True)