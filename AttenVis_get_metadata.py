import mne
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import helper_functions as tlbx
import AttenVis_config as cfg

participants_df, participants_to_study = tlbx.load_participants(cfg.participants_csv)

participant = '148501'
transcend_data_dir = os.path.join(cfg.transcend_data_dir,'AttenVis',participant)
metadata_tmin = -0.5
metadata_tmax = 3

# Load the data
diagnosis,study,visit_dir,subjID_date = tlbx.get_participant_details(participants_df,participant)
raw_sss_file = tlbx.find_files('_raw_tsss.fif',visit_dir)
raw_sss_file.sort()
epochs_list = []
epochs_list_baseline = []
for file in raw_sss_file:

    raw_sss = mne.io.read_raw_fif(file,preload=True,verbose=False)
    ica_file = file.replace('_raw_tsss.fif','_ica.fif')
    ica = mne.preprocessing.read_ica(ica_file)
    ica.apply(raw_sss)
    events_fname_tag = os.path.split(file)[1].replace('_raw_tsss.fif','_fixed_eve.fif')
    events_fname = tlbx.find_files(events_fname_tag,transcend_data_dir)
    events = mne.read_events(events_fname[0]) 
    row_events = [
        "target"
    ]
    keep_first = ["condition","response"]
    metadata, events_meta, event_id_meta = mne.epochs.make_metadata(
        events=events,
        event_id=cfg.event_dict,
        tmin=metadata_tmin,
        tmax=metadata_tmax,
        sfreq=raw_sss.info["sfreq"],
        row_events=row_events,
        keep_first=keep_first,
    )  
    metadata[['Condition','difficulty']] = metadata['first_condition'].str.split('/',expand = True)
    metadata['RT'] = metadata['response'] - metadata['condition']
    metadata.reset_index(drop=True,inplace=True)
    metadata = metadata.drop(columns=['condition/search/4','condition/search/6','condition/search/8','condition/search/10','condition/pop-out/4','condition/pop-out/6','condition/pop-out/8','condition/pop-out/10','target','response/right','response/left'])

    epochs_tmin, epochs_tmax = -0.5, 2.5 # epochs range: [-0.1, 0.4] s
    epochs = mne.Epochs(
        raw=raw_sss,
        events=events_meta,
        tmin=epochs_tmin,
        tmax=epochs_tmax,
        event_id=event_id_meta,
        reject = None,
        reject_by_annotation = False,
        picks="meg",
        baseline = None,
        on_missing="ignore",
        metadata=metadata,
    ).load_data()
    epochs_list.append(epochs)

    epochs_baseline = mne.Epochs(
        raw=raw_sss,
        events=events_meta,
        tmin=epochs_tmin,
        tmax=epochs_tmax,
        event_id=event_id_meta,
        reject = None,
        reject_by_annotation = False,
        picks="meg",
        baseline = (-0.3,-0.1),
        on_missing="ignore",
        metadata=metadata,
    ).load_data()
    epochs_list_baseline.append(epochs_baseline)
# epochs["(Condition == 'search')"].plot(n_epochs=10,events=events)
all_epochs = mne.concatenate_epochs(epochs_list)
all_epochs_baseline = mne.concatenate_epochs(epochs_list_baseline)
condition_epochs = tlbx.get_condition_epochs(all_epochs,"(Condition == 'search')")
evoked = tlbx.get_evoked(condition_epochs,filter=None,baseline=(-0.3,-0.1))

condition_epochs_baseline = tlbx.get_condition_epochs(all_epochs_baseline,"(Condition == 'search')")
evoked_baseline = tlbx.get_evoked(condition_epochs_baseline,filter=None,baseline=None)
#check covariance
fwd_fname = tlbx.find_files('_fwd.fif',visit_dir)[0]
fwd = mne.read_forward_solution(fwd_fname)

erm_fname = tlbx.find_files('_erm_raw_sss.fif',visit_dir)[0]
erm = mne.io.read_raw_fif(erm_fname,preload=True,verbose=False)
cov = mne.compute_raw_covariance(erm, tmin=0, method='auto', rank=None, tmax=None)
inv_fname = erm_fname.replace('_raw_sss.fif','_test_erm_inv.fif')
# mne.write_cov(os.path.join(visit_dir,covfname), cov, overwrite=True)
cov.plot(erm.info,proj=True)
inv_operator  = mne.minimum_norm.make_inverse_operator(erm.info, fwd, cov, loose=0.2, depth=0.8, rank='info')
mne.minimum_norm.write_inverse_operator(inv_fname,inv_operator)

#erm with ica applied
ica.apply(erm)
cov_ica = mne.compute_raw_covariance(erm, tmin=0, method='auto', rank=None, tmax=None)
inv_fname = erm_fname.replace('_raw_sss.fif','_test_erm_ica_inv.fif')
cov_ica.plot(erm.info,proj=True)
inv_operator  = mne.minimum_norm.make_inverse_operator(erm.info, fwd, cov_ica, loose=0.2, depth=0.8, rank='info')
mne.minimum_norm.write_inverse_operator(inv_fname,inv_operator)


noise_cov = mne.compute_covariance(all_epochs_baseline, tmax=0, method = "auto",rank = None)
noise_cov.plot(all_epochs_baseline.info,proj=True)
inv_operator  = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, noise_cov, loose=0.2, depth=0.8, rank='info')
inv_fname = erm_fname.replace('_raw_sss.fif','_test_prestim_baseline_inv.fif')
mne.minimum_norm.write_inverse_operator(inv_fname,inv_operator)


evoked.plot_white(cov,time_unit = "s")
evoked.plot_white(cov_ica,time_unit = "s")
evoked.plot_white(noise_cov,time_unit = "s")

metadata

