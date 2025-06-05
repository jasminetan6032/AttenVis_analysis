import mne
import helper_functions as tlbx
import AttenVis_config as cfg

participants_df, participants_to_study = tlbx.load_participants()
# participants_to_study = ['008301']
if cfg.inv_report_savename_hdf5 is None:
    report = mne.Report(title='AttenVis Prestimulus Inverses (stimuli-locked)')
    report.save(cfg.inv_report_savename_hdf5, overwrite=True)
else:
    report = mne.open_report(cfg.inv_report_savename_hdf5)

tlbx.initialize_error_log()

for participant in participants_to_study:
    visit_dir = participants_df[participants_df['Participant'] == participant]['Visit_Dir'].values[0]
    diagnosis = participants_df[participants_df['Participant'] == participant]['Diagnosis'].values[0]
    study = participants_df[participants_df['Participant'] == participant]['Study'].values[0]
    subjID_date = participants_df[participants_df['Participant'] == participant]['SubjID_Date'].values[0]
    dataname = tlbx.find_files('_nobaseline_nofilter_all_conditions_metadata_epo.fif',visit_dir)
    if not dataname or cfg.overwrite_epochs:
        print(f"No epochs found for participant {participant} or overwriting epochs. Getting epochs...")
        all_epochs = tlbx.epochs_metadata(participant,visit_dir,locked_to = 'stimuli',overwrite=True)
    else:
        all_epochs = mne.read_epochs(dataname[0],preload=True)
    condition_epochs = tlbx.get_condition_epochs(all_epochs.copy(),condition = None)
    evoked = tlbx.get_evoked(condition_epochs,filter=None,baseline=cfg.prestimulus_baseline)
    tlbx.inverse_from_prestimulus_baseline(participant,all_epochs,evoked,visit_dir,report,overwrite=False)
    tlbx.add_whitened_evoked_erm(participant,visit_dir,evoked,report)
tlbx.show_report(cfg.inv_report_savename_hdf5)