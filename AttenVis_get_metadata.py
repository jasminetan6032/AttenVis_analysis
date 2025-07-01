import mne
import helper_functions as tlbx
import AttenVis_config as cfg
from mne.parallel import parallel_func


participants_df, participants_to_study = tlbx.load_participants()
participants_to_study = participants_to_study[71:] 

tlbx.initialize_error_log()

def get_inverse(participant):
    """Get inverse operator for a participant."""
    diagnosis, study,visit_dir,subjID_date = tlbx.read_participant_details_from_dataframe(participants_df,participant)
    dataname = tlbx.find_files('_nobaseline_nofilter_all_conditions_metadata_epo.fif',visit_dir)
    if not dataname or cfg.overwrite_epochs:
        print(f"No epochs found for participant {participant} or overwriting epochs. Getting epochs...")
        all_epochs = tlbx.epochs_metadata(participant,visit_dir,locked_to = 'stimuli',overwrite=True)
    else:   
        all_epochs = mne.read_epochs(dataname[0],preload=True)
    condition_epochs = tlbx.get_condition_epochs(all_epochs.copy(),condition = None)
    evoked = tlbx.get_evoked(condition_epochs,filter=None,baseline=cfg.prestimulus_baseline)
    cov_fname = tlbx.inverse_from_prestimulus_baseline(all_epochs,visit_dir,overwrite=True)
    return participant, visit_dir, all_epochs, evoked, cov_fname

n_jobs = 2

debug = False
if debug:
    n_jobs = 1

parallel, run_func, _ = parallel_func(get_inverse, n_jobs=n_jobs)
results = parallel(run_func(participant) for participant in participants_to_study)

report = tlbx.generate_report(inv=True)

# Add evoked and covariance to report
for participant, visit_dir, all_epochs, evoked, cov_fname in results:
    tlbx.add_whitened_evoked_prestim_baseline(participant, cov_fname, all_epochs, evoked, report)
    tlbx.add_whitened_evoked_erm(participant, visit_dir, evoked, report)
tlbx.show_report(cfg.inv_report_savename_hdf5)
