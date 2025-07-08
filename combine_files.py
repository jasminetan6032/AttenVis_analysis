import mne
import os
import helper_functions as tlbx
from mne.parallel import parallel_func


participants_df, participants_to_study = tlbx.load_participants()
check_participants = []
def combine_epochs(participant):
    diagnosis, study,visit_dir,subjID_date = tlbx.read_participant_details_from_dataframe(participants_df,participant)
    epochs_search_name,epochs_search = tlbx.load_epochs('_AttenVis_nobaseline_nofilter_metadata_search_behaviour_cleaned_epo.fif', visit_dir)
    print(epochs_search_name)
    epochs_pop_out_name,epochs_pop_out = tlbx.load_epochs('_AttenVis_nobaseline_nofilter_metadata_pop-out_behaviour_cleaned_epo.fif', visit_dir)
    print(epochs_pop_out_name)
    combined_epochs = mne.concatenate_epochs([epochs_search, epochs_pop_out])
    combined_epochs.save(os.path.join(visit_dir, participant + '_AttenVis_nobaseline_nofilter_metadata_conditions_combined_behaviour_cleaned_epo.fif'), overwrite=True)

n_jobs = 8

debug = False
if debug:
    n_jobs = 1

parallel, run_func, _ = parallel_func(combine_epochs, n_jobs=n_jobs)
results = parallel(run_func(subject) for subject in participants_to_study)