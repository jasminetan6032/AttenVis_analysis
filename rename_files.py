
import helper_functions as tlbx

from mne.parallel import parallel_func

participants_df, participants_to_study = tlbx.load_participants()
parallel, run_func, _ = parallel_func(tlbx.rename_files, n_jobs=8)
parallel(run_func(sub_id,'cross_freq_stimuli_high_search_pop-out_V1_MNE_0.8_1.15_','high','low') for sub_id in participants_to_study)