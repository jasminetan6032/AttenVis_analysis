import os
import numpy as np
import pandas as pd

import helper_functions as tlbx
import AttenVis_connectivity_config as cfg
from mne.parallel import parallel_func

participants_df, participants_to_study = tlbx.load_participants()      

def calculate_zcoh(sub_id, overwrite_data=True):
    """
    Calculate the z-coherence for a given subject.
    """
    diagnosis, study,visit_dir,subjID_date = tlbx.read_participant_details_from_dataframe(participants_df,sub_id)
    participant_data_savename = os.path.join(visit_dir,cfg.data_fname.replace('.pkl','_' + sub_id + '.pkl'))
    df = pd.read_pickle(participant_data_savename)
    for hemi in cfg.hemisphere:
        for target_hemi in cfg.hemisphere:
            df_to_analyse = df[(df["hemisphere"]== hemi) & (df['target_hemi']== target_hemi)]
            zcoh_hemi = tlbx.get_zcoh(df_to_analyse,'search','pop-out')
            zcoh_data = [sub_id,subjID_date,diagnosis,hemi,target_hemi,zcoh_hemi,df['time'].values[0]]

    return zcoh_data

parallel, run_func, _ = parallel_func(calculate_zcoh, n_jobs=1)
results = parallel(run_func(sub_id) for sub_id in participants_to_study)

zcoh_rows = [res[0] for res in results] 
zcoh_df = pd.DataFrame(zcoh_rows, columns=['Participant','SubjID_Date','Diagnosis','hemisphere','target_hemi','zcoh','time'])
zcoh_df.to_pickle(cfg.connectivity_compare_data_savename.replace(".pkl","_zcoh.pkl"))
