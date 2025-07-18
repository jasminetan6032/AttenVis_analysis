import os
import numpy as np
import pandas as pd

import helper_functions as tlbx
import AttenVis_connectivity_config as cfg
import matplotlib.pyplot as plt   
import matplotlib as mpl 
from mne.parallel import parallel_func

participants_df, participants_to_study = tlbx.load_participants()      

def plot_participants_connectivity(sub_id):
    """
    Plot coherence line and tfrs for all participants.
    """
    diagnosis, study,visit_dir,subjID_date = tlbx.read_participant_details_from_dataframe(participants_df,sub_id)
    participant_data_savename = os.path.join(visit_dir,cfg.data_fname.replace('.pkl','_' + sub_id + '.pkl'))
    df = pd.read_pickle(participant_data_savename)
    pics = tlbx.plot_participant_coh_line(df, (6, 12), skip_brain_images=True)
    tfr_pics_bihemi = []
    for hemi in cfg.hemisphere:
        df_hemi = df[df["hemisphere"] == hemi]
        tfr_pics = tlbx.plot_participant_tfrs(df_hemi, sub_id, label_hemi=hemi)
        tfr_pics_bihemi.append(tfr_pics)
    return sub_id, pics, tfr_pics_bihemi


# parallel, run_func, _ = parallel_func(plot_participants_connectivity, n_jobs=3)
# results = parallel(run_func(sub_id) for sub_id in participants_to_study)

report = tlbx.generate_report()

# for sub_id, pics, tfr_pics_bihemi in results:
#     for pic,title in pics:
#         report.add_figure(fig=pic, title=sub_id + '_' + title, section=sub_id, tags=['coherence'], replace=True)
#         plt.close(pic)

#     for tfr_pics in tfr_pics_bihemi:
#         for pic,title,_,_ in tfr_pics:
#             report.add_figure(fig=pic, title=title, section=sub_id, tags=['tfr'], replace=True)
#             plt.close(pic)

# report.save(cfg.report_savename_hdf5, verbose=False, overwrite=True)

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