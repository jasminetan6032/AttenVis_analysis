import mne
import pandas as pd
import helper_functions as tlbx
import AttenVis_config as cfg
from mne.parallel import parallel_func
import matplotlib.pyplot as plt    

overwrite_data = False
participants_df, participants_to_study = tlbx.load_participants()
if cfg.rt_report_savename_hdf5 is None:
    report = mne.Report(title='AttenVis Reaction Times')
    report.save(cfg.rt_report_savename_hdf5, overwrite=True)
else:
    report = mne.open_report(cfg.rt_report_savename_hdf5)

def analyse_RTs(sub_id):
    #load epochs
    visit_dir = participants_df[participants_df['Participant'] == sub_id]['Visit_Dir'].values[0]
    diagnosis = participants_df[participants_df['Participant'] == sub_id]['Diagnosis'].values[0]
    study = participants_df[participants_df['Participant'] == sub_id]['Study'].values[0]
    subjID_date = participants_df[participants_df['Participant'] == sub_id]['SubjID_Date'].values[0]
    load_fname, epochs = tlbx.load_epochs('_nobaseline_nofilter_all_conditions_metadata_epo.fif',visit_dir,resample=True)
    metadata = epochs.metadata
    metadata['Participant'] = sub_id
    metadata['Diagnosis'] = diagnosis
    metadata['Study'] = study
    metadata['SubjID_Date'] = subjID_date

    cleaned_metadata,summary = tlbx.clean_metadata(metadata,rt_based=(0.15,1.2), percent=None, correct_answers_only=True)
    hist_fig = tlbx.plot_participant_RT_hist(metadata, cleaned_metadata)
    RT_fig = tlbx.plot_RT(cleaned_metadata)
    return sub_id, cleaned_metadata, summary, hist_fig, RT_fig

n_jobs = 6

debug = True
if debug:
    n_jobs = 1

if overwrite_data:
    parallel, run_func, _ = parallel_func(analyse_RTs, n_jobs=n_jobs)
    results = parallel(run_func(subject) for subject in participants_to_study)
    all_dfs = [entry[1] for entry in results]
    df = pd.concat(all_dfs, ignore_index=True)
    df.to_pickle(cfg.rt_data_savename)

    for sub_id, cleaned_metadata, summary, hist_fig, RT_fig in results:
        report.add_figure(fig=hist_fig, title='RT_hist', section=sub_id, tags=['histogram'], replace=True)
        report.add_figure(fig=RT_fig, title='RTs', section=sub_id, tags=['line_plot'], replace=True)
        tlbx.add_table_to_report(summary, report, sub_id)
        plt.close(hist_fig)
        plt.close(RT_fig)
    report.save(cfg.rt_report_savename_hdf5, verbose=False, overwrite=True)
else:
    df = pd.read_pickle(cfg.rt_data_savename)

fig = tlbx.plot_RT(df,group = True)
report.add_figure(fig=fig, title='RTs', section='gavg', tags=['line_plot'], replace=True)
plt.close(fig)

# fig = tlbx.plot_RT(df[df['Diagnosis']=='asd'])
# report.add_figure(fig=fig, title='RTs', section='gavg_asd', tags=['line_plot'], replace=True)
# plt.close(fig)
# fig = tlbx.plot_RT(df[df['Diagnosis']=='td'])
# report.add_figure(fig=fig, title='RTs', section='gavg_td', tags=['line_plot'], replace=True)
# plt.close(fig)

median_df = df.groupby(['Condition', 'difficulty','Diagnosis'])['RT'].median().reset_index()
min_df = df.groupby(['Condition', 'difficulty','Diagnosis'])['RT'].min().reset_index()

print(median_df)
print(min_df)

tlbx.add_table_to_report(median_df,report,'gavg_median')
tlbx.add_table_to_report(min_df,report,'gavg_min')
fig = tlbx.plot_participant_RT_hist(median_df,min_df)
report.add_figure(fig=fig, title='RT_hist', section='gavg_median_min', tags=['histogram'], replace=True)
plt.close(fig)

report.save(cfg.rt_report_savename_hdf5, verbose=False, overwrite=True)

tlbx.show_report(cfg.rt_report_savename_hdf5)