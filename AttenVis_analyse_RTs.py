import mne
import pandas as pd
import helper_functions as tlbx
import AttenVis_config as cfg

participants_df, participants_to_study = tlbx.load_participants()
# participants_to_study = ['108901']
if cfg.rt_report_savename_hdf5 is None:
    report = mne.Report(title='AttenVis Reaction Times')
    report.save(cfg.rt_report_savename_hdf5, overwrite=True)
else:
    report = mne.open_report(cfg.rt_report_savename_hdf5)

all_participants = []

for sub_id in participants_to_study:
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

    cleaned_metadata = tlbx.clean_metadata(metadata,0.1)
    tlbx.plot_participant_RT_hist(metadata, cleaned_metadata,report,sub_id)
    tlbx.plot_RT(cleaned_metadata,report,sub_id)
    all_participants.append(cleaned_metadata)

df = pd.concat(all_participants)
df.to_pickle(cfg.rt_data_savename)
tlbx.plot_RT(df,report,'gavg')
tlbx.plot_RT(df[df['Diagnosis']=='asd'],report,'gavg_asd')
tlbx.plot_RT(df[df['Diagnosis']=='td'],report,'gavg_td')

median_df = df.groupby(['Condition', 'difficulty','Diagnosis'])['RT'].median().reset_index()
print(median_df)

tlbx.show_report(cfg.rt_report_savename_hdf5)