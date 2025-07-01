#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 12:31:59 2024

@author: jwt30
"""

import mne
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib
matplotlib.use('Agg') 
import helper_functions as tlbx
import AttenVis_power_config as cfg
# import time

participants_df, participants_to_study = tlbx.load_participants()
report = tlbx.generate_report()

for sub_id in participants_to_study:
    participant_data = []
    diagnosis, study,visit_dir,subjID_date = tlbx.read_participant_details_from_dataframe(participants_df,sub_id)
    participant_data_savename = os.path.join(visit_dir,cfg.data_fname.replace('.pkl','_' + sub_id + '.pkl'))
    df_participant = pd.read_pickle(participant_data_savename)
    tlbx.add_participant_tfrs_to_report(df_participant,report,sub_id)

df = tlbx.collate_participants_data(participants_df,participants_to_study)

mpl.rcParams["svg.fonttype"] = "none"

tlbx.add_tfrs_to_report(df,report,'gavg')
tlbx.add_gavg_power_over_time_to_report(df,report,'gavg','Diagnosis',(40,80),ci=False)
tlbx.add_gavg_power_over_time_to_report(df,report,'gavg','Condition',(40,80),ci=False)

# tlbx.add_fsaverage_to_report(report,df,cfg.labels_of_interest[0] + '_grown')

tlbx.show_report(cfg.report_savename_hdf5)