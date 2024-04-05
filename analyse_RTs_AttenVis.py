#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 12:35:04 2024

@author: jwt30
"""
import pandas
import seaborn as sns
from scipy import stats
import scipy
import os
import matplotlib.pyplot as plt
import numpy

def find_files(search_string,data_dir):
    files = []
    for path, directory_names, filenames in os.walk(data_dir):
        for filename in filenames:
            if search_string in filename:
                file = os.path.join(path,filename)
                files.append(file)
                
    return files 

def import_redcap(file):
    redcap = pandas.DataFrame(pandas.read_csv(file,encoding = 'utf-8'))[["subject_id","asd","prescreen_survey_asd_v2","prescreen_survey_asd"]]
    redcap.loc[(redcap["asd"].isna() & redcap["prescreen_survey_asd_v2"].isna() & redcap["prescreen_survey_asd"].isna()),"to_delete"]=1
    redcap.loc[redcap["to_delete"].isna(),"to_delete"]=0
    redcap = redcap.loc[redcap["to_delete"]==0,:]
    redcap=redcap.replace({numpy.nan:None})
    redcap.loc[((redcap["asd"]==1) | (redcap["prescreen_survey_asd_v2"]==1) | (redcap["prescreen_survey_asd"]==1)),"asd_final"]=1
    redcap.loc[((redcap["asd"]==0) | (redcap["prescreen_survey_asd_v2"]==0) | (redcap["prescreen_survey_asd"]==0)),"asd_final"]=0
    redcap['subject_id'] = redcap['subject_id'].astype(str)
    redcap['subject_id'] = redcap['subject_id'].str.zfill(6)
    
    return redcap

    
def import_data(mat_file):
    mat = scipy.io.loadmat(mat_file)
    responses = pandas.DataFrame(numpy.stack((mat['triggers'][0],mat['correctTrials'][0],mat['reactionTime'][0]),axis=1))
    responses.columns = ["triggers","response_correct","RT"]
    
    responses.loc[responses["triggers"]==1,"condition"]="search"
    responses.loc[responses["triggers"]==2,"condition"]="search"
    responses.loc[responses["triggers"]==3,"condition"]="search"
    responses.loc[responses["triggers"]==4,"condition"]="search"
    responses.loc[responses["triggers"]==5,"condition"]="pop-out"
    responses.loc[responses["triggers"]==6,"condition"]="pop-out"
    responses.loc[responses["triggers"]==7,"condition"]="pop-out"
    responses.loc[responses["triggers"]==8,"condition"]="pop-out"
    responses.loc[responses["triggers"]==1,"difficulty"]=4
    responses.loc[responses["triggers"]==2,"difficulty"]=6
    responses.loc[responses["triggers"]==3,"difficulty"]=8
    responses.loc[responses["triggers"]==4,"difficulty"]=10
    responses.loc[responses["triggers"]==5,"difficulty"]=4
    responses.loc[responses["triggers"]==6,"difficulty"]=6
    responses.loc[responses["triggers"]==7,"difficulty"]=8
    responses.loc[responses["triggers"]==8,"difficulty"]=10
    
    responses["response_correct"].replace({0:"Incorrect Response",1:"Hit"},inplace=True)
    
    return responses


def add_group_id(redcap,dataset,dataset_file):
    participant = os.path.split(dataset_file)[1].split('_')[0]
    group = redcap.loc[redcap["subject_id"]==participant,"asd_final"].item()
    dataset["group"] = group
    
    return dataset

def clean_data(dataset,percent):
    correct_answers = dataset.loc[dataset['response_correct']=="Hit",:]
    correct_answers_lower_bound = correct_answers.loc[correct_answers["RT"]>0.1]
    convert_percent = 1-percent
    percent_cutoff = correct_answers_lower_bound["RT"].quantile(convert_percent)
    plt.figure()
    plt.hist(correct_answers["RT"],bins=30)
    correct_answers_cleaned = correct_answers_lower_bound.loc[correct_answers_lower_bound["RT"]<percent_cutoff]
    plt.figure()
    plt.hist(correct_answers_cleaned["RT"],bins=30)
    
    return correct_answers_cleaned

exclude_participants = ['073801','125401']

results_dir = "/autofs/space/transcend/MEG/AttenVis"

results = find_files('_behaviour.mat', results_dir) 
redcap_file = '/homes/7/jwt30/Downloads/TRANSCENDAllProjects-VisualSearchWithBeha_DATA_2024-01-30_1700.csv'
redcap = import_redcap(redcap_file)

all_participants = []
for file in results:
    participant = os.path.split(file)[1].split('_')[0]
    if participant not in exclude_participants:
        data = import_data(file)
        data_with_group = add_group_id(redcap,data,file)
        cleaned_data = clean_data(data_with_group,0.1).reset_index()
        all_participants.append(cleaned_data)
        all_participants_plotdata = pandas.concat(all_participants)
        
all_participants_plotdata["group"].replace({0:"TD",1:"ASD"},inplace=True)
#     average_RT = cleaned_data.groupby(by=["Condition","Noise"]).mean()
#     ttest_data = numpy.array([numpy.mean(average_RT['RT']['meaningful']),numpy.mean(average_RT['RT']['nonsense'])])
    
#     if cleaned_data['Group'][0]== 'TD':
#         TD_data.append(ttest_data)

#     elif cleaned_data['Group'][0]== 'ASD':
#         ASD_data.append(ttest_data)

    
# TD_data = numpy.vstack(TD_data) 
# ttest_result = stats.wilcoxon(TD_data[:,0], TD_data[:,1])
# result = [ttest_result.statistic,ttest_result.pvalue]
# print(result)
# ASD_data = numpy.vstack(ASD_data)
# ttest_result = stats.wilcoxon(ASD_data[:,0], ASD_data[:,1])
# result = [ttest_result.statistic,ttest_result.pvalue]
# print(result)



# sns.countplot(data=correct_answers_cleaned, x='Noise', hue = 'Condition')
# responses_by_condition = results.value_counts(["Noise","Condition","response_correct"], sort = False)

# print(responses_by_condition)

# def prep_data(dataframe):


    
#     dataframe.loc[(dataframe['Group'].str.contains('TD') & dataframe['Condition'].str.contains('meaningful')  & dataframe['Noise'].str.contains('quiet') ),
#                  'x_group'] = 0
#     dataframe.loc[(dataframe['Group'].str.contains('TD') & dataframe['Condition'].str.contains('nonsense')  & dataframe['Noise'].str.contains('quiet') ),
#                  'x_group'] = 0.1
#     dataframe.loc[(dataframe['Group'].str.contains('ASD') & dataframe['Condition'].str.contains('meaningful')  & dataframe['Noise'].str.contains('quiet') ),
#                  'x_group'] = 0.25
#     dataframe.loc[(dataframe['Group'].str.contains('ASD') & dataframe['Condition'].str.contains('nonsense')  & dataframe['Noise'].str.contains('quiet') ),
#                  'x_group'] = 0.35
    
#     dataframe.loc[(dataframe['Group'].str.contains('TD') & dataframe['Condition'].str.contains('meaningful')  & dataframe['Noise'].str.contains('low') ),
#                  'x_group'] = 1
#     dataframe.loc[(dataframe['Group'].str.contains('TD') & dataframe['Condition'].str.contains('nonsense')  & dataframe['Noise'].str.contains('low') ),
#                  'x_group'] = 1.1
#     dataframe.loc[(dataframe['Group'].str.contains('ASD') & dataframe['Condition'].str.contains('meaningful')  & dataframe['Noise'].str.contains('low') ),
#                  'x_group'] = 1.25
#     dataframe.loc[(dataframe['Group'].str.contains('ASD') & dataframe['Condition'].str.contains('nonsense')  & dataframe['Noise'].str.contains('low') ),
#                  'x_group'] = 1.35
    
#     dataframe.loc[(dataframe['Group'].str.contains('TD') & dataframe['Condition'].str.contains('meaningful')  & dataframe['Noise'].str.contains('high') ),
#                  'x_group'] = 2
#     dataframe.loc[(dataframe['Group'].str.contains('TD') & dataframe['Condition'].str.contains('nonsense')  & dataframe['Noise'].str.contains('high') ),
#                  'x_group'] = 2.1
#     dataframe.loc[(dataframe['Group'].str.contains('ASD') & dataframe['Condition'].str.contains('meaningful')  & dataframe['Noise'].str.contains('high') ),
#                  'x_group'] = 2.25
#     dataframe.loc[(dataframe['Group'].str.contains('ASD') & dataframe['Condition'].str.contains('nonsense')  & dataframe['Noise'].str.contains('high') ),
#                  'x_group'] = 2.35
    
#     dataframe = dataframe.loc[dataframe["Noise"]!='low',:]    
    
#     return dataframe
    
#     # sns.set(rc={'figure.figsize':(8.0,6.0)}) 
#     # sns.set(font_scale = 2)
#     # sns.set_style('white')
    
# plotdata = prep_data(all_participants_plotdata)
ax = sns.lineplot(data=plotdata, x='x_group', y='RT', ci=95, hue = 'Condition',
             err_style='bars',style = 'Group',estimator='mean')
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))




results_dir = input("Copy and paste the directory of the participant you want to analyse here")

results = load_files('answers.csv', results_dir) 
cleaned_data_all = clean_data(results)


noise_levels = list(set(results["Noise"]))
noise_levels.sort(reverse=True)

def ttest_by_condition(dataframe):

    ttest_results = []

    meaningful = dataframe.loc[dataframe["Condition"] == "meaningful",:]
    nonsense = dataframe.loc[dataframe["Condition"] == "nonsense",:]
    ttest_result = stats.mannwhitneyu(meaningful["RT"], nonsense["RT"])
    result = [ttest_result.statistic,ttest_result.pvalue]
    ttest_results.append(result)
        
    print(ttest_results)

TD_data = correct_answers_cleaned.loc[correct_answers_cleaned["Group"].str.contains('TD'),:]
ASD_data = correct_answers_cleaned.loc[correct_answers_cleaned["Group"].str.contains('ASD'),:]

ttest_by_condition(TD_data)
ttest_by_condition(ASD_data)

def ttest_by_condition(dataframe,condition,noise1,noise2):
    condition = dataframe.loc[dataframe["Condition"] == condition,:]
    noise1_data = condition.loc[condition["Noise"] == noise1,:]
    noise2_data = condition.loc[condition["Noise"] == noise2,:]
    ttest_result = stats.mannwhitneyu(noise1_data["RT"], noise2_data["RT"])
    result = [noise,ttest_result.statistic,ttest_result.pvalue]
    print(noise1,noise2,result)


ttest_by_condition(correct_answers_cleaned,"meaningful","quiet","high")
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    