import os
import pandas as pd
import matplotlib as plt

local_dir = '/local_mount/space/hypatia/2/users/Jasmine/'
paradigm = 'AttenVis'
analysis_type = 'power'
if paradigm == 'Misophonia_ASD_TD':
    data_dir = os.path.join(local_dir, 'Misophonia',paradigm)
else:
    data_dir = os.path.join(local_dir, paradigm)
savedir = os.path.join(data_dir,'analyses',analysis_type)

#experiment details 
# condition = {'target': {'label':'Target'},
#             'search/4': {'label':'Search 4'},
#             'search/6': {'label':'Search 6'},
#             'search/8': {'label':'Search 8'},
#             'search/10': {'label':'Search 10'},
#             'pop-out/4': {'label':'Pop-out 4'},   
#             'pop-out/6': {'label':'Pop-out 6'},
#             'pop-out/8': {'label':'Pop-out 8'},
#             'pop-out/10': {'label':'Pop-out 10'}
#              } #put the condition you want the label drawn in first

condition = {'search': {'label':'Search'},
             'pop-out': {'label':'Pop-out'}}
plot_selected_conditions = ['search','pop-out']
brain_selected_conditions = ['search','pop-out']
diagnoses = {'asd':{'label':'ASD'},
            'td':{'label':'TD'}}
plot_labels = condition | diagnoses
study = ['MisoNat','MisoNat2']

sensor_hemis = ['left','right']
hemisphere = ['lh','rh']
labels_of_interest = ['V1']
brain_view = 'caudal'

time_windows = [-1.5,0.5]
metadata_timewindow = [-5.0,0.0] #stim: [0.0,5.0]
prestimulus_baseline = (0.3, 0.5)

peak_time_window = [0.9,1.2]
peak_times_hemis = {key: {key:None for key in hemisphere} for key in hemisphere}
peak_labels_hemis = {key: {key:None for key in hemisphere} for key in hemisphere}
peak_morphed_labels_hemis = {key: {key:None for key in hemisphere} for key in hemisphere}

overwrite_report = True
overwrite_data = False
overwrite_epochs = False
redraw_labels = False

#connectivity settings                 
tmin_plot = -0.3
tmax_plot = 1.5
freq_min = 4
freq_max = 80
freq_min_plot = 4
freq_max_plot = 40
power_line_plot_ylims = (-0.2,0.3)
con_method = "dSPM"
fc_method = 'coh'
fc_mode = 'cwt_morlet'
con_n_cycles = 3
sfreq = 250
snr           = 0.3
lambda2       = 1.0 / snr**2
baseline = (-0.4,-0.2)

#plotting settings
vmin = -1.0
vmax = 1.0
fontsize = 20
confidence = 0.95 #ci interval for line plots
ylims = (0,15)
zcoh_ylims = (-2.0,2.0)

labels_dict = {
    'auditory'  :   ['S_temporal_transverse','G_temp_sup-G_T_transv'],
    'insula'    :   ['S_circular_insula_ant','G_insular_short', 'G_Ins_lg_and_S_cent_ins','S_circular_insula_sup'],  
    'frontal'   :   ['S_front_middle','G_front_middle'],
    'IFG'       :   ['S_precentral-inf-part', 'S_front_inf'],
    'postcentral':  ['G_postcentral'],
    'precentral':   ['G_precentral'],
    'central'   :   ['S_central'], #, 'S_central', 'G_postcentral','G_precentral'
    'intraparietal':['S_intrapariet_and_P_trans'],
    'superior_parietal':['G_parietal_sup'],
    'auditory_drawn_label':['aud'],
    'inf_central_sulcus':['inf_central_sulcus'],
    'ant_intrapariet':['ant_intrapariet'],
    'sel_IFG'   :   ['sel_IFG'],
    'DLPFC'     :   ['DLPFC'],
    'supp_motor' :  ['supp_motor'],
    'vmPFC'     :   ['vmPFC'],
    'ACC'       :   ['ACC'],
    'MCC'       :   ['MCC'],
    'V1'        :   ['V1'],
    'FEF'       :   ['FEF'],
    'LOC'       :   ['LOC'],
    'TPJ'       :   ['TPJ'],
    'visual_drawn_label':['vis']
}

if 'electrodes' not in labels_of_interest:
    labels_list = labels_dict[labels_of_interest[0]]

#for connectivity studies
connectivity_labels = ['auditory','vmPFC']
seed_label = ['auditory_grown']
target_label = labels_dict[connectivity_labels[1]]
selected_conditions = ['target','search','pop-out']

#save locations and names
if analysis_type == 'connectivity':
    labels_of_interest = connectivity_labels

if 'drawn_label' not in labels_of_interest:    
    output_dir = os.path.join(savedir,'_'.join([con_method]+labels_of_interest +['grown']))
else:
    output_dir = os.path.join(savedir,'_'.join([con_method]+labels_of_interest))

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

save_fname = '_'.join(selected_conditions + labels_of_interest + [con_method]+[str(time_windows[0]),str(time_windows[1])]) #[analysis_type] + 
data_fname = save_fname + '.pkl'
data_savename = os.path.join(output_dir,data_fname)
report_savename_html = os.path.join(output_dir,save_fname+brain_view + ".html")
report_savename_hdf5 = os.path.join(output_dir,save_fname+brain_view + ".hdf5")

morph_report_savename = labels_list[0] + '_morph_from_fsaverage'
morph_report_savename_html = os.path.join(output_dir,morph_report_savename+".html")
morph_report_savename_hdf5 = os.path.join(output_dir,morph_report_savename+".hdf5")

connectivity_save_fname = '_'.join(selected_conditions + labels_of_interest + [con_method]+[str(time_windows[0]),str(time_windows[1])])
connectivity_compare_data_fname = connectivity_save_fname + '.pkl'
connectivity_compare_data_savename = os.path.join(output_dir,connectivity_compare_data_fname)
con_report_savename_html = os.path.join(output_dir,connectivity_save_fname+".html")
con_report_savename_hdf5 = os.path.join(output_dir,connectivity_save_fname+".hdf5")

inv_report_savename_hdf5 = os.path.join(data_dir,'_'.join([str(prestimulus_baseline[0]),str(prestimulus_baseline[1]),"prestim_inverses_responses.hdf5"]))
rt_report_savename_hdf5 = os.path.join(data_dir,"RTs_all_answers.hdf5")
rt_data_savename = os.path.join(data_dir,"RTs_all_answers.pkl")


report_title = '_'.join(list(condition.keys()) + labels_of_interest + [con_method])

color_dict = {"search":"orchid",
            "pop-out":"limegreen",
            "target":"cyan",
            "asd":"darkorange",
            "td":"violet",
            "search/4":"violet",
            "search/6":"fuchsia",
            "search/8":"mediumorchid",
            "search/10":"darkmagenta",
            "pop-out/4":"greenyellow",
            "pop-out/6":"lawngreen",
            "pop-out/8":"forestgreen",
            "pop-out/10":"darkgreen"}

#recons and fsaverage directories
subj_dir = '/autofs/space/transcend/MRI/WMA/recons/'
fsaverageDir = '/local_mount/space/hypatia/2/users/Jasmine/MNE-sample-data/subjects/'
fname_fsaverage_src = os.path.join(fsaverageDir, "fsaverage" , "bem" , "fsaverage-ico-5-src.fif")

transcend_data_dir = '/autofs/space/transcend/MEG/'

#load analysed_participants_demographics file to get relevant info about participants
participants_csvs = {
    'AttenVis'  : '/local_mount/space/hypatia/2/users/Jasmine/AttenVis/analysed_participants_demographics.csv',
    'AttenAud'  : '/local_mount/space/hypatia/2/users/Jasmine/AttenAud/analysed_participants_demographics.csv', 
    'Misophonia': '/local_mount/space/hypatia/2/users/Jasmine/Misophonia/analysed_participants_demographics.csv',
    'Misophonia_ASD_TD' :'/local_mount/space/hypatia/2/users/Jasmine/Misophonia/Miso_TD_ASD.csv', 
}
participants_csv = participants_csvs[paradigm]

exclude_participants = {
    'AttenVis'  : ['000000','073801','125401','126801','110401','108901','900005','007501'],
    'AttenAud'  : ['000000','073801','125401','126801', '110401','KSU_te'], 
    'Misophonia': ['000000','113301','KSU_te'],
    'Misophonia_ASD_TD' :['000000','112601','KSU_te'], #113201 has asd, 113301 did not qualify as miso, 112601 had a problem with triggers
    'MisoNat'   : ['113201','113301','000000','112601','KSU_te'], #,'151101','150901','147401','146201'
    'MisoNat2'  : ['113201','113301','000000','112601','KSU_te']
    }
excluded_participants = exclude_participants[paradigm]

df_varnames = {'Condition': plot_selected_conditions,
               'Diagnosis': diagnoses}


all_event_dicts = {
    'AttenVis' : {'condition/search/4': 1,
                     'condition/search/6': 2,
                     'condition/search/8': 3,
                     'condition/search/10': 4,
                     'condition/pop-out/4': 5,
                     'condition/pop-out/6': 6,
                     'condition/pop-out/8': 7,
                     'condition/pop-out/10': 8,
                     'target': 32,
                     'response/right': 2048,
                     'response/left': 32768
                     },
    'AttenAud' : {'attendRight/standard/high/right': 1,
                     'attendRight/standard/low/right': 3,
                     'attendRight/target/high/right': 11,
                     'attendRight/target/low/right': 13,
                     'attendRight/beep/low/left': 5,
                     'attendRight/beep/high/left': 7,
                     'attendRight/dev/low/left': 35,
                     'attendRight/dev/high/left': 37,
                     'attendRight/novel/low/left': 25,
                     'attendRight/novel/high/left': 27,
                     'attendLeft/standard/high/left': 2,
                     'attendLeft/standard/low/left': 4,
                     'attendLeft/target/high/left': 12,
                     'attendLeft/target/low/left': 14,
                     'attendLeft/beep/low/right': 6,
                     'attendLeft/beep/high/right': 8,
                     'attendLeft/dev/low/right': 36,
                     'attendLeft/dev/high/right': 38,
                     'attendLeft/novel/low/right': 26,
                     'attendLeft/novel/high/right': 28,
                     }, 
    'Misophonia': {'attendRight/standard/high/right': 1,
                     'attendRight/standard/low/right': 3,
                     'attendRight/target/high/right': 11,
                     'attendRight/target/low/right': 13,
                     'attendRight/beep/low/left': 5,
                     'attendRight/beep/high/left': 7,
                     'attendRight/dev/low/left': 35,
                     'attendRight/dev/high/left': 37,
                     'attendRight/novel/low/left': 25,
                     'attendRight/novel/high/left': 27,
                     'attendRight/misophone/low/left': 45,
                     'attendRight/misophone/high/left': 47,
                     'attendLeft/standard/high/left': 2,
                     'attendLeft/standard/low/left': 4,
                     'attendLeft/target/high/left': 12,
                     'attendLeft/target/low/left': 14,
                     'attendLeft/beep/low/right': 6,
                     'attendLeft/beep/high/right': 8,
                     'attendLeft/dev/low/right': 36,
                     'attendLeft/dev/high/right': 38,
                     'attendLeft/novel/low/right': 26,
                     'attendLeft/novel/high/right': 28,
                     'attendLeft/misophone/low/right': 46,
                     'attendLeft/misophone/high/right': 48
                     }, 
    
    'ASSRnew_Jumps' : {'25Hz/jump/left/noModulation': 1,
                     '25Hz/jump/right/noModulation': 2,
                     '43Hz/jump/left/noModulation': 3,
                     '43Hz/jump/right/noModulation': 4,
                     '25Hz/stay/left/modulation': 5,
                     '43Hz/stay/left/modulation': 6,
                     '25Hz/stay/left/noModulation': 7,
                     '43Hz/stay/left/noModulation': 8,
                     '25Hz/stay/right/modulation': 9,
                     '43Hz/stay/right/modulation': 10,
                     '25Hz/stay/right/noModulation': 11,
                     '43Hz/stay/right/noModulation': 12
                     },

        'MisoNat' : {'miso' : 10,
                     'sound2' : 20,
                     'control' : 30,
                     'silence': 40},

        'MisoNat2' : {'miso/lip_smacking' : 11,
                      'miso/slurping'     : 12,
                     'sound2/ocean'       : 21,
                     'sound2/rain'        : 22,
                     'amp_mod/lip_smacking':31,
                     'amp_mod/slurping'    :32,
                     'control' : 41,
                     'silence': 99},

    }
event_dict = all_event_dicts[paradigm]
right_responses = [256,512,1024,2048]
left_responses = [4096,8192,16384,32768]
all_responses = right_responses + left_responses
