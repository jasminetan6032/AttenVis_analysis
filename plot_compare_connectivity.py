#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 17:14:57 2024

@author: jwt30
"""
import seaborn as sns, matplotlib.pyplot as plt
import pandas as pd

df  = pd.read_csv('/local_mount/space/hypatia/2/users/Jasmine/github/AttenVis_analyses/alpha_gamma_AttenVis_coh.csv')
df['Participant'] = df['Participant'].astype('string').str.zfill(6)
df_search = df[df['Condition'] == 'Search']
df_popout = df[df['Condition'] == 'Pop-Outs']

fig, ax = plt.subplots()
sns.set(rc={'figure.figsize':(13.7,8.27)})
sns.set(style="whitegrid")

df_search_td = df_search[df_search['Diagnosis'] == 'td']
df_search_asd = df_search[df_search['Diagnosis'] == 'asd']

ax = sns.lineplot(x="difficulty", y="alpha", data=df_search, hue = 'Diagnosis')
ax = sns.lineplot(x="difficulty", y="broad_gamma", data=df_popout, hue = 'Diagnosis')

ax = sns.catplot(data=df_search_td, x="difficulty", y="alpha", kind="point")
ax = sns.swarmplot(x="difficulty", y="alpha", data=df_search_td,alpha=.35).set(title = "beta connectivity for search condition")

ax = sns.catplot(data=df_search_asd, x="difficulty", y="alpha", kind="point", color = "g")
ax = sns.swarmplot(x="difficulty", y="alpha", data=df_search_asd,alpha=.35, color = "g").set(title = "alpha connectivity for search condition")

plt.figure()

ax = sns.lineplot(x="difficulty", y="gamma", data=df_popout, hue = 'Diagnosis')

