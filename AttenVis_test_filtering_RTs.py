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

def compare_before_and_after_filtering(df,cond1,cond2):
    """
    Compare the number of epochs before and after filtering.
    
    Parameters:
    summary (DataFrame): Summary DataFrame with counts before filtering.
    filtered_nepochs (DataFrame): DataFrame with counts after filtering.
    
    Returns:
    DataFrame: Merged DataFrame with percent loss calculated.
    """
    # Filter the summary DataFrame for the specified conditions
    if cond1 is None:
        first_df = df
    else:
        first_df = df.loc[cond1]
    summary1 = first_df.groupby(['Condition', 'Diagnosis']).size().reset_index(name='count_before')
    second_df = df.loc[cond2]
    summary2 = second_df.groupby(['Condition', 'Diagnosis']).size().reset_index(name='count_after')

    # Merge the two summaries on the same group keys
    merged = summary1.merge(summary2, on=['Condition', 'Diagnosis'], suffixes=('_before', '_after'))

    # Calculate percent loss
    merged['percent_loss'] = ((merged['count_before'] - merged['count_after']) / merged['count_before']) * 100

    # Optional: round for readability
    merged['percent_loss'] = merged['percent_loss'].round(2)

    return merged

def reaction_time_distribution_table(df, rt_col, condition_col, diagnosis_col, time_bins):
    """
    Create an HTML table showing % of RTs within given time bins by condition and diagnosis.

    Parameters:
    - df: pd.DataFrame with reaction time and grouping variables
    - rt_col: str, name of column with reaction times (in seconds)
    - condition_col: str, name of column with condition labels
    - diagnosis_col: str, name of column with diagnosis labels
    - time_bins: list of (min_time, max_time) tuples

    Returns:
    - html_block: str, styled HTML string suitable for use in MNE Report
    """
    # Grouped counts
    result_rows = []
    for (condition, diagnosis), group in df.groupby([condition_col, diagnosis_col]):
        total = len(group)
        row = {'Condition': condition, 'Diagnosis': diagnosis}
        for tmin, tmax in time_bins:
            label = f"{int(tmin*1000)}-{int(tmax*1000)} ms"
            count = group[(group[rt_col] >= tmin) & (group[rt_col] < tmax)].shape[0]
            row[label] = round(100 * count / total, 2) if total > 0 else 0.0
        result_rows.append(row)

    result_df = pd.DataFrame(result_rows)

    return result_df


def reaction_time_ttest_table(
    df,
    rt_col='reaction_time',
    subject_col='subject_id',
    diagnosis_col='diagnosis',
    condition_col='condition',
    difficulty_col='difficulty',
    include_within_subject=True
):
    from scipy.stats import ttest_ind, ttest_rel
    from itertools import combinations

    results = []
    conditions = df[condition_col].unique()
    difficulty_order = ['4','6','8','10']
    df['difficulty'] = pd.Categorical(df['difficulty'], categories=difficulty_order, ordered=True)
    difficulties = df['difficulty'].cat.categories.tolist()
    groups = df[diagnosis_col].unique()

    # BETWEEN-SUBJECT T-TESTS
    for cond in conditions:
        for diff in difficulties:
            subset = df[(df[condition_col] == cond) & (df[difficulty_col] == diff)]
            data_by_group = {
                grp: subset[subset[diagnosis_col] == grp][rt_col].dropna()
                for grp in groups
            }

            if all(len(data_by_group[grp]) > 1 for grp in groups):
                t_stat, p_val = ttest_ind(
                    data_by_group[groups[0]], data_by_group[groups[1]], equal_var=False
                )
            else:
                t_stat, p_val = float('nan'), float('nan')

            results.append({
                'Comparison': f'{groups[0]} vs {groups[1]}',
                'Condition': cond,
                'Difficulty': diff,
                f'{groups[0]} Mean': round(data_by_group[groups[0]].mean(), 3),
                f'{groups[1]} Mean': round(data_by_group[groups[1]].mean(), 3),
                't': round(t_stat, 3),
                'p': round(p_val, 4),
                f'n_{groups[0]}': len(data_by_group[groups[0]]),
                f'n_{groups[1]}': len(data_by_group[groups[1]]),
                'Type': 'Between'
            })

    # WITHIN-SUBJECT T-TESTS: Pairwise difficulty comparisons
    if include_within_subject:
        subj_avg = df.groupby([subject_col, condition_col, difficulty_col, diagnosis_col])[rt_col].mean().reset_index()

        for cond in conditions:
            for grp in groups:
                subset = subj_avg[
                    (subj_avg[condition_col] == cond) & (subj_avg[diagnosis_col] == grp)
                ]
                pivoted = subset.pivot(index=subject_col, columns=difficulty_col, values=rt_col)

                for diff1, diff2 in combinations(difficulties, 2):
                    if diff1 in pivoted.columns and diff2 in pivoted.columns:
                        paired = pivoted[[diff1, diff2]].dropna()
                        if len(paired) > 1:
                            t_stat, p_val = ttest_rel(paired[diff1], paired[diff2])
                        else:
                            t_stat, p_val = float('nan'), float('nan')

                        results.append({
                            'Comparison': f'{diff1} vs {diff2}',
                            'Condition': cond,
                            'Difficulty': '—',
                            f'{groups[0]} Mean': '',
                            f'{groups[1]} Mean': '',
                            't': round(t_stat, 3),
                            'p': round(p_val, 4),
                            f'n_{groups[0]}': '',
                            f'n_{groups[1]}': len(paired),
                            'Type': f'Within ({grp})'
                        })

    results_df = pd.DataFrame(results)
    return results_df

def collate_RTs(sub_id):
    #load epochs
    diagnosis, study,visit_dir,subjID_date = tlbx.read_participant_details_from_dataframe(participants_df,sub_id)
    load_fname, epochs = tlbx.load_epochs('_nobaseline_nofilter_all_conditions_metadata_epo.fif',visit_dir,resample=True)
    metadata = epochs.metadata
    metadata['Participant'] = sub_id
    metadata['Diagnosis'] = diagnosis
    metadata['Study'] = study
    metadata['SubjID_Date'] = subjID_date

    # cleaned_metadata,summary = tlbx.clean_metadata(metadata,rt_based=(0.15,1.2), percent=None, correct_answers_only=True)
    # hist_fig = tlbx.plot_participant_RT_hist(metadata, cleaned_metadata)
    # RT_fig = tlbx.plot_RT(cleaned_metadata)
    return sub_id, metadata,len(epochs) #cleaned_metadata, summary, hist_fig, RT_fig

n_jobs = 6

debug = True
if debug:
    n_jobs = 1

if overwrite_data:
    parallel, run_func, _ = parallel_func(collate_RTs, n_jobs=n_jobs)
    results = parallel(run_func(subject) for subject in participants_to_study)
    all_dfs = [entry[1] for entry in results]
    df = pd.concat(all_dfs, ignore_index=True)
    df.to_pickle(cfg.rt_data_savename)
else:
    df = pd.read_pickle(cfg.rt_data_savename)

# get summary before filtering
summary = df.groupby(['Condition','Diagnosis']).size().reset_index(name='count')

filtered_out_long_RTs = df.loc[(df["RT"] < 1.2)] 

# Loop through multiple RT cutoffs
cutoffs = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35,0.4,0.45,0.5,0.55]
all_results = []

for cutoff in cutoffs:
    cond2 = (df["RT"] > cutoff)
    result = compare_before_and_after_filtering(filtered_out_long_RTs, cond1=None, cond2=cond2)
    result['cutoff'] = cutoff  # add cutoff as a column
    all_results.append(result)

# Combine all results into one table
percent_loss_table = pd.concat(all_results, ignore_index=True)

# Optional: pivot for nicer display
percent_loss_pivot = percent_loss_table.pivot_table(
    index=['Condition', 'Diagnosis'],
    columns='cutoff',
    values='percent_loss'
).round(2)

tlbx.add_table_to_report(percent_loss_pivot, report, 'RTs_percent_loss_pivot')
tlbx.add_table_to_report(percent_loss_table, report, 'RTs_percent_loss')

bins = [(0, 0.1), (0.1, 0.15), (0.15, 0.2), (0.2, 0.25), (0.25, 0.3), (0.3, 0.35), (0.35, 0.4), (0.4, 0.45), (0.45, 0.5), (0.5, 0.55)]
rt_dist_table = reaction_time_distribution_table(filtered_out_long_RTs, 'RT', 'Condition', 'Diagnosis', bins)
tlbx.add_table_to_report(rt_dist_table, report, 'RTs_distribution')

median_df = filtered_out_long_RTs.groupby(['Condition', 'difficulty','Diagnosis'])['RT'].mean().reset_index()
tlbx.add_table_to_report(median_df,report,'gavg_median')

rt_ttest_table = reaction_time_ttest_table(
    filtered_out_long_RTs,
    rt_col='RT',
    subject_col='Participant',
    diagnosis_col='Diagnosis',
    condition_col='Condition',
    difficulty_col='difficulty',
    include_within_subject=False
)
tlbx.add_table_to_report(rt_ttest_table, report, 'RTs_ttest')

fig = tlbx.plot_RT(filtered_out_long_RTs,group = True)
report.add_figure(fig=fig, title='RTs', section='gavg', tags=['line_plot'], replace=True)
plt.close(fig)

report.save(cfg.rt_report_savename_hdf5, verbose=False, overwrite=True)

tlbx.show_report(cfg.rt_report_savename_hdf5)