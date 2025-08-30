import os
import numpy as np
from glob import glob
import json
from matplotlib import pyplot as plt
from matplotlib import cm
from copy import deepcopy

results_dict = {
    'dispersion': [],
    'dispersion_coeff': [],
    'dispersion_loc': [],
    'metrics': [],
}

empty_metrics_dict = {
    'step': [],
    'paloma_wikitext_103\nword_perplexity': {'mean': [], 'std': []},
    'lambada_openai\nacc': {'mean': [], 'std': []},
    'lambada_standard\nacc': {'mean': [], 'std': []},
    'medmcqa\nacc': {'mean': [], 'std': []},
    'mmlu\nacc': {'mean': [], 'std': []},
}

def sort_by_step(steps, means, stds):
    order = np.argsort(np.array(steps))
    return np.array(steps)[order], np.array(means)[order], np.array(stds)[order]

def run_label(dispersion, coeff, loc):
    if str(dispersion) == 'None':
        return 'None'
    return f'{dispersion}-{coeff}-{loc}'


if __name__ == '__main__':
    result_folder = './results/'
    run_folder_list = sorted(glob(os.path.join(result_folder, 'midtrain_gpt2_*')))

    for run_folder in run_folder_list:
        dispersion = run_folder.split('disp-')[1].split('-')[0]
        dispersion_coeff = run_folder.split(f'{dispersion}-')[1].split('-')[0]
        dispersion_loc = run_folder.split(f'{dispersion_coeff}-')[1].split('_')[0]

        results_dict['dispersion'].append(dispersion)
        results_dict['dispersion_coeff'].append(dispersion_coeff)
        results_dict['dispersion_loc'].append(dispersion_loc)
        results_dict['metrics'].append(deepcopy(empty_metrics_dict))

        eval_json_list = glob(os.path.join(run_folder, 'lm_eval_*.json'))

        for eval_json in eval_json_list:
            with open(eval_json, "r") as f:
                data_json = json.load(f)

            results_dict['metrics'][-1]['step'].append(int(eval_json.split('_')[-1].replace('.json', '')))
            for metric in empty_metrics_dict.keys():
                if metric != 'step':
                    metric_dataset = metric.split('\n')[0]
                    metric_measure = metric.split('\n')[1]
                    results_dict['metrics'][-1][metric]['mean'].append(
                        float(data_json['results'][metric_dataset][f'{metric_measure},none']))
                    std = data_json['results'][metric_dataset][f'{metric_measure}_stderr,none']
                    if std == 'N/A':
                        results_dict['metrics'][-1][metric]['std'].append(np.nan)
                    else:
                        results_dict['metrics'][-1][metric]['std'].append(float(std))

    all_metric_names = [k for k in results_dict['metrics'][0].keys() if k != 'step']

    baseline_idx = None
    for i, d in enumerate(results_dict['dispersion']):
        if d.lower() == "none":
            baseline_idx = i
            break
    if baseline_idx is None:
        raise RuntimeError("No baseline run found with dispersion == 'None'.")

    # group indices by dispersion (excluding baseline)
    rows_by_dispersion = {}
    for i, d in enumerate(results_dict['dispersion']):
        if i == baseline_idx:
            continue
        rows_by_dispersion.setdefault(d, []).append(i)

    # enforce the requested row order
    row_order = [d for d in ["Covariance", "Hinge", "InfoNCE_l2", "InfoNCE_cosine"] if d in rows_by_dispersion]

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    fig = plt.figure(figsize=(20, 12))

    for row_idx, row_disp in enumerate(row_order):
        run_indices = rows_by_dispersion[row_disp]
        # colors = ["#00C100", "#CE0E00", "#C8A300", "#005BC3"]
        cmap = cm.Reds

        for metric_idx, metric_name in enumerate(all_metric_names):
            ax = fig.add_subplot(len(row_order), len(all_metric_names), row_idx * len(all_metric_names) + metric_idx + 1)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # plot every run of this dispersion
            for idx in run_indices:
                steps = results_dict['metrics'][idx]['step']
                means = results_dict['metrics'][idx][metric_name]['mean']
                stds = results_dict['metrics'][idx][metric_name]['std']
                xs, means, stds = sort_by_step(steps, means, stds)
                label = run_label(results_dict['dispersion'][idx],
                                  results_dict['dispersion_coeff'][idx],
                                  results_dict['dispersion_loc'][idx])
                coeff_logscaled = (np.log10(float(results_dict['dispersion_coeff'][idx])) + 4) / 7
                ax.plot(xs, means, linewidth=2, label=label, color=cmap(coeff_logscaled))
                # ax.fill_between(xs, means - stds, means + stds, linewidth=2, color=cmap(coeff_logscaled), alpha=0.2)

            # overlay baseline on every plot
            baseline_steps = results_dict['metrics'][baseline_idx]['step']
            baseline_means = results_dict['metrics'][baseline_idx][metric_name]['mean']
            xs_base, means_base, _ = sort_by_step(baseline_steps, baseline_means, baseline_means)
            label = run_label(results_dict['dispersion'][baseline_idx],
                              results_dict['dispersion_coeff'][baseline_idx],
                              results_dict['dispersion_loc'][baseline_idx])
            ax.plot(xs_base, means_base, linestyle='--', linewidth=2, label=label, color='black', alpha=0.5)

            ax.set_xlabel("Step", fontsize=15)
            ax.set_ylabel(metric_name, fontsize=15)
            if metric_idx == 0:
                ax.legend(fontsize=10, frameon=False, loc='upper right')

    fig.tight_layout(pad=2)
    fig.savefig('results.png')
