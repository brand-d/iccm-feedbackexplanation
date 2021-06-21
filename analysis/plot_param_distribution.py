import os
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mpl
import seaborn as sns

def plot_distribution(dataset_name):
    filepath = 'benchmarks/parameters/{}'.format(dataset_name)
    
    # Set style to colorblind
    sns.set(style='whitegrid', palette='colorblind')

    # Load log data per model
    # PyPHM: p_entailment, A_conf, I_conf, E_conf, O_conf
    # mReasoner: epsilon, lambda, omega, sigma
    # TransSet: nvc_aversion, anchor_set, particularity, negativity


    # Individual analysis
    params_transset_indiv = []
    params_mreasoner_indiv = []
    params_phm_indiv = []

    for fname in os.listdir(filepath):
        if not fname.endswith('.json'):
            continue
        
        with open(filepath + "/" + fname) as fh:
            agg_log = json.load(fh)
            condition = fname.split("_")[1][:-5]

            # Extract TransSet
            for subj, data in agg_log['TransSet'].items():
                p_nvc_aversion = data['nvc_aversion']
                p_anchor_set = data['anchor_set']
                p_particularity = data['particularity']
                p_negativity = data['negativity']

                params_transset_indiv.append({
                    'model': 'TransSet',
                    'condition': condition,
                    'id': subj,
                    'nvc_aversion': p_nvc_aversion,
                    'anchor_set': p_anchor_set,
                    'particularity': p_particularity,
                    'negativity': p_negativity
                })
            
            # Extract mReasoner
            for subj, data in agg_log['mReasoner'].items():
                p_epsilon = data['epsilon']
                p_lambda = data['lambda']
                p_omega = data['omega']
                p_sigma = data['sigma']

                params_mreasoner_indiv.append({
                    'model': 'mReasoner',
                    'condition': condition,
                    'id': subj,
                    'epsilon': p_epsilon,
                    'lambda': p_lambda,
                    'omega': p_omega,
                    'sigma': p_sigma
                })
            
            # Extract PHM
            for subj, data in agg_log['PyPHM'].items():
                p_p_entailment = data['p_entailment']
                p_A_conf = data['A_conf']
                p_I_conf = data['I_conf']
                p_E_conf = data['E_conf']
                p_O_conf = data['O_conf']

                params_phm_indiv.append({
                    'model': 'PyPHM',
                    'condition': condition,
                    'id': subj,
                    'p_entailment': p_p_entailment,
                    'A_conf': p_A_conf,
                    'I_conf': p_I_conf,
                    'E_conf': p_E_conf,
                    'O_conf': p_O_conf
                })
            
    # Convert to dataframes
    df_params_transset_indiv = pd.DataFrame(params_transset_indiv)
    df_params_mreasoner_indiv = pd.DataFrame(params_mreasoner_indiv)
    df_params_phm_indiv = pd.DataFrame(params_phm_indiv)

    # Visualize distribution
    pnames_transset = ['nvc_aversion', 'anchor_set', 'particularity', 'negativity']
    pnames_mreasoner = ['epsilon', 'lambda', 'omega', 'sigma']
    greek_mreasoner = ['$\epsilon$', '$\lambda$', '$\omega$', '$\sigma$']
    pnames_phm = ['p_entailment', 'A_conf', 'I_conf', 'E_conf', 'O_conf']

    hue_order = ['control', 'feedback']
    plot_width = 8
    plot_height = 2

    # TransSet
    fig, axs = plt.subplots(1, 4, figsize=(plot_width, plot_height))

    for idx, pname in enumerate(pnames_transset):
        pdf = df_params_transset_indiv[['condition', pname]]
        
        plot_data = []
        for condition, condition_df in pdf.groupby('condition'):
            keys, cnts = np.unique(condition_df[pname], return_counts=True)
            cnts = cnts.astype('float')
            cnts /= cnts.sum()
            
            addendum = [{'condition': condition, 'label': x, 'value': y} for x, y in zip(keys, cnts)]
            plot_data.extend(addendum)
            
        df_plot = pd.DataFrame(plot_data)
        sns.barplot(x='label', y='value', hue='condition', data=df_plot, hue_order=hue_order, ax=axs[idx])
        
        axs[idx].get_legend().remove()
        axs[idx].set_xlabel(pname)
        axs[idx].set_ylim(0, 1)
        if idx > 0:
            axs[idx].set_ylabel('')
            axs[idx].set_yticklabels([])
        else:
            axs[idx].set_ylabel('Proportion')

    fig.suptitle('TransSet')
    plt.tight_layout(rect=(0,-0.03,1,0.95))
    fig.savefig('{}_transset.pdf'.format(dataset_name))
    plt.show()

    # mReasoner
    fig, axs = plt.subplots(1, 4, figsize=(plot_width, plot_height))
    df_params_mreasoner_indiv = df_params_mreasoner_indiv[pnames_mreasoner + ['condition']]

    for idx, pname in enumerate(pnames_mreasoner):
        sns.kdeplot(x=pname, hue='condition', data=df_params_mreasoner_indiv, hue_order=hue_order, ax=axs[idx], legend=False, common_norm=False)
        
        if idx > 0:
            axs[idx].set_ylabel('')
        axs[idx].set_xlabel(greek_mreasoner[idx])
    fig.suptitle('mReasoner')
    plt.tight_layout(rect=(0,-0.03,1,0.95))
    plt.savefig('{}_mreasoner.pdf'.format(dataset_name))
    plt.show()

    # PHM
    fig, axs = plt.subplots(1, 5, figsize=(plot_width, plot_height))

    for idx, pname in enumerate(pnames_phm):
        pdf = df_params_phm_indiv[['condition', pname]]
        
        plot_data = []
        for condition, condition_df in pdf.groupby('condition'):
            keys, cnts = np.unique(condition_df[pname], return_counts=True)
            cnts = cnts.astype('float')
            cnts /= cnts.sum()
            
            addendum = [{'condition': condition, 'label': x, 'value': y} for x, y in zip(keys, cnts)]
            plot_data.extend(addendum)
        
        df_plot = pd.DataFrame(plot_data)
        sns.barplot(x='label', y='value', hue='condition', data=df_plot, hue_order=hue_order, ax=axs[idx])
        
        axs[idx].get_legend().remove()
        axs[idx].set_xlabel(pname)
        axs[idx].set_ylim(0, 1)
        if idx > 0:
            axs[idx].set_ylabel('')
            axs[idx].set_yticklabels([])
        else:
            axs[idx].set_ylabel('Proportion')
        
    fig.suptitle('PHM')
    plt.tight_layout(rect=(0,-0.03,1,0.95))
    plt.savefig('{}_phm.pdf'.format(dataset_name))
    plt.show()
    
plot_distribution("dames2020")
plot_distribution("brand2021")