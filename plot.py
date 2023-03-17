import numpy as np
import matplotlib.pyplot as plt
import subprocess
import wandb
import os

from pathlib import Path
from tqdm.auto import tqdm
from matplotlib import cm
from matplotlib.lines import Line2D

from plot_utils import configure_plt, _legend_title_left

api = wandb.Api()

# Get the runs programmatically
# runs = api.runs('utimateteam/metaoptnet',
#     filters={'$and': [{
#         'config.network': {'$eq': 'ResNet'},
#         'config.head': {'$eq': 'Sparse-SVM'},
#         'tags': {'$nin': ['hidden']}
#     }]
# })

# Get the runs from a list of run IDs
run_ids = [
    '6mhjbs4z',
    'x66vofjg',
    '23wojcvb',
    '1485on1f',
    '7aweqltf',
    '3p4ras6v',
    '2kyxx42r',
    # '1zkxy09x',
    # '2shvnogq',
    # 'i0z370es',
    # '1xgx2xxr',
    # '2t373q1m',
    # 'c4xffwia',
    # 'ewgrzr8d'
    # '3ka93nnb',
]
# 'zqfum2lf',
# '30zar58q',
# '2nfm4d5e',
# '64lrdci0',
# '3m68byyk',
# '290c8dgz',
# '2klxyp5a',
# '71rzwrzo',
# '2nu5kh9k',
# '3m8fxufh',
# '2qwl1wmf'

configure_plt()

viridis = cm.get_cmap('viridis', 12)

fig, axarr = plt.subplots(
    1, 2, sharex=False, sharey=False, figsize=(18, 5), constrained_layout=True)

accuracies, sparsities, l1regs = [], [], []
accs_test, spars_test = [], []
for idx_run, run_id in enumerate(tqdm(run_ids)):
    run = api.run(f'utimateteam/metaoptnet/{run_id}')

    root = Path(os.getenv('SLURM_TMPDIR')) / run.id
    root.mkdir(exist_ok=True)

    try:
        # Get the sparsity patterns from wandb
        run.file('sparsity_patterns.npy').download(root=root, replace=True)
    except:
        # Run the test script
        subprocess.run(['python', 'test.py',
            '--run_id', f'utimateteam/metaoptnet/{run_id}',
            '--episode', '1000',
            '--way', '5',
            '--shot', '5',
            '--query', '15',
        ])
        run.file('sparsity_patterns.npy').download(root=root, replace=True)

    with open(root / 'sparsity_patterns.npy', 'rb') as f:
        patterns = np.load(f)

    # Get plam
    accuracies.append((
        run.summary['test/5-shot_5-way/accuracy/mean'],
        run.summary['test/5-shot_5-way/accuracy/ci95']
    ))
    sparsities.append(patterns.mean())
    plam = run.config['lambda1']
    l1regs.append(plam)

    frequencies = np.mean(patterns, axis=0)
    frequencies = -np.sort(-frequencies)
    axarr[0].step(
        np.arange(frequencies.size), frequencies, label=plam,
        color=viridis(idx_run / len(run_ids)))

axarr[0].set_xlabel('Features')
axarr[0].set_ylabel('Usage percentage')

accuracies_mean, accuracies_ci95 = map(np.array, zip(*accuracies))
sparsities_mean = np.array(sparsities)
l1regs = np.array(l1regs)

legend = fig.legend(
    axarr[0].lines,
    [line.get_label() for line in axarr[0].lines],
    ncol=8,
    loc='upper center',
    title='$\lambda/\lambda_{\max}$',
    bbox_to_anchor=(0.5, 1.17)
)
_legend_title_left(legend)

colors = viridis(np.arange(len(run_ids)) / len(run_ids))
axarr[1].scatter(sparsities_mean, accuracies_mean, color=colors, marker='^')
axarr[1].set_xlabel('Average sparsity level')
axarr[1].set_ylabel('Accuracy')

handles = [
    # Line2D([0], [0], marker='o', color='w', markerfacecolor='k', label='Meta-train', markersize=15),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='k', label='Meta-test', markersize=15),
]
axarr[1].legend(handles=handles, loc='lower left')

save_fig = True
if save_fig:
    fig_dir = ''
    fig_name = 'meta_learning_sparsity'

    print(fig_name)

    plt.savefig(
      fig_dir + fig_name + '.pdf', bbox_inches='tight',
      dpi=600
    )
