import os
import sys
import time as t
import numpy as np
import matplotlib.pyplot as plt


def div0(num, denom):
    """Divide operation that deals with a 0 value denominator.
    num: numerator.
    denom: denominator.
    Returns 0.0 if the denominator is 0, otherwise returns a float."""
    return 0.0 if denom == 0 else float(num) / denom


def out(message='', newline=1):
    msg = '\n' + message if newline == 1 else message
    sys.stdout.write(msg)
    sys.stdout.flush()
    return t.time()


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def plot_pr_curve(model, prec, rec, aupr, title='',
                  line='-', save=False, show_legend=False, show_grid=False,
                  more_ticks=False, out_dir='graphs/', domain=''):
    fname = out_dir + domain + '_aupr.pdf'

    set_plot_rc()
    plt.figure(2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title(title, fontsize=22)
    plt.xlabel('Recall', fontsize=22)
    plt.ylabel('Precision', fontsize=22)
    plt.tick_params(axis='both', labelsize=18)
    plt.plot(rec, prec, line, label=model + ' = %0.3f' % aupr)

    if show_legend:
        plt.legend(loc='lower left', prop={'size': 6})

    if show_grid:
        ax = plt.gca()
        ax.grid(b=True, which='major', color='#E5DCDA', linestyle='-')

    if more_ticks:
        plt.yticks(np.arange(0.0, 1.01, 0.1))
        plt.xticks(np.arange(0.0, 1.01, 0.1), rotation=70)

    if save:
        plt.savefig(fname, bbox_inches='tight', format='pdf')
        plt.clf()
        plt.close('all')


def plot_roc_curve(model, tpr, fpr, auroc, title='',
                   line='-', save=False, show_legend=False, show_grid=False,
                   more_ticks=False, out_dir='graphs/', domain=''):
    fname = out_dir + domain + '_auroc.pdf'

    set_plot_rc()
    plt.figure(2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title(title, fontsize=22)
    plt.xlabel('FPR', fontsize=22)
    plt.ylabel('TPR', fontsize=22)
    plt.tick_params(axis='both', labelsize=18)
    plt.plot(fpr, tpr, line, label=model + ' = %0.3f' % auroc)
    plt.plot([0, 1], [0, 1], 'k--')

    if show_legend:
        plt.legend(loc='lower left', prop={'size': 6})

    if show_grid:
        ax = plt.gca()
        ax.grid(b=True, which='major', color='#E5DCDA', linestyle='-')

    if more_ticks:
        plt.yticks(np.arange(0.0, 1.01, 0.1))
        plt.xticks(np.arange(0.0, 1.01, 0.1), rotation=70)

    if save:
        plt.savefig(fname, bbox_inches='tight', format='pdf')
        plt.clf()
        plt.close('all')


def set_plot_rc():
    """Corrects for embedded fonts for text in plots."""
    plt.rc('pdf', fonttype=42)
    plt.rc('ps', fonttype=42)


def time(t1, suffix='m'):
    elapsed = t.time() - t1

    if suffix == 'm':
        elapsed /= 60.0
    if suffix == 'h':
        elapsed /= 3600.0

    out('%.2f%s' % (elapsed, suffix), 0)
