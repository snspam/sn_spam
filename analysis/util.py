"""
Module of utility methods.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import os
import sys
import time
import pickle
import random
import scipy.sparse
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import termcolor
import sklearn.metrics as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV


class Util:

    def __init__(self):
        self.noise_limit = 0.000025
        self.timer = []
        self.dirs = []

    # public
    def check_file(self, file):
        """Checks to see if the file exists
        file: path of the file.
        Returns True if it exists, exits the application if not."""
        if os.path.exists(file):
            return True
        else:
            self.exit('cannot read ' + file)

    def clean_msg(self, msg):
        """Utility function to clean msg text by removing links, special
        characters using simple regex statements."""
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|\
            (\w+:\/\/\S+)", " ", msg).split())

    def close_writer(self, sw):
        """Closes a file writer.
        sw: file writer object."""
        sw.close()

    def colorize(self, string, color, display):
        """Gives the string the specified color if there is a display.
        string: string to colorize.
        color: color to give the string.
        display: boolean indicating if the application is run on a consolde.
        Returns a colorized string if there is a display, string otherwise."""
        s = string

        if display:
            s = termcolor.colored(string, color)
        return s

    def create_dirs(self, path):
        """Creates all directories in path that doesn't already exist.
        path: str, directory to create."""
        if not os.path.exists(path):
            os.makedirs(path)

    def div0(self, num, denom):
        """Divide operation that deals with a 0 value denominator.
        num: numerator.
        denom: denominator.
        Returns 0.0 if the denominator is 0, otherwise returns a float."""
        return 0.0 if denom == 0 else float(num) / denom

    def end(self, message='', fw=None):
        """Pop a start time and take the time difference from now.
        message: message to print."""
        unit = 's'
        elapsed = time.time() - self.timer.pop()
        if elapsed >= 60:
            elapsed /= 60
            unit = 'm'
        s = message + '%.2f' + unit + '\n'
        if fw is not None:
            fw.write(s % (elapsed))
        else:
            self.out(s % (elapsed))

    def evaluate(self, data, test_probs):
        """Evaluates the predictions against the true labels.
        data: tuple including test set labels and ids.
        test_probs: predictions to evaluate."""
        x, y, ids, feat_names = data

        if y is not None:
            t1 = self.out('evaluating...')
            auroc, aupr, p, r, mp, mr, t = self.compute_scores(test_probs, y)
            self.time(t1)
            self.print_scores(mp, mr, t, aupr, auroc)
            self.print_median_mean(ids, test_probs, y)

    def exit(self, message='Unexpected error occurred!'):
        """Convenience method to fail gracefully.
        message: messaage to display to the user as to the error."""
        self.out(message)
        self.out('exiting...')
        exit(0)

    def file_len(self, fname):
        """Counts the number of lines in a file.
        fname: path of the file.
        Returns the number of lines in the specified file."""
        lines = 0

        f = open(fname, 'r')
        lines = len(f.readlines())
        f.close()
        return lines

    def gen_noise(self, pred):
        """Returns a prediction with some noise added to it.
        pred: predicion (e.g. value between 0.0 and 1.0).
        Returns predictions with noise."""
        noise = random.uniform(-self.noise_limit, self.noise_limit)
        result = max(0.0, min(1.0, pred + noise))
        return result

    def get_comments_filename(self, modified):
        """Chooses the correct comments file to read
        modified: Boolean indicating to read the modified comments file.
        Returns the name of the appropriate comments file."""
        filename = 'comments.csv'
        if modified:
            filename = 'modified.csv'
        return filename

    def load(self, filename):
        """Loads a binary pickled object.
        filename: path of the file.
        Returns loaded object."""
        if self.check_file(filename):
            with open(filename, 'rb') as f:
                obj = pickle.load(f)
        return obj

    def load_sparse(self, filename):
        """Loads a sparse matrix object.
        filename: path to the sparse matrix object file.
        Returns sparse matrix object."""
        matrix = scipy.sparse.load_npz(filename)
        return matrix

    def mean(self, numbers):
        """Computes the mean for a list of numbers.
        numbers: list of numbers.
        Returns mean as a float."""
        return np.mean(numbers)

    def out(self, message='', newline=1):
        """Custom print method to print multiple times on one line.
        message: string to print immediately."""
        msg = '\n' + message if newline == 1 else message
        sys.stdout.write(msg)
        sys.stdout.flush()
        return time.time()

    def open_writer(self, name, mode='w'):
        f = open(name, mode)
        return f

    def percent(self, num, denom):
        """Turns fraction into a percent.
        num: numerator.
        denom: denominator.
        Returns float in percent form."""
        return self.div0(num, denom) * 100.0

    def plot_features(self, model, classifier, features, fname, save=True):
        """Plots relative feature importance.
        model: fitted model.
        classifier: specific model.
        features: list of feature names.
        fname: filename of where to store the plot.
        save: boolean of whether the plot should be saved."""
        if classifier == 'lr':
            feat_importance = model.coef_[0]
        elif classifier == 'rf' or classifier == 'lgb':
            feat_importance = model.feature_importances_
        elif classifier == 'xgb':
            try:
                ax = xgb.plot_importance(model._Booster)
                labels = ax.get_yticklabels()
                indices = [int(x.get_text().replace('f', '')) for x in labels]
                yticks = [features[ndx] for ndx in indices]
                ax.set_yticklabels(yticks)
                plt.savefig(fname + '_feats.png', bbox_inches='tight')
                plt.close('all')
            except ValueError:
                self.out('error plotting xgb feature importances...')
            return

        # normalize and rearrange features
        feat_norm = feat_importance / feat_importance.sum()
        sorted_idx = np.argsort(feat_norm)
        pos = np.arange(sorted_idx.shape[0]) + 0.5  # [0.5, 1.5, ...]
        feat_importance_sort = feat_norm[sorted_idx]
        feat_sort = np.asanyarray(features)[sorted_idx]

        # plot relative feature importance
        color = '#7A68A6'
        plt.figure(figsize=(12, 12))
        plt.barh(pos, feat_importance_sort, align='center', color=color)
        plt.yticks(pos, feat_sort)
        plt.xlabel('Relative Importance')
        plt.title('Feature Importance')
        plt.savefig(fname + '_feats.pdf', bbox_inches='tight', format='pdf')
        plt.close('all')

    def plot_pr_curve(self, model, fname, rec, prec, aupr, title='',
                      line='-', save=False, show_legend=False, show_grid=False,
                      more_ticks=False):
        """Plots a precision-recall curve.
        model: name of the model.
        fname: filename to save the plot.
        rec: recalls from the aupr.
        prec: precisions from the aupr.
        aupr: area under the pr curve.
        title: title of the plot.
        line: shape used to draw the curve.
        save: boolean specifying whether to save the plot."""
        self.set_plot_rc()
        # if ax is None:
        # fig, ax = plt.subplots()
        plt.figure(2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.title(title, fontsize=22)
        plt.xlabel('Recall', fontsize=22)
        plt.ylabel('Precision', fontsize=22)
        plt.tick_params(axis='both', labelsize=18)
        # else:
        # plt.figure(2)
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
            plt.savefig(fname + '.pdf', bbox_inches='tight', format='pdf')
            plt.clf()
            plt.close('all')

    def print_stats(self, df, r_df, relation, dset, fw=None):
        """Prints information about a relationship in the data.
        df: comments dataframe.
        r_df: df containing number of times relationship occurred.
        relation: name of relation (e.g. posts).
        dset: dataset (e.g. 'val' or 'test')."""
        spam = r_df['label'].sum()
        out_str = '\n\t[' + dset + '] ' + relation + ': >1: ' + str(len(r_df))
        out_str += ', spam: ' + str(spam)
        self.write(out_str, fw=fw)

    def pushd(self, dir):
        curd = os.getcwd()
        self.dirs.append(curd)
        os.chdir(dir)

    def popd(self):
        os.chdir(self.dirs.pop())

    def read_csv(self, filename):
        """Safe read for pandas dataframes.
        filename: path to data file.
        Returns dataframe if the file exists, None otherwise."""
        result = None

        if os.path.exists(filename):
            result = pd.read_csv(filename)
        return result

    def save(self, obj, filename):
        """Pickles an object to a binary file.
        obj: object to pickle.
        filename: path of the file."""
        with open(filename, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def save_sparse(self, matrix, filename):
        """Saves a sparse matrix object to a file.
        matrix: sparse matrix object.
        filename: path to the file to save the object to."""
        scipy.sparse.save_npz(filename, matrix)

    def set_noise_limit(self, noise_limit):
        """Setter for noise_limit."""
        self.noise_limit = noise_limit

    def start(self, message='', fw=None):
        """Pushes a start time onto a stack and print a specified message.
        message: message to print."""
        self.write(message=message, fw=fw)
        self.timer.append(time.time())

    def test(self, data, model, fsets=['all']):
        """Tests data using a trained model.
        data: tuple including data to classify.
        model: trained model.
        Returns predictions and ids associated with those predictions."""
        x, y, ids, feat_names = data

        if type(model) == xgb.XGBClassifier and \
                any(x in fsets for x in ['ngrams', 'all']):
            x = x.tocsc()  # bug in xgb, turn on when stacking is on.

        t1 = self.out('testing...')
        if type(model) == lgb.LGBMClassifier:
            ys = model.predict_proba(x, num_iteration=model.best_iteration_)
        else:
            ys = model.predict_proba(x)
        self.time(t1)
        return ys, ids

    def time(self, t):
        """Write time based on suffix."""
        elapsed = time.time() - t

        if elapsed < 60:
            suffix = 's'
        elif elapsed < 3600:
            suffix = 'm'
        else:
            suffix = 'h'

        if suffix == 'm':
            elapsed /= 60.0
        elif suffix == 'h':
            elapsed /= 3600.0

        self.out('%.2f%s' % (elapsed, suffix), 0)

    def train(self, data, clf='rf', param_search='single', tune_size=0.15,
              scoring='roc_auc', n_jobs=1, verbose=1):
        """Trains a classifier with the specified training data.
        data: tuple including training data.
        clf: string of {'rf' 'lr', 'xgb'}.
        Returns trained classifier."""
        x_train, y_train, _, features = data

        if param_search == 'single' or tune_size == 0:
            model, params = self.classifier(clf, param_search='single')
            model.set_params(**params)

        elif tune_size > 0:
            t1 = self.out('tuning...')
            model, params = self.classifier(clf, param_search=param_search)
            train_len = x_train.shape[0]

            split_ndx = train_len - int(train_len * tune_size)
            sm_x_train, x_val = x_train[:split_ndx], x_train[split_ndx:]
            sm_train_fold = np.full(sm_x_train.shape[0], -1)
            val_fold = np.full(x_val.shape[0], 0)

            predefined_fold = np.append(sm_train_fold, val_fold)
            ps = PredefinedSplit(predefined_fold)
            cv = ps.split(x_train, y_train)
            m = GridSearchCV(model, params, scoring=scoring, cv=cv,
                             verbose=verbose, n_jobs=n_jobs)
            m.fit(x_train, y_train)
            model = m.best_estimator_
            self.time(t1)

        t1 = self.out('training...')

        if clf == 'lgb':
            cat_feat = ['app', 'device', 'os', 'channel', 'hour']
            cat_feat_ndx = [features.index(x) for x in cat_feat]
            train_len = x_train.shape[0]
            split_ndx = train_len - int(train_len * tune_size)
            sm_x_train, x_val = x_train[:split_ndx], x_train[split_ndx:]
            sm_y_train, y_val = y_train[:split_ndx], y_train[split_ndx:]
            eval_set = (x_val, y_val)
            model = model.fit(sm_x_train, sm_y_train, eval_set=eval_set,
                              early_stopping_rounds=50, eval_metric='auc',
                              categorical_feature=cat_feat_ndx)
        else:
            model = model.fit(x_train, y_train)

        self.time(t1)
        self.out(str(model))
        return model

    def write(self, message='', fw=None):
        if fw is not None:
            fw.write(message)
        else:
            self.out(message)

    def classifier(self, classifier='rf', param_search='single'):
        """
        Defines model and parameters to tune.

        Parameters
        ----------
        classifier : str, {'rf', 'xgb', 'lr1', 'lr2'}, default: 'rf'
            Type of model to define.
        param_search : str, {'low', 'med', 'high'}, default: 'low'
            Level of parameters to tune.
        input_dim : int, default = 0
            Number of features input to the model.

        Returns
        -------
        Defined model and dictionary of parameters to tune.
        """
        if classifier == 'lr':
            clf = LogisticRegression()
            high = [{'penalty': ['l1', 'l2'],
                     'C': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5,
                           1.0, 2.0, 10.0, 50.0, 100.0, 500.0, 1000.0],
                     'solver': ['liblinear']},
                    {'penalty': ['l2'],
                     'C': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5,
                           1.0, 2.0, 10.0, 50.0, 100.0, 500.0, 1000.0],
                     'solver': ['newton-cg']}]
            med = [{'penalty': ['l1', 'l2'],
                    'C': [0.0001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
                    'solver': ['liblinear']},
                   {'penalty': ['l2'],
                    'C': [0.0001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
                    'solver': ['newton-cg']}]
            low = {'penalty': ['l2'],
                   'C': [0.0001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
                   'solver': ['liblinear'],
                   'class_weight': ['balanced']},
            single = {'penalty': 'l2', 'C': 1000.0, 'solver': 'liblinear',
                      'class_weight': 'balanced'}

        elif classifier == 'rf':
            clf = RandomForestClassifier()
            high = {'n_estimators': [10, 100, 1000], 'max_depth': [None, 2, 4]}
            med = {'n_estimators': [1000], 'max_depth': [None, 2]}
            low = {'n_estimators': [1000], 'max_depth': [None]}
            single = {'n_estimators': 100, 'max_depth': 4}

        elif classifier == 'lgb':
            clf = lgb.LGBMClassifier()
            high = {'max_depth': [3, 4, 6],
                    'n_estimators': [100, 1000],
                    'learning_rate': [0.3, 0.1, 0.05, 0.01, 0.005, 0.001],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]}
            med = {'max_depth': [4, 6], 'n_estimators': [10, 100, 1000],
                   'learning_rate': [0.005, 0.05, 0.1],
                   'subsample': [0.9, 1.0], 'colsample_bytree': [1.0]}
            low = {'max_depth': [4], 'boosting_type': ['gbdt'],
                   'n_estimators': [1500], 'subsample': [0.7],
                   'num_leaves': [7], 'colsample_bytree': [0.7, 0.9],
                   'min_child_samples': [100], 'max_bin': [100],
                   'learning_rate': [0.1, 0.2], 'min_child_weight': [0.0],
                   'scale_pos_weight': [500], 'verbose': [-1]}
            single = {'max_depth': 4, 'n_estimators': 1500,
                      'learning_rate': 0.1, 'scale_pos_weight': 500,
                      'num_leaves': 7, 'min_child_samples': 100,
                      'subsample': 0.7, 'colsample_bytree': 0.7,
                      'min_child_weight': 0.0, 'verbose': -1}
            # single = {'max_depth': 4, 'n_estimators': 1500,  # not adclicks
            #           'learning_rate': 0.1, 'scale_pos_weight': 500,
            #           'num_leaves': 7, 'min_child_samples': 20,
            #           'subsample': 0.7, 'colsample_bytree': 0.7,
            #           'min_child_weight': 0.0, 'verbose': -1}

        elif classifier == 'xgb':
            clf = xgb.XGBClassifier()
            high = {'max_depth': [3, 4, 6],
                    'n_estimators': [100, 1000],
                    'learning_rate': [0.3, 0.1, 0.05, 0.01, 0.005, 0.001],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]}
            med = {'max_depth': [4, 6], 'n_estimators': [10, 100, 1000],
                   'learning_rate': [0.005, 0.05, 0.1],
                   'subsample': [0.9, 1.0], 'colsample_bytree': [1.0]}
            low = {'max_depth': [6], 'n_estimators': [1000],
                   'learning_rate': [0.05], 'subsample': [0.9],
                   'colsample_bytree': [1.0]}
            single = {'max_depth': 4, 'n_estimators': 100,
                      'learning_rate': 0.1, 'subsample': 1.0,
                      'colsample_bytree': 1.0, 'scale_pos_weight': 500}

        param_dict = {'high': high, 'med': med, 'low': low, 'single': single}
        param_grid = param_dict[param_search]
        return (clf, param_grid)

    def compute_scores(self, probs, y):
        """Generates noisy predictions and computes various metrics.
        probs: predictions, shape=(2, <num_instances>).
        y: list of true labels.
        report: file to write performance to.
        dset: dataset (e.g. 'train', 'val', 'test').
        Returns auroc, aupr, recalls, precisions, max precision, max recall,
                and threshold where those max values take place."""
        prob_preds_noise = [self.gen_noise(pred) for pred in probs[:, 1]]
        fpr, tpr, tholds = sm.roc_curve(y, prob_preds_noise)
        prec, rec, tholds = sm.precision_recall_curve(y, prob_preds_noise)
        aupr = sm.average_precision_score(y, prob_preds_noise)
        auroc = sm.auc(fpr, tpr)
        max_p, max_r, thold = self.find_max_prec_recall(prec, rec, tholds)
        return auroc, aupr, prec, rec, max_p, max_r, thold

    def find_max_prec_recall(self, prec, rec, tholds):
        """Finds the precision and recall scores with the maximum amount of
        area and returns their values, including the threshold.
        prec: list of precisions from the pr curve.
        rec: list of recalls from the pr curve.
        tholds: list of thresholds from the pr curve.
        Returns max precision and recall scores, including their threshold."""
        max_val, max_prec, max_rec, max_thold = -1, -1, -1, -1

        if len(tholds) > 1:
            for i in range(len(prec)):
                val = prec[i] * rec[i]
                if val > max_val:
                    max_val = val
                    max_thold = tholds[i]
                    max_prec = prec[i]
                    max_rec = rec[i]
        return max_prec, max_rec, max_thold

    def save_preds(self, probs, ids, fold, pred_f, dset, eval='cc'):
        """Save predictions to a specified file.
        probs: array of binary predictions; shape=(2, <num_instances>).
        ids: list of identifiers for the data instances.
        pred_f: folder to save predictions to.
        dset: dataset (e.g. 'train', 'val', 'test')."""
        columns = ['com_id', 'ind_pred']
        fname = dset + '_' + fold + '_preds'

        t1 = self.out('saving predictions...')
        preds = list(zip(ids, probs[:, 1]))
        preds_df = pd.DataFrame(preds, columns=columns)
        preds_df.to_csv(pred_f + fname + '.csv', index=None)

        if eval == 'tt':
            preds_df.columns = ['click_id', 'is_attributed']
            preds_df.to_csv(pred_f + fname + '.csv.gz', index=None,
                            compression='gzip')
        self.time(t1)

    def set_plot_rc(self):
        """Corrects for embedded fonts for text in plots."""
        plt.rc('pdf', fonttype=42)
        plt.rc('ps', fonttype=42)

    def print_median_mean(self, ids, probs, y, fw=None):
        """Prints the median and mean independent predictions for spam and ham.
        ids: comment ids.
        probs: independent predictions.
        y: labels"""
        preds = list(zip(ids, probs[:, 1], y))
        df = pd.DataFrame(preds, columns=['com_id', 'ind_pred', 'label'])
        spam_med = df[df['label'] == 1]['ind_pred'].median()
        ham_med = df[df['label'] == 0]['ind_pred'].median()
        spam_mean = df[df['label'] == 1]['ind_pred'].mean()
        ham_mean = df[df['label'] == 0]['ind_pred'].mean()
        self.out('-> median spam: %.4f, ham: %.4f' % (spam_med, ham_med))
        self.out('-> mean spam: %.4f, ham: %.4f' % (spam_mean, ham_mean))

    def print_scores(self, max_p, max_r, thold, aupr, auroc, fw=None):
        """Print evaluation metrics to std out.
        max_p: maximum precision in pr curve at thold.
        max_r: maximum recall in pr curve at thold.
        thold: threshold where the maximum area is.
        aupr: area under the pr curve.
        auroc: area under the roc curve."""
        self.out('-> aupr: %.4f, auroc: %.4f' % (aupr, auroc))
        s = '-> max p: %.3f, max r: %.3f, area: %.3f, thold: %.3f'
        self.out(s % (max_p, max_r, max_p * max_r, thold))
