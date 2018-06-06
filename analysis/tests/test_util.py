"""
Tests the util module.
"""
import os
import mock
import unittest
import termcolor
import scipy.sparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as sm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from .context import util
from .context import test_utils as tu


class UtilTestCase(unittest.TestCase):
    def setUp(self):
        self.test_obj = util.Util()

    def tearDown(self):
        self.test_obj = None
        pd.DataFrame = pd.DataFrame

    def test_init(self):
        self.assertTrue(self.test_obj.noise_limit == 0.0025)

    @mock.patch('time.time')
    def test_start(self, mock_time):
        self.test_obj.out = mock.Mock()
        mock_time.return_value = 'time'

        self.test_obj.start('orange')

        mock_time.assert_called()
        self.test_obj.out.assert_called_with('orange')
        self.assertTrue(self.test_obj.timer == ['time'])

    @mock.patch('time.time')
    def test_end(self, mock_time):
        mock_time.return_value = 60.5
        self.test_obj.timer = [0.5]
        self.test_obj.out = mock.Mock()

        self.test_obj.end('banana ')

        mock_time.assert_called()
        self.test_obj.out.assert_called_with('banana 1.00m\n')

    def test_get_comments_filename_not_modified(self):
        result = self.test_obj.get_comments_filename(False)

        self.assertTrue(result == 'comments.csv')

    def test_get_comments_filename_modified(self):
        result = self.test_obj.get_comments_filename(True)

        self.assertTrue(result == 'modified.csv')

    def test_set_noise_limit(self):
        self.test_obj.set_noise_limit(0.69)

        self.assertTrue(self.test_obj.noise_limit == 0.69)

    def test_gen_noise_middle(self):
        result = self.test_obj.gen_noise(0.69)

        self.assertTrue(result <= 0.69 + 0.0025)
        self.assertTrue(result >= 0.69 - 0.0025)

    def test_gen_noise_high(self):
        result = self.test_obj.gen_noise(0.99)

        self.assertTrue(result <= 1.0)

    def test_gen_noise_low(self):
        result = self.test_obj.gen_noise(0.01)

        self.assertTrue(result >= 0.0)

    def test_plot_rc(self):
        plt.rc = mock.Mock()

        self.test_obj.set_plot_rc()

        ex = [mock.call('pdf', fonttype=42), mock.call('ps', fonttype=42)]
        self.assertTrue(plt.rc.call_args_list == ex)

    def test_plot_features(self):
        coef = np.array([[-7, 2, -3, 5]])
        features = ['banana', 'orange', 'dragonfruit', 'kiwi']
        g = mock.Mock(LogisticRegression)
        type(g).coef_ = mock.PropertyMock(return_value=coef)
        plt.figure = mock.Mock()
        plt.barh = mock.Mock()
        plt.yticks = mock.Mock()
        plt.xlabel = mock.Mock()
        plt.title = mock.Mock()
        plt.savefig = mock.Mock()

        self.test_obj.plot_features(g, 'lr', features, 'fname', save=True)

        pos = np.array([0.5, 1.5, 2.5, 3.5])
        feat_i_sort = np.array([-7, -3, 2, 5])
        feat_sort = ['banana', 'dragonfruit', 'orange', 'kiwi']
        plt.figure.assert_called_with(figsize=(12, 12))
        self.assertTrue(np.array_equal(plt.barh.call_args[0][0], pos))
        self.assertTrue(np.array_equal(plt.barh.call_args[0][1], feat_i_sort))
        self.assertTrue(np.array_equal(plt.barh.call_args[1]['align'],
                'center'))
        self.assertTrue(np.array_equal(plt.barh.call_args[1]['color'],
                '#7A68A6'))
        self.assertTrue(np.array_equal(plt.yticks.call_args[0][0], pos))
        self.assertTrue(np.array_equal(plt.yticks.call_args[0][1], feat_sort))
        plt.xlabel.assert_called_with('Relative Importance')
        plt.title.assert_called_with('Feature Importance')
        plt.savefig.assert_called_with('fname_feats.png', bbox_inches='tight')

    def test_plot_pr_curve(self):
        self.test_obj.set_plot_rc = mock.Mock()
        plt.figure = mock.Mock()
        plt.plot = mock.Mock()
        plt.xlim = mock.Mock()
        plt.ylim = mock.Mock()
        plt.title = mock.Mock()
        plt.xlabel = mock.Mock()
        plt.ylabel = mock.Mock()
        plt.tick_params = mock.Mock()
        plt.legend = mock.Mock()
        plt.savefig = mock.Mock()
        plt.clf = mock.Mock()
        ax = mock.Mock()
        plt.gca = mock.Mock(return_value=ax)
        ax.grid = mock.Mock()
        plt.yticks = mock.Mock()
        plt.xticks = mock.Mock()
        np.arange = mock.Mock(return_value='arange')

        self.test_obj.plot_pr_curve('model', 'fname', 'r', 'p', 0.77,
                title='title', line='line', save=True, show_legend=True,
                show_grid=True, more_ticks=True)

        exp_arange = [mock.call(0.0, 1.01, 0.1), mock.call(0.0, 1.01, 0.1)]
        self.test_obj.set_plot_rc.assert_called()
        plt.figure.assert_called_with(2)
        plt.plot.assert_called_with('r', 'p', 'line',
                label='model = 0.770')
        plt.xlim.assert_called_with([0.0, 1.0])
        plt.ylim.assert_called_with([0.0, 1.0])
        plt.title.assert_called_with('title', fontsize=22)
        plt.xlabel.assert_called_with('Recall', fontsize=22)
        plt.ylabel.assert_called_with('Precision', fontsize=22)
        plt.tick_params.assert_called_with(axis='both', labelsize=18)
        plt.legend.assert_called_with(loc='lower left', prop={'size': 6})
        plt.savefig.assert_called_with('fname.pdf', bbox_inches='tight',
                format='pdf')
        plt.clf.assert_called()
        plt.gca.assert_called()
        ax.grid.assert_called_with(b=True, which='major', color='#E5DCDA',
                linestyle='-')
        self.assertTrue(np.arange.call_args_list == exp_arange)
        plt.yticks.assert_called_with('arange')
        plt.xticks.assert_called_with('arange', rotation=70)

    @mock.patch('pandas.DataFrame')
    def test_save_preds(self, mock_DataFrame):
        probs = np.array([[0.2, 0.8], [0.4, 0.6], [0.7, 0.3], [0.9, 0.1]])
        ids = [1, 2, 3, 4]
        df = tu.sample_df(10)
        df.to_csv = mock.Mock()
        mock_DataFrame.return_value = df

        self.test_obj.save_preds(probs, ids, '1', 'pred/', 'dset')

        exp = [(1, 0.8), (2, 0.6), (3, 0.3), (4, 0.1)]
        mock_DataFrame.assert_called_with(exp, columns=['com_id', 'ind_pred'])
        df.to_csv.assert_called_with('pred/dset_1_preds.csv', index=None)

    def test_classifier_random_forest(self):
        result = self.test_obj.classifier('rf')

        self.assertTrue(isinstance(result, RandomForestClassifier))

    def test_classifier_logistic_regression(self):
        result = self.test_obj.classifier('lr')

        self.assertTrue(isinstance(result, LogisticRegression))

    def test_mean(self):
        result = self.test_obj.mean([0, 1, 2, 3])

        self.assertTrue(result == 1.5)

    def test_percent(self):
        result = self.test_obj.percent(50, 100)

        self.assertTrue(result == 50.0)

    def test_percent_zero_denom(self):
        result = self.test_obj.percent(50, 0)

        self.assertTrue(result == 0.0)

    def test_div0_denom_zero(self):
        result = self.test_obj.div0(100, 0)

        self.assertTrue(result == 0.0)

    def test_div0_denom_non_zero(self):
        result = self.test_obj.div0(100, 125)

        self.assertTrue(result == 0.8)

    def test_find_max_prec_recall_none(self):
        result = self.test_obj.find_max_prec_recall([], [], [])

        self.assertTrue(result == (-1, -1, None))

    def test_find_max_prec_recall_max(self):
        prec = [0.4, 0.7, 0.9, 0.2]
        rec = [0.9, 0.8, 0.7, 0.99]
        tholds = [0.2, 0.4, 0.7, 0.8]
        result = self.test_obj.find_max_prec_recall(prec, rec, tholds)

        self.assertTrue(result == (0.9, 0.7, 0.7))

    def test_colorize_no_display(self):
        termcolor.colored = mock.Mock()

        result = self.test_obj.colorize('dummy', 'green', False)

        termcolor.colored.assert_not_called()
        self.assertTrue(result == 'dummy')

    def test_colorize_display(self):
        termcolor.colored = mock.Mock(return_value='colorized!')

        result = self.test_obj.colorize('dummy', 'green', True)

        termcolor.colored.assert_called_with('dummy', 'green')
        self.assertTrue(result == 'colorized!')

    def test_compute_score(self):
        probs = np.array([[0.2, 0.8], [0.4, 0.6], [0.7, 0.3], [0.9, 0.1]])
        y = np.array([1, 0, 1, 0])
        self.test_obj.gen_noise = mock.Mock(return_value=0.77)
        sm.roc_curve = mock.Mock(return_value=('fpr', 'tpr', 'ts'))
        sm.precision_recall_curve = mock.Mock(return_value=('p', 'r', 'ts'))
        sm.average_precision_score = mock.Mock(return_value='aupr')
        sm.auc = mock.Mock(return_value='auroc')
        self.test_obj.find_max_prec_recall = mock.Mock(return_value=('mp',
                'mr', 't'))

        result = self.test_obj.compute_scores(probs, y)

        expected = [mock.call(0.8), mock.call(0.6), mock.call(0.3),
            mock.call(0.1)]
        ppn = [0.77, 0.77, 0.77, 0.77]
        self.assertTrue(result == ('auroc', 'aupr', 'p', 'r', 'mp', 'mr', 't'))
        self.assertTrue(self.test_obj.gen_noise.call_args_list == expected)
        sm.roc_curve.assert_called_with(y, ppn)
        sm.precision_recall_curve.assert_called_with(y, ppn)
        sm.average_precision_score.assert_called_with(y, ppn)
        sm.auc.assert_called_with('fpr', 'tpr')
        self.test_obj.find_max_prec_recall.assert_called_with('p', 'r', 'ts')

    def test_train(self):
        model1 = LogisticRegression()
        model2 = LogisticRegression()
        model1.fit = mock.Mock(return_value=model2)
        self.test_obj.start = mock.Mock()
        self.test_obj.classifier = mock.Mock(return_value=model1)
        self.test_obj.end = mock.Mock()
        data = ('1', '2', '3', '4', '5', '6')

        result = self.test_obj.train(data, classifier='rf', fw='fw')

        self.assertTrue(result == model2)
        self.test_obj.start.assert_called_with('training...', fw='fw')
        self.test_obj.classifier.assert_called_with('rf')
        model1.fit.assert_called_with('1', '2')
        self.test_obj.end.assert_called_with(fw='fw')

    def test_test(self):
        model = LogisticRegression()
        model.predict_proba = mock.Mock(return_value='test_probs')
        self.test_obj.start = mock.Mock()
        self.test_obj.end = mock.Mock()
        data = ('1', '2', '3', '4', '5', '6')

        result = self.test_obj.test(data, model, fw='fw')

        self.assertTrue(result == ('test_probs', '5'))
        self.test_obj.start.assert_called_with('testing...', fw='fw')
        model.predict_proba.assert_called_with('3')
        self.test_obj.end.assert_called_with(fw='fw')

    def test_evaluate(self):
        data = ('1', '2', '3', '4', '5', '6')
        scores = ('auroc', 'aupr', 'p', 'r', 'mp', 'mr', 't')
        self.test_obj.start = mock.Mock()
        self.test_obj.end = mock.Mock()
        self.test_obj.compute_scores = mock.Mock(return_value=scores)
        self.test_obj.print_scores = mock.Mock()
        self.test_obj.print_median_mean = mock.Mock()

        self.test_obj.evaluate(data, 'test_probs', fw='fw')

        self.test_obj.start.assert_called_with('evaluating...', fw='fw')
        self.test_obj.compute_scores.assert_called_with('test_probs', '4')
        self.test_obj.end.assert_called_with(fw='fw')
        self.test_obj.print_scores.assert_called_with('mp', 'mr', 't', 'aupr',
                'auroc', fw='fw')
        self.test_obj.print_median_mean.assert_called_with('5', 'test_probs',
                '4', fw='fw')


    # def test_classify(self):
    #     model1 = LogisticRegression()
    #     model2 = LogisticRegression()
    #     model1.fit = mock.Mock(return_value=model2)
    #     model2.predict_proba = mock.Mock(return_value='probs')
    #     self.test_obj.start = mock.Mock()
    #     self.test_obj.classifier = mock.Mock(return_value=model1)
    #     self.test_obj.save = mock.Mock()
    #     self.test_obj.end = mock.Mock()
    #     self.test_obj.compute_scores = mock.Mock(return_value=('auroc',
    #             'aupr', 'p', 'r', 'mp', 'mr', 't'))
    #     self.test_obj.print_scores = mock.Mock()
    #     self.test_obj.print_median_mean = mock.Mock()
    #     self.test_obj.plot_pr_curve = mock.Mock()
    #     self.test_obj.plot_features = mock.Mock()
    #     self.test_obj.save_preds = mock.Mock()
    #     data = 'x_tr', 'y_tr', 'x_te', 'y_te', 'id_te', 'feat_names'

    #     self.test_obj.classify(data, '1', 'feat_set', 'images/', 'pred/',
    #             'model/', save_pr_plot=True, line='-', save_feat_plot=True,
    #             save_preds=True, classifier='lr', dset='test', fw='fw')

    #     self.test_obj.classifier.assert_called_with('lr')
    #     model1.fit.assert_called_with('x_tr', 'y_tr')
    #     self.test_obj.save.assert_called_with(model2, 'model/test_lr_1.pkl')
    #     model2.predict_proba.assert_called_with('x_te')
    #     self.test_obj.compute_scores.assert_called_with('probs', 'y_te')
    #     self.test_obj.print_scores.assert_called_with('mp', 'mr', 't', 'aupr',
    #             'auroc', fw='fw')
    #     self.test_obj.print_median_mean.assert_called_with('id_te', 'probs',
    #             'y_te', fw='fw')
    #     self.test_obj.plot_pr_curve.assert_called_with('feat_set_1',
    #             'images/feat_set_1', 'r', 'p', 'aupr', title='feat_set',
    #             line='-', save=True)
    #     self.test_obj.plot_features.assert_called_with(model2, 'lr',
    #             'feat_names', 'images/feat_set_1', save=True)
    #     self.test_obj.save_preds.assert_called_with('probs', 'id_te', '1',
    #             'pred/', 'test')
    #     self.assertTrue(self.test_obj.start.call_args_list ==
    #             [mock.call('training...', fw='fw'),
    #             mock.call('testing...', fw='fw'),
    #             mock.call('evaluating...', fw='fw')])
    #     self.assertTrue(self.test_obj.end.call_args_list ==
    #             [mock.call(fw='fw'), mock.call(fw='fw'), mock.call(fw='fw')])

    def test_close_writer(self):
        sw = mock.Mock()
        sw.close = mock.Mock()

        self.test_obj.close_writer(sw)

        sw.close.assert_called()

    def test_check_file_exists(self):
        os.path.exists = mock.Mock(return_value=True)
        self.test_obj.exit = mock.Mock()

        result = self.test_obj.check_file('boogers')

        self.assertTrue(result)
        self.test_obj.exit.assert_not_called()

    def test_check_file_does_not_exist(self):
        os.path.exists = mock.Mock(return_value=False)
        self.test_obj.exit = mock.Mock()

        result = self.test_obj.check_file('boogers')

        self.assertTrue(not result)
        self.test_obj.exit.assert_called_with('cannot read boogers')

    @mock.patch('scipy.sparse.load_npz')
    def test_load_sparse(self, mock_load_npz):
        mock_load_npz.return_value = 'matrix'

        result = self.test_obj.load_sparse('filename')

        self.assertTrue(result == 'matrix')
        mock_load_npz.assert_called_with('filename')

    @mock.patch('scipy.sparse.save_npz')
    def test_save_sparse(self, mock_save_npz):

        self.test_obj.save_sparse('matrix', 'filename')

        mock_save_npz.assert_called_with('filename', 'matrix')

    @mock.patch('os.path.exists')
    def test_read_csv_none(self, mock_exists):
        mock_exists.return_value = False

        result = self.test_obj.read_csv('fname')

        self.assertTrue(result is None)

    @mock.patch('pandas.read_csv')
    @mock.patch('os.path.exists')
    def test_read_csv_exists(self, mock_exists, mock_read_csv):
        mock_exists.return_value = True
        mock_read_csv.return_value = 'df'

        result = self.test_obj.read_csv('fname')

        self.assertTrue(result == 'df')


def test_suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(UtilTestCase)
    return suite

if __name__ == '__main__':
    unittest.main()
