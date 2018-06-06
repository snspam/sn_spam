"""
Tests the evaluation module.
"""
import os
import mock
import unittest
from .context import evaluation
from .context import config
from .context import generator
from .context import util
from .context import test_utils as tu


class EvaluationTestCase(unittest.TestCase):
    def setUp(self):
        config_obj = tu.sample_config()
        generator_obj = generator.Generator()
        mock_util_obj = mock.Mock(util.Util)
        self.test_obj = evaluation.Evaluation(config_obj, generator_obj,
                mock_util_obj)

    def tearDown(self):
        self.test_obj = None

    def test_init(self):
        # setup
        result = self.test_obj

        # assert
        self.assertTrue(isinstance(result.config_obj, config.Config))
        self.assertTrue(isinstance(result.util_obj, util.Util))

    def test_settings(self):
        self.test_obj.util_obj.set_noise_limit = mock.Mock()

        self.test_obj.settings()

        self.test_obj.util_obj.set_noise_limit.assert_called_with(0.0025)

    def test_file_folders(self):
        os.makedirs = mock.Mock()

        result = self.test_obj.file_folders()

        self.assertTrue(result[0] == 'ind/data/soundcloud/')
        self.assertTrue(result[1] == 'ind/output/soundcloud/predictions/')
        self.assertTrue(result[2] == 'rel/output/soundcloud/predictions/')
        self.assertTrue(result[3] == 'rel/output/soundcloud/images/')

    def test_open_status_write(self):
        self.test_obj.util_obj.open_writer = mock.Mock(return_value='f')

        result = self.test_obj.open_status_writer('status/')

        exp_path = 'status/eval_1.txt'
        self.assertTrue(result == 'f')
        self.test_obj.util_obj.open_writer.assert_called_with(exp_path)

    def test_read_predictions(self):
        test_df = tu.sample_df(10)
        nps_df = tu.sample_df(10)
        r_df = tu.sample_df(10)
        self.test_obj.util_obj.read_csv = mock.Mock()
        self.test_obj.util_obj.read_csv.side_effect = [nps_df, None, r_df,
                None]

        result = self.test_obj.read_predictions(test_df, 'ind/', 'rel/')

        expected = [mock.call('ind/nps_test_1_preds.csv'),
                mock.call('ind/test_1_preds.csv'),
                mock.call('rel/predictions_1.csv'),
                mock.call('rel/mrf_preds_1.csv')]
        exp_preds = [(nps_df, 'nps_pred', 'No Pseudo', '-'),
                (r_df, 'rel_pred', 'Relational', ':')]
        self.assertTrue(self.test_obj.util_obj.read_csv.call_args_list ==
                expected)
        self.assertTrue(result == exp_preds)

    def test_merge_predictions(self):
        df = tu.sample_df(10)
        pred_df = tu.sample_df(10)
        pred_df.columns = ['com_id', 'ip']

        result = self.test_obj.merge_predictions(df, pred_df)

        self.assertTrue(list(result) == ['com_id', 'random', 'ip'])
        self.assertTrue(len(result) == 10)

    def test_apply_noise(self):
        df = tu.sample_df(10)
        df.columns = ['com_id', 'ind_pred']
        df['rel_pred'] = 100
        df2 = df.copy()

        result = self.test_obj.apply_noise(df, 'ind_pred')

        self.assertTrue(len(result) == 10)
        self.assertTrue(not result.equals(df2))

    def test_compute_scores(self):
        df = tu.sample_df(10)
        df['pred'] = [0.1, 0.7, 0.3, 0.4, 0.7, 0.8, 0.9, 0.2, 0.77, 0.88]
        df['label'] = [0, 1, 0, 1, 1, 1, 1, 0, 1, 1]

        result = self.test_obj.compute_scores(df, 'pred')

        self.assertTrue(result[0] == 0.99999999999999978)  # aupr
        self.assertTrue(result[1] == 1.0)  # auroc
        self.assertTrue(len(result[2]) == 7)  # recalls
        self.assertTrue(len(result[3]) == 7)  # precisions
        self.assertTrue(result[4] == 1.0)  # n-aupr

    def test_evaluate(self):
        preds = [('nps_df', 'nps_pred', 'No Pseudo', '-'),
                ('ind_df', 'ind_pred', 'Independent', '--')]
        df = tu.sample_df(10)
        df.copy = mock.Mock(return_value='t_df')
        self.test_obj.settings = mock.Mock()
        self.test_obj.file_folders = mock.Mock(return_value=('a/',
                'b/', 'c/', 'd/', 'e/'))
        self.test_obj.open_status_writer = mock.Mock(return_value='sw')
        self.test_obj.read_predictions = mock.Mock(return_value=preds)
        self.test_obj.read_modified = mock.Mock(return_value='mod_df')
        self.test_obj.merge_and_score = mock.Mock()
        self.test_obj.util_obj.close_writer = mock.Mock()

        self.test_obj.evaluate(df, modified=True)

        exp_ms = [mock.call('t_df', preds[0], 'd/pr_1', False, 'mod_df',
                'sw'), mock.call('t_df', preds[1], 'd/pr_1', True,
                'mod_df', 'sw')]
        df.copy.assert_called()
        self.test_obj.settings.assert_called()
        self.test_obj.file_folders.assert_called()
        self.test_obj.open_status_writer.assert_called_with('e/')
        self.test_obj.read_predictions.assert_called_with('t_df', 'b/', 'c/')
        self.assertTrue(self.test_obj.merge_and_score.call_args_list == exp_ms)
        self.test_obj.util_obj.close_writer.assert_called_with('sw')

    def test_merge_and_score(self):
        pred = ('nps_df', 'nps_pred', 'No Pseudo', '-')
        scores = ('i1', 'i2', 'i3', 'i4', 'i5')
        self.test_obj.merge_predictions = mock.Mock(return_value='m_df')
        self.test_obj.filter = mock.Mock(return_value='m2_df')
        self.test_obj.apply_noise = mock.Mock(return_value='noise_df')
        self.test_obj.compute_scores = mock.Mock(return_value=scores)
        self.test_obj.print_scores = mock.Mock()
        self.test_obj.util_obj.plot_pr_curve = mock.Mock()

        self.test_obj.merge_and_score('t_df', pred, 'fname', save=False,
                modified_df='mod_df', fw='fw')

        self.test_obj.merge_predictions.assert_called_with('t_df', 'nps_df')
        self.test_obj.filter.assert_called_with('m_df', 'mod_df')
        self.test_obj.apply_noise.assert_called_with('m2_df', 'nps_pred')
        self.test_obj.compute_scores.assert_called_with('noise_df', 'nps_pred')
        self.test_obj.print_scores.assert_called_with('No Pseudo', 'i1', 'i2',
                'i5', fw='fw')
        self.test_obj.util_obj.plot_pr_curve.assert_called_with('No Pseudo',
                'fname', 'i3', 'i4', 'i5', line='-', save=False)


def test_suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(EvaluationTestCase)
    return suite

if __name__ == '__main__':
    unittest.main()
