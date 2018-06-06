"""
Tests the training_exp module.
"""
import mock
import unittest
import pandas as pd
from .context import training_exp
from .context import config
from .context import runner
from .context import test_utils as tu


class Training_ExperimentTestCase(unittest.TestCase):
    def setUp(self):
        config_obj = tu.sample_config()
        mock_runner_obj = mock.Mock(runner.Runner)
        self.test_obj = training_exp.Training_Experiment(config_obj,
                mock_runner_obj)

    def tearDown(self):
        self.test_obj = None

    def test_init(self):
        # setup
        test_obj = self.test_obj

        # assert
        self.assertTrue(isinstance(test_obj.config_obj, config.Config))
        self.assertTrue(isinstance(test_obj.runner_obj, runner.Runner))
        self.assertTrue(test_obj.config_obj.modified)
        self.assertTrue(test_obj.config_obj.pseudo)
        self.assertTrue(not test_obj.config_obj.super_train)

    def test_divide_data_into_subsets(self):
        v = mock.Mock()
        v.merge = mock.Mock(return_value='v_df')
        t = mock.Mock()
        t.merge = mock.Mock(return_value='t_df')
        self.test_obj.config_obj.end = 2000
        self.test_obj.config_obj.start = 0
        self.test_obj.config_obj.fold = '0'
        self.test_obj.config_obj.ind_dir = 'ind/'
        self.test_obj.config_obj.val_size = 0.69
        self.test_obj.config_obj.domain = 'dom'
        self.test_obj.config_obj.val_size = 0.7
        self.test_obj.independent_run = mock.Mock(return_value=(v, t))
        self.test_obj.create_fold = mock.Mock()
        pd.read_csv = mock.Mock()
        pd.read_csv.side_effect = ['vp', 'tp']

        result = self.test_obj.divide_data_into_subsets(growth_factor=2,
                val_size=100)

        exp_read = [mock.call('ind/output/dom/predictions/val_0_preds.csv'),
            mock.call('ind/output/dom/predictions/test_0_preds.csv')]
        exp_fold = [mock.call('v_df', 't_df', 100, '0'),
                mock.call('v_df', 't_df', 200, '1'),
                mock.call('v_df', 't_df', 400, '2'),
                mock.call('v_df', 't_df', 800, '3')]
        self.test_obj.independent_run.assert_called_with()
        self.assertTrue(pd.read_csv.call_args_list == exp_read)
        v.merge.assert_called_with('vp', on='com_id', how='left')
        t.merge.assert_called_with('tp', on='com_id', how='left')
        self.assertTrue(self.test_obj.create_fold.call_args_list == exp_fold)
        self.assertTrue(result == ['0', '1', '2', '3'])

    def test_run_experiment(self):
        folds = ['1', '2']
        self.test_obj.change_config_fold = mock.Mock()
        self.test_obj.read_fold = mock.Mock()
        self.test_obj.read_fold.side_effect = [('v1', 't1'), ('v2', 't2')]
        self.test_obj.relational_run = mock.Mock()

        self.test_obj.run_experiment(folds)

        exp_ccf = [mock.call('1'), mock.call('2')]
        exp_rf = [mock.call('1'), mock.call('2')]
        exp_rr = [mock.call('v1', 't1'), mock.call('v2', 't2')]
        self.assertTrue(self.test_obj.change_config_fold.call_args_list ==
                    exp_ccf)
        self.assertTrue(self.test_obj.read_fold.call_args_list == exp_rf)
        self.assertTrue(self.test_obj.relational_run.call_args_list == exp_rr)

    def test_independent_run(self):
        dfs = ('v', 't')
        self.test_obj.runner_obj.run_independent = mock.Mock(return_value=dfs)

        result = self.test_obj.independent_run()

        self.assertTrue(result == ('v', 't'))
        self.test_obj.runner_obj.run_independent.assert_called_with()

    def test_read_fold(self):
        self.test_obj.config_obj.domain = 'dom'
        self.test_obj.config_obj.ind_dir = 'ind/'
        pd.read_csv = mock.Mock()
        pd.read_csv.side_effect = ['v', 't']

        result = self.test_obj.read_fold('2')

        exp_read = [mock.call('ind/data/dom/folds/val_2.csv'),
                mock.call('ind/data/dom/folds/test_2.csv')]
        self.assertTrue(pd.read_csv.call_args_list == exp_read)
        self.assertTrue(result == ('v', 't'))

    def test_create_fold(self):
        self.test_obj.config_obj.domain = 'dom'
        self.test_obj.config_obj.ind_dir = 'ind/'
        vf = mock.Mock()
        vf.to_csv = mock.Mock()
        v_df = mock.Mock()
        v_df.drop = mock.Mock(return_value=vf)
        v_df.to_csv = mock.Mock()
        v = mock.Mock()
        v.tail = mock.Mock(return_value=v_df)
        tf = mock.Mock()
        tf.to_csv = mock.Mock()
        t_df = mock.Mock()
        t_df.drop = mock.Mock(return_value=tf)
        t_df.to_csv = mock.Mock()

        self.test_obj.create_fold(v, t_df, 100, '2')

        exp_val_pred = 'ind/output/dom/predictions/val_2_preds.csv'
        exp_test_pred = 'ind/output/dom/predictions/test_2_preds.csv'
        v.tail.assert_called_with(100)
        v_df.drop.assert_called_with(['ind_pred'], axis=1)
        t_df.drop.assert_called_with(['ind_pred'], axis=1)
        vf.to_csv.assert_called_with('ind/data/dom/folds/val_2.csv',
                index=None, line_terminator='\n')
        tf.to_csv.assert_called_with('ind/data/dom/folds/test_2.csv',
                index=None, line_terminator='\n')
        v_df.to_csv.assert_called_with(exp_val_pred, index=None,
            line_terminator='\n', columns=['com_id', 'ind_pred'])
        t_df.to_csv.assert_called_with(exp_test_pred, index=None,
            line_terminator='\n', columns=['com_id', 'ind_pred'])

    def test_relational_run(self):
        self.test_obj.change_config_rel_op = mock.Mock()
        self.test_obj.runner_obj.run_relational = mock.Mock()
        self.test_obj.runner_obj.run_evaluation = mock.Mock()

        self.test_obj.relational_run('v', 't')

        exp_ccro = [mock.call(train=True), mock.call(train=False)]
        exp_rel = [mock.call('v', 't'), mock.call('v', 't')]
        self.assertTrue(self.test_obj.change_config_rel_op.call_args_list ==
                exp_ccro)
        self.assertTrue(self.test_obj.runner_obj.run_relational.call_args_list
                == exp_rel)
        self.test_obj.runner_obj.run_evaluation.assert_called_with('t')

    def test_change_config_fold(self):
        self.test_obj.change_config_fold('69')

        self.assertTrue(self.test_obj.config_obj.fold == '69')

    def test_change_config_rel_op(self):
        self.test_obj.change_config_rel_op(train=False)

        self.assertTrue(self.test_obj.config_obj.infer)


def test_suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(
            Training_ExperimentTestCase)
    return suite

if __name__ == '__main__':
    unittest.main()
