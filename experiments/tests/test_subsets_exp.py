"""
Tests the subsets_exp module.
"""
import mock
import unittest
from .context import subsets_exp
from .context import config
from .context import runner
from .context import test_utils as tu


class Subsets_ExperimentTestCase(unittest.TestCase):
    def setUp(self):
        config_obj = tu.sample_config()
        mock_runner_obj = mock.Mock(runner.Runner)
        self.test_obj = subsets_exp.Subsets_Experiment(config_obj,
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

    def test_divide_data_into_subsets(self):
        self.test_obj.config_obj.end = 4000
        self.test_obj.config_obj.start = 0
        self.test_obj.config_obj.fold = '0'

        result = self.test_obj.divide_data_into_subsets(num_subsets=4)

        exp = [(0, 1000, '0'), (1000, 2000, '1'), (2000, 3000, '2'),
                (3000, 4000, '3')]
        self.assertTrue(len(result) == 4)
        self.assertTrue(result == exp)

    def test_run_experiment(self):
        subsets = [(1, 2, '4'), (7, 77, '88'), (7, 88, '169')]
        self.test_obj.single_run = mock.Mock()
        self.test_obj.change_config_parameters = mock.Mock()

        self.test_obj.run_experiment(subsets)

        exp_ccp = [mock.call(1, 2, '4'), mock.call(7, 77, '88'),
                mock.call(7, 88, '169')]
        self.assertTrue(self.test_obj.single_run.call_count == 3)
        self.assertTrue(self.test_obj.change_config_parameters.call_args_list
                == exp_ccp)

    def test_single_run(self):
        self.test_obj.runner_obj.run_independent = mock.Mock()
        self.test_obj.runner_obj.run_independent.return_value = ('v', 't')
        self.test_obj.change_config_rel_op = mock.Mock()
        self.test_obj.runner_obj.run_relational = mock.Mock()
        self.test_obj.runner_obj.run_evaluation = mock.Mock()

        self.test_obj.single_run()

        exp_ccro = [mock.call(train=True), mock.call(train=False)]
        exp_rel = [mock.call('v', 't'), mock.call('v', 't')]
        self.test_obj.runner_obj.run_independent.assert_called_with()
        self.assertTrue(self.test_obj.change_config_rel_op.call_args_list ==
                exp_ccro)
        self.assertTrue(self.test_obj.runner_obj.run_relational.call_args_list
                == exp_rel)
        self.test_obj.runner_obj.run_evaluation.assert_called_with('t')

    def test_change_config_parameters(self):
        self.test_obj.change_config_parameters(2, 4, '69')

        self.assertTrue(self.test_obj.config_obj.start == 2)
        self.assertTrue(self.test_obj.config_obj.end == 4)
        self.assertTrue(self.test_obj.config_obj.fold == '69')

    def test_change_config_rel_op(self):
        self.test_obj.change_config_rel_op(train=False)

        self.assertTrue(self.test_obj.config_obj.infer)


def test_suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(
            Subsets_ExperimentTestCase)
    return suite

if __name__ == '__main__':
    unittest.main()
