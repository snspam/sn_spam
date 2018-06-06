"""
Tests the single_exp module.
"""
import mock
import unittest
from .context import single_exp
from .context import config
from .context import runner
from .context import test_utils as tu


class Single_ExperimentTestCase(unittest.TestCase):
    def setUp(self):
        config_obj = tu.sample_config()
        mock_runner_obj = mock.Mock(runner.Runner)
        self.test_obj = single_exp.Single_Experiment(config_obj,
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

    def test_run_experiment(self):
        self.test_obj.single_run = mock.Mock()

        self.test_obj.run_experiment()

        self.test_obj.single_run.assert_called_with()

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

    def test_change_config_rel_op(self):
        self.test_obj.change_config_rel_op(train=False)

        self.assertTrue(self.test_obj.config_obj.infer)


def test_suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(
            Single_ExperimentTestCase)
    return suite

if __name__ == '__main__':
    unittest.main()
