"""
Tests the runner module.
"""
import unittest
import mock
import pandas as pd
from context import runner
from context import independent
from context import relational
from context import analysis


class RunnerTestCase(unittest.TestCase):
    def setUp(self):
        mock_independent_obj = mock.Mock(independent.Independent)
        mock_relational_obj = mock.Mock(relational.Relational)
        mock_analysis_obj = mock.Mock(analysis.Analysis)
        self.test_obj = runner.Runner(mock_independent_obj,
                mock_relational_obj, mock_analysis_obj)

    def tearDown(self):
        self.test_obj = None

    def test_init(self):
        test_obj = self.test_obj

        self.assertTrue(isinstance(test_obj.independent_obj,
                independent.Independent))
        self.assertTrue(isinstance(test_obj.relational_obj,
                relational.Relational))
        self.assertTrue(isinstance(test_obj.analysis_obj, analysis.Analysis))

    def test_compile_reasoning_engine(self):
        self.test_obj.relational_obj.compile_reasoning_engine = mock.Mock()

        self.test_obj.compile_reasoning_engine()

        self.test_obj.relational_obj.compile_reasoning_engine.assert_called()

    def test_run_label(self):
        self.test_obj.analysis_obj.relabel = mock.Mock()

        self.test_obj.run_label()

        self.test_obj.analysis_obj.relabel.assert_called()

    def test_run_independent_main(self):
        df1, df2 = pd.DataFrame(), pd.DataFrame()
        self.test_obj.independent_obj.main = mock.Mock(return_value=(df1, df2))

        result = self.test_obj.run_independent()

        self.test_obj.independent_obj.main.assert_called()
        self.assertTrue(result[0].equals(df1.reset_index()))
        self.assertTrue(result[1].equals(df2.reset_index()))

    def test_run_purity(self):
        self.test_obj.analysis_obj.test_purity = mock.Mock()

        self.test_obj.run_purity('test_df')

        self.test_obj.analysis_obj.test_purity.assert_called_with('test_df')

    def test_run_relational(self):
        self.test_obj.relational_obj.main = mock.Mock()

        self.test_obj.run_relational('val_df', 'test_df')

        self.test_obj.relational_obj.main.assert_called_with('val_df',
                'test_df')

    def test_run_evaluation(self):
        self.test_obj.analysis_obj.evaluate = mock.Mock()

        self.test_obj.run_evaluation('df')

        self.test_obj.analysis_obj.evaluate.assert_called_with('df')

    def test_run_explanation(self):
        self.test_obj.analysis_obj.explain = mock.Mock()

        self.test_obj.run_explanation('df')

        self.test_obj.analysis_obj.explain.assert_called_with('df')


def test_suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(RunnerTestCase)
    return suite

if __name__ == '__main__':
    unittest.main()
