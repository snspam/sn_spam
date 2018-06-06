"""
Tests the analysis module.
"""
import unittest
import pandas as pd
import mock
from .context import analysis
from .context import evaluation
from .context import interpretability
from .context import label
from .context import purity
from .context import util
from .context import test_utils as tu


class AnalysisTestCase(unittest.TestCase):
    def setUp(self):
        config_obj = tu.sample_config()
        mock_label_obj = mock.Mock(label.Label)
        mock_purity_obj = mock.Mock(purity.Purity)
        mock_evaluation_obj = mock.Mock(evaluation.Evaluation)
        mock_interpretability_obj = mock.Mock(
                interpretability.Interpretability)
        util_obj = util.Util()
        self.test_obj = analysis.Analysis(config_obj, mock_label_obj,
                mock_purity_obj, mock_evaluation_obj,
                mock_interpretability_obj, util_obj)

    def tearDown(self):
        self.test_obj = None

    def test_init(self):
        # setup
        test_obj = self.test_obj

        # assert
        self.assertTrue(isinstance(test_obj.purity_obj,
                purity.Purity))
        self.assertTrue(isinstance(test_obj.evaluation_obj,
                evaluation.Evaluation))
        self.assertTrue(isinstance(test_obj.interpretability_obj,
                interpretability.Interpretability))

    def test_relabel(self):
        self.test_obj.label_obj.relabel = mock.Mock()

        self.test_obj.relabel()

        self.test_obj.label_obj.relabel.assert_called()

    def test_purity(self):
        self.test_obj.purity_obj.test_relations = mock.Mock()

        self.test_obj.test_purity('df')

        self.test_obj.purity_obj.test_relations.assert_called_with('df')

    def test_evaluate(self):
        self.test_obj.config_obj.modified = True
        self.test_obj.check_dataframe = mock.Mock(return_value='df2')
        self.test_obj.evaluation_obj.evaluate = mock.Mock()

        self.test_obj.evaluate('df')

        self.test_obj.check_dataframe.assert_called_with('df')
        self.test_obj.evaluation_obj.evaluate.assert_called_with('df2',
                modified=True)

    def test_explain(self):
        self.test_obj.interpretability_obj.explain = mock.Mock()
        self.test_obj.check_dataframe = mock.Mock(return_value='df2')

        self.test_obj.explain('df')

        self.test_obj.check_dataframe.assert_called_with('df')
        self.test_obj.interpretability_obj.explain.assert_called_with('df2')

    def test_check_dataframe(self):
        result = self.test_obj.check_dataframe('df')

        self.assertTrue(result == 'df')

    def test_check_dataframe_none(self):
        self.test_obj.define_file_folders = mock.Mock(return_value='folds/')
        self.test_obj.read_fold = mock.Mock(return_value='df')

        result = self.test_obj.check_dataframe(None)

        self.test_obj.define_file_folders.assert_called()
        self.test_obj.read_fold.assert_called_with('folds/')
        self.assertTrue(result == 'df')

    def test_define_file_folders(self):
        result = self.test_obj.define_file_folders()

        self.assertTrue(result == 'ind/data/soundcloud/folds/')

    def test_read_fold(self):
        self.test_obj.util_obj.check_file = mock.Mock(return_value=True)
        pd.read_csv = mock.Mock(return_value='df')

        result = self.test_obj.read_fold('f/')

        self.test_obj.util_obj.check_file.assert_called_with('f/test_1.csv')
        pd.read_csv.assert_called_with('f/test_1.csv', lineterminator='\n')
        self.assertTrue(result == 'df')


def test_suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(AnalysisTestCase)
    return suite

if __name__ == '__main__':
    unittest.main()
