"""
Tests the relational module.
"""
import os
import unittest
import numpy as np
import pandas as pd
import sklearn.metrics as sm
import mock
from .context import tuffy
from .context import config
from .context import pred_builder
from .context import util
from .context import test_utils as tu


class TuffyTestCase(unittest.TestCase):
    def setUp(self):
        config_obj = tu.sample_config()
        mock_pred_builder_obj = mock.Mock(pred_builder.PredicateBuilder)
        util_obj = util.Util()
        self.test_obj = tuffy.Tuffy(config_obj, mock_pred_builder_obj,
                util_obj)

    def tearDown(self):
        self.test_obj = None

    def test_init(self):
        # setup
        result = self.test_obj

        # assert
        self.assertTrue(isinstance(result.config_obj, config.Config))
        self.assertTrue(isinstance(result.pred_builder_obj,
                pred_builder.PredicateBuilder))
        self.assertTrue(isinstance(result.util_obj, util.Util))

    def test_clear_data(self):
        os.system = mock.Mock()

        self.test_obj.clear_data('t/')

        self.assertTrue(os.system.call_args_list ==
                [mock.call('rm t/*evidence.txt'),
                mock.call('rm t/*query.txt')])

    def test_gen_predicates(self):
        self.test_obj.pred_builder_obj.build_comments = mock.Mock()
        self.test_obj.pred_builder_obj.build_relations = mock.Mock()

        self.test_obj.gen_predicates('df', 'test', 'd/')

        expected = [mock.call('intext', 'text', 'text_id', 'df', 'test', 'd/',
                tuffy=True), mock.call('posts', 'user', 'user_id', 'df',
                'test', 'd/', tuffy=True)]
        self.test_obj.pred_builder_obj.build_comments.assert_called_with('df',
                'test', 'd/', tuffy=True)
        self.assertTrue(self.test_obj.pred_builder_obj.build_relations.
                call_args_list == expected)

    def test_run(self):
        os.chdir = mock.Mock()
        os.sytem = mock.Mock()

        self.test_obj.run('tuffy/')

        execute = 'java -jar tuffy.jar -i mln.txt -e test_evidence.txt '
        execute += '-queryFile test_query.txt -r out -dual > log'
        os.chdir.assert_called_with('tuffy/')
        os.system.assert_called_with(execute)

    def test_parse_output(self):
        mar_df = tu.sample_df(10)
        pred_df = tu.sample_df(10)
        pred_df['map'] = [1, 1, 1, 1, 1, 1, np.nan, 1, np.nan, 1]
        self.test_obj.parse_map_output = mock.Mock(return_value='map_df')
        self.test_obj.parse_marginal_output = mock.Mock(return_value=mar_df)
        mar_df.merge = mock.Mock(return_value=pred_df)

        result = self.test_obj.parse_output('tuffy/')

        exp = pd.Series([1, 1, 1, 1, 1, 1, 0, 1, 0, 1])
        self.test_obj.parse_map_output.assert_called_with('tuffy/out.map')
        self.test_obj.parse_marginal_output.assert_called_with(
                'tuffy/out.marginal')
        mar_df.merge.assert_called_with('map_df', on='com_id', how='left')
        self.assertTrue(result['map'].equals(exp))

    def test_evaluate(self):
        test_df = tu.sample_df(2)
        df = tu.sample_df(2)
        df['label'] = [1, 0]
        df['map'] = [1, 0]
        df['marginal'] = [0.77, 0.22]
        test_df.merge = mock.Mock(return_value=df)
        sm.average_precision_score = mock.Mock(return_value=0.777)
        sm.precision_score = mock.Mock(return_value=0.72)
        sm.recall_score = mock.Mock(return_value=0.27)

        self.test_obj.evaluate(test_df, 'p_df')

        test_df.merge.assert_called_with('p_df', on='com_id')
        sm.average_precision_score.assert_called_with(df['label'],
                df['marginal'])
        sm.precision_score.assert_called_with(df['label'], df['map'])
        sm.recall_score.assert_called_with(df['label'], df['map'])

    def test_parse_map_output(self):
        filename = 'relational/scripts/tests/resources/sample_out.map'

        result = self.test_obj.parse_map_output(filename)

        exp = pd.DataFrame([(1, 1), (2, 1), (7, 1), (8, 1)],
                columns=['com_id', 'map'])
        self.assertTrue(result.equals(exp))

    def test_parse_marginal_output(self):
        filename = 'relational/scripts/tests/resources/sample_out.marginal'

        result = self.test_obj.parse_marginal_output(filename)

        exp = pd.DataFrame([(78863, 1.0), (78777, 0.98), (78801, 0.96),
                (79236, 0.94), (78826, 0.69)], columns=['com_id', 'marginal'])
        self.assertTrue(result.equals(exp))


def test_suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(TuffyTestCase)
    return suite

if __name__ == '__main__':
    unittest.main()
