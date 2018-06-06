"""
Tests the interpretability module.
"""
import os
import unittest
import numpy as np
import pandas as pd
import mock
from .context import interpretability
from .context import config
from .context import connections
from .context import generator
from .context import pred_builder
from .context import util
from .context import test_utils as tu
from sklearn.linear_model import LinearRegression


class InterpretabilityTestCase(unittest.TestCase):
    def setUp(self):
        config_obj = tu.sample_config()
        connections_obj = connections.Connections()
        generator_obj = generator.Generator()
        mock_pred_builder_obj = mock.Mock(pred_builder.PredicateBuilder)
        util_obj = util.Util()
        self.test_obj = interpretability.Interpretability(config_obj,
                connections_obj, generator_obj, mock_pred_builder_obj,
                util_obj)

    def tearDown(self):
        self.test_obj = None

    def test_init(self):
        # setup
        test_obj = self.test_obj

        # assert
        self.assertTrue(isinstance(test_obj.config_obj, config.Config))
        self.assertTrue(isinstance(test_obj.connections_obj,
                connections.Connections))
        self.assertTrue(isinstance(test_obj.generator_obj,
                generator.Generator))
        self.assertTrue(isinstance(test_obj.pred_builder_obj,
                pred_builder.PredicateBuilder))
        self.assertTrue(isinstance(test_obj.util_obj, util.Util))
        self.assertTrue(test_obj.relations is None)

    def test_define_file_folders(self):
        # test
        result = self.test_obj.define_file_folders()

        # assert
        exp4 = 'rel/psl/data/soundcloud/interpretability/'
        self.assertTrue(result[0] == 'ind/output/soundcloud/predictions/')
        self.assertTrue(result[1] == 'rel/output/soundcloud/predictions/')
        self.assertTrue(result[2] == 'rel/output/soundcloud/interpretability/')
        self.assertTrue(result[3] == 'rel/psl/')
        self.assertTrue(result[4] == exp4)

    def test_clear_old_data(self):
        os.system = mock.Mock()

        self.test_obj.clear_old_data('data/')

        expected = [mock.call('rm data/*.csv'), mock.call('rm data/*.tsv'),
                mock.call('rm data/db/*.db')]
        self.assertTrue(os.system.call_args_list == expected)

    def test_read_predictions(self):
        df = tu.sample_df(10)
        df.columns = ['com_id', 'rel_pred']
        pd.read_csv = mock.Mock(return_value=df)

        ind_df, rel_df = self.test_obj.read_predictions('ind_f/', 'rel_f/')

        pd.read_csv.assert_called_with('rel_f/predictions_1.csv')
        self.assertTrue(list(rel_df) == ['com_id', 'rel_pred'])
        self.assertTrue(pd.read_csv.call_count == 2)

    def test_merge_predictions(self):
        df = tu.sample_df(10)
        ind_df = tu.sample_df(10)
        rel_df = tu.sample_df(10)
        ind_df.columns = ['com_id', 'ind_pred']
        rel_df.columns = ['com_id', 'rel_pred']

        result = self.test_obj.merge_predictions(df, ind_df, rel_df)

        result_columns = ['com_id', 'random', 'ind_pred', 'rel_pred']
        self.assertTrue(list(result.columns) == result_columns)

    def test_gen_group_ids(self):
        df = tu.sample_df(4)
        filled_df = tu.sample_df(4)
        df.copy = mock.Mock(return_value=filled_df)
        self.test_obj.generator_obj.gen_group_id = mock.Mock()
        self.test_obj.generator_obj.gen_group_id.side_effect = ['r1_df',
                'r2_df']

        result = self.test_obj.gen_group_ids(df)

        self.assertTrue(self.test_obj.generator_obj.gen_group_id.
                call_args_list == [mock.call(df, 'text_id'),
                mock.call('r1_df', 'user_id')])
        self.assertTrue(result == 'r2_df')

    def test_filter_comments(self):
        merged_df = tu.sample_df(10)
        connections = set({0, 1, 2, 3, 4, 5, 6, 7})

        result = self.test_obj.filter_comments(merged_df, connections)

        exp1 = pd.Series([0, 1, 2, 3, 4, 5, 6, 7])
        exp2 = pd.Series([10, 11, 12, 13, 14, 15, 16, 17])
        self.assertTrue(result['com_id'].equals(exp1))
        self.assertTrue(result['random'].equals(exp2))

    def test_write_predicates(self):
        self.test_obj.pred_builder_obj.build_comments = mock.Mock()
        self.test_obj.pred_builder_obj.build_relations = mock.Mock()
        self.test_obj.relations = [('posts', 'user', 'user_id')]

        self.test_obj.write_predicates('con_df', 'rel_data_f/')

        self.test_obj.pred_builder_obj.build_comments.assert_called_with(
                'con_df', 'test', 'rel_data_f/')
        self.test_obj.pred_builder_obj.build_relations.assert_called_with(
                'posts', 'user', 'user_id', 'con_df', 'test', 'rel_data_f/')
        self.assertTrue(self.test_obj.pred_builder_obj.build_relations.
                call_count == 1)

    def test_read_perturbed_labels(self):
        sample_df = tu.sample_df(10)
        os.chdir = mock.Mock()
        pd.read_csv = mock.Mock(return_value=sample_df)

        result = self.test_obj.read_perturbed_labels('rel_data_f/',
                'rel_out_f/')

        args_list = [mock.call('rel_out_f/labels_1.csv'),
                mock.call('rel_data_f/perturbed.csv')]
        os.chdir.assert_called_with('../scripts/')
        self.assertTrue(pd.read_csv.call_args_list == args_list)
        self.assertTrue(result[0].equals(sample_df))
        self.assertTrue(result[1].equals(sample_df))

    def test_preprocess(self):
        perturbed_df = tu.sample_perturbed_df()
        labels_df = tu.sample_df(5)
        labels_df.columns = ['com_id', 'pred']
        sim_dict = {'1': 1, '0': 2, '3': 0.5, '2': 2, '4': 1}

        result = self.test_obj.preprocess(perturbed_df, labels_df, sim_dict)

        x = np.array([[0.75, 0.75, 0.75, 0.75, 0.75],
                [0.85, 0.85, 0.85, 0.85, 0.85],
                [0.65, 0.65, 0.65, 0.65, 0.65],
                [0.95, 0.95, 0.95, 0.95, 0.95],
                [0.55, 0.55, 0.55, 0.55, 0.55]])
        y = np.array([[5], [6], [7], [8], [9]])
        self.assertTrue(len(result[0]) == len(result[1]))
        self.assertTrue(len(result[1]) == len(result[2]))
        self.assertTrue(np.array_equal(result[0], x))
        self.assertTrue(np.array_equal(result[1], y))
        self.assertTrue(result[2] == [0.5, 1.0, 0.5, 2.0, 1.0])
        self.assertTrue(result[3] == [100, 101, 102, 103, 104])

    def test_fit_linear_model(self):
        x = [[1, 2], [3, 4], [5, 6], [7, 8]]
        y = [[1], [0], [1], [0]]
        wgts = [0.75, 0.69, 0.88, 0.22]

        result = self.test_obj.fit_linear_model(x, y, wgts)

        self.assertTrue(isinstance(result, LinearRegression))

    def test_extract_and_sort_coefficients(self):
        coef = np.array([[-7, 2, -3, -5, 8, 6, -1, 4]])
        g = mock.Mock(LinearRegression)
        type(g).coef_ = mock.PropertyMock(return_value=coef)

        result = self.test_obj.extract_and_sort_coefficients(g)

        self.assertTrue(result[0] == [4, 0, 5, 3, 7, 2, 1, 6])
        self.assertTrue(result[1] == [8, -7, 6, -5, 4, -3, 2, -1])

    def test_rearrange_and_filter_features(self):
        features = ['orange', 'apple', 'banana', 'eagle']
        coef_indices = [2, 0, 3, 1]
        coef_values = [69, -22, 11, 2]

        result = self.test_obj.rearrange_and_filter_features(features,
                coef_indices, coef_values, k=2)

        self.assertTrue(len(result) == 2)
        self.assertTrue(result == [('banana', 69), ('orange', -22)])

    def test_read_subnetwork_relations(self):
        sample_df = tu.sample_df(10)
        sample_df.columns = ['intext', 'com_id']
        pd.read_csv = mock.Mock(return_value=sample_df)
        self.test_obj.relations = [('intext', 'text', 'text_id')]

        result = self.test_obj.read_subnetwork_relations('rel_data_f/')

        expected = [mock.call('rel_data_f/test_intext_1.tsv', sep='\t',
                header=None)]
        self.assertTrue(pd.read_csv.call_args_list == expected)
        self.assertTrue(result['intext'].equals(sample_df))


def test_suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(
            InterpretabilityTestCase)
    return suite

if __name__ == '__main__':
    unittest.main()
