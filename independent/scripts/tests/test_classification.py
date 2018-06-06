"""
Tests the classification module.
"""
import os
import unittest
import numpy as np
import scipy
import mock
from scipy.sparse import csr_matrix
from .context import classification
from .context import config
from .context import content_features
from .context import graph_features
from .context import relational_features
from .context import util
from .context import test_utils as tu


class ClassificationTestCase(unittest.TestCase):
    def setUp(self):
        config_obj = tu.sample_config()
        mock_content_features_obj = mock.Mock(content_features.ContentFeatures)
        mock_graph_features_obj = mock.Mock(graph_features.GraphFeatures)
        mock_rel_feats_obj = mock.Mock(relational_features.RelationalFeatures)
        mock_util_obj = mock.Mock(util.Util)
        self.test_obj = classification.Classification(config_obj,
                mock_content_features_obj, mock_graph_features_obj,
                mock_rel_feats_obj, mock_util_obj)

    def tearDown(self):
        self.test_obj = None

    def test_init(self):
        # setup
        test_obj = self.test_obj

        # assert
        self.assertTrue(isinstance(test_obj.config_obj, config.Config))
        self.assertTrue(isinstance(test_obj.cf_obj,
                                   content_features.ContentFeatures))
        self.assertTrue(isinstance(test_obj.gf_obj,
                                   graph_features.GraphFeatures))
        self.assertTrue(isinstance(test_obj.rf_obj,
                                   relational_features.RelationalFeatures))
        self.assertTrue(isinstance(test_obj.util_obj, util.Util))

    def test_file_folders(self):
        os.path.exists = mock.Mock(return_value=False)
        os.makedirs = mock.Mock()

        # test
        result = self.test_obj.file_folders()

        # assert
        self.assertTrue(result[0] == 'ind/output/soundcloud/images/')
        self.assertTrue(result[1] == 'ind/output/soundcloud/predictions/')
        self.assertTrue(result[2] == 'ind/output/soundcloud/models/')
        self.assertTrue(os.path.exists.call_count == 3)
        self.assertTrue(os.makedirs.call_count == 3)

    def test_merge(self):
        coms_df = tu.sample_df_with_com_id(10)
        c_df = tu.sample_df_with_com_id(10)
        r_df = tu.sample_df_with_com_id(10)
        g_df = tu.sample_df_with_com_id(10)

        result = self.test_obj.merge(coms_df, c_df, g_df, r_df)

        self.assertTrue(len(list(result)) == 5)
        self.assertTrue(len(result) == 10)

    def test_drop_columns(self):
        coms_df = tu.sample_df_with_com_id_and_user_id(10)
        c_df = tu.sample_df_with_com_id(10)
        r_df = tu.sample_df_with_com_id(10)
        g_df = tu.sample_df_with_com_id(5)
        feats_df = self.test_obj.merge(coms_df, c_df, g_df, r_df)
        features = ['random_x', 'random_y']

        result = self.test_obj.drop_columns(feats_df, features)

        self.assertTrue(list(result) == ['random_x', 'random_y'])

    def test_dataframe_to_matrix(self):
        coms_df = tu.sample_df_with_com_id_and_user_id(10)
        c_df = tu.sample_df_with_com_id(10)
        r_df = tu.sample_df_with_com_id(10)
        g_df = tu.sample_df_with_com_id(5)
        feats_df = self.test_obj.merge(coms_df, c_df, g_df, r_df)

        result = self.test_obj.dataframe_to_matrix(feats_df)

        self.assertTrue(type(result) == scipy.sparse.csr.csr_matrix)
        self.assertTrue(result.shape == (10, 5))

    def test_stack_matrices(self):
        feats_m = tu.sample_df_with_com_id_and_user_id(10).as_matrix()
        c_csr = csr_matrix(tu.sample_df_with_com_id(10).as_matrix())

        result = self.test_obj.stack_matrices(feats_m, c_csr)

        self.assertTrue(type(result) == scipy.sparse.csr.csr_matrix)
        self.assertTrue(result.shape == (10, 4))

    def test_extract_labels(self):
        coms_df = tu.sample_df_with_com_id_and_user_id(10)
        coms_df.columns = ['com_id', 'label']

        result = self.test_obj.extract_labels(coms_df)

        self.assertTrue(type(result[0]) == np.ndarray)
        self.assertTrue(result[0].shape == (10,))
        self.assertTrue(len(result[0]) == 10)

    def test_prepare(self):
        self.test_obj.merge = mock.Mock(return_value='f_df')
        self.test_obj.drop_columns = mock.Mock(return_value='f_df2')
        self.test_obj.dataframe_to_matrix = mock.Mock(return_value='f_m')
        self.test_obj.stack_matrices = mock.Mock(return_value='x')
        self.test_obj.extract_labels = mock.Mock(return_value=('y', 'ids'))

        result = self.test_obj.prepare('df', 'c_m', 'c_df', 'g_df', 'r_df',
                'feats_list')

        self.test_obj.merge.assert_called_with('df', 'c_df', 'g_df', 'r_df')
        self.test_obj.drop_columns.assert_called_with('f_df', 'feats_list')
        self.test_obj.dataframe_to_matrix.assert_called_with('f_df2')
        self.test_obj.stack_matrices.assert_called_with('f_m', 'c_m')
        self.test_obj.extract_labels.assert_called_with('df')
        self.assertTrue(result == ('x', 'y', 'ids'))

    def test_build_and_merge(self):
        self.test_obj.cf_obj.build = mock.Mock(return_value=('m_tr', 'm_te',
                'c_df', 'cf'))
        self.test_obj.gf_obj.build = mock.Mock(return_value=('g_df', 'gf'))
        self.test_obj.rf_obj.build = mock.Mock(return_value=('r_df', 'rf'))
        self.test_obj.util_obj.start = mock.Mock()
        self.test_obj.prepare = mock.Mock()
        self.test_obj.prepare.side_effect = [('x_tr', 'y_tr', ''),
                ('x_te', 'y_te', 'id_te')]
        self.test_obj.util_obj.end = mock.Mock()

        result = self.test_obj.build_and_merge('train', 'test', 'val', fw='fw')

        exp_res = ('x_tr', 'y_tr', 'x_te', 'y_te', 'id_te', 'cfgfrf')
        exp_pre = [mock.call('train', 'm_tr', 'c_df', 'g_df', 'r_df',
                'cfgfrf'), mock.call('test', 'm_te', 'c_df',
                'g_df', 'r_df', 'cfgfrf')]
        self.assertTrue(result == exp_res)
        self.test_obj.cf_obj.build.assert_called_with('train', 'test', 'val',
                fw='fw')
        self.test_obj.gf_obj.build.assert_called_with('train', 'test', fw='fw')
        self.test_obj.rf_obj.build.assert_called_with('train', 'test', 'val',
                fw='fw')
        self.test_obj.util_obj.start.assert_called_with('merging features...',
                fw='fw')
        self.assertTrue(self.test_obj.prepare.call_args_list == exp_pre)
        self.test_obj.util_obj.end.assert_called_with(fw='fw')

    def test_append_preds(self):
        df = tu.sample_df(2)
        probs = np.array([[0.2, 0.8], [0.6, 0.4]])
        id_te = [0, 1]

        result = self.test_obj.append_preds(df, probs, id_te)

        self.assertTrue(list(result['noisy_labels']) == [0.8, 0.4])

    def split_training_data_half(self):
        df = tu.sample_df(27)

        result = self.test_obj.split_training_data(df)

        self.assertTrue(len(result[0]) == 13)
        self.assertTrue(len(result[1]) == 14)

    def split_training_data_uneven(self):
        df = tu.sample_df(100)

        result = self.test_obj.split_training_data(df, split=0.25)

        self.assertTrue(len(result[0]) == 25)
        self.assertTrue(len(result[1]) == 75)

    def test_main_do_stacking(self):
        self.test_obj.config_obj.pseudo = True
        self.test_obj.do_stacking = mock.Mock()
        self.test_obj.do_normal = mock.Mock()

        self.test_obj.main('tr', 'te', dset='val', fw='fw')

        self.test_obj.do_stacking.assert_called_with('tr', 'te', 'val', 'fw')
        self.test_obj.do_normal.assert_not_called()

    def test_main_do_normal(self):
        self.test_obj.config_obj.pseudo = False
        self.test_obj.do_stacking = mock.Mock()
        self.test_obj.do_normal = mock.Mock()

        self.test_obj.main('tr', 'te', dset='val', fw='fw')

        self.test_obj.do_normal.assert_called_with('tr', 'te', 'val', 'fw')
        self.test_obj.do_stacking.assert_not_called()

    def test_do_normal(self):
        self.test_obj.file_folders = mock.Mock(return_value=('i/', 'p/', 'm/'))
        self.test_obj.build_and_merge = mock.Mock(return_value='d')
        self.test_obj.util_obj.train = mock.Mock(return_value='base_learner')
        self.test_obj.util_obj.test = mock.Mock(return_value=('p1', 'i1'))
        self.test_obj.util_obj.evaluate = mock.Mock()
        self.test_obj.util_obj.save_preds = mock.Mock()

        self.test_obj.do_normal('tr', 'te', dset='val', fw='fw')

        exp_bm = [mock.call('tr', 'te', 'val', fw='fw')]
        exp_tr = [mock.call('d', classifier='lr', fw='fw')]
        exp_test = [mock.call('d', 'base_learner', fw='fw')]
        self.test_obj.file_folders.assert_called_with()
        self.assertTrue(self.test_obj.build_and_merge.call_args_list == exp_bm)
        self.assertTrue(self.test_obj.util_obj.train.call_args_list == exp_tr)
        self.assertTrue(self.test_obj.util_obj.test.call_args_list == exp_test)
        self.test_obj.util_obj.evaluate.assert_called_with('d', 'p1', fw='fw')
        self.test_obj.util_obj.save_preds.assert_called_with('p1', 'i1', '1',
                'p/', 'val')

    def test_do_stacking(self):
        trainings = ('tr1', 'tr2')
        tests = [('p1', 'i1'), ('p2', 'i2'), ('p3', 'i3')]
        models = ['base_learner', 'real_learner', 'base_learner2']
        self.test_obj.file_folders = mock.Mock(return_value=('i/', 'p/', 'm/'))
        self.test_obj.split_training_data = mock.Mock(return_value=trainings)
        self.test_obj.build_and_merge = mock.Mock()
        self.test_obj.build_and_merge.side_effect = ['d1', 'd2', 'd3', 'd4']
        self.test_obj.util_obj.train = mock.Mock()
        self.test_obj.util_obj.train.side_effect = models
        self.test_obj.util_obj.test = mock.Mock()
        self.test_obj.util_obj.test.side_effect = tests
        self.test_obj.append_preds = mock.Mock()
        self.test_obj.append_preds.side_effect = ['new_tr2_df', 'new_test_df']
        self.test_obj.util_obj.evaluate = mock.Mock()
        self.test_obj.util_obj.save_preds = mock.Mock()

        self.test_obj.do_stacking('tr', 'te', dset='val', fw='fw')

        exp_bm = [mock.call('tr1', 'tr2', 'train1', fw='fw'),
                mock.call('new_tr2_df', 'te', 'train2', fw='fw'),
                mock.call('tr', 'te', 'val', fw='fw'),
                mock.call('new_tr2_df', 'new_test_df', 'val', fw='fw')]
        exp_tr = [mock.call('d1', classifier='lr', fw='fw'),
                mock.call('d2', classifier='lr', fw='fw'),
                mock.call('d3', classifier='lr', fw='fw')]
        exp_test = [mock.call('d1', 'base_learner', fw='fw'),
                mock.call('d3', 'base_learner2', fw='fw'),
                mock.call('d4', 'real_learner', fw='fw')]
        exp_ap = [mock.call('tr2', 'p1', 'i1'),
                mock.call('te', 'p2', 'i2')]
        self.test_obj.file_folders.assert_called_with()
        self.test_obj.split_training_data.assert_called_with('tr')
        self.assertTrue(self.test_obj.build_and_merge.call_args_list == exp_bm)
        self.assertTrue(self.test_obj.util_obj.train.call_args_list == exp_tr)
        self.assertTrue(self.test_obj.util_obj.test.call_args_list == exp_test)
        self.assertTrue(self.test_obj.append_preds.call_args_list == exp_ap)
        self.test_obj.util_obj.evaluate.assert_called_with('d4', 'p3', fw='fw')
        self.test_obj.util_obj.save_preds.assert_called_with('p3', 'i3', '1',
                'p/', 'val')


def test_suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(ClassificationTestCase)
    return suite

if __name__ == '__main__':
    unittest.main()
