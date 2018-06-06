"""
Tests the graph_features module.
"""
import mock
import unittest
from .context import graph_features
from .context import config
from .context import util
from .context import test_utils as tu


class GraphFeaturesTestCase(unittest.TestCase):
    def setUp(self):
        config_obj = tu.sample_config()
        util_obj = util.Util()
        self.test_obj = graph_features.GraphFeatures(config_obj, util_obj)

    def tearDown(self):
        self.test_obj = None

    def test_init(self):
        result = self.test_obj

        self.assertTrue(isinstance(result.config_obj, config.Config))
        self.assertTrue(isinstance(result.util_obj, util.Util))

    @mock.patch('pandas.concat')
    def test_build(self, mock_concat):
        self.test_obj.util_obj.start = mock.Mock()
        self.test_obj.build_features = mock.Mock()
        self.test_obj.build_features.side_effect = [('tr_f_df', ''),
                ('te_f_df', 'feats_list')]
        mock_concat.return_value = 'feats_df'
        self.test_obj.util_obj.end = mock.Mock()

        result = self.test_obj.build('train_df', 'test_df', fw='fw')

        self.assertTrue(result == ('feats_df', 'feats_list'))
        exp_start = 'loading graph features...'
        exp_bf = [mock.call('train_df'), mock.call('test_df')]
        self.test_obj.util_obj.start.assert_called_with(exp_start, fw='fw')
        self.assertTrue(self.test_obj.build_features.call_args_list == exp_bf)
        mock_concat.assert_called_with(['tr_f_df', 'te_f_df'])
        self.test_obj.util_obj.end.assert_called_with(fw='fw')

    def test_build_features_youtube(self):
        self.test_obj.config_obj.domain = 'youtube'
        self.test_obj.youtube = mock.Mock(return_value=('f_df', 'f_list'))
        self.test_obj.twitter = mock.Mock()

        result = self.test_obj.build_features('cf')

        self.assertTrue(result == ('f_df', 'f_list'))
        self.test_obj.youtube.assert_called_with('cf')
        self.test_obj.twitter.assert_not_called()

    @mock.patch('pandas.DataFrame')
    def test_soundcloud(self, mock_df):
        cf = tu.sample_df(10)
        mock_df.return_value = 'feats_df'

        result = self.test_obj.soundcloud(cf)

        exp_ga = ['pagerank', 'triangle_count', 'core_id', 'out_degree',
                'in_degree']
        self.assertTrue(result[0] == 'feats_df')
        self.assertTrue(result[1] == exp_ga)
        mock_df.assert_called_with(cf['com_id'])

    @mock.patch('pandas.DataFrame')
    def test_youtube(self, mock_df):
        cf = tu.sample_df(10)
        mock_df.return_value = 'feats_df'

        result = self.test_obj.youtube(cf)

        self.assertTrue(result[0] == 'feats_df')
        self.assertTrue(result[1] == [])
        mock_df.assert_called_with(cf['com_id'])

    @mock.patch('pandas.DataFrame')
    def test_twitter(self, mock_df):
        cf = tu.sample_df(10)
        mock_df.return_value = 'feats_df'

        result = self.test_obj.twitter(cf)

        exp_ga = ['pagerank', 'triangle_count', 'core_id', 'out_degree',
                'in_degree']
        self.assertTrue(result[0] == 'feats_df')
        self.assertTrue(result[1] == exp_ga)
        mock_df.assert_called_with(cf['com_id'])

    @mock.patch('pandas.DataFrame')
    def test_ifwe(self, mock_df):
        cf = tu.sample_df(10)
        mock_df.return_value = 'feats_df'

        result = self.test_obj.ifwe(cf)

        exp_ga = ['1_pagerank', '1_triangle_count', '1_core_id', '1_color_id',
                '1_component_id', '1_component_size', '1_out_degree',
                '1_in_degree', '2_pagerank', '2_triangle_count', '2_core_id',
                '2_color_id', '2_component_id', '2_component_size',
                '2_out_degree', '2_in_degree', '3_pagerank',
                '3_triangle_count', '3_core_id', '3_color_id',
                '3_component_id', '3_component_size', '3_out_degree',
                '3_in_degree', '4_pagerank', '4_triangle_count', '4_core_id',
                '4_color_id', '4_component_id', '4_component_size',
                '4_out_degree', '4_in_degree', '5_pagerank',
                '5_triangle_count', '5_core_id', '5_color_id',
                '5_component_id', '5_component_size', '5_out_degree',
                '5_in_degree', '6_pagerank', '6_triangle_count', '6_core_id',
                '6_color_id', '6_component_id', '6_component_size',
                '6_out_degree', '6_in_degree', '7_pagerank',
                '7_triangle_count', '7_core_id', '7_color_id',
                '7_component_id', '7_component_size', '7_out_degree',
                '7_in_degree']
        self.assertTrue(result[0] == 'feats_df')
        self.assertTrue(result[1] == exp_ga)
        mock_df.assert_called_with(cf['com_id'])

    @mock.patch('pandas.DataFrame')
    def test_yelp_hotel(self, mock_df):
        cf = tu.sample_df(10)
        mock_df.return_value = 'feats_df'

        result = self.test_obj.yelp_hotel(cf)

        self.assertTrue(result[0] == 'feats_df')
        self.assertTrue(result[1] == [])
        mock_df.assert_called_with(cf['com_id'])

    @mock.patch('pandas.DataFrame')
    def test_yelp_restaurant(self, mock_df):
        cf = tu.sample_df(10)
        mock_df.return_value = 'feats_df'

        result = self.test_obj.yelp_restaurant(cf)

        self.assertTrue(result[0] == 'feats_df')
        self.assertTrue(result[1] == [])
        mock_df.assert_called_with(cf['com_id'])


def test_suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(
            GraphFeaturesTestCase)
    return suite

if __name__ == '__main__':
    unittest.main()
