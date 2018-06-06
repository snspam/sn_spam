"""
Tests the pred_builder module.
"""
import os
import unittest
import mock
from .context import pred_builder
from .context import config
from .context import comments
from .context import generator
from .context import util
from .context import test_utils as tu


class PredicateBuilderTestCase(unittest.TestCase):
    def setUp(self):
        config_obj = tu.sample_config()
        mock_comments_obj = mock.Mock(comments.Comments)
        mock_generator_obj = mock.Mock(generator.Generator)
        mock_util_obj = mock.Mock(util.Util)
        self.test_obj = pred_builder.PredicateBuilder(config_obj,
                mock_comments_obj, mock_generator_obj, mock_util_obj)

    def tearDown(self):
        self.test_obj = None

    def test_init(self):
        # setup
        result = self.test_obj

        # assert
        self.assertTrue(isinstance(result.config_obj, config.Config))
        self.assertTrue(isinstance(result.comments_obj, comments.Comments))
        self.assertTrue(isinstance(result.util_obj, util.Util))

    def test_build_comments(self):
        self.test_obj.comments_obj.build = mock.Mock()

        self.test_obj.build_comments('df', 'dset', 'd/', tuffy=True)

        self.test_obj.comments_obj.build.assert_called_with('df', 'dset',
            'd/', tuffy=True)

    def test_build_relations_psl(self):
        self.test_obj.generator_obj.gen_group_id = mock.Mock(return_value='df')
        self.test_obj.group = mock.Mock(return_value=('g_df', 'r_df'))
        self.test_obj.util_obj.print_stats = mock.Mock()
        self.test_obj.write_files = mock.Mock()

        self.test_obj.build_relations('rel', 'group', 'group_id', 'df',
                'test', 'data/', fw='fw')

        self.test_obj.generator_obj.gen_group_id.assert_called_with('df',
                'group_id')
        self.test_obj.group.assert_called_with('df', 'group_id')
        self.test_obj.util_obj.print_stats.assert_called_with('df',
                'r_df', 'rel', 'test', fw='fw')
        self.test_obj.write_files.assert_called_with('test', 'r_df', 'g_df',
                'rel', 'group', 'group_id', 'data/')

    def test_build_relations_tuffy(self):
        self.test_obj.generator_obj.gen_group_id = mock.Mock(return_value='df')
        self.test_obj.group = mock.Mock(return_value=('g_df', 'r_df'))
        self.test_obj.util_obj.print_stats = mock.Mock()
        self.test_obj.write_tuffy_predicates = mock.Mock()

        self.test_obj.build_relations('rel', 'group', 'group_id', 'df',
                'test', 'data/', tuffy=True, fw='fw')

        self.test_obj.generator_obj.gen_group_id.assert_called_with('df',
                'group_id')
        self.test_obj.group.assert_called_with('df', 'group_id')
        self.test_obj.util_obj.print_stats.assert_called_with('df',
                'r_df', 'rel', 'test', fw='fw')
        self.test_obj.write_tuffy_predicates.assert_called_with('test', 'r_df',
                'rel', 'group_id', 'data/')

    def test_group(self):
        df = tu.sample_group_user_df()
        df['ind_pred'] = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]

        result = self.test_obj.group(df, 'user_id')

        self.assertTrue(len(result[0]) == 5)
        self.assertTrue(result[0]['size'].sum() == 10)
        self.assertTrue(list(result[0]) == ['user_id', 'size'])
        self.assertTrue(len(result[1]) == 10)
        self.assertTrue(list(result[1]) == list(df))

    def test_write_tuffy_predicates(self):
        r_df = tu.sample_df(2)
        r_df['g_id'] = ['77', '88']
        r_dir = 'relational/scripts/tests/resources/'
        dset = 'dset'
        os.system('rm ' + r_dir + dset + '_evidence.txt')

        self.test_obj.write_tuffy_predicates('dset', r_df, 'rel', 'g_id',
                r_dir)

    def test_write_files(self):
        r_df = tu.sample_df(10)
        g_df = tu.sample_group_df()
        r_df.to_csv = mock.Mock()
        g_df.to_csv = mock.Mock()

        self.test_obj.write_files('dset', r_df, g_df, 'rel', 'group',
                'group_id', 'dff/')

        r_df.to_csv.assert_called_with('dff/dset_rel_1.tsv', index=None,
                columns=['com_id', 'group_id'], header=None, sep='\t')
        g_df.to_csv.assert_called_with('dff/dset_group_1.tsv', index=None,
                columns=['group_id'], header=None, sep='\t')


def test_suite():
    s = unittest.TestLoader().loadTestsFromTestCase(PredicateBuilderTestCase)
    return s

if __name__ == '__main__':
    unittest.main()
