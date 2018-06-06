"""
Tests the purity module.
"""
import unittest
import numpy as np
import pandas as pd
import mock
from .context import purity
from .context import config
from .context import generator
from .context import util
from .context import test_utils as tu


class PurityTestCase(unittest.TestCase):
    def setUp(self):
        config_obj = tu.sample_config()
        generator_obj = generator.Generator()
        util_obj = util.Util()
        self.test_obj = purity.Purity(config_obj, generator_obj, util_obj)

    def tearDown(self):
        self.test_obj = None

    def test_init(self):
        # setup
        result = self.test_obj

        # assert
        self.assertTrue(isinstance(result.config_obj, config.Config))
        self.assertTrue(isinstance(result.generator_obj,
                generator.Generator))

    def test_relations(self):
        self.test_obj.file_folders = mock.Mock(return_value=('d/', 's/'))
        self.test_obj.open_status_writer = mock.Mock(return_value='sw')
        self.test_obj.util_obj.write = mock.Mock()
        self.test_obj.read_comments = mock.Mock(return_value='df2')
        self.test_obj.gen_group_ids = mock.Mock(return_value='f_df')
        self.test_obj.check_relations = mock.Mock()
        self.test_obj.util_obj.close_writer = mock.Mock()

        self.test_obj.test_relations('df')

        relations = self.test_obj.config_obj.relations
        s = 'Condition #1: How well does each relation separate spam/ham...'
        s += '\nScale is from 0.0 to 0.5, good to bad:'
        self.test_obj.file_folders.assert_called()
        self.test_obj.open_status_writer.assert_called_with('s/')
        self.test_obj.util_obj.write.assert_called_with(s, fw='sw')
        self.test_obj.read_comments.assert_called_with('df', 'd/')
        self.test_obj.gen_group_ids.assert_called_with('df2')
        self.test_obj.check_relations.assert_called_with('f_df', relations,
                fw='sw')
        self.test_obj.util_obj.close_writer.assert_called_with('sw')

    def test_file_folders(self):
        result = self.test_obj.file_folders()

        self.assertTrue(result[0] == 'ind/data/soundcloud/')
        self.assertTrue(result[1] == 'rel/output/soundcloud/status/')

    def test_open_status_writer(self):
        self.test_obj.util_obj.open_writer = mock.Mock(return_value='f')

        result = self.test_obj.open_status_writer('s/')

        self.assertTrue(result == 'f')
        self.test_obj.util_obj.open_writer.assert_called_with('s/purity_1.txt')

    def test_read_comments(self):
        result = self.test_obj.read_comments('df', 'data/')

        self.assertTrue(result == 'df')

    def test_read_comments_none(self):
        self.test_obj.config_obj.modified = True
        self.test_obj.config_obj.end = 7
        self.test_obj.config_obj.start = 2
        self.test_obj.util_obj.get_comments_filename = mock.Mock(
                return_value='comments_file')
        self.test_obj.util_obj.check_file = mock.Mock(return_value=True)
        df = tu.sample_df(4)
        pd.read_csv = mock.Mock(return_value=df)

        result = self.test_obj.read_comments(None, 'data/')

        self.test_obj.util_obj.get_comments_filename.assert_called_with(True)
        self.test_obj.util_obj.check_file.assert_called_with('data/comments_file')
        pd.read_csv.assert_called_with('data/comments_file', nrows=7)
        self.assertTrue(len(result) == 2)
        self.assertTrue(result['com_id'].equals(pd.Series([2, 3])))

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

    def test_check_purity(self):
        pass

    def test_check_relations(self):
        g_df = tu.sample_group_df()
        g_df.columns = ['text_id']
        relations = [self.test_obj.config_obj.relations[0]]
        self.test_obj.check_groups = mock.Mock(return_value=69)

        self.test_obj.check_relations(g_df, relations)

        self.test_obj.check_groups.assert_called_with(g_df, 'text_id',
                [1, 2, 3])

    def test_check_groups(self):
        g_df = tu.sample_group_df()
        g_df.columns = ['text_id']
        self.test_obj.check_group = mock.Mock()
        self.test_obj.check_group.side_effect = [77, 88]

        result = self.test_obj.check_groups(g_df, 'text_id', [1, 2])

        self.assertTrue(self.test_obj.check_group.call_count == 2)
        self.assertTrue(result == 165)

    def test_check_group(self):
        g_df = tu.sample_df(10)
        g_df['label'] = [1, 1, 1, 1, 0, 1, 0, 1, 0, 1]

        result = self.test_obj.check_group(g_df)

        self.assertTrue(np.array_equal(np.isclose([result], [3.0]), [True]))


def test_suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(PurityTestCase)
    return suite

if __name__ == '__main__':
    unittest.main()
