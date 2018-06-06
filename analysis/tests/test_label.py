"""
Tests the pred_builder module.
"""
import mock
import unittest
import numpy as np
import pandas as pd
from .context import label
from .context import config
from .context import generator
from .context import util
from .context import test_utils as tu


class LabelTestCase(unittest.TestCase):
    def setUp(self):
        config_obj = tu.sample_config()
        mock_generator_obj = mock.Mock(generator.Generator)
        util_obj = util.Util()
        self.test_obj = label.Label(config_obj, mock_generator_obj, util_obj)

    def tearDown(self):
        self.test_obj = None

    def test_init(self):
        # setup
        result = self.test_obj

        # assert
        self.assertTrue(isinstance(result.config_obj, config.Config))
        self.assertTrue(isinstance(result.generator_obj, generator.Generator))

    def test_relabel(self):
        self.test_obj.config_obj.relations = [('intext', 'text', 'text_id'),
                ('invideo', 'video', 'vid_id'),
                ('posts', 'user', 'user_id'),
                ('inment', 'mention', 'ment_id')]
        all_relations = self.test_obj.config_obj.relations
        relations = [('intext', 'text', 'text_id'),
                ('posts', 'user', 'user_id')]
        self.test_obj.util_obj.start = mock.Mock()
        self.test_obj.util_obj.end = mock.Mock()
        self.test_obj.define_file_folders = mock.Mock(return_value='data/')
        self.test_obj.read_comments = mock.Mock(return_value='df2')
        self.test_obj.filter_relations = mock.Mock(return_value=relations)
        self.test_obj.generator_obj.gen_group_ids = mock.Mock(
                return_value='f_df')
        self.test_obj.relabel_relations = mock.Mock(return_value=('labels_df',
                'new_df'))
        self.test_obj.write_new_dataframe = mock.Mock()

        self.test_obj.relabel('df')

        start = [mock.call(), mock.call('generating group ids...\n')]
        end = [mock.call('\ttime: '), mock.call('total time: ')]
        self.assertTrue(self.test_obj.util_obj.start.call_args_list == start)
        self.assertTrue(self.test_obj.util_obj.end.call_args_list == end)
        self.test_obj.define_file_folders.assert_called()
        self.test_obj.read_comments.assert_called_with('df', 'data/')
        self.test_obj.filter_relations.assert_called_with(all_relations)
        self.test_obj.generator_obj.gen_group_ids.assert_called_with('df2',
                relations)
        self.test_obj.relabel_relations.assert_called_with('f_df', relations)
        self.test_obj.write_new_dataframe.assert_called_with('new_df',
                'labels_df', 'data/')

    def test_define_file_folders(self):
        result = self.test_obj.define_file_folders()

        self.assertTrue(result == 'ind/data/soundcloud/')

    def test_read_comments(self):
        df = tu.sample_df(10)
        pd.read_csv = mock.Mock(return_value=df)
        self.test_obj.config_obj.start = 2
        self.test_obj.config_obj.end = 1000
        self.test_obj.util_obj.start = mock.Mock()
        self.test_obj.util_obj.end = mock.Mock()

        result = self.test_obj.read_comments(None, 'data/')

        self.test_obj.util_obj.start.assert_called_with('reading comments...')
        pd.read_csv.assert_called_with('data/comments.csv', nrows=1000)
        self.test_obj.util_obj.end.assert_called()
        self.assertTrue(len(result) == 8)

    def test_filter_relations(self):
        relations = [('intext', 'text', 'text_id'),
                ('invideo', 'video', 'vid_id'),
                ('posts', 'user', 'user_id'),
                ('inment', 'mention', 'ment_id')]

        result = self.test_obj.filter_relations(relations)

        exp = [('posts', 'user', 'user_id')]
        self.assertTrue(len(result) == 1)
        self.assertTrue(result == exp)

    def test_convert_dtypes(self):
        df = tu.sample_df(2)
        df['label'] = [1.0, 0.0]

        result = self.test_obj.convert_dtypes(df)

        self.assertTrue(list(result.dtypes) == [int, int, int])

    def test_relabel_relations(self):
        g_df = tu.sample_group_df()
        g_df.columns = ['user_id']
        g_df['com_id'] = [0, 1, 2, 3, 4, 5, 6]
        g_df['text_id'] = [1, 1, 1, 1, 2, 2, 7]
        g_df['label'] = [0, 1, np.nan, 0, 1, np.nan, 0]
        relations = [('intext', 'text', 'text_id'),
                ('posts', 'user', 'user_id')]
        self.test_obj.util_obj.start = mock.Mock()
        self.test_obj.util_obj.end = mock.Mock()

        result = self.test_obj.relabel_relations(g_df, relations)

        exp = 'checking if any comments need relabeling...'
        self.test_obj.util_obj.start.assert_called_with(exp)
        self.test_obj.util_obj.end.assert_called()
        self.assertTrue(list(result[0]['com_id']) == [0, 3])
        self.assertTrue(list(result[1]['com_id']) == [0, 1, 3, 4, 6])

    def test_write_new_dataframe(self):
        new_df = tu.sample_df(10)
        labels_df = tu.sample_df(10)
        new_df.to_csv = mock.Mock()
        labels_df.to_csv = mock.Mock()
        self.test_obj.util_obj.start = mock.Mock()
        self.test_obj.util_obj.end = mock.Mock()

        self.test_obj.write_new_dataframe(new_df, labels_df, 'data/')

        exp = 'writing relabeled comments...'
        self.test_obj.util_obj.start.assert_called_with(exp)
        new_df.to_csv.assert_called_with('data/modified.csv',
            encoding='utf-8', line_terminator='\n', index=None)
        labels_df.to_csv.assert_called_with('data/labels.csv', index=None)
        self.test_obj.util_obj.end.assert_called()


def test_suite():
    s = unittest.TestLoader().loadTestsFromTestCase(LabelTestCase)
    return s

if __name__ == '__main__':
    unittest.main()
