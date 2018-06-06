"""
Tests the comments module.
"""
import os
import unittest
import mock
from .context import comments
from .context import config
from .context import util
from .context import test_utils as tu


class CommentsTestCase(unittest.TestCase):
    def setUp(self):
        config_obj = tu.sample_config()
        util_obj = util.Util()
        self.test_obj = comments.Comments(config_obj, util_obj)

    def tearDown(self):
        self.test_obj = None

    def test_init(self):
        # setup
        result = self.test_obj

        # assert
        self.assertTrue(isinstance(result.config_obj, config.Config))
        self.assertTrue(isinstance(result.util_obj, util.Util))

    def test_build_no_data_f(self):
        self.test_obj.define_file_folders = mock.Mock(return_value='d/')
        self.test_obj.drop_duplicate_comments = mock.Mock(
                return_value='unique_df')
        self.test_obj.write_predicates = mock.Mock()

        self.test_obj.build('df', 'dset')

        self.test_obj.define_file_folders.assert_called()
        self.test_obj.drop_duplicate_comments.assert_called_with('df')
        self.test_obj.write_predicates.assert_called_with('unique_df',
                'dset', 'd/')

    def test_build_with_data_f(self):
        self.test_obj.define_file_folders = mock.Mock()
        self.test_obj.drop_duplicate_comments = mock.Mock(
                return_value='unique_df')
        self.test_obj.write_predicates = mock.Mock()

        self.test_obj.build('df', 'dset', data_f='b/')

        self.test_obj.define_file_folders.assert_not_called()
        self.test_obj.drop_duplicate_comments.assert_called_with('df')
        self.test_obj.write_predicates.assert_called_with('unique_df',
                'dset', 'b/')

    def test_build_no_data_tuffy(self):
        self.test_obj.define_file_folders = mock.Mock(return_value='d/')
        self.test_obj.drop_duplicate_comments = mock.Mock(
                return_value='unique_df')
        self.test_obj.write_tuffy_predicates = mock.Mock()

        self.test_obj.build('df', 'dset', tuffy=True)

        self.test_obj.define_file_folders.assert_called()
        self.test_obj.drop_duplicate_comments.assert_called_with('df')
        self.test_obj.write_tuffy_predicates.assert_called_with('unique_df',
                'dset', 'd/')

    def test_define_file_folders(self):
        os.makedirs = mock.Mock()

        result = self.test_obj.define_file_folders()

        self.assertTrue(result == 'rel/data/soundcloud/')

    def test_drop_duplicate_comments(self):
        df = tu.sample_df(10)
        temp_df = tu.sample_df(10)
        unique_df = tu.sample_df(10)
        df.filter = mock.Mock(return_value=temp_df)
        temp_df.drop_duplicates = mock.Mock(return_value=unique_df)

        result = self.test_obj.drop_duplicate_comments(df)

        df.filter.assert_called_with(['com_id', 'ind_pred', 'label'], axis=1)
        temp_df.drop_duplicates.assert_called()
        self.assertTrue(result.equals(unique_df))

    def test_write_predicates(self):
        df = tu.sample_df(10)
        df.to_csv = mock.Mock()

        self.test_obj.write_predicates(df, 'dset', 'd/')

        expected = [mock.call('d/dset_no_label_1.tsv', columns=['com_id'],
                sep='\t', header=None, index=None),
                mock.call('d/dset_1.tsv', columns=['com_id', 'label'],
                sep='\t', header=None, index=None),
                mock.call('d/dset_pred_1.tsv', columns=['com_id', 'ind_pred'],
                sep='\t', header=None, index=None)]
        self.assertTrue(df.to_csv.call_args_list == expected)

    def test_write_tuffy_predicates(self):
        df = tu.sample_df(2)
        df['ind_pred'] = [0.77, 0.27]
        resource_dir = 'relational/scripts/tests/resources/'

        self.test_obj.write_tuffy_predicates(df, 'test', resource_dir)


def test_suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(CommentsTestCase)
    return suite

if __name__ == '__main__':
    unittest.main()
