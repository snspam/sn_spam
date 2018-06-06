"""
Tests the independent module.
"""
import os
import mock
import unittest
import pandas as pd
from .context import independent
from .context import config
from .context import classification
from .context import util
from .context import test_utils as tu


class IndependentTestCase(unittest.TestCase):
    def setUp(self):
        config_obj = tu.sample_config()
        mock_classification_obj = mock.Mock(classification.Classification)
        mock_util_obj = mock.Mock(util.Util)
        self.test_obj = independent.Independent(config_obj,
                mock_classification_obj, mock_util_obj)

    def tearDown(self):
        self.test_obj = None

    def test_init(self):
        # setup
        test_obj = self.test_obj

        # assert
        self.assertTrue(isinstance(test_obj.config_obj, config.Config))
        self.assertTrue(isinstance(test_obj.classification_obj,
                classification.Classification))
        self.assertTrue(isinstance(test_obj.util_obj, util.Util))

    def test_file_folders(self):
        os.path.exists = mock.Mock(return_value=False)
        os.makedirs = mock.Mock()

        # test
        result = self.test_obj.file_folders()

        # assert
        self.assertTrue(result[0] == 'ind/data/soundcloud/')
        self.assertTrue(result[1] == 'ind/data/soundcloud/folds/')
        self.assertTrue(os.path.exists.called)
        self.assertTrue(os.makedirs.called)

    def test_open_status_writer(self):
        self.test_obj.util_obj.open_writer = mock.Mock(return_value='f')

        result = self.test_obj.open_status_writer('status/')

        exp_fname = 'status/ind_1.txt'
        self.test_obj.util_obj.open_writer.assert_called_with(exp_fname)
        self.assertTrue(result == 'f')

    def test_read_file(self):
        df = tu.simple_df()
        pd.read_csv = mock.Mock(return_value=df)
        self.test_obj.config_obj.end = 7

        result = self.test_obj.read_file('boogers.txt')

        pd.read_csv.assert_called_with('boogers.txt', lineterminator='\n',
                nrows=7)
        self.assertTrue(result.equals(tu.simple_df()))

    def test_split_coms(self):
        df = tu.simple_df()
        self.test_obj.config_obj.start = 2
        self.test_obj.config_obj.train_size = 0.5
        self.test_obj.config_obj.val_size = 0.2

        result = self.test_obj.split_coms(df)
        self.assertTrue(len(result[0]) == 4)
        self.assertTrue(len(result[1]) == 1)
        self.assertTrue(len(result[2]) == 3)

    def test_alter_user_ids(self):
        coms_df = tu.sample_df_with_user_id(5)
        test_df = tu.sample_df_with_user_id(3)
        test_df['label'] = [1, 1, 0]

        self.test_obj.alter_user_ids(coms_df, test_df)

        self.assertTrue(list(test_df['user_id']) == [5, 6, 2])

    def test_write_folds(self):
        val_df = tu.simple_df()
        test_df = tu.simple_df()
        val_df.to_csv = mock.Mock()
        test_df.to_csv = mock.Mock()

        self.test_obj.write_folds(val_df, test_df, 'data/')

        val_df.to_csv.assert_called_with('data/val_1.csv', index=None,
                line_terminator='\n')
        test_df.to_csv.assert_called_with('data/test_1.csv', index=None,
                line_terminator='\n')

    def test_print_subsets(self):
        train_df = tu.simple_df()
        val_df = tu.simple_df()
        test_df = tu.simple_df()
        train_df.columns = ['label']
        val_df.columns = ['label']
        test_df.columns = ['label']
        self.test_obj.util_obj.div0 = mock.Mock(return_value=0.69)

        self.test_obj.print_subsets(train_df, val_df, test_df)

        self.test_obj.util_obj.div0.assert_called_with(1, 10)
        self.assertTrue(self.test_obj.util_obj.div0.call_count == 3)

    @mock.patch('pandas.concat')
    def test_main(self, mock_concat):
        tr = mock.Mock()
        tr.copy = mock.Mock(return_value='super_train')
        self.test_obj.config_obj.super_train = True
        self.test_obj.config_obj.alter_user_ids = True
        self.test_obj.util_obj.start = mock.Mock()
        self.test_obj.file_folders = mock.Mock(return_value=(
                'a/', 'b/', 'c/'))
        self.test_obj.open_status_writer = mock.Mock(return_value='sw')
        self.test_obj.util_obj.get_comments_filename = mock.Mock(
                return_value='fname')
        self.test_obj.read_file = mock.Mock(return_value='df')
        self.test_obj.split_coms = mock.Mock(return_value=(tr, 'va', 'te'))
        self.test_obj.alter_user_ids = mock.Mock()
        self.test_obj.write_folds = mock.Mock()
        self.test_obj.print_subsets = mock.Mock()
        self.test_obj.classification_obj.main = mock.Mock()
        mock_concat.return_value = 'super_tr'
        self.test_obj.util_obj.end = mock.Mock()
        self.test_obj.util_obj.close_writer = mock.Mock()

        result = self.test_obj.main()

        s_args = [mock.call(), mock.call('\nvalidation set:\n', fw='sw'),
                mock.call('\ntest set:\n', fw='sw')]
        main_args = [mock.call(tr, 'va', dset='val', fw='sw'),
                mock.call('super_tr', 'te', dset='test', fw='sw')]
        end_args = [mock.call('time: ', fw='sw'), mock.call('time: ', fw='sw'),
                mock.call('total independent model time: ', fw='sw')]
        self.test_obj.util_obj.start.assert_called()
        self.test_obj.file_folders.assert_called()
        self.test_obj.util_obj.get_comments_filename.assert_called_with(False)
        self.test_obj.read_file.assert_called_with('a/fname', 'sw')
        self.test_obj.split_coms.assert_called_with('df')
        self.test_obj.alter_user_ids.assert_called_with('df', 'te')
        self.test_obj.write_folds.assert_called_with('va', 'te', 'b/')
        self.test_obj.print_subsets.assert_called_with(tr, 'va', 'te',
                fw='sw')
        mock_concat.assert_called_with([tr, 'va'])
        self.assertTrue(self.test_obj.classification_obj.main.call_args_list ==
                main_args)
        self.assertTrue(self.test_obj.util_obj.start.call_args_list == s_args)
        self.assertTrue(self.test_obj.util_obj.end.call_args_list == end_args)
        self.assertTrue(result[0] == 'va')
        self.assertTrue(result[1] == 'te')


def test_suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(IndependentTestCase)
    return suite

if __name__ == '__main__':
    unittest.main()
