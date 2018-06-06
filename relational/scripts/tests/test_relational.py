"""
Tests the relational module.
"""
import unittest
import pandas as pd
import mock
from .context import relational
from .context import config
from .context import psl
from .context import tuffy
from .context import mrf
from .context import util
from .context import test_utils as tu


class RelationalTestCase(unittest.TestCase):
    def setUp(self):
        config_obj = tu.sample_config()
        mock_psl_obj = mock.Mock(psl.PSL)
        mock_tuffy_obj = mock.Mock(tuffy.Tuffy)
        mock_mrf_obj = mock.Mock(mrf.MRF)
        util_obj = util.Util()
        self.test_obj = relational.Relational(config_obj, mock_psl_obj,
                mock_tuffy_obj, mock_mrf_obj, util_obj)

    def tearDown(self):
        self.test_obj = None

    def test_init(self):
        # setup
        result = self.test_obj

        # assert
        self.assertTrue(isinstance(result.config_obj, config.Config))

    @mock.patch('os.makedirs')
    @mock.patch('os.path.exists')
    def test_file_folders(self, mock_exists, mock_makedirs):
        # setup
        mock_exists.return_value = False

        # test
        result = self.test_obj.folders()

        # assert
        exp_paths = [mock.call('rel/psl/data/soundcloud/'),
                mock.call('rel/output/soundcloud/status/')]
        self.assertTrue(mock_exists.call_args_list == exp_paths)
        self.assertTrue(mock_makedirs.call_args_list == exp_paths)
        self.assertTrue(result[0] == 'rel/psl/')
        self.assertTrue(result[1] == 'rel/psl/data/soundcloud/')
        self.assertTrue(result[2] == 'rel/tuffy/')
        self.assertTrue(result[3] == 'rel/mrf/')
        self.assertTrue(result[4] == 'ind/data/soundcloud/folds/')
        self.assertTrue(result[5] == 'ind/output/soundcloud/predictions/')
        self.assertTrue(result[6] == 'rel/output/soundcloud/predictions/')
        self.assertTrue(result[7] == 'rel/output/soundcloud/status/')

    def test_open_status_writer(self):
        self.test_obj.config_obj.infer = False
        self.test_obj.util_obj.open_writer = mock.Mock(return_value='f')

        result = self.test_obj.open_status_writer('s/', mode='a')

        self.assertTrue(result == 'f')
        self.test_obj.util_obj.open_writer.assert_called_with('s/train_1.txt',
                'a')

    def test_check_dataframes_none(self):
        pd.read_csv = mock.Mock()
        pd.read_csv.side_effect = ['val_df', 'test_df']

        result = self.test_obj.check_dataframes(None, 'df', 'folds/')

        exp = [mock.call('folds/val_1.csv'), mock.call('folds/test_1.csv')]
        self.assertTrue(result[0] == 'val_df')
        self.assertTrue(result[1] == 'test_df')
        self.assertTrue(pd.read_csv.call_args_list == exp)

    def test_merge_ind_preds(self):
        test_df = tu.sample_df(10)
        test_df.columns = ['com_id', 'rando']
        pred_df = tu.sample_df(10)
        pd.read_csv = mock.Mock(return_value=pred_df)

        result = self.test_obj.merge_ind_preds(test_df, 'test', 'ind_pred/')

        self.assertTrue(len(list(result)) == 3)
        self.assertTrue(len(result) == 10)
        pd.read_csv.assert_called_with('ind_pred/test_1_preds.csv')

    def test_compile_reasoning_engine(self):
        folders = ('psl/', 'a/', 'b/', 'c/', 'd/', 'e/', 'f/', 'g/')
        self.test_obj.file_folders = mock.Mock(return_value=folders)
        self.test_obj.psl_obj.compile = mock.Mock()

        self.test_obj.compile_reasoning_engine()

        self.test_obj.file_folders.assert_called()
        self.test_obj.psl_obj.compile.assert_called_with('psl/')

    def test_main(self):
        folders = ('a/', 'b/', 'c/', 'd/', 'e/', 'f/', 'g/', 'h/')
        self.test_obj.folders = mock.Mock(return_value=folders)
        self.test_obj.open_status_writer = mock.Mock(return_value='sw')
        self.test_obj.util_obj.start = mock.Mock()
        self.test_obj.check_dataframes = mock.Mock(return_value=('v', 't'))
        self.test_obj.merge_ind_preds = mock.Mock()
        self.test_obj.merge_ind_preds.side_effect = ['v_df', 't_df']
        self.test_obj.run_relational_model = mock.Mock()
        self.test_obj.util_obj.end = mock.Mock()
        self.test_obj.util_obj.close_writer = mock.Mock()

        self.test_obj.main('v_df', 't_df')

        exp = 'total relational model time: '
        exp_sw = [mock.call('h/'), mock.call('h/', mode='a')]
        self.test_obj.folders.assert_called_with()
        self.assertTrue(self.test_obj.open_status_writer.call_args_list ==
                exp_sw)
        self.test_obj.util_obj.start.assert_called_with(fw='sw')
        self.test_obj.check_dataframes.assert_called_with('v_df', 't_df', 'e/')
        self.assertTrue(self.test_obj.merge_ind_preds.call_args_list ==
                [mock.call('v', 'val', 'f/'),
                mock.call('t', 'test', 'f/')])
        self.test_obj.run_relational_model.assert_called_with('v_df',
                't_df', 'a/', 'b/', 'c/', 'd/', 'g/', fw='sw')
        self.test_obj.util_obj.end.assert_called_with(exp, fw='sw')
        self.test_obj.util_obj.close_writer.assert_called_with('sw')

    def test_run_psl(self):
        self.test_obj.util_obj.start = mock.Mock()
        self.test_obj.util_obj.end = mock.Mock()
        self.test_obj.psl_obj.clear_data = mock.Mock()
        self.test_obj.psl_obj.gen_predicates = mock.Mock()
        self.test_obj.psl_obj.gen_model = mock.Mock()
        self.test_obj.psl_obj.network_size = mock.Mock()
        self.test_obj.util_obj.close_writer = mock.Mock()
        self.test_obj.psl_obj.run = mock.Mock()

        self.test_obj.run_psl('v_df', 't_df', 'psl/', 'psl_data/', fw='fw')

        exp = '\nbuilding predicates...'
        self.test_obj.util_obj.start.assert_called_with(exp, fw='fw')
        self.test_obj.psl_obj.clear_data.assert_called_with('psl_data/',
                fw='fw')
        self.assertTrue(self.test_obj.psl_obj.gen_predicates.call_args_list ==
                [mock.call('v_df', 'val', 'psl_data/', fw='fw'),
                mock.call('t_df', 'test', 'psl_data/', fw='fw')])
        self.test_obj.psl_obj.gen_model.assert_called_with('psl_data/')
        self.test_obj.psl_obj.network_size.assert_called_with('psl_data/',
                fw='fw')
        self.test_obj.util_obj.close_writer('fw')
        self.test_obj.util_obj.end.assert_called_with('\n\ttime: ', fw='fw')
        self.test_obj.psl_obj.run.assert_called_with('psl/')

    def test_run_tuffy(self):
        self.test_obj.util_obj.start = mock.Mock()
        self.test_obj.util_obj.end = mock.Mock()
        self.test_obj.tuffy_obj.clear_data = mock.Mock()
        self.test_obj.tuffy_obj.gen_predicates = mock.Mock()
        self.test_obj.tuffy_obj.run = mock.Mock()
        self.test_obj.tuffy_obj.parse_output = mock.Mock(return_value='p_df')
        self.test_obj.tuffy_obj.evaluate = mock.Mock()

        self.test_obj.run_tuffy('v_df', 't_df', 't/', fw='fw')

        exp = '\nbuilding predicates...'
        self.test_obj.tuffy_obj.clear_data.assert_called_with('t/')
        self.test_obj.util_obj.start.assert_called_with(exp, fw='fw')
        self.assertTrue(self.test_obj.tuffy_obj.gen_predicates.call_args_list
                == [mock.call('v_df', 'val', 't/'),
                mock.call('t_df', 'test', 't/')])
        self.test_obj.util_obj.end.assert_called_with('\n\ttime: ', fw='fw')
        self.test_obj.tuffy_obj.run.assert_called_with('t/')
        self.test_obj.tuffy_obj.parse_output.assert_called_with('t/')
        self.test_obj.tuffy_obj.evaluate.assert_called_with('t_df', 'p_df')

    def test_run_relational_model_psl(self):
        self.test_obj.run_psl = mock.Mock()
        self.test_obj.run_tuffy = mock.Mock()

        self.test_obj.run_relational_model('v', 't', 'p/', 'd/', 't/', 'm/',
            'rp/', fw='fw')

        self.test_obj.run_psl.assert_called_with('v', 't', 'p/', 'd/', fw='fw')
        self.test_obj.run_tuffy.assert_not_called()

    def test_run_relational_model_tuffy(self):
        self.test_obj.config_obj.engine = 'tuffy'
        self.test_obj.run_psl = mock.Mock()
        self.test_obj.run_tuffy = mock.Mock()

        self.test_obj.run_relational_model('v', 't', 'p/', 'd/', 't/', 'm/',
                'rp/', fw='fw')

        self.test_obj.run_tuffy.assert_called_with('v', 't', 't/', fw='fw')
        self.test_obj.run_psl.assert_not_called()


def test_suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(RelationalTestCase)
    return suite

if __name__ == '__main__':
    unittest.main()
