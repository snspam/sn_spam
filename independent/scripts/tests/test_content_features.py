"""
Tests the content_features module.
"""
import unittest
import numpy as np
import scipy.sparse as ss
import mock
from sklearn.feature_extraction.text import CountVectorizer
from .context import content_features
from .context import config
from .context import util
from .context import test_utils as tu


class ContentFeaturesTestCase(unittest.TestCase):
    def setUp(self):
        config_obj = tu.sample_config()
        util_obj = util.Util()
        self.test_obj = content_features.ContentFeatures(config_obj, util_obj)

    def tearDown(self):
        self.test_obj = None

    def test_init(self):
        result = self.test_obj

        self.assertTrue(isinstance(result.config_obj, config.Config))

    @mock.patch('pandas.concat')
    def test_build(self, mock_concat):
        self.test_obj.config_obj.ngrams = True
        self.test_obj.util_obj.start = mock.Mock()
        self.test_obj.define_file_folders = mock.Mock(return_value='f/')
        self.test_obj.settings = mock.Mock(return_value='ngram_params')
        self.test_obj.basic = mock.Mock(return_value=('tr', 'te', 'feats'))
        mock_concat.side_effect = ['coms_df', 'feats_df']
        self.test_obj.ngrams = mock.Mock(return_value=('tr_m', 'te_m'))
        self.test_obj.util_obj.end = mock.Mock()

        result = self.test_obj.build('train_df', 'test_df', 'test', fw='fw')

        exp_start = 'building content features...'
        exp_concat = [mock.call(['train_df', 'test_df']),
                mock.call(['tr', 'te'])]
        self.test_obj.util_obj.start.assert_called_with(exp_start, fw='fw')
        self.test_obj.define_file_folders.assert_called()
        self.test_obj.settings.assert_called()
        self.test_obj.basic.assert_called_with('train_df', 'test_df',
                'train_test_1', '_content.pkl', 'f/')
        self.assertTrue(mock_concat.call_args_list == exp_concat)
        self.test_obj.ngrams.assert_called_with('coms_df', 'train_df',
                'test_df', 'ngram_params', 'train_test_1', '_ngrams.npz',
                'f/', fw='fw')
        self.test_obj.util_obj.end.assert_called_with(fw='fw')
        self.assertTrue(result == ('tr_m', 'te_m', 'feats_df', 'feats'))

    def test_define_file_folders(self):
        result = self.test_obj.define_file_folders()

        self.assertTrue(result == 'ind/output/soundcloud/features/')

    def test_settings(self):
        setting_dict = {'stop_words': 'english', 'ngram_range': (3, 3),
                        'max_features': 10000, 'analyzer': 'char_wb',
                        'min_df': 1, 'max_df': 1.0, 'binary': True,
                        'vocabulary': None, 'dtype': np.int32}

        result = self.test_obj.settings()

        self.assertTrue(result == setting_dict)

    def test_basic(self):
        self.test_obj.util_obj.load = mock.Mock()
        self.test_obj.build_features = mock.Mock()
        self.test_obj.build_features.side_effect = [('tr', ''), ('te', 'fts')]
        self.test_obj.util_obj.save = mock.Mock()

        result = self.test_obj.basic('train', 'test', 'fn', '_ext', 'f/')

        exp_bf = [mock.call('train'), mock.call('test')]
        self.assertTrue(result == ('tr', 'te', 'fts'))
        self.assertTrue(self.test_obj.build_features.call_args_list == exp_bf)
        self.test_obj.util_obj.load.assert_not_called()

    def test_ngrams_none(self):
        self.test_obj.config_obj.ngrams = False

        result = self.test_obj.ngrams('coms', 'train', 'test', 'np', 'fn',
                '_ext', 'f/')

        self.assertTrue(result == (None, None))

    def test_ngrams(self):
        self.test_obj.config_obj.ngrams = True
        self.test_obj.util_obj.load_sparse = mock.Mock()
        self.test_obj.build_ngrams = mock.Mock(return_value='ngrams')
        self.test_obj.util_obj.save_sparse = mock.Mock()
        self.test_obj.split_mat = mock.Mock(return_value=('tr_m', 'te_m'))

        result = self.test_obj.ngrams('coms', 'train', 'test', 'np', 'fn',
                '_ext', 'f/', fw='fw')

        self.assertTrue(result == ('tr_m', 'te_m'))
        self.test_obj.util_obj.load_sparse.assert_not_called()
        self.test_obj.build_ngrams.assert_called_with('coms', 'np', fw='fw')
        self.test_obj.util_obj.save_sparse.assert_called_with('ngrams',
                'f/fn_ext')
        self.test_obj.split_mat.assert_called_with('ngrams', 'train', 'test')

    def test_count_vectorizer(self):
        setting_dict = {'stop_words': 'english', 'ngram_range': (3, 3),
                'max_features': 10000, 'analyzer': 'char_wb',
                'min_df': 6, 'max_df': 0.1, 'binary': True,
                'vocabulary': None, 'dtype': np.int32}

        result = self.test_obj.count_vectorizer(setting_dict)

        self.assertTrue(isinstance(result, CountVectorizer))

    def test_build_ngrams(self):
        setting_dict = {'stop_words': 'english', 'ngram_range': (3, 3),
                'max_features': 10000, 'analyzer': 'char_wb',
                'min_df': 6, 'max_df': 0.1, 'binary': True,
                'vocabulary': None, 'dtype': np.int32}
        matrix = mock.Mock(np.matrix)
        matrix.tocsr = mock.Mock(return_value='ngrams_csr')
        df = tu.sample_df(2)
        df['text'] = ['banana', 'orange']
        cv = mock.Mock(CountVectorizer)
        cv.fit_transform = mock.Mock(return_value='ngrams_m')
        self.test_obj.count_vectorizer = mock.Mock(return_value=cv)
        ss.lil_matrix = mock.Mock(return_value='id_m')
        ss.hstack = mock.Mock(return_value=matrix)

        result = self.test_obj.build_ngrams(df, setting_dict)

        self.test_obj.count_vectorizer.assert_called_with(setting_dict)
        cv.fit_transform.assert_called_with(['banana', 'orange'])
        ss.lil_matrix.assert_called_with((2, 1))
        ss.hstack.assert_called_with(['id_m', 'ngrams_m'])
        matrix.tocsr.assert_called()
        self.assertTrue(result == 'ngrams_csr')

    def test_split_mat(self):
        df1 = tu.sample_df(4)
        df2 = tu.sample_df(2)
        m = np.matrix([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])

        result = self.test_obj.split_mat(m, df1, df2)

        self.assertTrue(result[0].shape == (4, 2))
        self.assertTrue(result[1].shape == (2, 2))

    def test_build_features(self):
        df = tu.sample_df(2)
        df['text'] = ['banana', 'kiwi']
        self.test_obj.soundcloud = mock.Mock(return_value=('f', 'l'))
        self.test_obj.youtube = mock.Mock()
        self.test_obj.twitter = mock.Mock()
        self.test_obj.yelp_hotel = mock.Mock()
        self.test_obj.yelp_restaurant = mock.Mock()

        result = self.test_obj.build_features(df)

        df['text'] = df['text'].fillna('')
        self.assertTrue(result == ('f', 'l'))
        self.test_obj.soundcloud.assert_called_with(df)
        self.test_obj.youtube.assert_not_called()
        self.test_obj.twitter.assert_not_called()
        self.test_obj.yelp_hotel.assert_not_called()
        self.test_obj.yelp_restaurant.assert_not_called()

    def test_soundcloud(self):
        df = tu.sample_df(2)
        df['text'] = ['banana', 'orange']

        result = self.test_obj.soundcloud(df)

        self.assertTrue(len(result[0] == 2))
        self.assertTrue(result[1] == ['com_num_chars', 'com_has_link'])

    def test_youtube(self):
        df = tu.sample_df(2)
        df['text'] = ['banana', 'orange']
        df['timestamp'] = ['2011-10-31 13:37:50', '2011-10-31 13:47:50']

        result = self.test_obj.youtube(df)

        self.assertTrue(len(result[0] == 2))
        self.assertTrue(result[1] == ['com_num_chars', 'com_weekday',
                'com_hour'])

    def test_twitter(self):
        df = tu.sample_df(2)
        df['text'] = ['bana@na', '#orange']
        df['timestamp'] = ['2011-10-31 13:37:50', '2011-10-31 13:47:50']

        result = self.test_obj.twitter(df)

        self.assertTrue(len(result[0] == 2))
        self.assertTrue(result[1] == ['com_num_chars', 'com_num_hashtags',
                'com_num_mentions', 'com_num_links', 'com_num_retweets'])

    @mock.patch('pandas.DataFrame')
    def test_ifwe(self, mock_df):
        cf = tu.sample_df(10)
        mock_df.return_value = 'feats_df'

        result = self.test_obj.ifwe(cf)

        exp_list = ['sex_id', 'time_passed_id', 'age_id']
        self.assertTrue(result == ('feats_df', exp_list))
        mock_df.assert_called_with(cf['com_id'])

    def test_yelp_hotel(self):
        df = tu.sample_df(2)
        df['text'] = ['bana@na', '#orange!']

        result = self.test_obj.yelp_hotel(df)

        self.assertTrue(len(result[0] == 2))
        self.assertTrue(list(result[0]['com_num_chars']) == [7, 8])
        self.assertTrue(result[1] == ['com_num_chars', 'com_num_links'])

    def test_yelp_restaurant(self):
        df = tu.sample_df(2)
        df['text'] = ['bana@na', '#orange']

        result = self.test_obj.yelp_restaurant(df)

        self.assertTrue(len(result[0] == 2))
        self.assertTrue(list(result[0]['com_num_chars']) == [7, 7])
        self.assertTrue(result[1] == ['com_num_chars', 'com_num_links'])


def test_suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(
            ContentFeaturesTestCase)
    return suite

if __name__ == '__main__':
    unittest.main()
