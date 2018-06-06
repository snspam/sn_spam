"""
Tests the generator module.
"""
import re
import unittest
import numpy as np
import pandas as pd
import mock
from .context import generator
from .context import test_utils as tu


class GeneratorTestCase(unittest.TestCase):
    def setUp(self):
        self.test_obj = generator.Generator()

    def tearDown(self):
        self.test_obj = None

    def test_gen_group_ids(self):
        relations = [('intext', 'text', 'text_id'),
                ('posts', 'user', 'user_id')]
        self.test_obj.gen_group_id = mock.Mock()
        self.test_obj.gen_group_id.side_effect = ['df2', 'df3']

        result = self.test_obj.gen_group_ids('df', relations)

        exp = [mock.call('df', 'text_id'), mock.call('df2', 'user_id')]
        self.assertTrue(result == 'df3')
        self.assertTrue(self.test_obj.gen_group_id.call_args_list == exp)

    def test_gen_group_id_gen_string_ids_called(self):
        first_df = tu.sample_df(2)
        df = tu.sample_df(2)
        df.columns = ['com_id', 'text_id']
        df['ind_pred'] = [0.5, 0.5]
        filled_df = tu.sample_df(2)
        df2 = tu.sample_df(2)
        df2.columns = ['com_id', 'ind_pred']
        df2['hash_id'] = [np.nan, np.nan]
        first_df.copy = mock.Mock(return_value=df)
        self.test_obj.gen_string_ids = mock.Mock(return_value=filled_df)
        df.merge = mock.Mock(return_value=df2)

        result = self.test_obj.gen_group_id(first_df, 'hash_id')

        exp = pd.Series(['empty', 'empty'])
        first_df.copy.assert_called()
        self.test_obj.gen_string_ids.assert_called_with(df, 'hash_id',
                regex=r'(#\w+)')
        df.merge.assert_called_with(filled_df, on='com_id', how='left')
        self.assertTrue(result['hash_id'].equals(exp))

    def test_gen_text_ids_with_matches(self):
        series = pd.Series([1, 1, 5, 5, 3, 3, 2, 2, 4, 4])
        g_df = tu.sample_text_df()

        result = self.test_obj.gen_text_ids(g_df, 'text_id')

        self.assertTrue(result['text_id'].equals(series))

    def test_gen_text_ids_no_matches(self):
        series = pd.Series([1, 2])
        g_df = pd.DataFrame(['cool', 'pool'], columns=['text'])

        result = self.test_obj.gen_text_ids(g_df, 'text_id')

        self.assertTrue(result['text_id'].equals(series))

    def test_gen_hour_ids(self):
        df = pd.DataFrame(['2011-10-31 13:37:50', '2011-10-31 17:37:50'])
        df.columns = ['timestamp']

        result = self.test_obj.gen_hour_ids(df, 'hour_id')

        self.assertTrue(result['hour_id'].equals(pd.Series(['13', '17'])))

    def test_gen_string_ids(self):
        df = pd.DataFrame([('1', 'my #hashtaG', 0.77, 1),
                           ('2', '#orange #sooo', 0.01, 0)])
        df.columns = ['com_id', 'text', 'ind_pred', 'label']
        series = pd.Series(['#hashtag', '#orange#sooo'])

        result = self.test_obj.gen_string_ids(df, 'hash_id')

        self.assertTrue(len(result) == 2)
        self.assertTrue(result['hash_id'].equals(series))

    def test_gen_string_ids_no_items(self):
        df = pd.DataFrame([('1', 'my hashtaG', 0.77, 1),
                           ('2', 'orange sooo', 0.01, 0)])
        df.columns = ['com_id', 'text', 'ind_pred', 'label']

        result = self.test_obj.gen_string_ids(df, 'hash_id')

        self.assertTrue(len(result) == 0)

    def test_get_items_hashtags(self):
        regex = re.compile(r"(#\w+)")
        text = 'here is #Hashtag #wooHoo'

        result = self.test_obj.get_items(text, regex, str_form=True)

        self.assertTrue(result == '#hashtag#woohoo')


def test_suite():
    s = unittest.TestLoader().loadTestsFromTestCase(GeneratorTestCase)
    return s

if __name__ == '__main__':
    unittest.main()
