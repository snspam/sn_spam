"""
Tests the connections module.
"""
import unittest
import mock
from .context import connections
from .context import test_utils as tu


class ConnectionsTestCase(unittest.TestCase):
    def setUp(self):
        self.test_obj = connections.Connections()

    def tearDown(self):
        self.test_obj = None

    def test_subnetwork_group(self):
        self.test_obj.size_threshold = 4
        self.test_obj.direct_connections = mock.Mock(return_value=('12', 'r'))
        self.test_obj.group = mock.Mock(return_value=('result'))
        self.test_obj.iterate = mock.Mock()

        result = self.test_obj.subnetwork(69, 'df', 'rels', debug=True)

        self.assertTrue(result == 'result')
        self.test_obj.direct_connections.assert_called_with(69, 'df', 'rels')
        self.test_obj.group.assert_called_with(69, 'df', 'rels', debug=True)
        self.test_obj.iterate.assert_not_called()

    def test_subnetwork_iterate(self):
        self.test_obj.size_threshold = 1
        self.test_obj.direct_connections = mock.Mock(return_value=('12', 'r'))
        self.test_obj.group = mock.Mock()
        self.test_obj.iterate = mock.Mock(return_value='result')

        result = self.test_obj.subnetwork(69, 'df', 'rels', debug=True)

        self.assertTrue(result == 'result')
        self.test_obj.direct_connections.assert_called_with(69, 'df', 'rels')
        self.test_obj.group.assert_not_called()
        self.test_obj.iterate.assert_called_with(69, 'df', 'rels', debug=True)

    def test_direct_connections_empty_sets(self):
        df = tu.sample_df(10)
        df.columns = ['com_id', 'user_id']
        rels = [('posts', 'user', 'user_id')]

        result = self.test_obj.direct_connections(2, df, rels)

        self.assertTrue(len(result[0]) == 0)
        self.assertTrue(len(result[1]) == 0)

    def test_direct_connections_non_empty_sets(self):
        l = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        df = tu.sample_group_df(l)
        df['user_id'] = df['com_id']
        df['com_id'] = range(len(df))
        df['text_id'] = range(100, 100 + len(df))
        rels = [('posts', 'user', 'user_id'), ('intext', 'text', 'text_id')]

        result = self.test_obj.direct_connections(2, df, rels)

        self.assertTrue(result[0] == set({2, 3}))
        self.assertTrue(result[1] == set({'posts'}))

    def test_iterate_no_connections(self):
        df = tu.sample_df(10)
        df.columns = ['com_id', 'user_id']
        rels = [('posts', 'user', 'user_id')]

        result = self.test_obj.iterate(2, df, rels)

        self.assertTrue(result == (set({2}), set()))

    def test_iterate_with_direct_connections(self):
        l = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        df = tu.sample_group_df(l)
        df['user_id'] = df['com_id']
        df['com_id'] = range(len(df))
        df['text_id'] = range(100, 100 + len(df))
        rels = [('posts', 'user', 'user_id'), ('intext', 'text', 'text_id')]

        result = self.test_obj.iterate(2, df, rels)

        self.assertTrue(result == (set({2, 3}), set({'posts'})))

    def test_iterate_with_indirect_connections(self):
        l = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        df = tu.sample_group_df(l)
        df['user_id'] = df['com_id']
        df['com_id'] = range(len(df))
        df['text_id'] = [100, 101, 102, 103, 104, 105, 106, 107, 103, 109]
        rels = [('posts', 'user', 'user_id'), ('intext', 'text', 'text_id')]

        result = self.test_obj.group(2, df, rels)

        self.assertTrue(result[0] == set({8, 9, 2, 3}))
        self.assertTrue(result[1] == set({'intext', 'posts'}))

    def test_group_no_connections(self):
        df = tu.sample_df(10)
        df.columns = ['com_id', 'user_id']
        rels = [('posts', 'user', 'user_id')]

        result = self.test_obj.group(2, df, rels)

        self.assertTrue(result == (set({2}), set()))

    def test_group_with_direct_connections(self):
        l = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        df = tu.sample_group_df(l)
        df['user_id'] = df['com_id']
        df['com_id'] = range(len(df))
        df['text_id'] = range(100, 100 + len(df))
        rels = [('posts', 'user', 'user_id'), ('intext', 'text', 'text_id')]

        result = self.test_obj.group(2, df, rels)

        self.assertTrue(result == (set({2, 3}), set({'posts'})))

    def test_group_with_indirect_connections(self):
        l = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        df = tu.sample_group_df(l)
        df['user_id'] = df['com_id']
        df['com_id'] = range(len(df))
        df['text_id'] = [100, 101, 102, 103, 104, 105, 106, 107, 103, 109]
        rels = [('posts', 'user', 'user_id'), ('intext', 'text', 'text_id')]

        result = self.test_obj.group(2, df, rels)

        self.assertTrue(result[0] == set({8, 9, 2, 3}))
        self.assertTrue(result[1] == set({'intext', 'posts'}))


def test_suite():
    s = unittest.TestLoader().loadTestsFromTestCase(ConnectionsTestCase)
    return s

if __name__ == '__main__':
    unittest.main()
