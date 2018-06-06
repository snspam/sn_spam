"""
Tests the relational module.
"""
import os
import unittest
import mock
from .context import psl
from .context import config
from .context import pred_builder
from .context import util
from .context import test_utils as tu


class PSLTestCase(unittest.TestCase):
    def setUp(self):
        config_obj = tu.sample_config()
        mock_pred_builder_obj = mock.Mock(pred_builder.PredicateBuilder)
        util_obj = util.Util()
        self.test_obj = psl.PSL(config_obj, mock_pred_builder_obj, util_obj)

    def tearDown(self):
        self.test_obj = None

    def test_init(self):
        # setup
        result = self.test_obj

        # assert
        self.assertTrue(isinstance(result.config_obj, config.Config))
        self.assertTrue(isinstance(result.pred_builder_obj,
                pred_builder.PredicateBuilder))
        self.assertTrue(isinstance(result.util_obj, util.Util))

    def test_compile(self):
        os.chdir = mock.Mock()
        os.system = mock.Mock()

        self.test_obj.compile('psl/')

        build = 'mvn dependency:build-classpath '
        build += '-Dmdep.outputFile=classpath.out -q'
        expected = [mock.call('mvn compile -q'), mock.call(build)]
        os.chdir.assert_called_with('psl/')
        self.assertTrue(os.system.call_args_list == expected)

    def test_run_infer(self):
        os.chdir = mock.Mock()
        os.system = mock.Mock()
        self.test_obj.config_obj.infer = True

        self.test_obj.run('psl/')

        execute = 'java -Xmx60g -cp ./target/classes:`cat classpath.out` '
        execute += 'spam.Infer 1 soundcloud intext posts'
        os.chdir.assert_called_with('psl/')
        os.system.assert_called_with(execute)

    def test_clear_data(self):
        psl_data_f = 'test_psl/'
        os.system = mock.Mock()

        self.test_obj.clear_data(psl_data_f)

        expected = [mock.call('rm test_psl/*.tsv'),
                mock.call('rm test_psl/*.txt'),
                mock.call('rm test_psl/db/*.db')]
        self.assertTrue(os.system.call_args_list == expected)

    def test_gen_predicates(self):
        self.test_obj.pred_builder_obj.build_comments = mock.Mock()
        self.test_obj.pred_builder_obj.build_relations = mock.Mock()

        self.test_obj.gen_predicates('df', 'test', 'd/', fw='fw')

        expected = [mock.call('intext', 'text', 'text_id', 'df', 'test',
                'd/', fw='fw'), mock.call('posts', 'user', 'user_id', 'df',
                'test', 'd/', fw='fw')]
        self.test_obj.pred_builder_obj.build_comments.assert_called_with('df',
                'test', 'd/')
        self.assertTrue(self.test_obj.pred_builder_obj.build_relations.
                call_args_list == expected)

    def test_gen_model(self):
        self.test_obj.priors = mock.Mock(return_value=['n', 'p'])
        self.test_obj.map_relation_to_rules = mock.Mock()
        self.test_obj.map_relation_to_rules.side_effect = [['r1', 'r2'],
                ['a1', 'a2']]
        self.test_obj.write_model = mock.Mock()

        self.test_obj.gen_model('d/')

        exp = ['n', 'p', 'r1', 'r2', 'a1', 'a2']
        self.test_obj.priors.assert_called()
        self.assertTrue(self.test_obj.map_relation_to_rules.call_args_list ==
                [mock.call('intext', 'text'), mock.call('posts', 'user')])
        self.test_obj.write_model.assert_called_with(exp, 'd/')

    def test_network_size(self):
        self.test_obj.util_obj.file_len = mock.Mock()
        self.test_obj.util_obj.file_len.side_effect = [2, 4, 8]
        self.test_obj.config_obj.relations = [('posts', 'user', 'user_id')]

        self.test_obj.network_size('d/')

        exp_fl = [mock.call('d/val_1.tsv'), mock.call('d/val_posts_1.tsv'),
                mock.call('d/val_user_1.tsv')]
        self.assertTrue(self.test_obj.util_obj.file_len.call_args_list ==
                exp_fl)

    def test_priors_no_sq(self):
        self.test_obj.sq = False

        result = self.test_obj.priors()

        exp = ['1.0: ~spam(Com)', '1.0: indpred(Com) -> spam(Com)']

        self.assertTrue(result == exp)

    def test_priors_sq_diff_weights(self):
        self.test_obj.wgt = 2.0

        result = self.test_obj.priors()

        exp = ['2.0: ~spam(Com) ^2', '2.0: indpred(Com) -> spam(Com) ^2']
        self.assertTrue(result == exp)

    def test_map_relation_to_rules_no_sq(self):
        self.test_obj.sq = False

        result = self.test_obj.map_relation_to_rules('intext', 'text')

        r1 = '1.0: intext(Com, Text) & spammytext(Text) -> spam(Com)'
        r2 = '1.0: intext(Com, Text) & spam(Com) -> spammytext(Text)'
        self.assertTrue(result == [r1, r2])

    def test_map_relation_to_rules_sq_diff_weights(self):
        self.test_obj.wgt = 2.0

        result = self.test_obj.map_relation_to_rules('intext', 'text')

        r1 = '2.0: intext(Com, Text) & spammytext(Text) -> spam(Com) ^2'
        r2 = '2.0: intext(Com, Text) & spam(Com) -> spammytext(Text) ^2'
        self.assertTrue(result == [r1, r2])


def test_suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(PSLTestCase)
    return suite

if __name__ == '__main__':
    unittest.main()
