"""
Tests the run script.
"""
import os
import mock
import unittest
import pandas as pd
from context import run
from context import runner
from context import config
from context import test_utils as tu


class RunTestCase(unittest.TestCase):
    def test_directories(self):
        # test
        result = run.directories('this')

        # assert
        self.assertTrue(result[0] == 'this/app/')
        self.assertTrue(result[1] == 'this/independent/')
        self.assertTrue(result[2] == 'this/relational/')
        self.assertTrue(result[3] == 'this/analysis/')

    def test_init_dependencies(self):
        result = run.init_dependencies()

        self.assertTrue(isinstance(result[0], runner.Runner))
        self.assertTrue(isinstance(result[1], config.Config))

    def test_global_settings(self):
        os.isatty = mock.Mock(return_value=True)
        read = mock.Mock()
        popen = mock.Mock()
        read.split = mock.Mock(return_value=('69', '77'))
        popen.read = mock.Mock(return_value=read)
        os.popen = mock.Mock(return_value=popen)
        pd.set_option = mock.Mock()
        config_obj = tu.sample_config()
        config_obj.set_display = mock.Mock()

        run.global_settings(config_obj)

        os.isatty.assert_called_with(0)
        os.popen.assert_called_with('stty size', 'r')
        pd.set_option.assert_called_with('display.width', 77)
        config_obj.set_display.assert_called_with(True)


def test_suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(RunTestCase)
    return suite

if __name__ == '__main__':
    unittest.main()
