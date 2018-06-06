import unittest
import test_run
import test_runner
import test_config
from independent.scripts.tests import test_classification
from independent.scripts.tests import test_content_features
from independent.scripts.tests import test_graph_features
from independent.scripts.tests import test_independent
from independent.scripts.tests import test_relational_features
from relational.scripts.tests import test_comments
from relational.scripts.tests import test_generator
from relational.scripts.tests import test_pred_builder
from relational.scripts.tests import test_psl
from relational.scripts.tests import test_relational
from relational.scripts.tests import test_tuffy
from analysis.tests import test_analysis
from analysis.tests import test_connections
from analysis.tests import test_evaluation
from analysis.tests import test_interpretability
from analysis.tests import test_label
from analysis.tests import test_purity
from analysis.tests import test_util
from experiments.tests import test_single_exp
from experiments.tests import test_subsets_exp
from experiments.tests import test_training_exp
from experiments.tests import test_robust_exp

if __name__ == '__main__':
    suites = []

    # app package
    suites.append(test_run.test_suite())
    suites.append(test_runner.test_suite())
    suites.append(test_config.test_suite())

    # independent package
    suites.append(test_independent.test_suite())
    suites.append(test_classification.test_suite())
    suites.append(test_content_features.test_suite())
    suites.append(test_relational_features.test_suite())
    suites.append(test_graph_features.test_suite())

    # relational package
    suites.append(test_comments.test_suite())
    suites.append(test_generator.test_suite())
    suites.append(test_pred_builder.test_suite())
    suites.append(test_relational.test_suite())
    suites.append(test_psl.test_suite())
    suites.append(test_tuffy.test_suite())

    # analysis package
    suites.append(test_analysis.test_suite())
    suites.append(test_connections.test_suite())
    suites.append(test_evaluation.test_suite())
    suites.append(test_interpretability.test_suite())
    suites.append(test_label.test_suite())
    suites.append(test_purity.test_suite())
    suites.append(test_util.test_suite())

    # experiments package
    suites.append(test_single_exp.test_suite())
    suites.append(test_subsets_exp.test_suite())
    suites.append(test_training_exp.test_suite())
    suites.append(test_robust_exp.test_suite())

    all_tests = unittest.TestSuite(suites)
    unittest.TextTestRunner().run(all_tests)
