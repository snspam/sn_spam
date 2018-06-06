"""
This module handles which parts of the appliction to run.
"""


class Runner:
    """Runs the independent and relational models."""

    def __init__(self, independent_obj, relational_obj, analysis_obj):
        """Initializes the independent and relational objects."""

        self.independent_obj = independent_obj
        """Object to access modules in the 'independent' package."""
        self.relational_obj = relational_obj
        """Object to access modules in the 'relational' package."""
        self.analysis_obj = analysis_obj
        """Container object holding modules in the 'analysis' package."""

    def compile_reasoning_engine(self):
        """Compiles PSL and the Groovy scripts."""
        self.relational_obj.compile_reasoning_engine()

    def run_label(self):
        """Runs the relabeling module to relabel the specified data."""
        self.analysis_obj.relabel()

    def run_independent(self, data, stacking=0):
        """Runs the independent model and returns the training and test
        dataframes used in classification."""
        val_df, test_df = self.independent_obj.main(data)
        return val_df, test_df

    def run_purity(self, test_df):
        """Checks to see how many comments that are most susceptible to change
                are helped and hurt by a potential relational model.
        test_df: comments dataframe."""
        self.analysis_obj.test_purity(test_df)

    def run_relational(self, val_df, test_df):
        """Runs the relational model with the specified training and test.
        val_df: validation dataframe.
        test_df: testing dataframe."""
        self.relational_obj.main(val_df, test_df)

    def run_evaluation(self, test_df):
        """Evaluates how the models perform.
        test_df: comments dataframe."""
        score_dict = self.analysis_obj.evaluate(test_df)
        return score_dict

    def run_explanation(self, test_df):
        """Provides explanations for relational model predictions.
        test_df: comments dataframe."""
        self.analysis_obj.explain(test_df)
