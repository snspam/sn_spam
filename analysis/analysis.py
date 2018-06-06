"""
Module that provides easy acess to the modules in the 'analysis' package.
"""
import pandas as pd


class Analysis:
    """Class containing handles to modules in the 'analysis' package."""

    def __init__(self, config_obj, label_obj, purity_obj, evaluation_obj,
                 interpretability_obj, util_obj):
        """Initializes all object dependencies for this object."""

        self.config_obj = config_obj
        """User settings."""
        self.label_obj = label_obj
        """Relabels data."""
        self.purity_obj = purity_obj
        """Tests purity of relational groups for low predicted spam."""
        self.evaluation_obj = evaluation_obj
        """Evaluates the performance of both models."""
        self.interpretability_obj = interpretability_obj
        """Can provide an explanation for relational model predictions."""
        self.util_obj = util_obj
        """General utility methods."""

    # public
    def relabel(self):
        """Runs the relabeling module to relabel a dataset."""
        self.label_obj.relabel()

    def test_purity(self, df):
        """Checks to see how many comments that are most susceptible to change
                are helped and hurt by a potential relational model.
        df: comments dataframe."""
        df = self.check_dataframe(df)
        self.purity_obj.test_relations(df)

    def evaluate(self, df):
        """Convenience method to evaluate model performance.
        df: dataframe containing comments with both sets of predictions."""
        df = self.check_dataframe(df)
        score_dict = self.evaluation_obj.evaluate(df)
        return score_dict

    def explain(self, df):
        """Convencience method to explain model predictions.
        df: dataframe containing comments with both sets of predictions."""
        df = self.check_dataframe(df)
        self.interpretability_obj.explain(df)

    # private
    def check_dataframe(self, df):
        """If there is no dataframe, then reads it in, otherwise returns df.
        df: dataframe.
        Returns dataframe object."""
        if df is None:
            folds_f = self.define_file_folders()
            df = self.read_fold(folds_f)
        return df

    def define_file_folders(self):
        """Returns absolute path to folds data folder."""
        ind_dir = self.config_obj.ind_dir
        domain = self.config_obj.domain

        folds_f = ind_dir + 'data/' + domain + '/folds/'
        return folds_f

    def read_fold(self, folds_f, dset='test'):
        """Reads the comments for a specified dataset.
        folds_f: folder where the datasets are.
        dset: dataset to read (e.g. 'val', 'test')
        Returns comments dataframe if it exists."""
        fold = self.config_obj.fold
        filename = folds_f + dset + '_' + fold + '.csv'

        if self.util_obj.check_file(filename):
            df = pd.read_csv(filename, lineterminator='\n')
        return df
