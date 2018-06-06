"""
This module handles all operations to run the relational model using Tuffy.
"""
import os
import time as t
import pandas as pd
import sklearn.metrics as sm


class Tuffy:
    """Class that handles all operations pertaining to Tuffy."""

    def __init__(self, config_obj, pred_builder_obj, util_obj):
        """Initialize all object dependencies for this class."""

        self.config_obj = config_obj
        """User settings."""
        self.pred_builder_obj = pred_builder_obj
        """Predicate builder."""
        self.util_obj = util_obj
        """General utility methods."""

    # public
    def clear_data(self, tuffy_f):
        """Clears any old predicate and model data.
        tuffy_data_f: folder where all tuffy related files are stored."""
        print('Clearing out old data...')
        os.system('rm ' + tuffy_f + '*evidence.txt')
        os.system('rm ' + tuffy_f + '*query.txt')

    def gen_predicates(self, df, dset, rel_data_f):
        """Generates all necessary predicates for the relational model.
        df: original validation dataframe.
        dset: dataset (e.g. 'val', 'test').
        rel_data_f: folder to save predicate data to."""
        self.pred_builder_obj.build_comments(df, dset, rel_data_f, tuffy=True)
        for relation, group, group_id in self.config_obj.relations:
            self.pred_builder_obj.build_relations(relation, group, group_id,
                    df, dset, rel_data_f, tuffy=True)

    def run(self, tuffy_f):
        """Runs the tuffy model using java.
        tuffy_f: folder with the tuffy source code."""
        print('running relational model...',)
        start = t.time()
        execute = 'java -jar tuffy.jar -i mln.txt -e test_evidence.txt '
        execute += '-queryFile test_query.txt -r out -dual > log'

        os.chdir(tuffy_f)  # change directory to tuffy
        os.system(execute)
        print('%ds' % (t.time() - start))

    def parse_output(self, tuffy_f):
        """Parses the map and marginal outputs from tuffy.
        tuffy_f: tuffy folder.
        Returns dataframe with map and marginal predictions."""
        map_filename = tuffy_f + 'out.map'
        marginal_filename = tuffy_f + 'out.marginal'

        map_df = self.parse_map_output(map_filename)
        marginal_df = self.parse_marginal_output(marginal_filename)
        pred_df = marginal_df.merge(map_df, on='com_id', how='left')
        pred_df['map'] = pred_df['map'].fillna(0).apply(int)
        return pred_df

    def evaluate(self, test_df, pred_df):
        """Evaluates tuffy performance.
        test_df: testing dataframe.
        pred_df: predictions dataframe."""
        df = test_df.merge(pred_df, on='com_id')  # add noise
        aupr = sm.average_precision_score(df['label'], df['marginal'])
        precision = sm.precision_score(df['label'], df['map'])
        recall = sm.recall_score(df['label'], df['map'])
        result = (aupr, precision, recall)
        print('aupr: %.4f, precision: %.4f, recall: %.4f' % result)

    # private
    def parse_map_output(self, filename):
        """Extracts the com_id from the map file.
        filename: name of the map output file.
        Returns dataframe with the com_id and its map assignment."""
        spam_list = []

        with open(filename, 'r') as mp:
            for line in mp.readlines():
                start = line.find('(') + 1
                end = line.find(')')
                com_id = int(line[start:end])
                spam_list.append((com_id, 1))

        map_df = pd.DataFrame(spam_list, columns=['com_id', 'map'])
        return map_df

    def parse_marginal_output(self, filename):
        """Extracts the com_id and its marginal prediction.
        filename: name of the marginal output file.
        Returns dataframe of com_ids and their marginal predictions."""
        spam_list = []

        with open(filename, 'r') as mp:
            for line in mp.readlines():
                end1 = line.find('\t')
                start2 = line.find('(') + 1
                end2 = line.find(')')
                pred = float(line[:end1])
                com_id = int(line[start2:end2])
                spam_list.append((com_id, pred))

        marginal_df = pd.DataFrame(spam_list, columns=['com_id', 'marginal'])
        return marginal_df
