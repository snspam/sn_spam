"""
Class that applies the LIME framework to a collective classification task.
"""
import os
import random
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.linear_model import LinearRegression

import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Lime:
    """Class that handles all operations to apply LIME to graphical models."""

    def __init__(self, config_obj, connections_obj, generator_obj,
            pred_builder_obj, util_obj):
        """Initializes all object dependencies for this object."""

        self.config_obj = config_obj
        """User settings."""
        self.generator_obj = generator_obj
        """Finds and generates relational ids."""
        self.connections_obj = connections_obj
        """Finds subnetworks for a specific data points."""
        self.pred_builder_obj = pred_builder_obj
        """Predicate Builder."""
        self.util_obj = util_obj
        """Utility methods."""
        self.relations = None
        """Relations present in the subnetwork, can be modified."""

    # public
    def explain(self, test_df):
        """Produce an explanation for a specific comment in the test set by
        applying the LIME framework.
        test_df: dataframe containing test comments."""
        print('\nEXPLANATION')
        test_df = test_df.copy()

        # merge independent and relational model predictions together.
        p, samples, k = self.settings()
        ip_f, rp_f, r_out_f, psl_f, psl_data_f = self.define_file_folders()
        ind_df, rel_df = self.read_predictions(ip_f, rp_f)
        merged_df = self.merge_predictions(test_df, ind_df, rel_df)
        self.show_biggest_improvements(merged_df)

        com_id = self.user_input(merged_df)
        while com_id != -1:
            self.display_raw_instance_to_explain(merged_df, com_id)

            # identify subnetwork pertaining to comment needing explanation.
            expanded_df = self.gen_group_ids(merged_df)
            connections = self.retrieve_all_connections(com_id, expanded_df)
            filtered_df = self.filter_comments(merged_df, connections)
            altered_df = self.perturb_input(filtered_df, samples, p)
            similarities, sample_ids = self.compute_similarity(altered_df,
                    samples)

            # write predicate data pertaining to subnetwork.
            self.clear_old_data(psl_data_f)
            self.write_predicates(filtered_df, psl_data_f)
            self.write_perturbations(altered_df, sample_ids, psl_data_f)

            # generate labels for perturbed instances, then fit a linear model.
            self.compute_labels_for_perturbed_instances(com_id, psl_f)
            labels_df, perturbed_df = self.read_perturbed_labels(psl_data_f,
                    r_out_f)
            x, y, wgts, features = self.preprocess(perturbed_df, labels_df,
                    similarities)
            g = self.fit_linear_model(x, y, wgts)

            # sort features by importance, indicated by their coefficients.
            coef_indices, coef_values = self.extract_and_sort_coefficients(g)
            top_features = self.rearrange_and_filter_features(features,
                    coef_indices, coef_values, k=k)

            # obtain relation dataframes and display explanation.
            relation_dict = self.read_subnetwork_relations(psl_data_f)
            self.display_median_predictions(merged_df)
            self.display_top_features(top_features, merged_df, relation_dict)
            self.display_subnetwork(com_id, filtered_df)
            com_id = self.user_input(merged_df)

    # private
    def settings(self):
        """Settings for the LIME approximation.
        Returns max amount of perturbation, number of samples,
        and top k features."""
        p, samples, k = 1.0, 100, 50
        return p, samples, k

    def define_file_folders(self):
        """Returns absolute path directories."""
        ind_dir = self.config_obj.ind_dir
        rel_dir = self.config_obj.rel_dir
        domain = self.config_obj.domain

        ind_pred_f = ind_dir + 'output/' + domain + '/predictions/'
        rel_pred_f = rel_dir + 'output/' + domain + '/predictions/'
        rel_out_f = rel_dir + 'output/' + domain + '/interpretability/'
        psl_f = rel_dir + 'psl/'
        psl_data_f = psl_f + 'data/' + domain + '/interpretability/'
        if not os.path.exists(psl_data_f):
            os.makedirs(psl_data_f)
        if not os.path.exists(rel_out_f):
            os.makedirs(rel_out_f)
        return ind_pred_f, rel_pred_f, rel_out_f, psl_f, psl_data_f

    def read_predictions(self, ind_pred_f, rel_pred_f):
        """Reads in the predictions from the independent and relational models.
        ind_pred_f: folder to the independent predictions.
        rel_pred_f: folder to the relational predictons.
        Returns predictions for each model in their respective dataframes."""
        fold = self.config_obj.fold
        dset = 'test'

        ind_df = pd.read_csv(ind_pred_f + dset + '_' + fold + '_preds.csv')
        rel_df = pd.read_csv(rel_pred_f + 'predictions_' + fold + '.csv')
        return ind_df, rel_df

    def merge_predictions(self, test_df, ind_df, rel_df):
        """Merges independent and relational prediction dataframes.
        test_df: original testing dataframe.
        ind_df: dataframe with independent predictions.
        rel_df: dataframe with relational predictions.
        Returns merged dataframe."""
        rel_df['rel_pred'] = rel_df['rel_pred'] / rel_df['rel_pred'].max()
        temp_df = test_df.merge(ind_df).merge(rel_df)
        return temp_df

    def show_biggest_improvements(self, df):
        """Presents the user with spam comments that were difficult to detect.
        df: comments dataframe."""
        temp_df = df[df['label'] == 1]
        temp_df['diff'] = temp_df['rel_pred'] - temp_df['ind_pred']
        temp_df = temp_df.sort_values('diff', ascending=False)
        print(temp_df.head(10))

    def gen_group_ids(self, df):
        """Generates any missing group_id columns.
        df: comments dataframe with predictions.
        Returns dataframe with filled in group_ids."""
        for relation, group, group_id in self.config_obj.relations:
            df = self.generator_obj.gen_group_id(df, group_id)
        return df

    def user_input(self, merged_df):
        """Takes input about which comment to show an explanation for.
        merged_df: testing dataframe with predictions.
        Returns comment id of the comment needing an explanation."""
        com_id = None
        com_ids = list(merged_df['com_id'].apply(str))

        while com_id != '-1' and com_id not in com_ids:
            s = '\nEnter com_id for an explanation (-1 to quit): '
            com_id = str(input(s))
        return int(com_id)

    def retrieve_all_connections(self, com_id, expanded_df):
        """Recursively obtain all relations to this comment and all of those
                comments' relations, and so on.
        expanded_df: comments dataframe with multiple same com_id rows.
        com_id: comment to be explained.
        Returns subnetwork of com_ids directly or indirectly connected to
                com_id."""
        debug = True
        possible_relations = self.config_obj.relations
        print('\nextracting subnetwork...')

        connections, relations = self.connections_obj.subnetwork(com_id,
                expanded_df, possible_relations, debug=debug)
        self.relations = [r for r in possible_relations if r[0] in relations]
        print('subnetwork size: ' + str(len(connections)))
        return connections

    def filter_comments(self, merged_df, connections):
        """Filters comments from test set to subnetwork.
        merged_df: test dataframe with predictions.
        connections: set of com_ids in the subnetwork.
        Returns: subnetwork dataframe with predictions."""
        filtered_df = merged_df[merged_df['com_id'].isin(connections)]
        return filtered_df

    def perturb_input(self, df, samples=100, p=1.0):
        """Perturb the independent predictions to generate similar instances.
        df: dataframe containing comments of the subnetwork.
        samples: number of times to perturb the original instance.
        p: amount to perturb each prediction.
        Returns a dataframe with perturbed samples."""
        # perturb = lambda x: max(0, min(1, x + random.uniform(-p, p)))
        temp_df = df.copy()
        perturb = lambda x: random.uniform(0.0, p)

        for i in range(samples):
            temp_df[str(i)] = temp_df['ind_pred'].apply(perturb)
        return temp_df

    def compute_similarity(self, df, samples=100):
        """Computes how similar each perturbed example is from the original.
        df: dataframe with original and perturbed instances.
        samples: number of perturbed samples.
        Returns a dict of similarity scores, a list of sample ids."""
        similarities = {}
        sample_ids = []

        for i in range(samples):
            sample_id = str(i)
            temp_df = df[['ind_pred', sample_id]]
            similarities[sample_id] = pdist(temp_df.values.T)[0]
            sample_ids.append(sample_id)
        return similarities, sample_ids

    def clear_old_data(self, rel_data_f):
        """Clears out old predicate data and database stores.
        rel_data_f: relational model data folder."""
        os.system('rm ' + rel_data_f + '*.csv')
        os.system('rm ' + rel_data_f + '*.tsv')
        os.system('rm ' + rel_data_f + 'db/*.db')

    def write_predicates(self, df, rel_data_f):
        """Writes predicate data for relations to be used by relational model.
        df: subnetwork dataframe.
        rel_data_f: relational model data folder."""
        dset = 'test'
        self.pred_builder_obj.build_comments(df, dset, rel_data_f)
        for relation, group, group_id in self.relations:
            self.pred_builder_obj.build_relations(relation, group, group_id,
                    df, dset, rel_data_f)

    def write_perturbations(self, df, sample_ids, rel_data_f):
        """Writes perturbed instances in a way to be easily loaded by the
                relational model.
        df: subnetwork dataframe.
        rel_data_f: relational model data folder."""
        temp_df = df.filter(items=['com_id'] + sample_ids)
        temp_df.to_csv(rel_data_f + 'perturbed.csv', index=None)

    def compute_labels_for_perturbed_instances(self, com_id, psl_f):
        """Calls relational model to produce labels for the altered instances.
        com_id: id of the comment needing explainaing."""
        fold = self.config_obj.fold
        domain = self.config_obj.domain
        relations = [r[0] for r in self.relations]

        arg_list = [str(com_id), fold, domain] + relations
        execute = 'java -Xmx60g -cp ./target/classes:`cat classpath.out` '
        execute += 'spam.Lime ' + ' '.join(arg_list)

        os.chdir(psl_f)
        os.system(execute)

    def read_perturbed_labels(self, rel_data_f, rel_out_f):
        """Read in the generated labels for the perturbed instances.
        rel_data_f: relational model data folder.
        rel_out_f: relational model output folder.
        Returns dataframe with labels, dataframe with perturbed isntances."""
        fold = self.config_obj.fold
        os.chdir('../scripts/')

        labels_df = pd.read_csv(rel_out_f + 'labels_' + fold + '.csv')
        perturbed_df = pd.read_csv(rel_data_f + 'perturbed.csv')
        return labels_df, perturbed_df

    def preprocess(self, perturbed_df, labels_df, similarities):
        """Processes the perturbed instances, generated labels, and
                perturbed similarities to be fitted by the interpretable model.
        perturbed_df: dataframe of perturbed instances.
        labels_df: dataframe of generated labels for perturbed instances.
        similarities: dict of similarity scores of each perturbed instance
                to the original: (key, value) = (sample_id, sim_score).
        Returns feature data, labels, weights, and feature ids."""
        features = list(perturbed_df['com_id'])

        temp_df = perturbed_df.drop(['com_id'], axis=1)
        temp_df = temp_df.transpose()
        x = temp_df.values

        temp_df = labels_df.filter(items=['pred'])
        y = temp_df.values

        temp_list = list(similarities.items())
        temp_list = sorted(temp_list, key=lambda x: int(x[0]))
        weights = [self.util_obj.div0(1.0, v) for k, v in temp_list]
        return x, y, weights, features

    def fit_linear_model(self, x, y, weights):
        """Fits a linear model using features of the perturbed instances to
                the labels generated for those perturbed instances.
        x: 2d array containing perturbed instance features.
        y: 2d array containing labels for the perturbed instances.
        wgts: 1d array containing weights for each perturbed instance.
        Returns fitted linear model."""
        print('\nFitting linear model...')
        g = LinearRegression()
        g = g.fit(x, y, weights)
        return g

    def extract_and_sort_coefficients(self, g):
        """Sort coefficients by absolute value and then rearrange features to
                line up with their respective coefficient.
        g: fitted linear model.
        Returns list of coefficient indices, list of coefficient values."""
        abs_val = lambda x: abs(x[1])
        coef_list = enumerate(g.coef_[0])  # [(coef_ndx, coef_val), ...]
        coef_sorted = sorted(coef_list, key=abs_val, reverse=True)
        coef_indices = [c[0] for c in coef_sorted]
        coef_values = [c[1] for c in coef_sorted]
        return coef_indices, coef_values

    def rearrange_and_filter_features(self, features, coef_indices,
            coef_values, k=10):
        """Rearrange features to line up with their coefficient values.
        features: attributes used in linear model.
        coef_indices: indices of coeeficients after sorting.
        coef_values: values of coefficients after sorting.
        k: number of features to include in explanation.
        Returns the top k features."""
        features_sorted = [features[ndx] for ndx in coef_indices]  # rearrange
        important_features = list(zip(features_sorted, coef_values))
        top_features = important_features[:k]
        return top_features

    def read_subnetwork_relations(self, rel_data_f):
        """Read in relation files corresponding to comments in the subnetwork.
        rel_data_f: relational model data folder.
        Returns a dictionary containing the relation dataframes."""
        fold = self.config_obj.fold
        relation_dict = {}

        for relation, group, group_id in self.relations:
            filename = rel_data_f + 'test_' + relation + '_' + fold + '.tsv'
            df = pd.read_csv(filename, sep='\t', header=None)
            df.columns = ['com_id', relation]
            relation_dict[relation] = df
        return relation_dict

    def display_raw_instance_to_explain(self, df, com_id):
        """Displays the comment in raw form to the user.
        df: comments dataframe.
        com_id: comment identifier."""
        display = self.config_obj.display
        print('\nInstance to explain:')

        temp_df = df[df['com_id'] == com_id]
        for column in list(temp_df):
            col_val = temp_df[column].values[0]
            col = self.util_obj.colorize(column, 'grey', display)
            print(col + ': ' + str(col_val))

    def display_median_predictions(self, df):
        """Displays the median spam and ham predictions from the independent
                model."""
        spam_ind = df[df['label'] == 1]['ind_pred'].median()
        ham_ind = df[df['label'] == 0]['ind_pred'].median()
        spam_rel = df[df['label'] == 1]['rel_pred'].median()
        ham_rel = df[df['label'] == 0]['rel_pred'].median()

        s = 'median spam: %.4f, median ham: %.4f'
        print('\nLinear explanation:')
        print('(Independent) ' + s % (spam_ind, ham_ind))
        print('(Relational) ' + s % (spam_rel, ham_rel))

    def display_top_features(self, features, df, relation_dict):
        """Shows the interpretable model features that contribute most to the
                original instance prediction.
        features: linear model features.
        df: subnetwork dataframe.
        relation_dict: dictionary of dataframes for every relation."""
        display = self.config_obj.display
        max_val = abs(max(features, key=lambda x: abs(x[1]))[1])

        for feature, value in features:
            ind_pred = df[df['com_id'] == feature]['ind_pred']
            s = 'IndPred(%d) (%.4f): (%7.4f) '

            # adds pictorial view to normalized coefficients.
            bars = ''
            norm_val = int(((value / max_val) * 100.0) / 10)
            color = 'red' if norm_val < 0 else 'green'
            for i in range(abs(norm_val)):
                bars += '-' if color == 'red' else '+'
            s += self.util_obj.colorize(bars, color, display)

            # adds relational evidence information.
            r = ''
            for relation, group, group_id in self.relations:
                rel_df = relation_dict[relation]
                temp_df = rel_df[rel_df['com_id'] == feature]
                if len(temp_df) > 0:
                    vals = [str(x) for x in temp_df[relation].values]
                    rel_id = ' '.join(vals)
                    r += relation + '(' + rel_id + ', ' + str(feature) + '); '

            s += r.rjust(len(r) + (10 - len(bars)) + 2)
            print(s % (int(feature), ind_pred, value))

    def display_subnetwork(self, com_id, df):
        df = self.gen_group_ids(df)

        g = nx.Graph()
        colors = {}
        sizes = {}
        com_df = df[df['com_id'] == com_id]
        ind_val = com_df['ind_pred'].values[0]
        com_node = str(com_id) + ':\n{:.2f}'.format(ind_val)
        g.add_node(com_node)
        colors[com_node] = ind_val

        for rel, group, g_id in self.relations:
            g_df = df[df[g_id].isin(com_df[g_id])]
            if len(g_df) > 1:
                group_val = com_df[g_id].values[0]
                group_node = g_id + ':\n{:d}'.format(group_val)
                colors[group_node] = 0.0
                g.add_edge(com_node, group_node)
                sizes[group_node] = 1000

                for ndx, row in g_df.iterrows():
                    other_id = row['com_id']

                    if other_id != com_id:
                        ind_val = float(row['ind_pred'])
                        other_node = str(other_id) + ':\n{:.2f}'.format(ind_val)
                        g.add_node(other_node)
                        colors[other_node] = ind_val
                        g.add_edge(group_node, other_node)

        color_map = [colors.get(node, 0.25) for node in g.nodes()]
        size_map = [sizes.get(node, 300) for node in g.nodes()]

        print('plot...')
        cmap = plt.get_cmap('bwr')
        nx.draw(g, node_color=color_map, node_size=size_map, with_labels=True,
                cmap=cmap)
        plt.show()
