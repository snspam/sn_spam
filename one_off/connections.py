"""
Module to find subnetwork of a data points based on its relationships.
"""
import math
import numpy as np
import networkx as nx
from collections import Counter
import util as ut


class Connections:

    def __init__(self):
        self.size_threshold = 100

    # public
    def build_networkx_graph(self, df, relations):
        g = nx.Graph()
        h = {h: i + 1 for i, h in enumerate(list(df))}

        for r in df.itertuples():
            msg_id = str(r[h['com_id']]) + '+' + str(r[h['label']])

            g.add_node(msg_id)

            for rel, group, g_id in relations:
                gids = r[h[g_id]]

                if type(gids) == list:
                    for gid in gids:
                        g.add_edge(msg_id, rel + '_' + str(gid))
                elif gids != -1:
                    g.add_edge(msg_id, rel + '_' + str(gids))
        return g

    def build_networkx_graph_rdfs(self, r_dfs, relations):
        g = nx.Graph()

        rdict = {g_id: rel for rel, group, g_id in relations}

        for r_df in r_dfs:
            rel = list(r_df)[1]
            for r in r_df.itertuples():
                msg_id, g_id = r[1], r[2]
                g.add_edge(msg_id, rdict[rel] + '_' + str(g_id))
        return g

    def consolidate(self, subgraphs, max_size=40000, div=2):
        """Combine subgraphs into larger sets to reduce total number of
        subgraphs to do inference over."""
        t1 = ut.out('consolidating subgraphs...')

        sgs = []
        new_ids, new_hubs = set(), set()
        new_rels, new_edges = set(), 0

        for ids, hubs, rels, edges in subgraphs:
            size = int(len(new_ids) / div) + int(len(ids) / div)
            size += new_edges + edges

            if size < max_size:  # keep adding to new
                new_ids.update(ids)
                new_rels.update(rels)
                new_hubs.update(hubs)
                new_edges += edges
            elif new_edges == 0 and size > max_size:  # subgraph too big
                new_ids.update(ids)
                new_hubs.update(hubs)
                new_rels.update(rels)
                new_edges += edges
            else:  # new is full
                sgs.append((new_ids, new_hubs, new_rels, new_edges))
                new_ids, new_hubs = ids, hubs
                new_rels, new_edges = rels, edges

        if len(new_ids) > 0:
            sgs.append((new_ids, new_hubs, new_rels, new_edges))

        ut.time(t1)
        self._print_subgraphs_size(sgs)

        return sgs

    def find_subgraphs(self, df, relations, max_size=40000, verbose=False):
        if verbose:
            t1 = ut.out('finding subgraphs...')

        if verbose:
            t1 = ut.out('building networkx graph...')
        g = self.build_networkx_graph(df, relations)
        ccs = list(nx.connected_components(g))
        if verbose:
            ut.time(t1)

        if verbose:
            t1 = ut.out('processing connected components...')
        subgraphs = self._process_components(ccs, g)
        if verbose:
            ut.time(t1)

        # t1 = ut.out('filtering redundant subgraphs...')
        # subgraphs = self._filter_redundant_subgraphs(subgraphs, df)
        # ut.time(t1)

        # t1 = ut.out('removing single edge hubs...')
        # subgraphs = self._remove_single_edge_hubs(subgraphs, g)
        # ut.time(t1)

        # t1 = ut.out('compiling single node subgraphs...')
        # subgraphs += self._single_node_subgraphs(subgraphs, df, max_size)
        # ut.time(t1)

        if verbose:
            self._print_subgraphs_size(subgraphs)
        return g, subgraphs

    def find_target_subgraph(self, com_id, df, relations, debug=False):
        """Public interface to find a subnetwork given a specified comment.
        com_id: identifier of target comment.
        df: comments dataframe.
        relations: list of relations as tuples.
        debug: boolean to print extra information.
        Returns set of comment ids in subnetwork, and relations used."""
        direct, rels, edges = self._direct_connections(com_id, df, relations)

        if len(direct) < self.size_threshold:
            result = self._group(com_id, df, relations)
        else:
            result = self._iterate(com_id, df, relations)
        return result

    # private
    def _aggregate_single_node_subgraphs(self, subnets):
        no_rel_ids = [s.pop() for s, r, e in subnets if r == set()]
        no_rel_sg = (set(no_rel_ids), set(), 0)
        rel_sgs = [(s, r, e) for s, r, e in subnets if r != set()]

        subgraphs = rel_sgs.copy()
        subgraphs.append(no_rel_sg)
        return subgraphs

    def _direct_connections(self, com_id, df, possible_relations):
        com_df = df[df['com_id'] == com_id]
        subnetwork, relations, edges = set(), set(), 0

        list_filter = lambda l, v: True if v in l else False

        for relation, group, group_id in possible_relations:
            g_vals = com_df[group_id].values

            if len(g_vals) > 0:
                vals = g_vals[0]

                for val in vals:
                    rel_df = df[df[group_id].apply(list_filter, args=(val,))]
                    if len(rel_df) > 1:
                        relations.add(relation)
                        subnetwork.update(set(rel_df['com_id']))
                        edges += 1
        return subnetwork, relations, edges

    def _filter_redundant_subgraphs(self, subgraphs, df, rel_tol=1e-2):
        filtered_subgraphs = []

        for msg_nodes, hub_nodes, rels, edges in subgraphs:
            redundant = True

            qf = df[df['com_id'].isin(msg_nodes)]
            mean = qf['ind_pred'].mean()
            preds = list(qf['ind_pred'])

            for p in preds:
                if not math.isclose(p, mean, rel_tol=rel_tol):
                    redundant = False

            if not redundant:
                filtered_subgraphs.append((msg_nodes, hub_nodes, rels, edges))
        return filtered_subgraphs

    def _single_node_subgraphs(self, subgraphs, df, max_size=69000):
        single_node_subgraphs = []

        msgs = set()
        for msg_nodes, hub_nodes, rels, edges in subgraphs:
            msgs.update(msg_nodes)

        qf = df[~df['com_id'].isin(msgs)]
        msg_nodes = set(qf['com_id'])

        if len(msg_nodes) <= max_size:
            single_node_subgraphs.append((msg_nodes, set(), set(), 0))
        else:
            num_splits = int(len(msg_nodes) / max_size)
            num_splits = 2 if num_splits == 1 else num_splits
            single_node_lists = np.array_split(list(msg_nodes), num_splits)
            for msgs in single_node_lists:
                single_node_subgraphs.append((set(msgs), set(), set(), 0))

        return single_node_subgraphs

    def _group(self, target_id, df, relations, debug=False):
        subnetwork = set({target_id})
        frontier, direct_connections = set({target_id}), set()
        relations_present = set()
        edges = 0
        tier = 1

        while len(frontier) > 0:
            com_id = frontier.pop()
            connections, r, e = self._direct_connections(com_id, df, relations)
            unexplored = [c for c in connections if c not in subnetwork]

            # switch to iteration method if subnetwork is too large.
            if len(connections) >= self.size_threshold:
                return self._iterate(target_id, df, relations, debug)

            # update sets.
            subnetwork.update(unexplored)
            direct_connections.update(unexplored)
            relations_present.update(r)
            edges += e

            # tier is exhausted, move to next level.
            if len(frontier) == 0 and len(direct_connections) > 0:
                id_df = df[df['com_id'].isin(list(direct_connections))]
                id_list = id_df['com_id'].values
                frontier = direct_connections.copy()
                direct_connections.clear()

                if debug:
                    print('\nTier ' + str(tier))
                    print(id_list, len(id_list))
                tier += 1

        return subnetwork, relations_present, edges

    def _iterate(self, com_id, df, relations, debug=False):
        """Finds all comments directly and indirectly connected to com_id.
        com_id: identifier of target comment.
        df: comments dataframe.
        relations: list of relations as tuples.
        debug: boolean to print extra information.
        Returns set of comment ids in subnetwork, and relations used."""
        g_ids = [r[2] for r in relations]
        com_df = df[df['com_id'] == com_id]

        first_pass = True
        converged = False
        passes = 0

        headers = list(df)
        h = {h: i + 1 for i, h in enumerate(headers)}

        # init group values as sets in a dict.
        g_dict, g_cnt = {}, {}
        for rel, g, g_id in relations:
            g_vals = com_df[g_id].values
            vals = g_vals[0] if len(g_vals) > 0 else []
            g_dict[g_id] = set(vals)
            g_cnt[g_id] = 0

        total_cc = set()
        edges = 0

        while first_pass or not converged:
            passes += 1
            cc = set({com_id})

            if debug:
                print('pass ' + str(passes))

            for r in df.itertuples():
                connected = False

                # method
                for g_id in g_ids:
                    r_vals = r[h[g_id]]

                    if any(r_val in g_dict[g_id] for r_val in r_vals):
                        connected = True
                        cc.add(r[h['com_id']])
                        if r[h['com_id']] not in total_cc\
                                and r[h['com_id']] != com_id:
                            g_cnt[g_id] += 1
                        break

                # method
                if connected is True:
                    for g_id in g_ids:
                        r_vals = r[h[g_id]]
                        [g_dict[g_id].add(r_val) for r_val in r_vals]

            if first_pass:
                first_pass = False
                total_cc = cc
            else:
                if len(cc) != len(total_cc):
                    total_cc = cc
                else:
                    converged = True
        rels = set([r for r, g, gid in relations if g_cnt[gid] > 0])
        for g_id, ids in g_dict.items():
            edges += len(ids)
        return total_cc, rels, edges

    def _print_subgraphs_size(self, subgraphs):
        tot_m, tot_h, tot_e = 0, 0, 0

        for ids, hubs, rels, edges in subgraphs:
            tot_m += len(ids)
            tot_h += len(hubs)
            tot_e += edges

        t = (len(subgraphs), tot_m, tot_h, tot_e)
        ut.out('subgraphs: %d, msgs: %d, hubs: %d, edges: %d' % t)

    def _process_components(self, ccs, g):
        subgraphs = []

        for cc in ccs:
            rels, edges = set(), 0
            msg_nodes = {x for x in cc if '_' not in str(x)}
            hub_nodes = {x for x in cc if '_' in str(x)}

            for hub_node in hub_nodes:
                rels.add(hub_node.split('_')[0])
                edges += g.degree(hub_node)

            subgraphs.append((msg_nodes, hub_nodes, rels, edges))

        return subgraphs

    def _remove_single_edge_hubs(self, subgraphs, g):
        new_subgraphs = []

        for msg_nodes, hub_nodes, rels, edges in subgraphs:
            all_nodes = msg_nodes.union(hub_nodes)
            new_hub_nodes = hub_nodes.copy()
            sg = g.subgraph(all_nodes)

            for n in sg:
                if '_' in str(n):
                    hub_deg = sg.degree(n)

                    if hub_deg == 1:
                        new_hub_nodes.remove(n)
                        edges -= 1

            new_subgraphs.append((msg_nodes, new_hub_nodes, rels, edges))
        return new_subgraphs

    def _validate_subgraphs(self, subgraphs):
        id_list = []

        for ids, hubs, rels, edges in subgraphs:
            id_list.extend(list(ids))

        for v in Counter(id_list).values():
            assert v == 1
