"""
Module to draw relational connected components in inference graph.
"""
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Draw:

    def __init__(self, util_obj):
        self.util_obj = util_obj

    # public
    def draw_graphs(self, df, g, subgraphs, rels, dir='graphs/',
                    col='ind_pred', rel_margs=None):
        self.util_obj.create_dirs(dir)

        # filtering
        subgraphs = [s for s in subgraphs if s[3] != 0]  # no edges
        subgraphs = self._filter_non_target_subgraphs(subgraphs, df)

        for i, (msgs, hubs, relations, edges) in enumerate(subgraphs):
            self.util_obj.out('drawing subgraph %d...' % i)
            sg_nodes = msgs.union(hubs)
            sg = g.subgraph(sg_nodes)

            f, (ax1, ax2) = plt.subplots(1, 2)
            self._draw_graph(df, sg, rels, ax=ax1, col='ind_pred')
            self._draw_graph(df, sg, rels, ax=ax2, col=col, rm=rel_margs)

            f.suptitle('test subgraph %d - only nodes with rels' % i)
            f.savefig(dir + 'sg_%d.pdf' % i, format='pdf', bbox_inches='tight')

            plt.clf()
            plt.close('all')

            if i == 25:
                break

    # private
    def _draw_graph(self, df, sg, rels, col='ind_pred', ax=None, rm=None):
        clist = ['purple', 'blue', 'yellow', 'teal', 'green']
        pos = nx.shell_layout(sg)

        # draw msg nodes
        n, nc, nl = self.get_node_map(sg, df, 'msg', col=col)
        ndeg = sg.degree(n)
        nx.draw_networkx_nodes(sg, nodelist=n, node_color=nc,
                               pos=pos, font_size=6, node_shape='s',
                               node_size=[v * 50 for v in ndeg.values()],
                               label='msgs', ax=ax)
        nx.draw_networkx_labels(sg, labels=nl, pos=pos, font_size=6, ax=ax)

        # draw hub nodes
        for j, (rel, group, g_id) in enumerate(rels):
            h, hc, hl = self.get_node_map(sg, df, 'hub', clist[j], rel, rm=rm)
            hdeg = sg.degree(h)
            nx.draw_networkx_nodes(sg, nodelist=h, node_color=hc,
                                   pos=pos, font_size=6,
                                   node_size=[v * 50 for v in hdeg.values()],
                                   label=group, ax=ax)
            nx.draw_networkx_labels(sg, labels=hl, pos=pos, font_size=6, ax=ax)

        # draw edges
        ec = self.get_edge_map(sg, rels)
        nx.draw_networkx_edges(sg, edge_color=ec, pos=pos, alpha=0.8, ax=ax)

        # ax specific settings
        ax.set_title(col)
        ax.legend(prop={'size': 6})

    def _filter_non_target_subgraphs(self, subgraphs, df):
        filtered = []

        for subgraph in subgraphs:
            msg_nodes = subgraph[0]
            qf = df[df['com_id'].isin(msg_nodes)]

            if qf['label'].sum() > 0:
                filtered.append(subgraph)
        return filtered

    def get_edge_map(self, g, relations):
        colors = []
        clist = ['purple', 'blue', 'yellow', 'teal', 'green']
        cd = {r[0]: clist[i] for i, r in enumerate(relations)}

        for n1, n2 in g.edges():
            if '_' in str(n1):
                assert '_' not in str(n2)
                color = cd[n1.split('_')[0]]
                colors.append(color)
            elif '_' in str(n2):
                assert '_' not in str(n1)
                color = cd[n2.split('_')[0]]
                colors.append(color)
        return colors

    def get_node_map(self, g, df, type='msg', color='green', rel='',
                     col='ind_pred', rm=None):
        nodes, colors, labels = [], [], {}
        cmap = plt.get_cmap('Reds')

        for n in g:
            if type == 'msg':
                if '_' not in str(n):
                    com_df = df[df['com_id'] == n]
                    ind_pred = com_df[col].values[0]
                    label = com_df['label'].values[0]
                    nodes.append(n)
                    colors.append(list(cmap(ind_pred)))
                    labels[n] = '%.2f:%d' % (ind_pred, label)
                    labels[n] += '\n' + str(n)
            elif type == 'hub':
                if '_' in str(n) and n.split('_')[0] == rel:
                    nodes.append(n)
                    colors.append(color)
                    labels[n] = rm.get(n, '') if rm is not None else ''
                    labels[n] = str(labels[n]) + '\n' + str(n)
        return nodes, colors, labels
