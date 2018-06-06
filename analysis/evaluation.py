"""
This module evaluates the predictions from the independent and relational
models.
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.metrics import average_precision_score


class Evaluation:

    def __init__(self, config_obj, generator_obj, connections_obj, util_obj):
        self.config_obj = config_obj
        self.gen_obj = generator_obj
        self.con_obj = connections_obj
        self.util_obj = util_obj

    # public
    def evaluate(self, test_df):
        """Evaluation of both the indeendent and relational models.
        test_df: testing dataframe."""
        fold = self.config_obj.fold
        test_df = test_df.copy()

        self._settings()
        data_f, ip_f, rp_f, image_f, status_f = self._file_folders()
        preds = self._read_predictions(test_df, ip_f, rp_f)

        ind_df, col, model, line = preds[0]
        df = self._merge_predictions(test_df, ind_df)
        analysis_dict = self._analyze(df, col)
        results_dict = {model: analysis_dict}

        if self.config_obj.analyze_subgraphs:
            df = self._merge_predictions(test_df, ind_df)
            spread_dict = self._spread(df)
            results_dict[model].update(spread_dict)

            df = self._merge_predictions(test_df, ind_df)
            approx_dict = self._approximations(df)
            results_dict[model].update(approx_dict)

        # score_dict = {}
        fname = image_f + 'pr_' + fold
        for pred in preds:
            pred_df, col, model, line = pred
            save = True if pred[1] in preds[-1][1] else False
            # save = True
            scores = self._merge_and_score(test_df, pred, fname, save)
            if model in results_dict.keys():
                results_dict[model].update(scores)
            else:
                results_dict[model] = scores

        return results_dict

    # private
    def _settings(self):
        noise_limit = 0.000025
        self.util_obj.set_noise_limit(noise_limit)

    def _file_folders(self):
        ind_dir = self.config_obj.ind_dir
        rel_dir = self.config_obj.rel_dir
        domain = self.config_obj.domain

        ind_data_f = ind_dir + 'data/' + domain + '/'
        ind_pred_f = ind_dir + 'output/' + domain + '/predictions/'
        rel_out_f = rel_dir + 'output/' + domain + '/'
        rel_pred_f = rel_out_f + 'predictions/'
        rel_image_f = rel_out_f + 'images/'
        status_f = rel_dir + 'output/' + domain + '/status/'
        if not os.path.exists(rel_image_f):
            os.makedirs(rel_image_f)
        if not os.path.exists(status_f):
            os.makedirs(status_f)
        return ind_data_f, ind_pred_f, rel_pred_f, rel_image_f, status_f

    def _read_predictions(self, test_df, ind_pred_f, rel_pred_f, dset='test'):
        fold = self.config_obj.fold
        engine = self.config_obj.engine
        util = self.util_obj
        fname = dset + '_' + fold
        preds = []

        ind_df = util.read_csv(ind_pred_f + fname + '_preds.csv')
        if ind_df is not None and len(ind_df) == len(test_df):
            preds.append((ind_df, 'ind_pred', 'ind', 'b--'))

        if engine in ['psl', 'all']:
            psl_df = util.read_csv(rel_pred_f + 'psl_preds_' + fold + '.csv')
            if psl_df is not None and len(psl_df) == len(test_df):
                preds.append((psl_df, 'psl_pred', 'psl', 'g:'))

        if engine in ['mrf', 'all']:
            mrf_df = util.read_csv(rel_pred_f + 'mrf_preds_' + fold + '.csv')
            if mrf_df is not None and len(mrf_df) == len(test_df):
                preds.append((mrf_df, 'mrf_pred', 'mrf', 'c-.'))

        return preds

    def _merge_and_score(self, test_df, pred, fname, save=False):
        pred_df, col, name, line = pred

        merged_df = self._merge_predictions(test_df, pred_df)

        noise_df = self._apply_noise(merged_df, col)
        pr, roc, r, p, npr = self._compute_scores(noise_df, col)
        self.util_obj.plot_pr_curve(name, fname, r, p, npr, line=line,
                                    save=save)
        scores = {'aupr': round(pr, 7), 'auroc': round(roc, 7),
                  'naupr': round(npr, 7)}
        return scores

    def _merge_predictions(self, test_df, pred_df):
        merged_df = test_df.merge(pred_df, on='com_id', how='left')
        return merged_df

    def _apply_noise(self, merged_df, col):
        merged_df[col] = merged_df[col].apply(self.util_obj.gen_noise)
        return merged_df

    def _compute_scores(self, pf, col):
        fpr, tpr, _ = roc_curve(pf['label'], pf[col])
        prec, rec, _ = precision_recall_curve(pf['label'], pf[col])
        nPre, nRec, _ = precision_recall_curve(pf['label'], 1 - pf[col],
                                               pos_label=0)
        auroc, aupr, nAupr = auc(fpr, tpr), auc(rec, prec), auc(nRec, nPre)
        return aupr, auroc, rec, prec, nAupr

    def _print_scores(self, name, aupr, auroc, naupr):
        s = name + ' evaluation...AUPR: %.4f, AUROC: %.4f, N-AUPR: %.4f'
        self.util_obj.out(s % (aupr, auroc, naupr))

    def _analyze(self, df, col, samples=100):
        ut = self.util_obj
        relations = self.config_obj.relations
        gids = [r[2] for r in relations]

        if len(relations) == 0:
            return {}

        ut.out('\nANALYSIS...\n')

        t1 = ut.out('computing messages missed most often...')

        p, r, ts = precision_recall_curve(df['label'], df[col])
        aupr = average_precision_score(df['label'], df[col])
        mp = 1.0 - aupr

        corrects = []
        step = int(len(ts) / 100) if len(ts) > 100 else 1
        for i in range(0, len(ts), step):
            t = ts[i]
            df['pred'] = np.where(df[col] > t, 1, 0)
            correct = df['pred'] == df['label']
            corrects.append(correct.apply(int))

        total_corrects = [sum(x) for x in zip(*corrects)]
        df['correct'] = total_corrects

        # extract bottom x% data
        df = df.sort_values('correct', ascending=False)
        ndx = len(df) - int(len(df) * mp)
        qf1, qf2 = df[ndx:], df[:ndx]
        # dfs = df[df['label'] == 1]
        qf1s = qf1[qf1['label'] == 1]  # low performers
        qf1o = qf1[qf1['label'] == 0]  # low performers
        qf2s = qf2[qf2['label'] == 1]  # high performers
        qf2o = qf2[qf2['label'] == 0]  # high performers
        ut.time(t1)

        # ut.out('spam in bot %.2f%%: %d' % (mp * 100, len(qf1s)))
        # ut.out('ham in bot %.2f%%: %d' % (mp * 100, len(qf1o)))

        t1 = ut.out('computing messages with a relation...')
        r1s, r1sf = self._msgs_with_rel(qf1s, gids, mp, 'bot', 'spam')
        r1o, r1of = self._msgs_with_rel(qf1o, gids, mp, 'bot', 'ham')
        r2s, r2sf = self._msgs_with_rel(qf2s, gids, mp, 'top', 'spam')
        r2o, r2of = self._msgs_with_rel(qf2o, gids, mp, 'top', 'ham')
        ut.time(t1)

        # ut.out()

        t1 = ut.out('computing messages with an outside relation...')
        rr1sof = self._rm_in_sect(df, qf1s, qf2, gids, mp, r1s, 'bot', 'spam')
        rr1oof = self._rm_in_sect(df, qf1o, qf2, gids, mp, r1o, 'bot', 'ham')
        rr2sof = self._rm_in_sect(df, qf2s, qf1, gids, mp, r2s, 'top', 'spam')
        rr2oof = self._rm_in_sect(df, qf2o, qf1, gids, mp, r2o, 'top', 'ham')
        # rr1sif = self._rm_in_sect(df, qf1s, qf1, gids, mp, r1s, 'bot', 'spam',
        #                           'inside')
        # rr1oif = self._rm_in_sect(df, qf1o, qf1, gids, mp, r1o, 'bot', 'ham',
        #                           'inside')

        sd = {}
        sd['bot_spam_rels'] = round(r1sf, 4)
        sd['bot_ham_rels'] = round(r1of, 4)
        sd['top_spam_rels'] = round(r2sf, 4)
        sd['top_ham_rels'] = round(r2of, 4)
        sd['bot_spam_rels_out'] = round(rr1sof, 4)
        sd['bot_ham_rels_out'] = round(rr1oof, 4)
        sd['top_spam_rels_out'] = round(rr2sof, 4)
        sd['top_ham_rels_out'] = round(rr2oof, 4)
        # sd['bot_spam_rels_in'] = rr1sif
        # sd['bot_ham_rels_in'] = rr1oif

        ut.time(t1)
        return sd

    def _msgs_with_rel(self, df, gids, miss_pct, sect='bot', lbl='spam'):
        ut = self.util_obj
        n = len(df[(df[gids] != -1).any(axis=1)])
        frac = ut.div0(n, len(df))
        # sect_pct = 1 - miss_pct if sect == 'top' else miss_pct
        # t = (sect, sect_pct * 100, lbl, frac * 100)
        # ut.out('%s %.2f%% %s w/ relations: %.2f%%' % t)
        return n, frac

    def _rm_in_sect(self, df, tgt_df, cmp_df, gids, miss_pct, num_rel_msgs,
                    sect='bot', lbl='spam', boundary='outside'):
        ut = self.util_obj

        fraction = set()
        tgt_msgs = set(tgt_df['com_id'])
        cmp_msgs = set(cmp_df['com_id'])

        for gid in gids:
            tgt_gids = {x for x in tgt_df[gid] if x != -1}

            for temp_gid, qf in df.groupby(gid):
                if temp_gid in tgt_gids:
                    grp_msgs = set(qf['com_id'])
                    other = grp_msgs.intersection(cmp_msgs)
                    if len(other) > 0:
                        fraction.update(grp_msgs.intersection(tgt_msgs))

        n = ut.div0(len(fraction), num_rel_msgs)
        # sect_pct = 1 - miss_pct if sect == 'top' else miss_pct
        # t = (sect, sect_pct * 100, lbl, boundary, sect, n * 100)
        # self.util_obj.out('%s %.2f%% %s w/ rels %s %s: %.2f%%' % t)
        return n

    def _spread(self, df, col='ind_pred'):
        """This'll give some post-hoc test-set analysis, when running this,
        keep track of the test sets that improved using relational modeling,
        then average those test set statistics together to compare to the test
        sets that did not improve."""

        ut = self.util_obj
        t1 = ut.out('computing subgraph statistics...')

        relations = self.config_obj.relations
        gids = [r[2] for r in relations]
        g, sgs = self.con_obj.find_subgraphs(df, relations, verbose=False)
        spread_dict = {}

        sg_list = []
        for i, sg in enumerate(sgs):
            if sg[3] > 0:  # num edges > 0
                sg_list.extend([(x, i) for x in sg[0]])  # give sg_id

        if len(sg_list) == 0:
            return spread_dict

        sg_df = pd.DataFrame(sg_list, columns=['com_id', 'sg_id'])
        df = df.merge(sg_df, how='left')
        df['sg_id'] = df['sg_id'].fillna(-1).apply(int)

        p, r, ts = precision_recall_curve(df['label'], df[col])
        aupr = average_precision_score(df['label'], df[col])
        mp = 1.0 - aupr

        corrects = []
        step = int(len(ts) / 100) if len(ts) > 100 else 1
        for i in range(0, len(ts), step):
            t = ts[i]
            df['pred'] = np.where(df[col] > t, 1, 0)
            correct = df['pred'] == df['label']
            corrects.append(correct.apply(int))

        total_corrects = [sum(x) for x in zip(*corrects)]
        df['correct'] = total_corrects

        # extract bottom x% data
        df = df.sort_values('correct', ascending=False)
        ndx = len(df) - int(len(df) * mp)
        qfs = df[df['label'] == 1]
        qfo = df[df['label'] == 0]
        qf1, qf2 = df[ndx:], df[:ndx]
        qf1s = qf1[qf1['label'] == 1]  # low performers
        qf1o = qf1[qf1['label'] == 0]  # low performers
        qf2s = qf2[qf2['label'] == 1]  # high performers
        qf2o = qf2[qf2['label'] == 0]  # high performers

        spread_dict['spam_mean'] = round(qfs['ind_pred'].mean(), 4)
        spread_dict['spam_median'] = round(qfs['ind_pred'].median(), 4)
        spread_dict['ham_mean'] = round(qfo['ind_pred'].mean(), 4)
        spread_dict['ham_median'] = round(qfo['ind_pred'].median(), 4)

        for nm, temp_df in [('bot_spam', qf1s), ('bot_ham', qf1o),
                            ('top_spam', qf2s), ('top_ham', qf2o)]:
            wf = temp_df[(temp_df[gids] != -1).any(axis=1)]
            sg_mean = wf.groupby('sg_id')['ind_pred'].mean().reset_index()\
                .rename(columns={'ind_pred': 'sg_mean'})
            sg_std = wf.groupby('sg_id')['ind_pred'].std().reset_index()\
                .rename(columns={'ind_pred': 'sg_std'})
            sg_median = wf.groupby('sg_id')['ind_pred'].median().reset_index()\
                .rename(columns={'ind_pred': 'sg_median'})
            sg_min = wf.groupby('sg_id')['ind_pred'].min().reset_index()\
                .rename(columns={'ind_pred': 'sg_min'})
            sg_max = wf.groupby('sg_id')['ind_pred'].max().reset_index()\
                .rename(columns={'ind_pred': 'sg_max'})
            wf = wf.merge(sg_mean).merge(sg_std).merge(sg_median)\
                .merge(sg_min).merge(sg_max)
            wf['sg_spread'] = wf['sg_max'] - wf['sg_min']

            spread_dict[nm + '_sg_mean'] = round(np.mean(wf['sg_mean']), 4)
            spread_dict[nm + '_sg_std'] = round(np.mean(wf['sg_std']), 4)
            spread_dict[nm + '_sg_median'] = round(np.mean(wf['sg_median']), 4)
            spread_dict[nm + '_sg_min'] = round(np.mean(wf['sg_min']), 4)
            spread_dict[nm + '_sg_max'] = round(np.mean(wf['sg_max']), 4)
            spread_dict[nm + '_sg_spread'] = round(np.mean(wf['sg_spread']), 4)

        ut.time(t1)
        return spread_dict

    def _approximations(self, df):
        ut = self.util_obj
        t1 = ut.out('approximating relational with mean, max, median...')

        relations = self.config_obj.relations
        g, sgs = self.con_obj.find_subgraphs(df, relations, verbose=False)
        approx_dict = {}

        sg_list = []
        for i, sg in enumerate(sgs):
            if sg[3] > 0:  # num edges > 0
                sg_list.extend([(x, i) for x in sg[0]])  # give sg_id

        if len(sg_list) == 0:
            return approx_dict

        sg_df = pd.DataFrame(sg_list, columns=['com_id', 'sg_id'])
        df = df.merge(sg_df, how='left')
        df['sg_id'] = df['sg_id'].fillna(-1).apply(int)

        sg_mean = df.groupby('sg_id')['ind_pred'].mean().reset_index()\
            .rename(columns={'ind_pred': 'sg_mean_pred'})
        sg_median = df.groupby('sg_id')['ind_pred'].median().reset_index()\
            .rename(columns={'ind_pred': 'sg_median_pred'})
        sg_max = df.groupby('sg_id')['ind_pred'].max().reset_index()\
            .rename(columns={'ind_pred': 'sg_max_pred'})
        df = df.merge(sg_mean).merge(sg_median).merge(sg_max)

        filler = lambda x, c: x['ind_pred'] if x['sg_id'] == -1 else x[c]
        for col in ['sg_mean_pred', 'sg_median_pred', 'sg_max_pred']:
            cols = ['ind_pred', col, 'sg_id']
            df[col] = df[cols].apply(filler, axis=1, args=(col,))
            approx_dict[col] = round(average_precision_score(df['label'],
                                     df[col]), 4)

        ut.time(t1)
        return approx_dict
