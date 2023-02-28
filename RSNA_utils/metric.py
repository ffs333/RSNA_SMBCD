import numpy as np


def pfbeta(labels, preds, beta=1):
    preds = preds.clip(0, 1)
    y_true_count = labels.sum()
    ctp = preds[labels == 1].sum()
    cfp = preds[labels == 0].sum()
    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if c_precision > 0 and c_recall > 0:
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0.0


def get_df_scores_patient(val_folds):
    best_, best_raw = 0, 0
    best_params = None

    score_max, params, score_raw = get_scores(val_folds['prediction_max'].values, val_folds['cancer'].values,
                                              agg='max', best_cur=best_, use_prints=False)

    if score_max > best_:
        best_ = score_max
        best_params = params

    if score_raw > best_raw:
        best_raw = score_raw

    score_mean, params, score_raw = get_scores(val_folds['prediction_mean'].values, val_folds['cancer'].values,
                                               agg='mean', best_cur=best_, use_prints=False)
    if score_mean > best_:
        best_ = score_mean
        best_params = params

    if score_raw > best_raw:
        best_raw = score_raw

    score_median, params, score_raw = get_scores(val_folds['prediction_median'].values, val_folds['cancer'].values,
                                                 agg='median', best_cur=best_, use_prints=False)
    if score_median > best_:
        best_ = score_median
        best_params = params

    if score_raw > best_raw:
        best_raw = score_raw

    print(f'Best params: {best_params}')
    print(f'Best RAW score: {best_raw}')
    return best_, best_params, best_raw


def get_df_scores(val_folds):

    best_, best_raw = 0, 0
    best_params = None
    df_max = val_folds.groupby(by=['patient_id', 'laterality']).max()
    df_mean = val_folds.groupby(by=['patient_id', 'laterality']).mean()
    df_median = val_folds.groupby(by=['patient_id', 'laterality']).median()

    score_max, params, score_raw = get_scores(df_max['prediction'].values, df_max['cancer'].values,
                                              agg='max', best_cur=best_, use_prints=False)
    if score_max > best_:
        best_ = score_max
        best_params = params

    if score_raw > best_raw:
        best_raw = score_raw

    score_mean, params, score_raw = get_scores(df_mean['prediction'].values, df_mean['cancer'].values,
                                               agg='mean', best_cur=best_, use_prints=False)
    if score_mean > best_:
        best_ = score_mean
        best_params = params

    if score_raw > best_raw:
        best_raw = score_raw

    score_median, params, score_raw = get_scores(df_median['prediction'].values, df_median['cancer'].values,
                                                 agg='median', best_cur=best_, use_prints=False)
    if score_median > best_:
        best_ = score_median
        best_params = params

    if score_raw > best_raw:
        best_raw = score_raw

    print(f'Best params: {best_params}')
    print(f'Best RAW score: {best_raw}')
    return best_, best_params, best_raw


def get_scores(preds, labels, agg, best_cur=0, use_prints=False):
    thresholds = np.arange(0.005, 0.995, 0.005)
    params = {'agg': agg, 'variant': None, 'threshold': None, 'score': best_cur}

    raw_score = pfbeta(labels, preds)
    if raw_score > best_cur:
        best_cur = raw_score
        params = {'agg': agg, 'variant': 1, 'threshold': None, 'score': best_cur}
        if use_prints:
            print(params)

    for thr in thresholds:
        cur_score = pfbeta(labels, np.where(preds > thr, 1, 0))
        if cur_score > best_cur:
            best_cur = cur_score
            params = {'agg': agg, 'variant': 2, 'threshold': thr, 'score': best_cur}
            if use_prints:
                print(params)

    return best_cur, params, raw_score


def get_scores_attn(preds, labels, use_prints=False):
    thresholds = np.arange(0.005, 0.995, 0.0025)
    best_cur = 0
    params = {'agg': 'attention', 'variant': None, 'threshold': None, 'score': best_cur}

    raw_score = pfbeta(labels, preds)
    if raw_score > best_cur:
        best_cur = raw_score
        params = {'agg': 'attention', 'variant': 1, 'threshold': None, 'score': best_cur}
        if use_prints:
            print(params)

    for thr in thresholds:
        cur_score = pfbeta(labels, np.where(preds > thr, 1, 0))
        if cur_score > best_cur:
            best_cur = cur_score
            params = {'agg': 'attention', 'variant': 2, 'threshold': thr, 'score': best_cur}
            if use_prints:
                print(params)

    return best_cur, params, raw_score


def get_metric_infer(dc, val_folds):
    if dc['agg'] == 'max':
        vl_df = val_folds.groupby(by=['patient_id', 'laterality']).max()
    elif dc['agg'] == 'mean':
        vl_df = val_folds.groupby(by=['patient_id', 'laterality']).mean()
    elif dc['agg'] == 'median':
        vl_df = val_folds.groupby(by=['patient_id', 'laterality']).median()
    else:
        raise KeyError(f'Wrong "agg" : {dc["agg"]}')

    preds, labels = vl_df['prediction'].values, vl_df['cancer'].values
    if dc['variant'] == 1:
        score = pfbeta(labels, preds)
    elif dc['variant'] == 2:
        score = pfbeta(labels, np.where(preds > dc['threshold'], 1, 0))
    elif dc['variant'] == 3:
        score = pfbeta(labels, np.where(preds > dc['threshold'], 1, preds))
    elif dc['variant'] == 4:
        score = pfbeta(labels, np.where(preds < dc['threshold'], 0, preds))
    elif dc['variant'] == 5:
        now_pred = np.where(preds < dc['threshold'][0], 0, preds)
        now_pred = np.where(now_pred > dc['threshold'][1], 1, now_pred)
        score = pfbeta(labels, now_pred)
    else:
        raise KeyError(f'Wrong "variant" : {dc["variant"]}')
    return score
