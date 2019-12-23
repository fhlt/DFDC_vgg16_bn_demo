from sklearn import metrics


def get_roc_auc(y_true, y_pos_score):
    return metrics.roc_auc_score(y_true, y_pos_score)


def get_f1(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred)


def get_acc(y_true, y_pred):
    return metrics.accuracy_score(y_true, y_pred)


def evaluate(y_true, y_pred, y_pos_score):
    acc = get_acc(y_true, y_pred)
    f1 = get_f1(y_true, y_pred)
    roc_auc = get_roc_auc(y_true, y_pos_score)
    return acc, f1, roc_auc
