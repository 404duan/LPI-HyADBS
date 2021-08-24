from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


def F1(y_pred, y_true):
    return f1_score(y_true, y_pred)

def recall(y_pred, y_true):
    return recall_score(y_true, y_pred)

def precision(y_pred, y_true):
    return precision_score(y_true, y_pred)

def accuracy(y_proba, y_true):
    y_proba = y_proba > 0.5
    acc = (y_proba == y_true).mean()
    return acc

def auc(y_proba, y_true, i, j):
    fpr, tpr, _ = metrics.roc_curve(y_true, y_proba)
    auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr)
    plt.title("No.{} Time No.{} Fold AUC: {:.4f}".format(i+1, j+1, auc))
    plt.savefig("./Figure/AUC_Time{}_Fold{}.jpg".format(i+1, j+1))
    plt.close()
    return auc, fpr, tpr

def aupr(y_proba, y_true, i, j):
    pre, rec, _ = metrics.precision_recall_curve(y_true, y_proba)
    aupr = metrics.auc(rec, pre)
    plt.plot(rec, pre)
    plt.title("No.{} Time No.{} Fold AUPR: {:.4f}".format(i+1, j+1, aupr))
    plt.savefig("./Figure/AUPR_Time{}_Fold{}.jpg".format(i+1, j+1))
    plt.close()
    return aupr, pre, rec
