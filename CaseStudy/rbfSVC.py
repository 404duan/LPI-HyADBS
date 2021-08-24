from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline


def RBFkernelSVC(gamma):
    return Pipeline([
        ("std", StandardScaler()),
        ("svc", SVC(kernel="rbf", gamma=gamma, probability=True))
    ])
