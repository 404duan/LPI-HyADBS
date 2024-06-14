from torch.optim import Adam
from torch.nn import BCELoss
from model import Model
from loadData import loadData, loadcv4Data, loadcv3Data, loadcv2Data, loadcv1Data
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import evaluate
import numpy as np
import pandas as pd


def get_model(lr, inputNode, hidden, outputNode):
    model = Model(inputNode, hidden, outputNode)
    opt = Adam(model.parameters(), lr=lr)
    return model, opt


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 0.0001
loss_fn = BCELoss()
batch = 128
epochs = 100


def KFold(K):
    preArray = np.zeros((20, K), dtype=float)
    recallArray = np.zeros((20, K), dtype=float)
    accArray = np.zeros((20, K), dtype=float)
    F1Array = np.zeros((20, K), dtype=float)
    aucArray = np.zeros((20, K), dtype=float)
    auprArray = np.zeros((20, K), dtype=float)
    for i in range(20):
        mm = 1
        cv = 3
        cross_x, cross_y = loadcv3Data("./data/"+str(mm)+"/")
        for j in range(K):
            train_x = []
            train_y = []
            test_x = []
            test_y = []
            for itest_x in cross_x[j]:
                test_x.extend(itest_x)
            for itest_y in cross_y[j]:
                test_y.extend(itest_y)
            test_x = np.array(test_x, dtype=float).reshape(-1, 653)
            test_y = np.array(test_y, dtype=int).reshape(-1, 1)
            train_x.extend(cross_x[c] for c in range(len(cross_x)) if c != j)
            train_y.extend(cross_y[c] for c in range(len(cross_y)) if c != j)
            train_x = str(train_x)
            train_y = str(train_y)
            train_x = train_x.replace("[", "")
            train_x = train_x.replace("]", "")
            train_x = list(eval(train_x))
            train_y = train_y.replace("[", "")
            train_y = train_y.replace("]", "")
            train_y = list(eval(train_y))
            train_x = np.array(train_x, dtype=float).reshape(-1, 653)
            train_y = np.array(train_y, dtype=int).reshape(-1, 1)

            train_ds = TensorDataset(torch.from_numpy(train_x).type(torch.FloatTensor), torch.from_numpy(train_y).type(torch.FloatTensor))
            train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, drop_last=False)
            test_ds = TensorDataset(torch.from_numpy(test_x).type(torch.FloatTensor), torch.from_numpy(test_y).type(torch.FloatTensor))
            test_dl = DataLoader(test_ds, batch_size=5000, shuffle=True, drop_last=False)

            print("# No.{} Time No.{} Fold".format(i+1, j+1))
            model, optim = get_model(lr, 653, 653 * 2, 1)
            model.train()
            print("******** DNN start to train ********")
            for epoch in range(epochs):
                # DNN train
                for x, y in train_dl:
                    y_pred = model(x)
                    loss = loss_fn(y_pred, y)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                with torch.no_grad():
                    epoch_accuracy = 0
                    epoch_loss = 0
                    count = 0
                    for x, y in train_dl:
                        count += 1
                        pred_y = model(x)
                        epoch_accuracy += evaluate.accuracy(pred_y.numpy(), y.numpy())
                        epoch_loss = loss_fn(pred_y, y).data
                    epoch_accuracy = epoch_accuracy / count
                    epoch_loss = epoch_loss / count

                    print("epoch", epoch+1, ": ", "train_loss: ", round(epoch_loss.item(), 4),
                        "train_acc: ", round(epoch_accuracy.item(), 4))
                    if epoch_accuracy >= 0.99:
                        break
            print("******** DNN train end ********")

            print("******** SVC start to train ********")
            from rbfSVC import RBFkernelSVC

            sv = RBFkernelSVC(gamma="auto")
            sv.fit(train_x, train_y.flatten())
            print("******** SVC train end ********")

            print("******** XGBoost start to train ********")
            from XGBoost import xgbmodel

            xgbmodel.fit(train_x,
                  train_y.flatten(),
                  eval_set = [(test_x, test_y.flatten())],
                  eval_metric = "logloss",
                  early_stopping_rounds = 10,
                  verbose = False)
            print("******** XGBoost train end ********")

            with torch.no_grad():
                precision = 0
                recall = 0
                test_accuracy = 0
                Fscore = 0
                auc = 0
                aupr = 0
                count = 0
                model.eval()
                for x, y in test_dl:
                    predClassResult = []
                    probaResult = []
                    count += 1
                    pred_y = model(x)

                    # predClassSVC = sv.predict(x.numpy())
                    probaSVC = sv.predict_proba(x.numpy())
                    y_predSVC = []
                    for iSVC in probaSVC:
                        y_predSVC.append(iSVC[1])

                    # predClassXGB = xgbmodel.predict(x.numpy())
                    probaXGB = xgbmodel.predict_proba(x.numpy())
                    y_predXGB = []
                    for iXGB in probaXGB:
                        y_predXGB.append(iXGB[1])

                    for iRes in range(len(y)):
                        probaResult.append(0.4 * pred_y[iRes][0] + 0.3 * y_predSVC[iRes] + 0.3 * y_predXGB[iRes])
                    for iRes in probaResult:
                        predClassResult.append(iRes > 0.5 and 1 or 0)

                    precision += evaluate.precision(np.array(predClassResult), y.numpy().flatten())
                    recall += evaluate.recall(np.array(predClassResult), y.numpy().flatten())
                    test_accuracy += evaluate.accuracy(np.array(probaResult), y.numpy().flatten())
                    Fscore += evaluate.F1(np.array(predClassResult), y.numpy().flatten())

                    au, fpr, tpr = evaluate.auc(np.array(probaResult), y.numpy().flatten(), i, j)
                    curve_1 = np.vstack([fpr, tpr])
                    curve_1 = pd.DataFrame(curve_1.T)
                    curve_1.to_csv(str(mm) + 'c' + str(cv) + '_au' + str(au) + '.csv', header=None, index=None)
                    auc += au

                    apr, pre, rec_ = evaluate.aupr(np.array(probaResult), y.numpy().flatten(), i, j)
                    curve_2 = np.vstack([rec_, pre])
                    curve_2 = pd.DataFrame(curve_2.T)
                    curve_2.to_csv(str(mm) + 'c' + str(cv) + '_aupr' + str(apr) + '.csv', header=None, index=None)
                    aupr += apr
                
                precision = precision / count
                recall = recall / count  
                test_accuracy = test_accuracy / count
                Fscore = Fscore / count
                auc = auc / count
                aupr = aupr / count

                preArray[i][j] = precision
                recallArray[i][j] = recall
                accArray[i][j] = test_accuracy
                F1Array[i][j] = Fscore
                aucArray[i][j] = auc
                auprArray[i][j] = aupr
                print("# No.{} time No.{} fold evaluate:".format(i+1, j+1), end=" ")
                print("Precision: {:.4f} Recall: {:.4f} Accuracy: {:.4f} F1-score: {:.4f} AUC: {:.4f} AUPR: {:.4f}"
                    .format(precision, recall, test_accuracy, Fscore, auc, aupr) )

    preArray = preArray.flatten()
    recallArray = recallArray.flatten()
    accArray = accArray.flatten()
    F1Array = F1Array.flatten()
    aucArray = aucArray.flatten()
    auprArray = auprArray.flatten()

    print("preArray: {:.4f}±{:.4f}".format(preArray.mean(), preArray.std()))
    print("recallArray: {:.4f}±{:.4f}".format(recallArray.mean(), recallArray.std()))
    print("accArray: {:.4f}±{:.4f}".format(accArray.mean(), accArray.std()))
    print("F1Array: {:.4f}±{:.4f}".format(F1Array.mean(), F1Array.std()))
    print("aucArray: {:.4f}±{:.4f}".format(aucArray.mean(), aucArray.std()))
    print("auprArray: {:.4f}±{:.4f}".format(auprArray.mean(), auprArray.std()))


if __name__ == '__main__':
    KFold(K=5)
