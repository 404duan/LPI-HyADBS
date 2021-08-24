from torch.optim import Adam
from torch.nn import BCELoss
from model import Model
from loadData import loadTrainData, loadTestData, loadTrainData2, loadTestData2, loadTrainData3, loadTestData3
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import evaluate
import numpy as np


def get_model(lr, inputNode, hidden, outputNode):
    model = Model(inputNode, hidden, outputNode)
    opt = Adam(model.parameters(), lr=lr)
    return model, opt


lr = 0.0001
loss_fn = BCELoss()
batch = 128
epochs = 100


def trainOnce():
    train_x, train_y = loadTrainData("./data/1/")

    train_ds = TensorDataset(torch.from_numpy(train_x).type(torch.FloatTensor), torch.from_numpy(train_y).type(torch.FloatTensor))
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, drop_last=False)

    test_x, test_y, RNAindex = loadTestData("./data/1/")
    test_ds = TensorDataset(torch.from_numpy(test_x).type(torch.FloatTensor), torch.from_numpy(test_y).type(torch.FloatTensor))
    test_dl = DataLoader(test_ds, batch_size=5000, shuffle=False, drop_last=False)

    model, optim = get_model(lr, train_x.shape[1], train_x.shape[1] * 2, 1)
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

    xgbmodel.fit(train_x, train_y.flatten())
    print("******** XGBoost train end ********")

    with torch.no_grad():
        for x, y in test_dl:
            # predClassResult = []
            # probaResult = []
            pred_y = model(x)

            # predClassSVC = sv.predict(x.numpy())
            probaSVC = sv.predict_proba(x.numpy())
            y_predSVC = []
            for i in probaSVC:
                y_predSVC.append(i[1])

            # predClassXGB = xgbmodel.predict(x.numpy())
            probaXGB = xgbmodel.predict_proba(x.numpy())
            y_predXGB = []
            for i in probaXGB:
                y_predXGB.append(i[1])

            result = []
            for i in range(len(y)):
                tmp = (RNAindex[i], 0.4 * pred_y[i][0] + 0.3 * y_predSVC[i] + 0.3 * y_predXGB[i], y[i][0])
                result.append(tmp)
            result = sorted(result, key=lambda x: x[1], reverse=True)
            top = 0
            with open("data1_case1.csv", "w") as f:
                for i in result:
                    top += 1
                    print("Top{}\tP35637 with {}\tMyModel proba: {:.4f}\t True value: {}"
                        .format(top ,i[0], i[1], i[2]))
                    f.write(str(i[0])+","+str(i[1])[7:-1]+","+str(i[2])[7:-2]+"\n")

def trainCase2(dataNum):
    train_x, train_y = loadTrainData2("./data/"+dataNum+"/")

    train_ds = TensorDataset(torch.from_numpy(train_x).type(torch.FloatTensor), torch.from_numpy(train_y).type(torch.FloatTensor))
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, drop_last=False)

    test_x, test_y, Proindex = loadTestData2("./data/"+dataNum+"/")
    test_ds = TensorDataset(torch.from_numpy(test_x).type(torch.FloatTensor), torch.from_numpy(test_y).type(torch.FloatTensor))
    test_dl = DataLoader(test_ds, batch_size=5000, shuffle=False, drop_last=False)

    model, optim = get_model(lr, train_x.shape[1], train_x.shape[1] * 2, 1)
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

    xgbmodel.fit(train_x, train_y.flatten())
    print("******** XGBoost train end ********")

    with torch.no_grad():
        for x, y in test_dl:
            # predClassResult = []
            # probaResult = []
            pred_y = model(x)

            # predClassSVC = sv.predict(x.numpy())
            probaSVC = sv.predict_proba(x.numpy())
            y_predSVC = []
            for i in probaSVC:
                y_predSVC.append(i[1])

            # predClassXGB = xgbmodel.predict(x.numpy())
            probaXGB = xgbmodel.predict_proba(x.numpy())
            y_predXGB = []
            for i in probaXGB:
                y_predXGB.append(i[1])

            result = []
            for i in range(len(y)):
                tmp = (Proindex[i], 0.4 * pred_y[i][0] + 0.3 * y_predSVC[i] + 0.3 * y_predXGB[i], y[i][0])
                result.append(tmp)
            result = sorted(result, key=lambda x: x[1], reverse=True)
            top = 0
            with open("data"+dataNum+"_case2.csv", "w") as f:
                for i in result:
                    top += 1
                    print("Top {}\tRNase MRP RNA with Protein {}\tMyModel proba:{:.4f}\tTrue value:{}"
                        .format(top ,i[0], i[1], i[2]))
                    f.write(str(i[0])+","+str(i[1])[7:-1]+","+str(i[2])[7:-2]+"\n")

def trainCase3(dataNum):
    train_x, train_y = loadTrainData3("./data/"+dataNum+"/")

    train_ds = TensorDataset(torch.from_numpy(train_x).type(torch.FloatTensor), torch.from_numpy(train_y).type(torch.FloatTensor))
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, drop_last=False)

    test_x, test_y, Proindex, RNAindex = loadTestData3("./data/"+dataNum+"/")
    test_ds = TensorDataset(torch.from_numpy(test_x).type(torch.FloatTensor), torch.from_numpy(test_y).type(torch.FloatTensor))
    test_dl = DataLoader(test_ds, batch_size=100000, shuffle=False, drop_last=False)

    model, optim = get_model(lr, train_x.shape[1], train_x.shape[1] * 2, 1)
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

    xgbmodel.fit(train_x, train_y.flatten())
    print("******** XGBoost train end ********")

    with torch.no_grad():
        for x, y in test_dl:
            # predClassResult = []
            # probaResult = []
            pred_y = model(x)

            # predClassSVC = sv.predict(x.numpy())
            probaSVC = sv.predict_proba(x.numpy())
            y_predSVC = []
            for i in probaSVC:
                y_predSVC.append(i[1])

            # predClassXGB = xgbmodel.predict(x.numpy())
            probaXGB = xgbmodel.predict_proba(x.numpy())
            y_predXGB = []
            for i in probaXGB:
                y_predXGB.append(i[1])

            result = []
            for i in range(len(y)):
                tmp = (Proindex[i], 0.4 * pred_y[i][0] + 0.3 * y_predSVC[i] + 0.3 * y_predXGB[i], y[i][0], RNAindex[i])
                result.append(tmp)
            result = sorted(result, key=lambda x: x[1], reverse=True)
            top = 0
            with open("data"+dataNum+"_case3.csv", "w") as f:
                for i in result:
                    top += 1
                    print("Top {}\tProtein-{} with RNA-{}\tproba:{:.4f}\tTrue value:{}"
                        .format(top , i[0], i[3], i[1], i[2]))
                    f.write(str(i[0])+","+str(i[3])+","+str(i[1])[7:-1]+","+str(i[2])[7:-2]+"\n")


if __name__ == '__main__':
    trainCase2("1")
    trainCase2("2")
    trainCase2("3")
