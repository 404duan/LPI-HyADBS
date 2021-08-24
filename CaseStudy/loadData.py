import pandas as pd
import numpy as np


def loadTrainData(filePath):
    pliData = pd.read_csv(filePath+"plimat.csv", header=None).to_numpy().tolist()
    nonpliData = pd.read_csv(filePath+"nonplimat.csv", header=None).to_numpy().tolist()

    from random import sample, shuffle
    nonpliData = sample(nonpliData, len(pliData))
    shuffle(pliData)

    proData = pd.read_csv(filePath + "ProteinFeature/optimumDataset.csv", header=None).values.tolist()
    rnaData = pd.read_csv(filePath + "RNAFeature/optimumDataset.csv", header=None).values.tolist()

    train_x = []
    train_y = []
    
    for i in range(len(pliData)):
        tmpJX = []
        tmpJX.extend(proData[pliData[i][0]-1])
        tmpJX.extend(rnaData[pliData[i][1]-1])
        train_x.append(tmpJX)
        train_y.append([1])

        tmpJX = []
        tmpJX.extend(proData[nonpliData[i][0]-1])
        tmpJX.extend(rnaData[nonpliData[i][1]-1])
        train_x.append(tmpJX)
        train_y.append([0])

    train_x = np.array(train_x, dtype=float)
    train_y = np.array(train_y, dtype=int)

    return train_x, train_y


def loadTestData(filePath):
    pliData = pd.read_csv(filePath+"Proplimat.csv", header=None).to_numpy().tolist()
    nonpliData = pd.read_csv(filePath+"Prononplimat.csv", header=None).to_numpy().tolist()

    from random import shuffle
    # nonpliData = sample(nonpliData, len(pliData))
    shuffle(pliData)
    shuffle(nonpliData)

    proData = pd.read_csv(filePath + "ProteinFeature/optimumDataset.csv", header=None).values.tolist()
    rnaData = pd.read_csv(filePath + "RNAFeature/optimumDataset.csv", header=None).values.tolist()

    test_x = []
    test_y = []
    RNAindex = []

    for i in range(len(pliData)):
        tmpJX = []
        tmpJX.extend(proData[pliData[i][0]-1])
        tmpJX.extend(rnaData[pliData[i][1]-1])
        test_x.append(tmpJX)
        test_y.append([1])
        RNAindex.append(pliData[i][1])

    for i in range(len(nonpliData)):
        tmpJX = []
        tmpJX.extend(proData[nonpliData[i][0]-1])
        tmpJX.extend(rnaData[nonpliData[i][1]-1])
        test_x.append(tmpJX)
        test_y.append([0])
        RNAindex.append(nonpliData[i][1])

    test_x = np.array(test_x, dtype=float)
    test_y = np.array(test_y, dtype=int)

    return test_x, test_y, RNAindex

def loadTrainData2(filePath):
    pliData = pd.read_csv(filePath+"plimat2.csv", header=None).to_numpy().tolist()
    nonpliData = pd.read_csv(filePath+"nonplimat2.csv", header=None).to_numpy().tolist()

    from random import sample, shuffle
    nonpliData = sample(nonpliData, len(pliData))
    shuffle(pliData)

    proData = pd.read_csv(filePath + "ProteinFeature/optimumDataset.csv", header=None).values.tolist()
    rnaData = pd.read_csv(filePath + "RNAFeature/optimumDataset.csv", header=None).values.tolist()

    train_x = []
    train_y = []
    
    for i in range(len(pliData)):
        tmpJX = []
        tmpJX.extend(proData[pliData[i][1]-1])
        tmpJX.extend(rnaData[pliData[i][0]-1])
        train_x.append(tmpJX)
        train_y.append([1])

        tmpJX = []
        tmpJX.extend(proData[nonpliData[i][1]-1])
        tmpJX.extend(rnaData[nonpliData[i][0]-1])
        train_x.append(tmpJX)
        train_y.append([0])

    train_x = np.array(train_x, dtype=float)
    train_y = np.array(train_y, dtype=int)

    return train_x, train_y


def loadTestData2(filePath):
    pliData = pd.read_csv(filePath+"RNAplimat.csv", header=None).to_numpy().tolist()
    nonpliData = pd.read_csv(filePath+"RNAnonplimat.csv", header=None).to_numpy().tolist()

    from random import shuffle
    # nonpliData = sample(nonpliData, len(pliData))
    shuffle(pliData)
    shuffle(nonpliData)

    proData = pd.read_csv(filePath + "ProteinFeature/optimumDataset.csv", header=None).values.tolist()
    rnaData = pd.read_csv(filePath + "RNAFeature/optimumDataset.csv", header=None).values.tolist()

    test_x = []
    test_y = []
    Proindex = []

    for i in range(len(pliData)):
        tmpJX = []
        tmpJX.extend(proData[pliData[i][1]-1])
        tmpJX.extend(rnaData[pliData[i][0]-1])
        test_x.append(tmpJX)
        test_y.append([1])
        Proindex.append(pliData[i][1])

    for i in range(len(nonpliData)):
        tmpJX = []
        tmpJX.extend(proData[nonpliData[i][1]-1])
        tmpJX.extend(rnaData[nonpliData[i][0]-1])
        test_x.append(tmpJX)
        test_y.append([0])
        Proindex.append(nonpliData[i][1])

    test_x = np.array(test_x, dtype=float)
    test_y = np.array(test_y, dtype=int)

    return test_x, test_y, Proindex

def loadTrainData3(filePath):
    pliData = pd.read_csv(filePath+"plimat3.csv", header=None).to_numpy().tolist()
    nonpliData = pd.read_csv(filePath+"nonplimat3.csv", header=None).to_numpy().tolist()

    from random import sample, shuffle
    nonpliData = sample(nonpliData, len(pliData))
    shuffle(pliData)

    proData = pd.read_csv(filePath + "ProteinFeature/optimumDataset.csv", header=None).values.tolist()
    rnaData = pd.read_csv(filePath + "RNAFeature/optimumDataset.csv", header=None).values.tolist()

    train_x = []
    train_y = []
    
    for i in range(len(pliData)):
        tmpJX = []
        tmpJX.extend(proData[pliData[i][0]-1])
        tmpJX.extend(rnaData[pliData[i][1]-1])
        train_x.append(tmpJX)
        train_y.append([1])

        tmpJX = []
        tmpJX.extend(proData[nonpliData[i][0]-1])
        tmpJX.extend(rnaData[nonpliData[i][1]-1])
        train_x.append(tmpJX)
        train_y.append([0])

    c = list(zip(train_x, train_y))
    shuffle(c)
    train_x[:], train_y[:] = zip(*c)
    
    train_x = np.array(train_x, dtype=float)
    train_y = np.array(train_y, dtype=int)

    return train_x, train_y


def loadTestData3(filePath):
    pliData = pd.read_csv(filePath+"plimat3.csv", header=None).to_numpy().tolist()
    nonpliData = pd.read_csv(filePath+"nonplimat3.csv", header=None).to_numpy().tolist()

    from random import shuffle
    # nonpliData = sample(nonpliData, len(pliData))
    shuffle(pliData)
    shuffle(nonpliData)

    proData = pd.read_csv(filePath + "ProteinFeature/optimumDataset.csv", header=None).values.tolist()
    rnaData = pd.read_csv(filePath + "RNAFeature/optimumDataset.csv", header=None).values.tolist()

    test_x = []
    test_y = []
    Proindex = []
    RNAindex = []

    for i in range(len(pliData)):
        tmpJX = []
        tmpJX.extend(proData[pliData[i][0]-1])
        tmpJX.extend(rnaData[pliData[i][1]-1])
        test_x.append(tmpJX)
        test_y.append([1])
        Proindex.append(pliData[i][0])
        RNAindex.append(pliData[i][1])

    for i in range(len(nonpliData)):
        tmpJX = []
        tmpJX.extend(proData[nonpliData[i][0]-1])
        tmpJX.extend(rnaData[nonpliData[i][1]-1])
        test_x.append(tmpJX)
        test_y.append([0])
        Proindex.append(nonpliData[i][0])
        RNAindex.append(nonpliData[i][1])

    c = list(zip(test_x, test_y))
    shuffle(c)
    test_x[:], test_y[:] = zip(*c)

    test_x = np.array(test_x, dtype=float)
    test_y = np.array(test_y, dtype=int)

    return test_x, test_y, Proindex, RNAindex
