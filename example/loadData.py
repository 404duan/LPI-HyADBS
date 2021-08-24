import pandas as pd
import numpy as np


def divideToFive(listTemp):
    from math import ceil
    n = ceil(len(listTemp) / 5)
    resules = []
    for i in range(0, len(listTemp), n):
        temp = listTemp[i:i + n]
        resules.append(temp)
    return resules


def loadcv4Data(filePath):
    pliData = pd.read_csv(filePath+"plimat.csv", header=None).to_numpy().tolist()
    nonpliData = pd.read_csv(filePath+"nonplimat.csv", header=None).to_numpy().tolist()

    protein = []
    for i in range(max(pliData[-1][0], nonpliData[-1][0])):
        protein.append(i+1)
    rna = []
    for i in range(max(pliData[-1][1], nonpliData[-1][1])):
        rna.append(i+1)
    from random import sample, shuffle
    shuffle(protein)
    shuffle(rna)
    proteinList = divideToFive(protein)
    rnaList = divideToFive(rna)
    posData = []
    negData = []
    for i in range(5):
        tmpPosData = []
        tmpNegData = []
        for j in range(len(proteinList[i])):
            for k in range(len(rnaList[i])):
                tmpList = [proteinList[i][j], rnaList[i][k]]
                if tmpList in pliData:
                    tmpPosData.append(tmpList)
                else:
                    tmpNegData.append(tmpList)
        posData.append(tmpPosData)
        negData.append(tmpNegData)
    proData = pd.read_csv(filePath + "ProteinFeature/optimumDataset.csv", header=None).values.tolist()
    rnaData = pd.read_csv(filePath + "RNAFeature/optimumDataset.csv", header=None).values.tolist()
    cross_x = []
    cross_y = []
    for i in range(5):
        if len(posData[i]) == 0:
            negData[i] = sample(negData[i], 1)
        elif len(negData[i]) == 0:
            posData[i] = sample(posData[i], 1)
        elif len(posData[i]) > len(negData[i]):
            posData[i] = sample(posData[i], len(negData[i]))
        else:
            negData[i] = sample(negData[i], len(posData[i]))

    for i in range(5):
        tmpIX = []
        tmpIY = []
        if len(posData[i]) == 0:
            tmpJX = []
            tmpJX.extend(proData[negData[i][0][0]-1])
            tmpJX.extend(rnaData[negData[i][0][1]-1])
            tmpIX.append(tmpJX)
            tmpIY.append([0])
            cross_x.append(tmpIX)
            cross_y.append(tmpIY)
        elif len(negData[i]) == 0:
            tmpJX = []
            tmpJX.extend(proData[posData[i][0][0]-1])
            tmpJX.extend(rnaData[posData[i][0][1]-1])
            tmpIX.append(tmpJX)
            tmpIY.append([1])
            cross_x.append(tmpIX)
            cross_y.append(tmpIY)
        else:
            for j in range(len(posData[i])):
                tmpJX = []
                tmpJX.extend(proData[posData[i][j][0]-1])
                tmpJX.extend(rnaData[posData[i][j][1]-1])
                tmpIX.append(tmpJX)
                tmpIY.append([1])

                tmpJX = []
                tmpJX.extend(proData[negData[i][j][0]-1])
                tmpJX.extend(rnaData[negData[i][j][1]-1])
                tmpIX.append(tmpJX)
                tmpIY.append([0])
            cross_x.append(tmpIX)
            cross_y.append(tmpIY)

    return cross_x, cross_y


def loadcv3Data(filePath):
    pliData = pd.read_csv(filePath+"plimat.csv", header=None).to_numpy().tolist()
    nonpliData = pd.read_csv(filePath+"nonplimat.csv", header=None).to_numpy().tolist()

    from random import sample, shuffle
    nonpliData = sample(nonpliData, len(pliData))
    shuffle(pliData)

    pliData = divideToFive(pliData)
    nonpliData = divideToFive(nonpliData)

    proData = pd.read_csv(filePath + "ProteinFeature/optimumDataset.csv", header=None).values.tolist()
    rnaData = pd.read_csv(filePath + "RNAFeature/optimumDataset.csv", header=None).values.tolist()
    cross_x = []
    cross_y = []
    for i in range(5):
        tmpIX = []
        tmpIY = []
        for j in range(len(pliData[i])):
            tmpJX = []
            tmpJX.extend(proData[pliData[i][j][0]-1])
            tmpJX.extend(rnaData[pliData[i][j][1]-1])
            tmpIX.append(tmpJX)
            tmpIY.append([1])

            tmpJX = []
            tmpJX.extend(proData[nonpliData[i][j][0]-1])
            tmpJX.extend(rnaData[nonpliData[i][j][1]-1])
            tmpIX.append(tmpJX)
            tmpIY.append([0])
        cross_x.append(tmpIX)
        cross_y.append(tmpIY)

    return cross_x, cross_y


def loadcv2Data(filePath):
    with open(filePath+"plimatCV2.txt", "r") as f:
        pli = f.read()
    pli = pli.split("\n")
    pliData = []
    for i in pli:
        tmp = i.split("\t")
        tmp.pop()
        for j in range(len(tmp)):
            tmp[j] = int(tmp[j])
        pliData.append(tmp)
    pliData.pop()

    with open(filePath+"nonplimatCV2.txt", "r") as f:
        nonpli = f.read()
    nonpli = nonpli.split("\n")
    nonpliData = []
    for i in nonpli:
        tmp = i.split("\t")
        tmp.pop()
        for j in range(len(tmp)):
            tmp[j] = int(tmp[j])
        nonpliData.append(tmp)
    nonpliData.pop()

    from random import shuffle, randint, seed, sample
    randnum = randint(0, 100)
    seed(randnum)
    shuffle(pliData)
    seed(randnum)
    shuffle(nonpliData)
    pliData = divideToFive(pliData)
    nonpliData = divideToFive(nonpliData)

    indexPli = []
    for i in pliData:
        noILayer = []
        for j in i:
            for k in range(len(j)-1):
                noILayer.extend([[j[0], j[k+1]]])
        indexPli.extend([noILayer])

    indexNonPli = []
    for i in nonpliData:
        noILayer = []
        for j in i:
            for k in range(len(j)-1):
                noILayer.extend([[j[0], j[k+1]]])
        indexNonPli.extend([noILayer])
    for i in range(5):
        if len(indexPli[i]) < len(indexNonPli[i]):
            indexNonPli[i] = sample(indexNonPli[i], len(indexPli[i]))
        else:
            indexPli[i] = sample(indexPli[i], len(indexNonPli[i]))

    proData = pd.read_csv(filePath + "ProteinFeature/optimumDataset.csv", header=None).values.tolist()
    rnaData = pd.read_csv(filePath + "RNAFeature/optimumDataset.csv", header=None).values.tolist()
    cross_x = []
    cross_y = []
    for i in range(5):
        tmpIX = []
        tmpIY = []
        for j in range(len(indexPli[i])):
            tmpJX = []
            tmpJX.extend(proData[indexPli[i][j][0]-1])
            tmpJX.extend(rnaData[indexPli[i][j][1]-1])
            tmpIX.append(tmpJX)
            tmpIY.append([1])

            tmpJX = []
            tmpJX.extend(proData[indexNonPli[i][j][0]-1])
            tmpJX.extend(rnaData[indexNonPli[i][j][1]-1])
            tmpIX.append(tmpJX)
            tmpIY.append([0])
        cross_x.append(tmpIX)
        cross_y.append(tmpIY)

    return cross_x, cross_y


def loadcv1Data(filePath):
    with open(filePath+"plimatCV1.txt", "r") as f:
        pli = f.read()
    pli = pli.split("\n")
    pliData = []
    for i in pli:
        tmp = i.split("\t")
        tmp.pop()
        for j in range(len(tmp)):
            tmp[j] = int(tmp[j])
        pliData.append(tmp)
    pliData.pop()

    with open(filePath+"nonplimatCV1.txt", "r") as f:
        nonpli = f.read()
    nonpli = nonpli.split("\n")
    nonpliData = []
    for i in nonpli:
        tmp = i.split("\t")
        tmp.pop()
        for j in range(len(tmp)):
            tmp[j] = int(tmp[j])
        nonpliData.append(tmp)
    nonpliData.pop()

    from random import shuffle, randint, seed, sample
    randnum = randint(0, 100)
    seed(randnum)
    shuffle(pliData)
    seed(randnum)
    shuffle(nonpliData)
    pliData = divideToFive(pliData)
    nonpliData = divideToFive(nonpliData)

    indexPli = []
    for i in pliData:
        noILayer = []
        for j in i:
            for k in range(len(j)-1):
                noILayer.extend([[j[0], j[k+1]]])

        indexPli.extend([noILayer])

    indexNonPli = []
    for i in nonpliData:
        noILayer = []
        for j in i:
            for k in range(len(j)-1):
                noILayer.extend([[j[0], j[k+1]]])
        indexNonPli.extend([noILayer])

    for i in range(5):
        if len(indexPli[i]) < len(indexNonPli[i]):
            indexNonPli[i] = sample(indexNonPli[i], len(indexPli[i]))
        else:
            indexPli[i] = sample(indexPli[i], len(indexNonPli[i]))

    proData = pd.read_csv(filePath + "ProteinFeature/optimumDataset.csv", header=None).values.tolist()
    rnaData = pd.read_csv(filePath + "RNAFeature/optimumDataset.csv", header=None).values.tolist()
    cross_x = []
    cross_y = []
    for i in range(5):
        tmpIX = []
        tmpIY = []
        for j in range(len(indexPli[i])):
            tmpJX = []
            tmpJX.extend(proData[indexPli[i][j][1]-1])
            tmpJX.extend(rnaData[indexPli[i][j][0]-1])
            tmpIX.append(tmpJX)
            tmpIY.append([1])

            tmpJX = []
            tmpJX.extend(proData[indexNonPli[i][j][1]-1])
            tmpJX.extend(rnaData[indexNonPli[i][j][0]-1])
            tmpIX.append(tmpJX)
            tmpIY.append([0])
        cross_x.append(tmpIX)
        cross_y.append(tmpIY)

    return cross_x, cross_y
