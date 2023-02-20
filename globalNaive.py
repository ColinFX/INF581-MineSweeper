import random
import time
import numpy as np
from scipy.special import comb


class globalNaive:
    def __init__(self, info, ratio, cut):
        self.info = info

        self.height = info["StageConfig"].height
        self.width = info["StageConfig"].width
        self.mineNumber = info["StageConfig"].mineNumber
        self.doFirstFlip = info["StageConfig"].doFirstFlip

        self.mineField = info["mineField"]
        self.state = "start"
        self.infoState = info["state"]

        self.restMine = self.mineNumber
        self.possibleMineField = np.copy(self.mineField)
        self.unknown = np.ones(shape=(self.height, self.width))
        self.possibility = np.zeros(shape=(self.height, self.width))

        self.ratio = ratio
        self.cut = cut

    def decision(self, info):
        np.random.seed(int(time.time()))
        self.mineField = info["mineField"]
        self.possibility = np.zeros(shape=(self.height, self.width))

        if not self.doFirstFlip and self.state == "start":
            self.state = self.infoState
            return [[1, np.random.randint(low=0, high=self.height), np.random.randint(low=0, high=self.width)]]
        self.state = self.infoState

        if not self.restMine:
            allRestPosition = []
            for i in range(self.height):
                for j in range(self.width):
                    if self.mineField[i][j] == '':
                        allRestPosition.append([1, i, j])
            return allRestPosition

        restAction = []
        for i in range(self.height):
            for j in range(self.width):
                if '1' <= self.mineField[i][j] <= '9':
                    closeBlocks = self.getAllCloseBlock(i, j)
                    contentBlocks = [self.mineField[i][j] for i, j in closeBlocks]
                    if '' in contentBlocks:
                        if contentBlocks.count('*') == int(self.mineField[i][j]):
                            restAction.append([3, i, j])
                            for i, j in closeBlocks:
                                if self.mineField[i][j] == '':
                                    restAction.append([1, i, j])
        if restAction:
            return restAction

        for i in range(self.height):
            for j in range(self.width):
                if '1' <= self.mineField[i][j] <= '9':
                    availablePosition, lack = self.checkNumber(i, j)
                    if lack:
                        if len(availablePosition) == lack:
                            allDecision = []
                            for l in range(lack):
                                if self.mineField[availablePosition[l][0]][availablePosition[l][1]] == '':
                                    allDecision.append([2, availablePosition[l][0], availablePosition[l][1]])
                                    self.restMine -= 1
                            for l in range(lack):
                                for b in self.getAllCloseBlock(availablePosition[l][0], availablePosition[l][1]):
                                    if '1' <= self.mineField[b[0]][b[1]] <= '9':
                                        allDecision.append([3, b[0], b[1]])
                            return allDecision
                        else:
                            if availablePosition:
                                if self.possibility[i][j] < lack / len(availablePosition):
                                    for b in self.getAllCloseBlock(i, j):
                                        if self.mineField[b[0]][b[1]] == '':
                                            self.possibility[b[0]][b[1]] = lack / len(availablePosition)

        i, j = np.where(self.possibility == np.max(self.possibility))
        randomPosition = np.random.randint(0, len(i))
        i, j = i[randomPosition], j[randomPosition]
        closeNumberList = self.getAllCloseNumber(i, j)
        random.shuffle(closeNumberList)
        i, j = closeNumberList[0]
        return self.globalDecision(i, j)

    def globalDecision(self, i, j):
        self.possibility = np.zeros(shape=(self.height, self.width))
        self.possibleMineField = np.copy(self.mineField)

        allRelativeNumber = [[i, j]]
        relativeNumber = [[i, j]]
        allRelativeEmpty = []
        relativeEmpty = []
        allPossibleSolution = []
        while True:
            for k in relativeNumber:
                for m in self.getAllCloseEmpty(k[0], k[1]):
                    if m not in allRelativeEmpty:
                        relativeEmpty.append(m)
                        allRelativeEmpty.append(m)
            relativeNumber = []
            for k in relativeEmpty:
                for m in self.getAllCloseNumber(k[0], k[1]):
                    if m not in allRelativeNumber:
                        relativeNumber.append(m)
                        allRelativeNumber.append(m)
            if not relativeEmpty and not relativeNumber:
                break
            relativeEmpty = []

        mineList = [0 for _ in allRelativeEmpty]
        for k in range(1, min(len(allRelativeEmpty) + 1, self.cut + 1)):
            for m in range(k):
                mineList[m] = 1
            while mineList != -1:
                possibleMinePosition = [allRelativeEmpty[m] for m in range(len(mineList)) if mineList[m]]
                self.possibleMineField = np.copy(self.mineField)
                for p in possibleMinePosition:
                    self.possibleMineField[p[0]][p[1]] = '?'
                possibleResult = True
                for p in allRelativeNumber:
                    if int(self.possibleMineField[p[0]][p[1]]) != len(self.getAllClosePossibleMine(p[0], p[1])):
                        possibleResult = False
                        break
                if possibleResult:
                    allPossibleSolution.append(possibleMinePosition)
                mineList = self.getNextList(mineList)
            mineList = [0 for _ in allRelativeEmpty]

        for i in range(self.height):
            for j in range(self.width):
                if self.mineField[i][j] != '':
                    self.unknown[i][j] = False
                elif [i, j] in allRelativeEmpty:
                    self.unknown[i][j] = False

        amountPossible = {}
        amountLen = {}
        Couts = {}
        for p in allPossibleSolution:
            if len(p) not in amountPossible:
                Cout, amountPossible[len(p)] = self.getAmountPossible(len(allRelativeEmpty), int(np.round(np.sum(self.unknown))), len(p), self.restMine - len(p))
                Couts[len(p)] = Cout
                amountLen[len(p)] = 0
            amountLen[len(p)] += 1
        sumValue = np.sum(list(amountPossible.values()))
        for p in amountPossible:
            if sumValue:
                amountPossible[p] /= sumValue
        Cout = 0
        for p in Couts:
            Cout += Couts[p] * amountPossible[p]
        for p in amountPossible:
            amountPossible[p] /= amountLen[p]
        for p in allPossibleSolution:
            for q in p:
                self.possibility[q[0]][q[1]] += amountPossible[len(p)]

        i, j = np.where(self.possibility == np.max(self.possibility))
        randomPosition = np.random.randint(0, len(i))
        i, j = i[randomPosition], j[randomPosition]
        if np.sum(self.unknown) < 1 or Cout <= np.max(self.possibility) * self.ratio:
            self.restMine -= 1
            allDecision = [[2, i, j]]
            allDecision += [[3, b[0], b[1]] for b in self.getAllCloseNumber(i, j)]
            return allDecision
        else:
            while True:
                i, j = np.random.randint(low=0, high=self.height), np.random.randint(low=0, high=self.width)
                if self.unknown[i][j]:
                    return [[1, i, j]]

    def checkNumber(self, i, j):
        lack = int(self.mineField[i][j])
        availablePosition = []
        for b in self.getAllCloseBlock(i, j):
            if self.mineField[b[0]][b[1]] == '':
                availablePosition.append(b)
            elif self.mineField[b[0]][b[1]] == '*':
                lack -= 1
        return availablePosition, lack

    def getAllCloseBlock(self, i, j):
        blockList = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if not di and not dj:
                    continue
                if i + di < 0 or i + di >= self.height:
                    continue
                if j + dj < 0 or j + dj >= self.width:
                    continue
                blockList.append([i + di, j + dj])
        return blockList

    def getAllCloseNumber(self, i, j):
        blockList = self.getAllCloseBlock(i, j)
        numberList = []
        for b in blockList:
            if '1' <= self.mineField[b[0]][b[1]] <= '9':
                numberList.append(b)
        return numberList

    def getAllCloseEmpty(self, i, j):
        blockList = self.getAllCloseBlock(i, j)
        emptyList = []
        for b in blockList:
            if self.mineField[b[0]][b[1]] == '':
                emptyList.append(b)
        return emptyList

    def getAllCloseMine(self, i, j):
        blockList = self.getAllCloseBlock(i, j)
        closeMineList = []
        for b in blockList:
            if self.mineField[b[0]][b[1]] == '*':
                closeMineList.append(b)
        return closeMineList

    def getAllClosePossibleMine(self, i, j):
        blockList = self.getAllCloseBlock(i, j)
        possibleMineList = []
        for b in blockList:
            if self.possibleMineField[b[0]][b[1]] == '*' or self.possibleMineField[b[0]][b[1]] == '?':
                possibleMineList.append(b)
        return possibleMineList

    def getNextList(self, prelist):
        preLen = 0
        if len(prelist) > self.cut:
            preLen = len(prelist)
            prelist = prelist[: self.cut]

        back = 0
        position = len(prelist) - 1
        while position != -1 and prelist[position] != 0:
            back += 1
            position -= 1
        while position != -1 and prelist[position] == 0:
            position -= 1

        if position == -1:
            return -1

        nextlist = prelist.copy()
        nextlist[position] = 0
        nextlist[position + 1] = 1
        for k in range(position + 2, position + back + 2):
            nextlist[k] = 1
        for k in range(position + back + 2, len(prelist)):
            nextlist[k] = 0

        if preLen:
            nextlist += [0 for _ in range(preLen - self.cut)]

        return nextlist

    def getAmountPossible(self, insideEmpty, outsideEmpty, insideMine, outsideMine):
        if outsideEmpty < outsideMine:
            return 1, 0
        if outsideEmpty == 0:
            return 1, 1
        if outsideMine < 0:
            return 0, 0

        Cin = comb(insideEmpty, insideMine)
        Cout = comb(outsideEmpty, outsideMine)
        Call = comb(insideEmpty + outsideEmpty, insideMine + outsideMine)

        return max(0, 1 - outsideMine / outsideEmpty), Cin * Cout / Call
