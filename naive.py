import numpy as np
import time


class naive:
    def __init__(self, info, ratio):
        self.info = info

        self.height = info["StageConfig"].height
        self.width = info["StageConfig"].width
        self.mineNumber = info["StageConfig"].mineNumber
        self.doFirstFlip = info["StageConfig"].doFirstFlip

        self.mineField = info["mineField"]
        self.state = "start"
        self.infoState = info["state"]

        self.restMine = self.mineNumber
        self.unknown = np.ones(shape=(self.height, self.width))
        self.possibility = np.zeros(shape=(self.height, self.width))
        self.ratio = ratio

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

        for i in range(self.height):
            for j in range(self.width):
                if self.mineField[i][j] != '':
                    self.unknown[i][j] = False
                else:
                    for b in self.getAllCloseBlock(i, j):
                        if '1' <= self.mineField[b[0]][b[1]] <= '9':
                            self.unknown[i][j] = False
                            break

        i, j = np.where(self.possibility == np.max(self.possibility))
        randomPosition = np.random.randint(0, len(i))
        i, j = i[randomPosition], j[randomPosition]
        if np.sum(self.unknown) < 1 or (self.restMine - 1) / np.sum(self.unknown) < np.max(self.possibility) * self.ratio:
            self.restMine -= 1
            allDecision = [[2, i, j]]
            for b in self.getAllCloseBlock(i, j):
                if '1' <= self.mineField[b[0]][b[1]] <= '9':
                    allDecision.append([3, b[0], b[1]])
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
