import numpy as np


class StageConfig:
    height = 16
    width = 30
    mineNumber = 99
    doFirstFlip = True


class Stage:
    def __init__(self, seed):
        np.random.seed(seed)
        self.height = StageConfig.height
        self.width = StageConfig.width
        self.mineNumber = StageConfig.mineNumber
        self.initStage()

    def initStage(self):
        if not self.isMineNumberLegal():
            raise Exception("Wrong Mine Number!")
        self.generateField()
        self.generateMine()
        self.generateMarkNumber()
        self.generateMaskField()
        self.initFlagInfo()
        if StageConfig.doFirstFlip:
            safeIdx = np.where(self.mineField != '*')
            safeIdx_8 = []
            for index in range(len(safeIdx[0])):
                if not self.getAllCloseMine(safeIdx[0][index], safeIdx[1][index]):
                    safeIdx_8.append(index)
            if safeIdx_8:
                randomSelectIdx = np.random.choice(safeIdx_8)
            else:
                randomSelectIdx = np.random.randint(0, len(safeIdx[0]))
            self.flipGrid(safeIdx[0][randomSelectIdx], safeIdx[1][randomSelectIdx])
        self.gameoverState = False

    def getAllCloseMine(self, i, j):
        blockList = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if not di and not dj:
                    continue
                if i + di < 0 or i + di >= self.height:
                    continue
                if j + dj < 0 or j + dj >= self.width:
                    continue
                if self.mineField[i][j] == '*':
                    blockList.append([i + di, j + dj])
        return blockList

    def isMineNumberLegal(self):
        if self.width * self.height <= self.mineNumber:
            return False
        if self.mineNumber <= 0:
            return False
        return True

    def generateField(self):
        self.mineField = np.zeros(shape=(self.height, self.width), dtype=str)

    def generateMine(self):
        indexList = list(np.arange(self.width * self.height, dtype=int))
        mineList = []
        for mineID in range(self.mineNumber):
            thisMineIdx = indexList[int(np.random.rand() * len(indexList))]
            mineList.append(thisMineIdx)
            indexList.remove(thisMineIdx)
        for eachMinePosIdx in mineList:
            i = eachMinePosIdx // self.width
            j = eachMinePosIdx % self.width
            self.mineField[i][j] = '*'

    def generateMarkNumber(self):
        for i in range(self.height):
            for j in range(self.width):
                if self.mineField[i][j] == '*':
                    continue
                number = 0
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0 or not 0 <= i + di < self.height or not 0 <= j + dj < self.width:
                            continue
                        if self.mineField[i + di][j + dj] == '*':
                            number += 1
                self.mineField[i][j] = str(number)

    def generateMaskField(self):
        self.maskField = np.ones(shape=(self.height, self.width), dtype=bool)
    
    def initFlagInfo(self):
        self.flagContainer = []
        self.flagLimit = self.mineNumber
        self.currentFlagNumber = 0

    def flipGrid(self, i, j):
        if not self.isMasked(i, j):
            return
        if [i, j] in self.flagContainer:
            return
        if self.mineField[i][j] == '*':
            self.gameover()
        self.updateMask(i, j)

    def isGameover(self):
        if self.gameoverState:
            return True
        return False

    def gameover(self):
        self.gameoverState = True
        # print("gameover")

    def updateMask(self, i, j):
        queue = [[i, j]]
        while queue:
            p = queue[0]
            queue.remove(queue[0])
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    i = p[0]
                    j = p[1]
                    if di == 0 and dj == 0 or not 0 <= i + di < self.height or not 0 <= j + dj < self.width:
                        continue
                    if di and dj:
                        pass
                    if self.maskField[i + di, j + dj]:
                        if self.mineField[i, j] not in ['0', '*'] and self.mineField[i + di, j + dj] == '0':
                            queue.append([i + di, j + dj])
                        if self.mineField[i, j] == '0' and self.mineField[i + di, j + dj] not in ['*']:
                            queue.append([i + di, j + dj])
            if [i, j] in self.flagContainer:
                self.flagGrid(i, j)
            self.maskField[i, j] = False

    def isMasked(self, i, j):
        if self.maskField[i][j]:
            return True
        return False

    def flagGrid(self, i, j):
        if not self.maskField[i][j]:
            return
        self.updateFlag(i, j)

    def isFlagReachMax(self):
        if self.currentFlagNumber >= self.flagLimit:
            return True
        return False

    def updateFlag(self, i, j):
        if [i, j] in self.flagContainer:
            self.flagContainer.remove([i, j])
            self.currentFlagNumber -= 1
        else:
            if not self.isFlagReachMax():
                self.flagContainer.append([i, j])
                self.currentFlagNumber += 1

    def cmdShow(self):
        print("X\t", end="")
        for j in range(self.width):
            print("\033[33m%s\033[0m\t" % j, end="")
        print("")
        for i in range(self.height):
            print("\033[33m%s\033[0m\t" % i, end="")
            for j in range(self.width):
                if self.maskField[i][j]:
                    if [i, j] in self.flagContainer:
                        print("F\t", end="")
                    else:
                        print(".\t", end="")
                else:
                    print("%s\t" % self.mineField[i][j], end="")
            print("")

    def swampGrid(self, i, j):
        if self.maskField[i][j]:
            return
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0 or not 0 <= i + di < self.height or not 0 <= j + dj < self.width:
                    continue
                if [i + di, j + dj] in self.flagContainer and self.mineField[i + di][j + dj] != "*":
                    return
                if [i + di, j + dj] not in self.flagContainer and self.mineField[i + di][j + dj] == "*":
                    if self.isFlagReachMax():
                        self.flipGrid(i + di, j + dj)
                    return
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0 or not 0 <= i + di < self.height or not 0 <= j + dj < self.width:
                    continue
                if self.maskField[i + di][j + dj] and [i + di, j + dj] not in self.flagContainer:
                    self.flipGrid(i + di, j + dj)

    def isGamewin(self):
        if sum(sum(self.maskField)) == self.mineNumber:
            return True
        return False

    def gamewin(self):
        for i in range(self.height):
            for j in range(self.width):
                if self.maskField[i][j] and [i, j] not in self.flagContainer:
                    self.flagContainer.append([i, j])
        # print("game win!")

    def action(self, signal, i, j):
        if signal == 1:
            self.flipGrid(i, j)
        elif signal == 2:
            self.flagGrid(i, j)
        elif signal == 3:
            self.swampGrid(i, j)

    def transform_all_info(self):
        info = {}

        info["mineField"] = np.zeros(shape=(self.height, self.width), dtype=str)
        info["state"] = "playing"
        info["StageConfig"] = StageConfig

        for line in range(self.height):
            for col in range(self.width):
                if not self.maskField[line][col]:
                    info["mineField"][line][col] = self.mineField[line][col]
                elif [line, col] in self.flagContainer:
                    info["mineField"][line][col] = "*"

        if self.isGameover():
            info["state"] = "gameover"
            info["mineField"] = self.mineField
        elif self.isGamewin():
            info["state"] = "gamewin"
            info["mineField"] = self.mineField

        return info
