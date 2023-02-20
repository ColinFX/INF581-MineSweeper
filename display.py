import os
import shutil

from stage import *
from naive import *
from globalNaive import *


class Display:
    def __init__(self, seed=0, auto=1, ratio=0.2, cut=10):
        self.seed = seed
        self.stage = Stage(self.seed)
        self.method = 0
        if auto == 1:
            self.method = naive(self.stage.transform_all_info(), ratio)
            if os.path.isdir("results/naive/" + str(self.seed)):
                shutil.rmtree("results/naive/" + str(self.seed))
            os.makedirs("results/naive/" + str(self.seed))
        elif auto == 2:
            self.method = globalNaive(self.stage.transform_all_info(), ratio, cut)
            if os.path.isdir("results/globalNaive/" + str(self.seed)):
                shutil.rmtree("results/globalNaive/" + str(self.seed))
            os.makedirs("results/globalNaive/" + str(self.seed))
        self.note = []

    def autoOperation(self):
        actionOneStep = self.method.decision(self.stage.transform_all_info())
        for a in actionOneStep:
            self.stage.action(a[0], a[1], a[2])
        self.note.append(self.stage.transform_all_info())

    def gameStage(self):
        while True:
            if not self.stage.isGameover() and not self.stage.isGamewin():
                self.autoOperation()
                if self.stage.isGamewin():
                    self.stage.gamewin()
            if self.stage.isGameover() or self.stage.isGamewin():
                return self.note

    def mainLoop(self):
        return self.gameStage()
