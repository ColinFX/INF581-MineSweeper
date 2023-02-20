import numpy as np
import pygame

import stage


class render:
    def __init__(self):
        self.initForm()
        self.render_init()

    def initForm(self):
        pygame.init()
        self.gridSize = 50
        self.width = stage.StageConfig.width * self.gridSize
        self.height = stage.StageConfig.height * self.gridSize + 50
        self.screen = pygame.display.set_mode((self.width, self.height))

    def render_init(self):
        self.maskedGrid_img = pygame.image.load("./img/maskedGrid.png")
        self.maskedGrid_rect = self.maskedGrid_img.get_rect()
        self.unmaskedGrid_img = pygame.image.load("./img/unmaskedGrid.png")
        self.unmaskedGrid_rect = self.unmaskedGrid_img.get_rect()
        self.flag_img = pygame.image.load("./img/flag.png")
        self.flag_rect = self.flag_img.get_rect()
        self.mine_img = pygame.image.load("./img/mine.png")
        self.mine_rect = self.mine_img.get_rect()

    def gameStageDrawAuto(self, info, seed, number, location):
        self.screen.fill((0, 0, 0))

        info["maskField"] = np.zeros(shape=(self.height, self.width))
        info["flagContainer"] = []
        for line in range(info["StageConfig"].height):
            for col in range(info["StageConfig"].width):
                if info["mineField"][line][col] == '':
                    info["maskField"][line][col] = 1
                elif info["mineField"][line][col] == '*':
                    info["flagContainer"].append([line, col])

        for i in range(info["StageConfig"].height):
            for j in range(info["StageConfig"].width):
                centerx = self.gridSize * (0.5 + j)
                centery = self.gridSize * (0.5 + i)
                if info["maskField"][i][j]:
                    self.maskedGrid_rect.centerx = centerx
                    self.maskedGrid_rect.centery = centery
                    self.screen.blit(self.maskedGrid_img, self.maskedGrid_rect)
                else:
                    self.unmaskedGrid_rect.centerx = centerx
                    self.unmaskedGrid_rect.centery = centery
                    self.screen.blit(self.unmaskedGrid_img, self.unmaskedGrid_rect)
                    if '1' <= info["mineField"][i][j] <= '9':
                        number_img = pygame.image.load("./img/%s.png" % info["mineField"][i][j])
                        number_rect = number_img.get_rect()
                        number_img = pygame.transform.scale(number_img, (
                        int(number_rect.size[0] * 0.6), int(number_rect.size[1] * 0.6)))
                        number_rect = number_img.get_rect()
                        number_rect.centerx = centerx
                        number_rect.centery = centery
                        self.screen.blit(number_img, number_rect)

                if info["state"] == "gameover" and info["mineField"][i][j] == '*':
                    self.mine_rect.centerx = centerx
                    self.mine_rect.centery = centery
                    self.screen.blit(self.mine_img, self.mine_rect)

        if info["state"] != "gameover":
            for eachFlagPoint in info["flagContainer"]:
                self.flag_rect.centerx = self.gridSize * (0.5 + eachFlagPoint[1])
                self.flag_rect.centery = self.gridSize * (0.5 + eachFlagPoint[0])
                self.screen.blit(self.flag_img, self.flag_rect)

        self.flag_rect.centerx = 25
        self.flag_rect.centery = self.gridSize * info["StageConfig"].height + 25
        self.screen.blit(self.flag_img, self.flag_rect)
        font = pygame.font.SysFont("arial", 35)
        img = font.render('x %s' % str(info["StageConfig"].mineNumber - len(info["flagContainer"])), True, (255, 255, 255))
        rect = img.get_rect()
        rect.left = 50
        rect.centery = self.gridSize * info["StageConfig"].height + 25
        self.screen.blit(img, rect)

        font = pygame.font.SysFont("arial", 35)
        if info["state"] == "gameover":
            img = font.render('Game Over', True, (255, 0, 0))
            rect = img.get_rect()
            rect.centerx = self.width / 2
            rect.centery = self.gridSize * info["StageConfig"].height + 25
            self.screen.blit(img, rect)
        elif info["state"] == "gamewin":
            img = font.render('Game Win', True, (0, 255, 0))
            rect = img.get_rect()
            rect.centerx = self.width / 2
            rect.centery = self.gridSize * info["StageConfig"].height + 25
            self.screen.blit(img, rect)

        if location == 1:
            pygame.image.save(self.screen, "results/naive/" + str(seed) + "/" + str(number) + ".png")
        elif location == 2:
            pygame.image.save(self.screen, "results/globalNaive/" + str(seed) + "/" + str(number) + ".png")