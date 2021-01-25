import pygame
import time
import random
import os

from datetime import datetime
from os import listdir
from pygame.locals import *
from button import *
from WFC import *

class MapGenerator:

    def __init__(self, game):
        self.game = game
        self.screen = game.screen
        self.images = []
        self.button1 = Button((600,550,105,25), 'Regenerate', self.game)
        self.button2 = Button((600,20,105,25), 'Save', self.game)

    def init(self, images):
        location = [(10,70), (210,70), (410, 70), (610, 70), (10, 300), (210,300), (410, 300), (610, 300)]
        index = 0
        self.images = []
        for image in images:
            mode = image.mode
            size = image.size
            data = image.tobytes()
            tile = pygame.image.fromstring(data, size, mode)
            tmp = {}
            tmp['tile'] = pygame.transform.scale(tile, (180,180))
            tmp['rect'] = pygame.Rect((location[index][0],location[index][1],180,180))
            tmp['location'] = location[index]
            tmp['selected'] = False
            self.images.append(tmp)
            index += 1


    # def process(self, imagefiles):
    def draw(self):
        self.button1.draw()
        self.button2.draw()
        for image in self.images:
            self.screen.blit(image['tile'], image['rect'])
            if image['selected'] == True:
                pygame.draw.rect(self.screen, Color('blue'), image['rect'], 2)

    def get_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for image in self.images:
                if image['rect'].collidepoint(pygame.mouse.get_pos()):
                    # print(image['name'])
                    image['selected'] = not image['selected']
            if self.button1.rect.collidepoint(pygame.mouse.get_pos()):
                self.button1updata()
            if self.button2.rect.collidepoint(pygame.mouse.get_pos()):
                self.button2updata()

    def button1updata(self):
        selectedImage = self.game.constrainGenerator.selectedImage
        constraints = self.game.constrainGenerator.constraints
        WFC = WaveFunctionCollaps(selectedImage, constraints)
        res = []
        for i in range(8):
            res.append(WFC.run())
        self.init(res)

    def button2updata(self):
        for image in self.images:
            if image['selected'] == True:
                time = str(datetime.now())
                name = time + str(image['location'])
                pygame.image.save(image['tile'], f"./ResultMap/{name}.png")



if __name__ == "__main__":
    print('hello')
    datetime = datetime.now()
    print(type(str(datetime)))
    # mapgenerator = MapGenerator()