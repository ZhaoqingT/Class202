import pygame
import time
import random
import os
import copy
import math

from os import listdir
from pygame.locals import *
from button import *
from WFC import *

NUMMAPGENERATE = 21

class ConstrainGenerator:

    def __init__(self, game):
        self.game = game
        self.screen = game.screen

        self.target = None
        self.lastTarget = False
        self.finish = False
        self.selectedImage = []
        self.constraints = []
        # self.targetRec = pygame.Rect((350,250,100,100))

    def init(self):
        self.image = []
        self.topimages = []
        self.botimages = []
        self.leftimages = []
        self.rightimages = []
        self.lastTarget = False
        self.button = Button((600,550,105,25), 'Next', self.game)
        self.images = self.getImages(self.game.imageSelector.images)
        self.topimages = self.buildTopImages()
        self.botimages = self.buildBotImages()
        self.leftimages = self.buildLeftImages()
        self.rightimages = self.buildRightImages()
        self.setTarget()
        

    def getImages(self, images):
        res = []
        for image in images:
            tmp = {}
            if image['selected'] == True:
                tmp['tile'] = pygame.transform.scale(image['tile'], (30,30))
                tmp['name'] = image['name']
                tmp['combined'] = False
                res.append(tmp)
        return res

    def setTarget(self):
        for image in self.topimages:
            if image['selected'] == True:
                image['selected'] = False
        for image in self.botimages:
            if image['selected'] == True:
                image['selected'] = False
        for image in self.leftimages:
            if image['selected'] == True:
                image['selected'] = False
        for image in self.rightimages:
            if image['selected'] == True:
                image['selected'] = False

        numOfTarget = 0
        self.hasTarget = False
        for image in self.images:
            if image['combined'] == False:
                numOfTarget += 1
        if numOfTarget == 1:
            print("last")
            self.button.setText('Finish!')
            self.lastTarget = True
        for image in self.images:
            if image['combined'] == False:
                self.target = copy.copy(image)
                self.target['tile'] = pygame.transform.scale(self.target['tile'], (100,100))
                self.hasTarget = True
                break

        self.targetloc = pygame.Rect((350,250,100,100))
        # print(self.target)
    
    def buildTopImages(self):
        res = []
        numEachLine = math.ceil(len(res)/3)
        hori = 300
        verti = 100
        for image in self.images:
            tmp = {}
            tmp['name'] = image['name']
            tmp['tile'] = image['tile']
            tmp['postion'] = 'top'
            tmp['selected'] = False
            tmp['rect'] = pygame.Rect((hori,verti,30,30))
            # print(tmp)
            res.append(tmp)
            # print(res)
            hori += 40
            if hori > 500: 
                hori = 300
                verti += 40
        # print(res)
        return res
    
    def buildBotImages(self):
        res = []
        numEachLine = math.ceil(len(res)/3)
        hori = 300
        verti = 400
        for image in self.images:
            tmp = {}
            tmp['name'] = image['name']
            tmp['tile'] = image['tile']
            tmp['postion'] = 'bot'
            tmp['selected'] = False
            tmp['rect'] = pygame.Rect((hori,verti,30,30))
            # print(tmp)
            res.append(tmp)
            # print(res)
            hori += 40
            if hori > 500: 
                hori = 300
                verti += 40
        # print(res)
        return res

    def buildLeftImages(self):
        res = []
        numEachLine = math.ceil(len(res)/3)
        hori = 100
        verti = 250
        for image in self.images:
            tmp = {}
            tmp['name'] = image['name']
            tmp['tile'] = image['tile']
            tmp['postion'] = 'left'
            tmp['selected'] = False
            tmp['rect'] = pygame.Rect((hori,verti,30,30))
            # print(tmp)
            res.append(tmp)
            # print(res)
            verti += 40
            if verti > 500: 
                hori += 40
                verti = 250
        # print(res)
        return res
    
    def buildRightImages(self):
        res = []
        numEachLine = math.ceil(len(res)/3)
        hori = 600
        verti = 250
        for image in self.images:
            tmp = {}
            tmp['name'] = image['name']
            tmp['tile'] = image['tile']
            tmp['postion'] = 'right'
            tmp['selected'] = False
            tmp['rect'] = pygame.Rect((hori,verti,30,30))
            # print(tmp)
            res.append(tmp)
            # print(res)
            verti += 40
            if verti > 500: 
                hori += 40
                verti = 250
        # print(res)
        return res
    
    def drawAround(self):
        for image in self.topimages:
            self.screen.blit(image['tile'], image['rect'])
            if image['selected'] == True:
                pygame.draw.rect(self.screen, Color('red'), image['rect'], 2)
        for image in self.botimages:
            self.screen.blit(image['tile'], image['rect'])
            if image['selected'] == True:
                pygame.draw.rect(self.screen, Color('red'), image['rect'], 2)
        for image in self.leftimages:
            self.screen.blit(image['tile'], image['rect'])
            if image['selected'] == True:
                pygame.draw.rect(self.screen, Color('red'), image['rect'], 2)
        for image in self.rightimages:
            self.screen.blit(image['tile'], image['rect'])
            if image['selected'] == True:
                pygame.draw.rect(self.screen, Color('red'), image['rect'], 2)

    def draw(self):
        self.button.draw()
        self.drawAround()
        if self.target is not None:
            # self.targetloc.center = (400,300)
            self.screen.blit(self.target['tile'], self.targetloc)

        


    def get_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for image in self.topimages:
                if image['rect'].collidepoint(pygame.mouse.get_pos()):
                    # print(image['name'])
                    image['selected'] = not image['selected']
            for image in self.botimages:
                if image['rect'].collidepoint(pygame.mouse.get_pos()):
                    # print(image['name'])
                    image['selected'] = not image['selected']
            for image in self.leftimages:
                if image['rect'].collidepoint(pygame.mouse.get_pos()):
                    # print(image['name'])
                    image['selected'] = not image['selected']
            for image in self.rightimages:
                if image['rect'].collidepoint(pygame.mouse.get_pos()):
                    # print(image['name'])
                    image['selected'] = not image['selected']
            if self.button.rect.collidepoint(pygame.mouse.get_pos()):
                    self.buttonupdata()

    def recordConstraints(self):
        target = self.target
        top = []
        bot = []
        left = []
        right = []
        for image in self.topimages:
            if image['selected'] == True:
                top.append(image['name'])
        for image in self.botimages:
            if image['selected'] == True:
                bot.append(image['name'])
        for image in self.leftimages:
            if image['selected'] == True:
                left.append(image['name'])
        for image in self.rightimages:
            if image['selected'] == True:
                right.append(image['name'])
        self.constraints.append({'target':target['name'], "top":top, 'bot':bot, "left":left, 'right':right})


    def buttonupdata(self):
        # print("Test")
        for image in self.images:
            if image['name'] == self.target['name']:
                image['combined'] = True
        self.recordConstraints()

        if self.lastTarget:
            self.selectedImage = [image['name'] for image in self.images]
            self.game.mapGenerator.init()
            self.game.scene1 = False
            self.game.scene2 = False
            self.game.scene3 = True

        self.setTarget()



if __name__ == "__main__":
    # files = listdir('./Images')
    # print(files)
    # imagefiles = []
    # for filename in files:
    #     if '.png' in filename:
    #         imagefiles.append(filename)
    # print(imagefiles)
    print(min(1,2))