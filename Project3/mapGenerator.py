import pygame
import time
import random
import os

from datetime import datetime
from os import listdir
from pygame.locals import *
from button import *
from WFC import *
from Classifier import *
from GeneticAlgorithm import *
MAPSIZE = 100
NUMMAPGENERATE = 100
NUMREQUIRED = 21
STARTPOPFORGENE = 5

class MapGenerator:

    def __init__(self, game):
        self.game = game
        self.screen = game.screen
        self.images = []
        self.button1 = Button((500,480,205,25), 'WFC Regeration', self.game)
        self.button2 = Button((500,520,205,25), 'WFC+ModelRank Regeneration', self.game)
        self.button3 = Button((500,560,205,25), 'Genetic Regeneration', self.game)
        self.button4 = Button((550,20,205,25), 'Save for Training', self.game)
        self.button5 = Button((40,20,205,25), 'Save for Usage', self.game)
        self.button6 = Button((100,480,205,25), 'Train the Model', self.game)
        self.gene = None

    def init(self):
        self.generateWithWFC()

    def generateWithWFC(self):
        selectedImage = self.game.constrainGenerator.selectedImage
        constraints = self.game.constrainGenerator.constraints
        probs = [1] * len(selectedImage)
        selectedImageWithProbs = [selectedImage, probs]
        WFC = WaveFunctionCollaps(selectedImageWithProbs, constraints)
        res = []
        for i in range(NUMREQUIRED):
            res.append(WFC.run())

        location = [(10,50), (120,50), (230, 50), (340, 50), (450, 50), (560, 50), (670,50), 
                    (10,200), (120,200), (230, 200), (340, 200), (450, 200), (560, 200), (670,200), 
                    (10,350), (120,350), (230, 350), (340, 350), (450, 350), (560, 350), (670,350)]
        index = 0
        self.images = []
        # print('res length:', len(res))
        for image in res:
            mode = image.mode
            size = image.size
            data = image.tobytes()
            tile = pygame.image.fromstring(data, size, mode)
            tmp = {}
            tmp['tile'] = pygame.transform.scale(tile, (MAPSIZE,MAPSIZE))
            tmp['rect'] = pygame.Rect((location[index][0],location[index][1],MAPSIZE,MAPSIZE))
            tmp['location'] = location[index]
            tmp['selected'] = False
            tmp['probs'] = probs
            self.images.append(tmp)
            index += 1

    def generateWithModel(self):
        selectedImage = self.game.constrainGenerator.selectedImage
        constraints = self.game.constrainGenerator.constraints
        probs = [1] * len(selectedImage)
        selectedImageWithProbs = [selectedImage, probs]
        WFC = WaveFunctionCollaps(selectedImageWithProbs, constraints)
        imagesToRank = []
        for i in range(NUMMAPGENERATE):
            imagesToRank.append(WFC.run())

        clf = Classifier()
        clf.loadModel()
        res = clf.predict(imagesToRank, NUMREQUIRED)

        location = [(10,50), (120,50), (230, 50), (340, 50), (450, 50), (560, 50), (670,50), 
                    (10,200), (120,200), (230, 200), (340, 200), (450, 200), (560, 200), (670,200), 
                    (10,350), (120,350), (230, 350), (340, 350), (450, 350), (560, 350), (670,350)]
        index = 0
        self.images = []
        for image in res:
            mode = image.mode
            size = image.size
            data = image.tobytes()
            tile = pygame.image.fromstring(data, size, mode)
            tmp = {}
            tmp['tile'] = pygame.transform.scale(tile, (MAPSIZE,MAPSIZE))
            tmp['rect'] = pygame.Rect((location[index][0],location[index][1],MAPSIZE,MAPSIZE))
            tmp['location'] = location[index]
            tmp['selected'] = False
            tmp['probs'] = probs
            self.images.append(tmp)
            index += 1

    def generateWithGeneticAlg(self):
        selectedImage = self.game.constrainGenerator.selectedImage
        constraints = self.game.constrainGenerator.constraints
        if self.gene == None:
            self.gene = Genetic(len(selectedImage), STARTPOPFORGENE)
            probs = self.gene.init()
        else:
            probsSelected = []
            for image in self.images:
                if image['selected'] == True:
                    probsSelected.append(image['probs'])
            probs = self.gene.next(probsSelected)
        res = []
        probslist = []
        for i in range(NUMREQUIRED):
            probsIndex = np.random.choice(len(probs))
            selectedImageWithProbs = [selectedImage, probs[probsIndex]]
            probslist.append(probs[probsIndex])
            WFC = WaveFunctionCollaps(selectedImageWithProbs, constraints)
            res.append(WFC.run())

        location = [(10,50), (120,50), (230, 50), (340, 50), (450, 50), (560, 50), (670,50), 
                    (10,200), (120,200), (230, 200), (340, 200), (450, 200), (560, 200), (670,200), 
                    (10,350), (120,350), (230, 350), (340, 350), (450, 350), (560, 350), (670,350)]
        index = 0
        self.images = []
        for i in range(len(res)):
            mode = res[i].mode
            size = res[i].size
            data = res[i].tobytes()
            tile = pygame.image.fromstring(data, size, mode)
            tmp = {}
            tmp['tile'] = pygame.transform.scale(tile, (MAPSIZE,MAPSIZE))
            tmp['rect'] = pygame.Rect((location[index][0],location[index][1],MAPSIZE,MAPSIZE))
            tmp['location'] = location[index]
            tmp['selected'] = False
            tmp['probs'] = probslist[i]
            self.images.append(tmp)
            index += 1
            # print(tmp['location'], tmp['probs'])

    # def process(self, imagefiles):
    def draw(self):
        self.button1.draw()
        self.button2.draw()
        self.button3.draw()
        self.button4.draw()
        self.button5.draw()
        self.button6.draw()
        for image in self.images:
            self.screen.blit(image['tile'], image['rect'])
            if image['selected'] == True:
                pygame.draw.rect(self.screen, Color('red'), image['rect'], 5)

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
            if self.button3.rect.collidepoint(pygame.mouse.get_pos()):
                self.button3updata()
            if self.button4.rect.collidepoint(pygame.mouse.get_pos()):
                self.button4updata()
            if self.button5.rect.collidepoint(pygame.mouse.get_pos()):
                self.button5updata()
            if self.button6.rect.collidepoint(pygame.mouse.get_pos()):
                self.button6updata()

    def button1updata(self):
        self.generateWithWFC()

    def button2updata(self):
        self.generateWithModel()
    
    def button3updata(self):
        self.generateWithGeneticAlg()

    def button4updata(self):
        for image in self.images:
            if image['selected'] == True:
                time = str(datetime.now())
                name = time + str(image['location'])
                pygame.image.save(image['tile'], f"./data/train/like/{name}.jpg")
            else:
                time = str(datetime.now())
                name = time + str(image['location'])
                pygame.image.save(image['tile'], f"./data/train/dislike/{name}.jpg")
    
    def button5updata(self):
        for image in self.images:
            if image['selected'] == True:
                time = str(datetime.now())
                name = time + str(image['location'])
                pygame.image.save(image['tile'], f"./Result/{name}.jpg")
    
    def button6updata(self):
        clf = Classifier()
        clf.main()



if __name__ == "__main__":
    print('hello')
    datetime = datetime.now()
    print(type(str(datetime)))
    # mapgenerator = MapGenerator()