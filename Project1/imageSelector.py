import pygame
import time
import random
import os

from os import listdir
from pygame.locals import *
class ImageSelector:

    def __init__(self, path, game):
        self.game = game
        self.screen = game.screen

        self.images = self.loadImages(path)

    def loadImages(self, path):
        files = listdir(path)
        imagefiles = []
        for filename in files:
            if '.png' in filename:
                imagefiles.append(filename)

        images = self.process(imagefiles)
        return images

    def process(self, imagefiles):
        segment = len(imagefiles)
        width, height = pygame.display.get_surface().get_size()
        print(width,height)
        horicenter =  width / (segment+1)
        verticenter = (height - 50) / 5
        diff = int(2 * min(horicenter, verticenter) / 5)
        images = []
        index = 0

        curverticenter = verticenter
        for angle in range(0, 360, 90):
            curhoricenter = horicenter
            for imagefile in imagefiles:
                image = {}
                originalTile = pygame.image.load(os.path.join("./Images", imagefile)).convert()
                rotateTile = pygame.transform.rotate(originalTile, angle)
                image['name'] = "./Images/" + imagefile + "," + str(angle)
                image['tile'] = pygame.transform.scale(rotateTile, (2*diff,2*diff))
                image['rect'] = pygame.Rect((curhoricenter - diff,curverticenter - diff,2 * diff,2 * diff))
                image['selected'] = False
                image['combined'] = False
                images.append(image)
                index += 1
                curhoricenter += horicenter
            curverticenter += verticenter

        return images


    def draw(self):
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

if __name__ == "__main__":
    # files = listdir('./Images')
    # print(files)
    # imagefiles = []
    # for filename in files:
    #     if '.png' in filename:
    #         imagefiles.append(filename)
    # print(imagefiles)
    print(min(1,2))