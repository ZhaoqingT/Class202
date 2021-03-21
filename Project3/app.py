import pygame
import time
import random
from pygame.locals import *   

from button import *
from imageSelector import *
from constrainGenerator import *
from WFC import *
from mapGenerator import *
 
 
display_width = 800
display_height = 600
 
black = (0,0,0)
white = (255,255,255)

red = (200,0,0)
green = (0,200,0)

bright_red = (255,0,0)
bright_green = (0,255,0)
 
block_color = (53,115,255)

def text_objects(text, font):
    textSurface = font.render(text, True, black)
    return textSurface, textSurface.get_rect()

class Game:

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((display_width,display_height))
        self.clock = pygame.time.Clock()
        self.button1 = Button((350,550,105,25), 'Select', self)
        self.button2 = Button((100,550,105,25), 'Reset', self)
        self.imageSelector = ImageSelector('./Images', self)
        self.constrainGenerator = ConstrainGenerator(self)
        self.mapGenerator = MapGenerator(self)

        self.done = False
        self.scene1 = True
        self.scene2 = False
        self.scene3 = False
        

        pygame.display.set_caption('Map Generator')

    def main_loop(self):
        """Game() main loop"""        
        while not self.done:
            # get key input, move, draw.
            self.handle_events()
            self.draw()                      
            self.clock.tick(60)

    def draw(self):
        """render screen"""
        if self.scene1 == True:
            self.screen.fill(Color("white")) 
            self.button1.draw()
            self.imageSelector.draw()
            pygame.display.flip()
        elif self.scene2 == True:
            self.screen.fill(Color("white")) 
            self.constrainGenerator.draw()
            self.button2.draw()
            pygame.display.flip()
        elif self.scene3 == True:
            self.screen.fill(Color("white")) 
            self.button2.draw()
            self.mapGenerator.draw()
            pygame.display.flip()

    def handle_events(self):
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT: 
                pygame.quit()
                quit()
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE: self.done = True
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.button1.rect.collidepoint(pygame.mouse.get_pos()) and self.scene1 == True:
                    self.button1updata()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.button2.rect.collidepoint(pygame.mouse.get_pos()):
                    self.button2updata()
            if self.scene1 == True:
                self.imageSelector.get_event(event)
            elif self.scene2 == True:
                self.constrainGenerator.get_event(event)
            elif self.scene3 == True:
                self.mapGenerator.get_event(event)
    
    def button1updata(self):
        self.scene1 = False
        self.scene2 = True
        self.scene3 = False
        self.constrainGenerator.init()
    
    def button2updata(self):
        self.scene1 = True
        self.scene2 = False
        self.scene3 = False

if __name__ == "__main__":
    game = Game()
    game.main_loop()