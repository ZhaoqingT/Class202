import pygame
import time
import random

def text_objects(text, font):
    textSurface = font.render(text, True, (0, 0, 0))
    return textSurface, textSurface.get_rect()

class Button:
    def __init__(self, rect, text, game):
        self.color = (255,0,0)
        self.rect = pygame.Rect(rect)
        self.centerx = self.rect.centerx
        self.centery = self.rect.centery
        self.image = pygame.Surface(self.rect.size)
        self.image.fill(self.color)

        self.game = game
        self.screen = game.screen

        smallText = pygame.font.Font("freesansbold.ttf",20)
        self.textSurf, self.textRect = text_objects(text, smallText)
        self.textRect.center = (self.centerx, self.centery)

    def setText(self, text):
        smallText = pygame.font.Font("freesansbold.ttf",20)
        self.textSurf, self.textRect = text_objects(text, smallText)
        self.textRect.center = (self.centerx, self.centery)

    def draw(self):
        self.screen.blit(self.image, self.rect)
        self.screen.blit(self.textSurf, self.textRect)

    # def get_event(self, event):
    #     if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
    #         if self.rect.collidepoint(pygame.mouse.get_pos()):
    #             self.game.scene1 = not self.game.scene1
