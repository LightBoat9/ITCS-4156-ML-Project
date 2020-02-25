import pygame
import random
from typing import Iterable
import numpy as np

GRID_SIZE = np.array([16, 9])
CELL_SIZE = np.array([16, 16])

clear_color = [115, 209, 94, 255]
screen = None
objects = []

class GridObject(object):
    def _update(self, delta: float) -> None:
        pass

    def _draw(self) -> None:
        pass

    def _input(self, event: pygame.event.Event):
        pass

    def move(self, direction: Iterable) -> None:
        self.grid_position = np.clip(self.grid_position + direction, 0, GRID_SIZE)

class Bot(GridObject, pygame.sprite.Sprite):
    def __init__(self) -> None:
        pygame.sprite.Sprite.__init__(self)
        
        self.image = pygame.image.load("robot.png")
        self.grid_position = np.array([8, 4])
        pygame.sprite.Group().add(self)

    def _update(self, delta: float) -> None:
        self.move(np.array([random.randint(-1, 1), random.randint(-1, 1)]))

    def _draw(self) -> None:
        draw_surface(self.image, self.grid_position * CELL_SIZE)

    def _input(self, event: pygame.event.Event):
        pass

def draw_surface(surface: pygame.Surface, position: Iterable) -> None:
    screen.blit(surface, position)

def main() -> None:
    global screen

    pygame.init()
    icon = pygame.image.load("icon.png")
    pygame.display.set_icon(icon)
    pygame.display.set_caption("Machine Learning Farming")

    screen = pygame.display.set_mode((GRID_SIZE[0] * 16, GRID_SIZE[1] * 16), pygame.RESIZABLE)
    running = True
    objects.append(Bot())

    frame_ticks_last = pygame.time.get_ticks()

    while running:
        frame_ticks = pygame.time.get_ticks()
        delta_time = (frame_ticks - frame_ticks_last) / 1000.0
        frame_ticks_last = frame_ticks

        for obj in objects:
            obj._update(delta_time)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            for obj in objects:
                obj._input(event)
            
        screen.fill(clear_color)
        for obj in objects:
            obj._draw()
        pygame.display.update()

if __name__ == "__main__":
    main()