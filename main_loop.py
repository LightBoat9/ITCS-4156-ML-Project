import pygame
import random
from typing import Iterable
import numpy as np
import time

GRID_SIZE = np.array([16, 9])
CELL_SIZE = np.array([16, 16])

clear_color = [115, 209, 94, 255]
screen = None
objects = []
paused = False
steps = 0

key_items = {
    "hoe": None,
    "seeds": None,
}

class GridObject(object):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = None
        self.grid_position = np.array([0, 0])
        pygame.sprite.Group().add(self)

    def _update(self, delta: float) -> None:
        pass

    def _draw(self) -> None:
        if self.image:
            draw_surface(self.image, self.grid_position * CELL_SIZE)

    def _input(self, event: pygame.event.Event):
        pass

    def move(self, direction: Iterable) -> None:
        self.grid_position = np.clip(self.grid_position + direction, 0, GRID_SIZE-1)

class Bot(GridObject, pygame.sprite.Sprite):
    MOVE_LEFT = 0
    MOVE_RIGHT = 1
    MOVE_UP = 2
    MOVE_DOWN = 3
    PICKUP_HOE = 4
    PICKUP_SEEDS = 5
    PICKUP_WATER = 6
    PLANT_SEED = 7
    WATER_SEED = 8
    DROP_ITEM = 9

    def __init__(self) -> None:
        GridObject.__init__(self)
        self.image = pygame.image.load("robot.png")
        self.grid_position = np.array([8, 4])
        self.holding = None

    def _update(self, delta: float) -> None:
        available_actions = self.get_available_actions()
        avail = []
        for k, v in available_actions.items():
            if v:
                avail.append(k)

        action = np.random.choice(avail)
        self.do_action(action)

    def do_action(self, action: int) -> None:
        if action == Bot.MOVE_LEFT:
            self.move(np.array([-1, 0]))
        elif action == Bot.MOVE_RIGHT:
            self.move(np.array([1, 0]))
        elif action == Bot.MOVE_UP:
            self.move(np.array([0, -1]))
        elif action == Bot.MOVE_DOWN:
            self.move(np.array([0, 1]))
        elif action == Bot.PICKUP_HOE:
            self.holding = key_items["hoe"]
            remove_object(key_items["hoe"])
        elif action == Bot.PICKUP_SEEDS:
            self.holding = key_items["seeds"]
            remove_object(key_items["seeds"])
        elif action == Bot.DROP_ITEM:
            self.holding.grid_position = self.grid_position
            add_object(self.holding)
            self.holding = None

    def get_available_actions(self) -> int:
        actions = {
            Bot.MOVE_LEFT: False,
            Bot.MOVE_RIGHT: False,
            Bot.MOVE_UP: False,
            Bot.MOVE_DOWN: False,
            Bot.PICKUP_HOE: False,
            Bot.PICKUP_SEEDS: False,
            Bot.PICKUP_WATER: False,
            Bot.PLANT_SEED: False,
            Bot.WATER_SEED: False,
        }

        dirs = {
            Bot.MOVE_LEFT: [-1, 0], 
            Bot.MOVE_RIGHT: [1, 0], 
            Bot.MOVE_UP: [0, -1], 
            Bot.MOVE_DOWN: [0, 1]
        }
        for k, v in dirs.items():
            pos = self.grid_position + np.array(v)
            if is_inside_grid(pos) and not is_collision_at(pos):
                actions[k] = True

        if self.holding == None:
            for obj in get_objects_at(self.grid_position):
                if isinstance(obj, Hoe):
                    actions[Bot.PICKUP_HOE] = True
                elif isinstance(obj, Seeds):
                    actions[Bot.PICKUP_SEEDS] = True
        else:
            actions[Bot.DROP_ITEM] = True
        
        return actions

    def _draw(self) -> None:
        GridObject._draw(self)
        if self.holding and self.holding.image:
            draw_surface(self.holding.image, self.grid_position * CELL_SIZE + np.array([8, 0]))

class Hoe(GridObject, pygame.sprite.Sprite):
    def __init__(self) -> None:
        GridObject.__init__(self)
        self.image = pygame.image.load("hoe.png")
        self.grid_position = np.array([0, 4])

class Seeds(GridObject, pygame.sprite.Sprite):
    def __init__(self) -> None:
        GridObject.__init__(self)
        self.image = pygame.image.load("seeds.png")
        self.grid_position = np.array([GRID_SIZE[0]-1, 4])

def draw_surface(surface: pygame.Surface, position: Iterable) -> None:
    screen.blit(surface, position)

def is_inside_grid(position: Iterable) -> bool:
    return (position[0] >= 0 and position[1] >= 0 and position[0] < GRID_SIZE[0] and position[1] < GRID_SIZE[1])

def get_objects_at(position: Iterable) -> Iterable:
    objs = []
    for o in objects:
        if np.array_equal(o.grid_position, position):
            objs.append(o)
    return objs

def is_collision_at(position: Iterable) -> bool:
    for obj in get_objects_at(position):
        pass
    
    return False

def add_object(obj: GridObject) -> None:
    objects.append(obj)

def remove_object(obj: GridObject) -> None:
    if obj in objects:
        objects.remove(obj)

def main() -> None:
    global screen
    global steps
    global key_items

    pygame.init()
    icon = pygame.image.load("icon.png")
    pygame.display.set_icon(icon)
    pygame.display.set_caption("Machine Learning Farming")

    screen = pygame.display.set_mode((GRID_SIZE[0] * 16, GRID_SIZE[1] * 16), pygame.SCALED | pygame.RESIZABLE)
    running = True
    add_object(Bot())
    key_items["hoe"] = Hoe()
    add_object(key_items["hoe"])
    key_items["seeds"] = Seeds()
    add_object(key_items["seeds"])

    frame_ticks_last = pygame.time.get_ticks()

    while running:
        frame_ticks = pygame.time.get_ticks()
        delta_time = (frame_ticks - frame_ticks_last) / 1000.0
        frame_ticks_last = frame_ticks

        if not paused:
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
        
        steps += 1
        #time.sleep(0.01)

if __name__ == "__main__":
    main()