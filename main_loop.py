import pygame
import random
from typing import Iterable
import numpy as np
import time

GRID_SIZE = np.array([16, 9])
CELL_SIZE = np.array([16, 16])

clear_color = [115, 209, 94, 255]
screen = None
objects = {}
paused = False
steps = 0
font = None
text_mode = False

key_items = {
    "bot": None,
    "hoe": None,
    "seeds": None,
    "water": None,
}

class GridObject(object):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = None
        self.text_image = None
        self.grid_position = np.array([0, 0])
        pygame.sprite.Group().add(self)

    def _update(self, delta: float) -> None:
        pass

    def _draw(self) -> None:
        if self.image:
            draw_surface(self.text_image if self.text_image else self.image, self.grid_position * CELL_SIZE)

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
    TIL_GROUND = 7
    PLANT_SEED = 8
    WATER_SEED = 9
    DROP_ITEM = 10

    def __init__(self) -> None:
        GridObject.__init__(self)
        self.image = pygame.image.load("robot.png")
        self.text_image = font.render("@", False, pygame.Color(255, 255, 255))
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
        elif action == Bot.PICKUP_WATER:
            self.holding = key_items["water"]
            remove_object(key_items["water"])
        elif action == Bot.TIL_GROUND:
            plant = Plant()
            plant.grid_position = self.grid_position
            add_object(plant, self.grid_position)
        elif action == Bot.PLANT_SEED:
            ground_objects = get_objects_at(self.grid_position)
            for o in ground_objects:
                if isinstance(o, Plant):
                    o.stage = Plant.STAGE_PLANTED
                    break
        elif action == Bot.WATER_SEED:
            ground_objects = get_objects_at(self.grid_position)
            for o in ground_objects:
                if isinstance(o, Plant):
                    o.stage = Plant.STAGE_GROWN
                    break
        elif action == Bot.DROP_ITEM:
            self.holding.grid_position = self.grid_position
            add_object(self.holding, self.holding.grid_position)
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

        ground_objects = get_objects_at(self.grid_position)
        
        for k, v in dirs.items():
            pos = self.grid_position + np.array(v)
            if is_inside_grid(pos) and not is_collision_at(pos):
                actions[k] = True

        if self.holding == None:
            for obj in ground_objects:
                if isinstance(obj, Hoe):
                    actions[Bot.PICKUP_HOE] = True
                elif isinstance(obj, Seeds):
                    actions[Bot.PICKUP_SEEDS] = True
                elif isinstance(obj, Water):
                    actions[Bot.PICKUP_WATER] = True
        else:
            actions[Bot.DROP_ITEM] = True

        plant_state = -1
        for o in ground_objects:
            if isinstance(o, Plant):
                plant_state = o.stage
                break

        if plant_state == -1:
            if isinstance(self.holding, Hoe):
                actions[Bot.TIL_GROUND] = True
        else:
            if plant_state == Plant.STAGE_PLANTED and isinstance(self.holding, Water):
                actions[Bot.WATER_SEED] = True
            elif plant_state == Plant.STAGE_TILLED and isinstance(self.holding, Seeds):
                actions[Bot.PLANT_SEED] = True
        
        return actions

    def _draw(self) -> None:
        GridObject._draw(self)
        if self.holding and self.holding.image:
            draw_surface(self.holding.image, self.grid_position * CELL_SIZE + np.array([8, 0]))

class Hoe(GridObject, pygame.sprite.Sprite):
    def __init__(self) -> None:
        GridObject.__init__(self)
        self.image = pygame.image.load("hoe.png")
        self.text_image = font.render("/", False, pygame.Color(162, 42, 42))
        self.grid_position = np.array([0, 4])

class Seeds(GridObject, pygame.sprite.Sprite):
    def __init__(self) -> None:
        GridObject.__init__(self)
        self.image = pygame.image.load("seeds.png")
        self.text_image = font.render("*", False, pygame.Color(40, 255, 40))
        self.grid_position = np.array([GRID_SIZE[0]-1, 4])

class Water(GridObject, pygame.sprite.Sprite):
    def __init__(self) -> None:
        GridObject.__init__(self)
        self.image = pygame.image.load("water.png")
        self.text_image = font.render("6", False, pygame.Color(0, 0, 255))
        self.grid_position = np.array([8, 0])

class Plant(GridObject, pygame.sprite.Sprite):
    STAGE_TILLED = 0
    STAGE_PLANTED = 1
    STAGE_GROWN = 2

    stage_surfaces = {}

    def __init__(self) -> None:
        GridObject.__init__(self)
        self._stage = 0
        self.stage_surfaces[self.STAGE_TILLED] = pygame.image.load("tilled.png"), font.render("~", False, pygame.Color(162, 42, 42)),
        self.stage_surfaces[self.STAGE_PLANTED] = pygame.image.load("planted.png"), font.render(",", False, pygame.Color(0, 255, 0)),
        self.stage_surfaces[self.STAGE_GROWN] = pygame.image.load("grown.png"), font.render("\"", False, pygame.Color(255, 0, 0))
        self.stage = Plant.STAGE_TILLED


    @property
    def stage(self) -> int:
        return self._stage

    @stage.setter
    def stage(self, to: int) -> None:
        self._stage = to
        self.image = self.stage_surfaces[to][0]
        self.text_image = self.stage_surfaces[to][1]

def draw_surface(surface: pygame.Surface, position: Iterable) -> None:
    screen.blit(surface, position)

def is_inside_grid(position: Iterable) -> bool:
    return (position[0] >= 0 and position[1] >= 0 and position[0] < GRID_SIZE[0] and position[1] < GRID_SIZE[1])

def get_objects_at(position: Iterable) -> Iterable:
    return objects[(position[0], position[1])]

def is_collision_at(position: Iterable) -> bool:
    for obj in get_objects_at(position):
        pass
    
    return False

def move_object(obj: GridObject, position: Iterable) -> None:
    remove_object(obj)
    add_object(obj, position)

def add_object(obj: GridObject, position: Iterable) -> None:
    obj.grid_position = position
    objects[(position[0], position[1])].append(obj)

def remove_object(obj: GridObject) -> None:
    position = obj.grid_position
    if obj in objects[(position[0], position[1])]:
        objects[(position[0], position[1])].remove(obj)

def main() -> None:
    global screen
    global steps
    global key_items
    global font

    for y in range(GRID_SIZE[1]):
        for x in range(GRID_SIZE[0]):
            objects[(x, y)] = []

    pygame.init()
    icon = pygame.image.load("icon.png")
    pygame.display.set_icon(icon)
    pygame.display.set_caption("Machine Learning Farming")
    font = pygame.font.Font("kenney_pixel_square.ttf", 16)

    screen = pygame.display.set_mode((GRID_SIZE[0] * 16, GRID_SIZE[1] * 16), pygame.RESIZABLE)#, pygame.SCALED)
    running = True
    key_items["bot"] = Bot()
    add_object(key_items["bot"], [8, 4])
    key_items["hoe"] = Hoe()
    add_object(key_items["hoe"], [0, 4])
    key_items["seeds"] = Seeds()
    add_object(key_items["seeds"], [15, 4])
    key_items["water"] = Water()
    add_object(key_items["water"], [8, 0])

    frame_ticks_last = pygame.time.get_ticks()

    while running:
        frame_ticks = pygame.time.get_ticks()
        delta_time = (frame_ticks - frame_ticks_last) / 1000.0
        frame_ticks_last = frame_ticks

        for i in range(1000):
            if not paused:
                for key in objects.keys():
                    for obj in objects[key]:
                        obj._update(delta_time)
            
            steps += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                print(steps)

            for key in objects.keys():
                for obj in objects[key]:
                    obj._input(event)
            
        screen.fill(clear_color)
        for key in objects.keys():
            for obj in objects[key]:
                obj._draw()
        pygame.display.update()

if __name__ == "__main__":
    main()