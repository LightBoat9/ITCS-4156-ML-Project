import pygame
import random
from typing import Iterable
import numpy as np
import time
from keras.utils import to_categorical
import collections
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.core import Dense, Dropout

SKIP_STEPS = 1
GENERATIONS = 100
EPSILON_DECAY = 1/(GENERATIONS*2)
MEM_SIZE = 500

GRID_SIZE = np.array([3, 3], dtype=int)
CELL_SIZE = np.array([16, 16], dtype=int)
STATE_COUNT = 8

clear_color = [115, 209, 94, 255]
screen = None
objects = {}
paused = False
steps = 0
generation = 0
font = None

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
    USE_ITEM = 7
    DROP_ITEM = 8

    def __init__(self) -> None:
        GridObject.__init__(self)
        self.image = pygame.image.load("robot.png")
        self.grid_position = np.array([8, 4])
        self.holding = None
        
        self.epsilon = 1
        self.gamma = 0.9
        self.reward = 0
        self.memory = collections.deque(maxlen=MEM_SIZE)

        self._init_model()

    def _init_model(self) -> None:
        self.model = Sequential()
        self.model.add(Dense(output_dim=120, activation="relu", input_dim=STATE_COUNT))
        self.model.add(Dense(output_dim=120, activation="relu"))
        self.model.add(Dense(output_dim=120, activation="relu"))
        self.model.add(Dense(output_dim=9, activation="softmax"))
        self.model.compile(Adam(0.0005), loss="mean_squared_error")

    def _update(self, delta: float) -> None:
        global steps

        self.epsilon = 1 - (generation * EPSILON_DECAY)

        start_actions = self.get_available_actions()
        available_actions = []
        for k, v in start_actions.items():
            if v == 1:
                available_actions.append(k)
        
        start_state = self.get_state(list(start_actions.values()))

        if random.randint(0, 1) < self.epsilon:
            action = np.random.choice(available_actions)
        else:
            a = np.asarray(start_state).reshape((1, STATE_COUNT))
            p = self.model.predict(a)
            action = np.argmax(p[0])
            print(action)

        self.do_action(action, available_actions)

        end_actions = self.get_available_actions()
        end_state = self.get_state(list(end_actions.values()))

        complete = check_win()

        s = np.array(start_state).reshape((1, STATE_COUNT))
        e = np.array(end_state).reshape((1, STATE_COUNT))

        self._train_short(s, e, action, complete)
        self._train_long(s, e, action, complete)

        if complete:
            print(steps)
            paused = True
            key_items["bot"]._redo_memory()
            key_items["bot"].model.save_weights("weights/w1.hdf5")
            init_grid()

    def _train_short(self, start_actions, end_actions, action, complete) -> None:
        t = self.reward
        if not complete:
            t = self.reward + self.gamma * np.amax(self.model.predict(end_actions)[0])

        target = self.model.predict(start_actions)
        target[0][np.argmax(action)] = t

        self.model.fit(start_actions, target, epochs=5, verbose=0)

    def _train_long(self, start_actions, end_actions, action, complete) -> None:
        self.memory.append((start_actions, end_actions, action, self.reward, complete))

    def _redo_memory(self):
        if len(self.memory) > MEM_SIZE:
            minibatch = random.sample(self.memory, MEM_SIZE)
        else:
            minibatch = self.memory

        for start_actions, end_actions, action, reward, complete in minibatch:
            t = reward
            if not complete:
                t = reward + self.gamma * np.amax(self.model.predict(end_actions)[0])
            
            target = self.model.predict(start_actions)
            target[0][np.argmax(action)] = t

            self.model.fit(start_actions, target, epochs=1, verbose=0)

    def do_action(self, action: int, viable_actions: list) -> None:
        self.reward = 0
        
        if action not in viable_actions:
            self.reward = -10
            return
        
        needs_water = False
        needs_tilling = False
        needs_seeds = False

        start_pos = self.grid_position

        for y in range(GRID_SIZE[1]):
            for x in range(GRID_SIZE[0]):
                has_plant = False
                
                for o in get_objects_at([x, y]):
                    if isinstance(o, Plant):
                        has_plant = True

                        plant_state = o.stage
                        if plant_state == Plant.STAGE_PLANTED:
                            needs_tilling = True
                        elif plant_state == Plant.STAGE_TILLED:
                            needs_water = True

                if not has_plant:
                    needs_tilling = True

        if action == Bot.MOVE_LEFT:
            self.move(np.array([-1, 0]))
            self.reward = -100
            # if start_pos[0] == self.grid_position[0] and start_pos[1] == self.grid_position[1]:
            #     self.reward = -10
        elif action == Bot.MOVE_RIGHT:
            self.move(np.array([1, 0]))
            self.reward = -100
            # if start_pos[0] == self.grid_position[0] and start_pos[1] == self.grid_position[1]:
            #     self.reward = -10
        elif action == Bot.MOVE_UP:
            self.move(np.array([0, -1]))
            self.reward = -100
            # if start_pos[0] == self.grid_position[0] and start_pos[1] == self.grid_position[1]:
            #     self.reward = -10
        elif action == Bot.MOVE_DOWN:
            self.move(np.array([0, 1]))
            self.reward = -100
            # if start_pos[0] == self.grid_position[0] and start_pos[1] == self.grid_position[1]:
            #     self.reward = -10
        elif action == Bot.PICKUP_HOE:
            ground_objects = get_objects_at(self.grid_position)

            found_hoe = False

            for o in ground_objects:
                if isinstance(o, Hoe):
                    self.holding = key_items["hoe"]
                    remove_object(key_items["hoe"])
                    found_hoe = True
                    break

            if found_hoe:
                if isinstance(self.holding, Hoe):
                    self.reward = -10 if needs_tilling else 10
            else:
                self.reward = -10
        elif action == Bot.PICKUP_SEEDS:
            ground_objects = get_objects_at(self.grid_position)

            found_seeds = False

            for o in ground_objects:
                if isinstance(o, Seeds):
                    self.holding = key_items["seeds"]
                    remove_object(key_items["seeds"])
                    found_seeds = True
                    break

            if found_seeds:
                if isinstance(self.holding, Seeds):
                    self.reward = -10 if needs_seeds else 10
            else:
                self.reward = -10
        elif action == Bot.PICKUP_WATER:
            ground_objects = get_objects_at(self.grid_position)

            found_water = False

            for o in ground_objects:
                if isinstance(o, Water):
                    self.holding = key_items["water"]
                    remove_object(key_items["water"])
                    found_water = True
                    break

            if found_water:
                if isinstance(self.holding, Water):
                    self.reward = -10 if needs_water else 10
            else:
                self.reward = -10
        elif action == Bot.USE_ITEM:
            self.reward = -10

            if isinstance(self.holding, Hoe):
                ground_objects = get_objects_at(self.grid_position)
                has_plant = False
                for o in ground_objects:
                    if isinstance(o, Plant):
                        has_plant = True
                        break
                
                if not has_plant:
                    self.reward = 10
                    plant = Plant()
                    plant.grid_position = self.grid_position
                    add_object(plant, self.grid_position)
            elif isinstance(self.holding, Seeds):
                ground_objects = get_objects_at(self.grid_position)
                for o in ground_objects:
                    if isinstance(o, Plant):
                        o.stage = Plant.STAGE_PLANTED
                        self.reward = 10
                        break
            elif isinstance(self.holding, Water):
                ground_objects = get_objects_at(self.grid_position)
                for o in ground_objects:
                    if isinstance(o, Plant):
                        self.reward = 10
                        o.stage = Plant.STAGE_GROWN
                        break
        elif action == Bot.DROP_ITEM:
            if (isinstance(self.holding, Hoe) and needs_tilling or
                isinstance(self.holding, Seeds) and needs_seeds or
                isinstance(self.holding, Water) and needs_water):
                self.reward = -10
            
            if (isinstance(self.holding, Hoe) and not needs_tilling or
                isinstance(self.holding, Seeds) and not needs_seeds or
                isinstance(self.holding, Water) and not needs_water):
                self.reward = 10

            self.holding.grid_position = self.grid_position
            add_object(self.holding, self.holding.grid_position)
            self.holding = None

    def get_state(self, actions) -> list:
        global key_items

        needs_water = False
        needs_tilling = False
        needs_seeds = False

        for y in range(GRID_SIZE[1]):
            for x in range(GRID_SIZE[0]):
                has_plant = False
                
                for o in get_objects_at([x, y]):
                    if isinstance(o, Plant):
                        has_plant = True

                        plant_state = o.stage
                        if plant_state == Plant.STAGE_PLANTED:
                            needs_water = True
                        elif plant_state == Plant.STAGE_TILLED:
                            needs_seeds = True

                if not has_plant:
                    needs_tilling = True

        state = [
            # # Hoe Direction
            # key_items["hoe"].grid_position[0] < self.grid_position[0],
            # key_items["hoe"].grid_position[1] < self.grid_position[1],
            # key_items["hoe"].grid_position[0] > self.grid_position[0],
            # key_items["hoe"].grid_position[1] > self.grid_position[1],
            
            # # Seeds Direction
            # key_items["seeds"].grid_position[0] < self.grid_position[0],
            # key_items["seeds"].grid_position[1] < self.grid_position[1],
            # key_items["seeds"].grid_position[0] > self.grid_position[0],
            # key_items["seeds"].grid_position[1] > self.grid_position[1],
            
            # # Water Direction
            # key_items["water"].grid_position[0] < self.grid_position[0],
            # key_items["water"].grid_position[1] < self.grid_position[1],
            # key_items["water"].grid_position[0] > self.grid_position[0],
            # key_items["water"].grid_position[1] > self.grid_position[1],

            self.holding != None,
            isinstance(self.holding, Hoe),
            isinstance(self.holding, Seeds),
            isinstance(self.holding, Water),

            needs_tilling,
            needs_seeds,
            needs_water,

            self.holding_matches_at(self.grid_position),
            # self.holding_matches_at(self.grid_position + np.array([-1, 0])),
            # self.holding_matches_at(self.grid_position + np.array([1, 0])),
            # self.holding_matches_at(self.grid_position + np.array([0, 1])),
            # self.holding_matches_at(self.grid_position + np.array([0, -1])),
            
            # is_inside_grid(self.grid_position + np.array([-1, 0])),
            # is_inside_grid(self.grid_position + np.array([1, 0])),
            # is_inside_grid(self.grid_position + np.array([0, 1])),
            # is_inside_grid(self.grid_position + np.array([0, -1])),
        ]

        for i in range(len(state)):
            state[i] = int(state[i])

        return state# + actions

    def holding_matches_at(self, pos) -> bool:
        needs = get_needs_at(pos)
        return ((isinstance(self.holding, Hoe) and needs[0]) or
            (isinstance(self.holding, Seeds) and needs[1]) or
            (isinstance(self.holding, Water) and needs[2]))


    def get_available_actions(self) -> dict:
        actions = {
            Bot.MOVE_LEFT: 0,
            Bot.MOVE_RIGHT: 0,
            Bot.MOVE_UP: 0,
            Bot.MOVE_DOWN: 0,
            Bot.PICKUP_HOE: 0,
            Bot.PICKUP_SEEDS: 0,
            Bot.PICKUP_WATER: 0,
            Bot.USE_ITEM: 0,
            Bot.DROP_ITEM: 0,
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
                actions[k] = 1

        if self.holding == None:
            for obj in ground_objects:
                if isinstance(obj, Hoe):
                    actions[Bot.PICKUP_HOE] = 1
                if isinstance(obj, Seeds):
                    actions[Bot.PICKUP_SEEDS] = 1
                if isinstance(obj, Water):
                    actions[Bot.PICKUP_WATER] = 1

        if self.holding != None:
            actions[Bot.DROP_ITEM] = 1

        plant_state = -1
        for o in ground_objects:
            if isinstance(o, Plant):
                plant_state = o.stage
                break

        if plant_state == -1:
            if isinstance(self.holding, Hoe):
                actions[Bot.USE_ITEM] = 1
        else:
            if plant_state == Plant.STAGE_PLANTED and isinstance(self.holding, Water):
                actions[Bot.USE_ITEM] = 1
            elif plant_state == Plant.STAGE_TILLED and isinstance(self.holding, Seeds):
                actions[Bot.USE_ITEM] = 1
        
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

class Water(GridObject, pygame.sprite.Sprite):
    def __init__(self) -> None:
        GridObject.__init__(self)
        self.image = pygame.image.load("water.png")
        self.grid_position = np.array([8, 0])

class Plant(GridObject, pygame.sprite.Sprite):
    STAGE_TILLED = 0
    STAGE_PLANTED = 1
    STAGE_GROWN = 2
    STAGE_CRUSHED = 3

    stage_surfaces = {}

    def __init__(self) -> None:
        GridObject.__init__(self)
        self._stage = 0
        self.stage_surfaces[self.STAGE_TILLED] = pygame.image.load("tilled.png"), font.render("~", False, pygame.Color(162, 42, 42)),
        self.stage_surfaces[self.STAGE_PLANTED] = pygame.image.load("planted.png"), font.render(",", False, pygame.Color(0, 255, 0)),
        self.stage_surfaces[self.STAGE_GROWN] = pygame.image.load("grown.png"), font.render("\"", False, pygame.Color(255, 0, 0))
        self.stage_surfaces[self.STAGE_CRUSHED] = pygame.image.load("crushed.png"), font.render("X", False, pygame.Color(255, 0, 0))
        self.stage = Plant.STAGE_TILLED


    @property
    def stage(self) -> int:
        return self._stage

    @stage.setter
    def stage(self, to: int) -> None:
        self._stage = to
        self.image = self.stage_surfaces[to][0]

def draw_surface(surface: pygame.Surface, position: Iterable) -> None:
    screen.blit(surface, position)

def is_inside_grid(position: Iterable) -> bool:
    return (position[0] >= 0 and position[1] >= 0 and position[0] < GRID_SIZE[0] and position[1] < GRID_SIZE[1])

def get_objects_at(position: Iterable) -> Iterable:
    return objects[(position[0], position[1])]

def get_needs_at(position: Iterable) -> Iterable:
    needs = [False, False, False]

    if is_inside_grid(position):
        has_plant = False
        for o in get_objects_at(position):
            if isinstance(o, Plant):
                if o.stage == Plant.STAGE_PLANTED:
                    needs[2] = True
                elif o.stage == Plant.STAGE_TILLED:
                    needs[1] = True
        
        if not has_plant:
            needs[0] = True

    return needs

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

def check_win() -> None:
    for y in range(GRID_SIZE[1]):
        for x in range(GRID_SIZE[0]):
            has_plant = False

            for obj in get_objects_at([x, y]):
                if isinstance(obj, Plant) and (obj.stage == Plant.STAGE_GROWN or obj.stage == Plant.STAGE_CRUSHED):
                    has_plant = True
                    break

            if not has_plant:
                return False
        
    return True

def init_grid() -> None:
    global screen
    global key_items
    global font
    global paused
    global steps
    global generation

    generation += 1
    steps = 0
    paused = False

    objects.clear()
    for y in range(GRID_SIZE[1]):
        for x in range(GRID_SIZE[0]):
            objects[(x, y)] = []

    pygame.init()
    icon = pygame.image.load("icon.png")
    pygame.display.set_icon(icon)
    pygame.display.set_caption("Machine Learning Farming")
    font = pygame.font.Font("kenney_pixel_square.ttf", 16)

    screen = pygame.display.set_mode((GRID_SIZE[0] * 16, GRID_SIZE[1] * 16), pygame.RESIZABLE | pygame.SCALED)

    add_object(key_items["bot"], np.array([GRID_SIZE[0]-1, GRID_SIZE[1]/2], dtype=int))
    add_object(key_items["hoe"], np.array([GRID_SIZE[0]/2, 0], dtype=int))
    add_object(key_items["seeds"], np.array([GRID_SIZE[0]/2, GRID_SIZE[1]-1], dtype=int))
    add_object(key_items["water"], np.array([0, GRID_SIZE[1]/2], dtype=int))

def init_key_items() -> None:
    key_items["bot"] = Bot()
    key_items["hoe"] = Hoe()
    key_items["seeds"] = Seeds()
    key_items["water"] = Water()

def main() -> None:
    global screen
    global steps
    global key_items
    global font
    global paused

    init_key_items()
    init_grid()

    frame_ticks_last = pygame.time.get_ticks()

    running = True

    while running:
        frame_ticks = pygame.time.get_ticks()
        delta_time = (frame_ticks - frame_ticks_last) / 1000.0
        frame_ticks_last = frame_ticks

        for i in range(SKIP_STEPS):
            if not paused:
                for key in objects.keys():
                    for obj in objects[key]:
                        obj._update(delta_time)
            
            steps += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            for key in objects.keys():
                for obj in objects[key]:
                    obj._input(event)
            
        screen.fill(clear_color)
        for key in objects.keys():
            for obj in objects[key]:
                obj._draw()
        pygame.display.update()

        if generation >= GENERATIONS - 1:
            break

if __name__ == "__main__":
    main()