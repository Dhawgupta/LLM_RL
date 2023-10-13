import io
import random
import sys
import contextlib

from typing import Tuple

import numpy as np

import textworld
from textworld.generator.game import Event, Quest, Game
from textworld.generator.game import GameOptions


def compile_game(game, options: GameOptions, hide_location: bool = True) -> str:
    grammar_flags = {
        "theme": "house",
        "include_adj": False,
        "only_last_action": True,
        "blend_instructions": True,
        "blend_descriptions": True,
        "hide_location": hide_location,
        "instruction_extension": []
    }
    rng_grammar = np.random.RandomState(1234)
    grammar = textworld.generator.make_grammar(grammar_flags, rng=rng_grammar)
    game.change_grammar(grammar)

    game_file = textworld.generator.compile_game(game, options)
    return game_file


def build_game(options: GameOptions) -> Game:
    M = GameMaker()

    bedroom = M.new_room("bedroom")
    office = M.new_room("office")
    bathroom = M.new_room("bathroom")
    livingroom = M.new_room("living room")
    kitchen = M.new_room("kitchen")
    diningroom = M.new_room("dining room")
    garden = M.new_room("garden")
    backyard = M.new_room("backyard")

    hallway1 = M.connect(bedroom.west, livingroom.east)
    hallway2 = M.connect(livingroom.west, diningroom.east)
    hallway3 = M.connect(diningroom.west, kitchen.east)
    hallway4 = M.connect(kitchen.north, garden.south)
    hallway5 = M.connect(livingroom.south, office.north)
    hallway6 = M.connect(office.east, bathroom.west)
    hallway7 = M.connect(bedroom.south, bathroom.north)
    hallway8 = M.connect(kitchen.west, backyard.east)


    table1 = M.new(type='s', name='table') 
    table2 = M.new(type='s', name='table')
    table3 = M.new(type='s', name='table')
    chest1 = M.new(type='c', name='chest')
    fridge = M.new(type='c', name='fridge')
    M.add_fact("closed", chest1)
    M.add_fact("closed", fridge)
    
    bedroom.add(table1)
    office.add(chest1)
    livingroom.add(table2)
    diningroom.add(table3)
    kitchen.add(fridge)

    food = M.new(type="f", name="stale food")
    distractor1 = M.new(type="f", name="fresh food")
    distractor2 = M.new(type="o", name="bowl")
    distractor3 = M.new(type="o", name="coffee cup")
    distractor4 = M.new(type="o", name="plate")
    distractor5 = M.new(type="o", name="utensils")
    distractor6 = M.new(type="f", name="fruit")
    
    table2.add(food)
    table2.add(distractor1)
    table1.add(distractor2)
    table3.add(distractor3)
    table3.add(distractor4)
    chest1.add(distractor5)
    fridge.add(distractor6)

    random_room = random.choice([bedroom, bathroom, office, livingroom, kitchen, diningroom, garden, backyard])

    M.player = M.new(type='P')
    M.set_player(random_room)
    
    quests = []

    food_consumed = Event(conditions={M.new_fact("eaten", food)})
    quests.append(Quest(win_events=[], fail_events=[food_consumed]))

    holding_food = Event(conditions={M.new_fact("in", food, M.inventory)})
    quests.append(Quest(win_events=[holding_food]))

    fridge_closed_with_food = Event(
        conditions={
            M.new_fact("in", food, fridge),
            M.new_fact("closed", fridge)
        })
    quests.append(Quest(win_events=[fridge_closed_with_food], reward=1))
    M.quests = quests

    return game


def build_and_compile_game(hide_location: bool = True) -> Tuple[Game, str]:
    options = GameOptions()

    game = build_game(options)
    game_file = compile_game(game, options, hiden_location)
    return game, game_file
