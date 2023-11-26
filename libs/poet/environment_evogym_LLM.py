
import os
import copy
from itertools import count
import json
import pickle
import numpy as np
import random
import openai

import matplotlib.pyplot as plt

import neat_cppn


openai.api_key = os.getenv("OPENAI_API_KEY")

class EnvironmentEvogym: # class for the environment
    def __init__(self, key, LLM_env, prompt):
        self.key = key
        self.terrain = None
        self.prompt = prompt
        self.LLM_env = LLM_env # get the environment from LLM

    def make_terrain(self, decode_function, genome_config):# transform the environment to the format of CPPN(this function is not used in this code)
        terrain = decode_function(self.cppn_genome, genome_config, self.terrain_params)
        self.terrain = terrain
        for platform in terrain['objects'].values():
            indices = platform['indices']
            for i,nei in platform['neighbors'].items():
                for n in nei:
                    assert n in indices, f'{i}: n'
    
    def make_terrain_LLM(self,env):# transform the environment to the format of LLM
        self.terrain = env
        for platform in env['objects'].values():
            indices = platform['indices']
            for i,nei in platform['neighbors'].items():
                for n in nei:
                    assert n in indices, f'{i}: n'

    def archive(self):
        pass

    def admitted(self, config):
        pass

    
    def save(self, path):
        terrain_json = os.path.join(path, 'terrain.json')
        with open(terrain_json, 'w') as f:
            json.dump(self.terrain, f)

        terrain_figure = os.path.join(path, 'terrain.jpg')
        self.save_terrain_figure(terrain_figure)
    

    def save_terrain_figure(self, filename):# save the figure of the environment
        width, height = self.terrain['grid_width'], self.terrain['grid_height']+5
        fig, ax = plt.subplots(figsize=(width/8, height/8))
        for platform in self.terrain['objects'].values():
            for idx, t in zip(platform['indices'],platform['types']):
                x = idx % width
                y = idx // width

                if t==2:
                    color = [0.7,0.7,0.7]
                else:
                    color = 'k'
                ax.fill_between([x,x+1], [y+1, y+1], [y, y], fc=color)

        ax.set_xlim([0,width])
        ax.set_ylim([0,height])
        ax.grid()
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

    def get_env_info(self, config):# get the information of the environment

        env_kwargs = dict(**config.robot, terrain=self.terrain)

        make_env_kwargs = {
            'env_id': config.env_id,
            'env_kwargs': env_kwargs,
            'seed': 0,
        }
        return make_env_kwargs

    def reproduce(self, config):# regenerate the environment with LLM
        key = config.get_new_env_key()
        parent_prompt =  create_prompt(self.prompt)
        print(parent_prompt)
        self.save_prompt(parent_prompt)
        child_LLM = generate_env(parent_prompt)
        child = EnvironmentEvogym(key, child_LLM, parent_prompt)
        child.make_terrain_LLM(child_LLM)
        return child
    
    def save_prompt(self,prompt):# save the prompt
        with open('./out/prompts.txt', 'a') as f:
            f.write(prompt)
            f.write('\n')



class EnvrionmentEvogymConfig:
    def __init__(self,
                 robot,
                 neat_config,
                 prompt,
                 LLM_env,
                 env_id='Parkour-v0',
                 max_width=80,
                 first_platform=10):

        self.env_id = env_id
        self.robot = robot
        self.prompt = prompt
        self.neat_config = neat_config
        self.env_indexer = count(0)
        #self.cppn_indexer = count(0)
        #self.params_indexer = count(0)
        #decoder = EvogymTerrainDecoder(max_width, first_platform=first_platform)
        #self.decode_cppn = decoder.decode
        self.decode_cppn = LLM_env
    
    def get_new_env_key(self):
        return next(self.env_indexer)

    def make_init(self):
        env_key = next(self.env_indexer)
        environment = EnvironmentEvogym(env_key, self.decode_cppn, self.prompt)
        environment.make_terrain_LLM(self.decode_cppn)
        return environment
    
# function to create environment with LLM
def create_env(prompt):
    model_id = "ft:gpt-3.5-turbo-0613:webscience-lab::8DyQDcTj"

    response = openai.ChatCompletion.create(
        model = model_id,
        messages=[
            {"role": "system", "content": "The environment in reinforcement learning, where the agent progresses from left to right, is represented by a list of strings, one character per block. The following conditions are applied: '-' is a blank block, 'H' is a hard block, and 'S' is a soft block. The agent can walk on the 'H' and 'S' blocks and can exist in the '-' area. If there is no 'H' or 'S' block under the agent, it will fall. Please return a list that predicts what kind of environment it is from a prompt that describes the given environment. Please make all elements in the list have the same length. Also, only allow '-' , 'H', and 'S' characters in the elements. Please return with the specified character length. Do not output anything other than a list."},
            {"role": "user", "content": "100*20 size Evolution Gym environment that is simple."},
            {"role": "assistant", "content": "['----------------------------------------------------------------------------------------------------', '----------------------------------------------------------------------------------------------------', '----------------------------------------------------------------------------------------------------', '----------------------------------------------------------------------------------------------------', '----------------------------------------------------------------------------------------------------', '----------------------------------------------------------------------------------------------------', '----------------------------------------------------------------------------------------------------', '----------------------------------------------------------------------------------------------------', '----------------------------------------------------------------------------------------------------', '----------------------------------------------------------------------------------------------------', '----------------------------------------------------------------------------------------------------', '----------------------------------------------------------------------------------------------------', '----------------------------------------------------------------------------------------------------', 'HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH------------------------------------------------', '----------------------------------------------------HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH', '----------------------------------------------------------------------------------------------------', '----------------------------------------------------------------------------------------------------', '----------------------------------------------------------------------------------------------------', '----------------------------------------------------------------------------------------------------', '----------------------------------------------------------------------------------------------------']"},
            {"role": "user", "content": prompt}
        ]
    )
    env_list = eval(response['choices'][0]['message']['content'])
    return env_list

def adjust_list(lst):
    # Initialize with 5 lines of hyphens
    adjusted_list = ['-' * 100 for _ in range(5)]
    for s in lst:
        s = ''.join(c if c in "-HS" else '-' for c in s)
        
        if len(s) == 100:
            adjusted_list.append(s)
        elif len(s) < 100:
            adjusted_list.append(s + '-'*(100 - len(s)))
        else:
            adjusted_list.append(s[:100])
    # Extend with 7 lines of hyphens at the end
    adjusted_list.extend(['-' * 100 for _ in range(7)])   
    return adjusted_list


def check_columns(lst):
    num_hyphens = 0
    for i in range(len(lst[0])):
        for j in range(len(lst)):
            if lst[j][i] != '-':
                num_hyphens = 0
                break
        else:
            num_hyphens += 1

        if num_hyphens >= 5:
            return False

    return True


def process_neighbour(i, j, grid, object_cells, object_type):
    if i < 0 or j < 0 or i >= len(grid) or j >= len(grid[0]) or grid[i][j] == "-":
        return

    cell = (len(grid)-1-i)*len(grid[0]) + j  # fixed here

    if cell in object_cells:
        return

    object_cells.append(cell)

    # Add the current cell's type to the object_type list
    object_type.append(grid[i][j])

    process_neighbour(i+1, j, grid, object_cells, object_type)
    process_neighbour(i-1, j, grid, object_cells, object_type)
    process_neighbour(i, j+1, grid, object_cells, object_type)
    process_neighbour(i, j-1, grid, object_cells, object_type)


def create_json_file(env_list):
    grid = [list(row) for row in env_list]
    objects_dict = dict()
    object_counter = 1

    grid_width = len(grid[0])
    grid_height = len(grid)

    # Initialize start_height with the minimum value
    start_height = 1  # Start from the bottom row (1)
    for i in range(grid_height):
        for j in range(min(5, grid_width)):  # Check only the first 5 columns
            if grid[i][j] == 'H' or grid[i][j] == 'S':
                start_height = max(start_height, grid_height - i)

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == 'H' or grid[i][j] == 'S':
                new_object_indices = []
                new_object_type = []

                process_neighbour(i, j, grid, new_object_indices, new_object_type)

                # Initialize new_object_types with 'S' as 2 and 'H' as 5
                new_object_types = [5 if t == 'H' else 2 for t in new_object_type]
                # Modify the object type based on the requirements
                s_count = 0
                for k, t in enumerate(new_object_type):
                    if t == 'S':
                        s_count += 1
                        if k == 0 or k == len(new_object_type) - 1 or s_count == 5:
                            new_object_types[k] = 5  # Convert to 'H'
                            s_count = 0

                new_object = {}
                new_object['indices'] = new_object_indices
                new_object['types'] = new_object_types
                new_object['neighbors'] = {str(idx): [n for n in new_object_indices if (n//grid_width == idx//grid_width and abs(n%grid_width - idx%grid_width) == 1) or (abs(n//grid_width - idx//grid_width) == 1 and n%grid_width == idx%grid_width)] for idx in new_object_indices}

                objects_dict['new_object_' + str(object_counter)] = new_object
                object_counter += 1 

                for idx in new_object_indices:
                    grid[(len(grid)-1-idx//grid_width)][idx%grid_width] = "-"  # fixed here

    return_dict = {}
    return_dict['grid_width'] = grid_width
    return_dict['grid_height'] = grid_height
    return_dict['start_height'] = start_height
    return_dict['objects'] = objects_dict

    return return_dict

def generate_env(prompt):
    checked_list = False

    while not checked_list:
        try:
            env_list = create_env(prompt)
            json_env = create_json_file(fixed_list)
            fixed_list = adjust_list(env_list)
            checked_list = check_columns(fixed_list)
        except:
            print('re generate env')
            checked_list = False

    return json_env

def recreate_fixed_list(json_env):
    """
    Recreates the fixed_list from the json_env object with inverted y-axis.
    The json_env contains information about different objects, their indices in a grid,
    and their types (represented as 5 for 'H' and 2 for 'S').
    The grid is reconstructed into a list of strings, each representing a row, with an inverted y-axis.
    """
    grid_width = json_env['grid_width']
    grid_height = json_env['grid_height']
    objects = json_env['objects']

    # Initialize the grid with '-' (empty space)
    grid = [['-' for _ in range(grid_width)] for _ in range(grid_height)]

    # Populate the grid with objects, inverting the y-axis
    for obj in objects.values():
        indices = obj['indices']
        types = obj['types']

        for index, obj_type in zip(indices, types):
            x = index % grid_width
            y = grid_height - 1 - (index // grid_width)  # Invert the y-axis
            grid[y][x] = 'H' if obj_type == 5 else 'S'  # 'H' for 5 and 'S' for 2

    # Convert grid rows to strings
    fixed_list = [''.join(row) for row in grid]

    return fixed_list

def create_prompt(prompt):
    model_id = "gpt-4"
    # few shot prompting
    
    # 50% chance to return the original prompt
    if random.random() < 0.5:
        return prompt
    
    response = openai.ChatCompletion.create(
        model = model_id,
        messages=[
            {"role": "system", "content": "The prompt describes an two dimensional environment for reinforcement learning, where the task is to learn to move from left to right. Please return a prompt within 100 characters that describes a slightly more difficult environment than the one described in the sent prompt, using words like stairs, hole, wall, shaped mountain, shaped valley, easy, difficult to describe the shape. The reinforcement learning environment consists only of square hard and soft blocks, and the arrangement of these blocks is described in the prompt. There are no other functional blocks. There are no physical forces in this environment other than gravity. Do not alter the phrase 100*20 size Evolution Gym environment."},
            {"role": "user", "content": "100*20 size Evolution Gym environment that is simple."},
            {"role": "assistant", "content": "100*20 size Evolution Gym environment that is simple with some small holes."},
            {"role": "user", "content": "100*20 size Evolution Gym environment that is simple with some small holes."},
            {"role": "assistant", "content": "100*20 size Evolution Gym environment that is shaped like mountain."},
            {"role": "user", "content": "100*20 size Evolution Gym environment that is shaped like mountain."},
            {"role": "assistant", "content": "00*20 size Evolution Gym environment that is a little difficult."},            
            {"role": "user", "content": prompt}
        ]
    )
    mutated_prompt = response['choices'][0]['message']['content']
    return mutated_prompt
