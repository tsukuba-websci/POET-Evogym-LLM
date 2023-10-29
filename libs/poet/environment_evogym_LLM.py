
import os
import copy
from itertools import count
import json
import pickle
import numpy as np


import matplotlib.pyplot as plt

import neat_cppn


class EnvironmentEvogym: # CPPNの環境をLLMでの関数に置き換える。
    def __init__(self, key, LLM_env):
        self.key = key
        #self.cppn_genome = cppn_genome
        #self.terrain_params = terrain_params
        self.terrain = None
        self.LLM_env = LLM_env # LLMの環境を受け取る。

    def make_terrain(self, decode_function, genome_config):# 生成した環境をEvogym用に変換する(LLMでは関数でここまで行うので不要)
        terrain = decode_function(self.cppn_genome, genome_config, self.terrain_params)
        self.terrain = terrain
        for platform in terrain['objects'].values():
            indices = platform['indices']
            for i,nei in platform['neighbors'].items():
                for n in nei:
                    assert n in indices, f'{i}: n'
    
    def make_terrain_LLM(self,env):# LLMの環境をEvogym用に変換する。
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
    

    def save_terrain_figure(self, filename):# 環境の画像を保存する。
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

    def get_env_info(self, config):# 環境の情報を取得する。

        env_kwargs = dict(**config.robot, terrain=self.terrain)

        make_env_kwargs = {
            'env_id': config.env_id,
            'env_kwargs': env_kwargs,
            'seed': 0,
        }
        return make_env_kwargs

    def reproduce(self, config):#環境の再生成を行う。
        key = config.get_new_env_key()
        #child_cppn = config.reproduce_cppn_genome(self.cppn_genome)
        #child_params = config.reproduce_terrain_params(self.terrain_params)
        child_LLM = self.LLM_env# LLMの環境を受け取る。
        child = EnvironmentEvogym(key, child_LLM)
        #child.make_terrain(config.decode_cppn, config.neat_config.genome_config)
        child.make_terrain_LLM(child_LLM)
        return child



class EnvrionmentEvogymConfig:
    def __init__(self,
                 robot,
                 neat_config,
                 LLM_env,
                 env_id='Parkour-v0',
                 max_width=80,
                 first_platform=10):

        self.env_id = env_id
        self.robot = robot
        self.neat_config = neat_config
        self.env_indexer = count(0)
        self.cppn_indexer = count(0)
        self.params_indexer = count(0)

        #decoder = EvogymTerrainDecoder(max_width, first_platform=first_platform)
        #self.decode_cppn = decoder.decode
        self.decode_cppn = LLM_env
    
    def get_new_env_key(self):
        return next(self.env_indexer)

    def make_init(self):
        #cppn_key = self.get_new_env_key()
        #cppn_genome = self.neat_config.genome_type(cppn_key)
        #cppn_genome.configure_new(self.neat_config.genome_config)

        env_key = next(self.env_indexer)
        environment = EnvironmentEvogym(env_key, self.decode_cppn)
        #environment.make_terrain(self.decode_cppn, self.neat_config.genome_config)
        environment.make_terrain_LLM(self.decode_cppn)
        return environment