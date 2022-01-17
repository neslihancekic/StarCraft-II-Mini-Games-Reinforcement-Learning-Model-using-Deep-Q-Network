from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from base_env import SC2Env
from gym import spaces
import logging
import numpy as np
import random

logger = logging.getLogger(__name__)


class DREnv(SC2Env):
    metadata = {'render.modes': ['human']} #render to the current display or terminal and return nothing. Usually for human consumption.
    default_settings = {
        'map_name': "DefeatRoaches",
        'players': [sc2_env.Agent(sc2_env.Race.terran)], #specify type of players and list of agent object
        'agent_interface_format': features.AgentInterfaceFormat( #specify observation and action to be used by each agent 
            action_space=actions.ActionSpace.RAW,  #enables the raw actions instead of the regular screen and mini-map actions
            use_raw_units=True, #including raw unit data in observations.
            raw_resolution=64), #map_size 64 x 64 
        'realtime': False 
    }

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.environment = None 
        self.marines_list = [] #keep marines
        self.roaches_list = [] #keep roaches
        self.observation = None
        self.step_count = 0
        self.ep_reward = 0
        self.episode_count = 0
        self.action_space = spaces.Discrete(37) # 9 marines x 4 possible action and no op = 37 actions
        self.observation_space = spaces.Box(
            low=0,
            high=150,
            shape=(150,),
            dtype=np.uint8
        )

    def create_environment(self): #create environment function 
        args = {**self.default_settings, **self.kwargs}
        self.environment = sc2_env.SC2Env(**args)

    def reset(self): # When finish episode, mini game is reset 
        if self.environment is None:
            self.create_environment()
        self.roaches_list = []
        self.marines_list = []
        self.episode_count = self.episode_count + 1 #inrease episode number 
        self.step_count = 0 #reset step
        self.ep_reward = 0 #reset reward 

        return self.get_derived_obs(self.environment.reset()[0]) #???????????????????????
    
    def get_units_by_type(self, obs, unit_type, player_relative=0):  # return requested units 
        return [unit for unit in obs.observation.raw_units if unit.unit_type == unit_type and unit.alliance == player_relative]
    """
        NONE = 0
        SELF = 1
        ALLY = 2
        NEUTRAL = 3
        ENEMY = 4
    """

    def get_derived_obs(self, raw_obs):
        self.observation = raw_obs
        obs = np.zeros((50, 3), dtype=np.uint8)
        marines = self.get_units_by_type(raw_obs, units.Terran.Marine, 1)
        roaches = self.get_units_by_type(raw_obs, units.Zerg.Roach, 4)
        self.marines_list = []
        self.roaches_list = []

        for i, m in enumerate(marines):
            self.marines_list.append(m)
            obs[i] = np.array([m.x, m.y, m[2]]) #m[2] -> number of marines 

        for i, r in enumerate(roaches):
            self.roaches_list.append(r)
            obs[i + len(marines)] = np.array([r.x, r.y, r[2]])

        return obs.reshape(-1)

    def step(self, action):
        raw_obs = self.take_action(action)
        reward = raw_obs.reward
        obs = self.get_derived_obs(raw_obs)
        self.step_count = self.ep_reward +  1
        self.ep_reward =   self.ep_reward + reward
        self._total_reward = self._total_reward + reward
        done = raw_obs.last()
        info = self.get_info() if done else {} # each step will set the dictionary to emtpy
        return obs, reward, done, info

    def take_action(self, action):
        if action == 0:
            action_mapped = actions.RAW_FUNCTIONS.no_op()
        elif action <= 32: 
            derived_action = np.floor((action - 1) / 8)
            idx = (action - 1) % 8
            if derived_action == 0:
                action_mapped = self.move_up(idx)
            elif derived_action == 1:
                action_mapped = self.move_down(idx)
            elif derived_action == 2:
                action_mapped = self.move_left(idx)
            else:
                action_mapped = self.move_right(idx)
        else:
            action_mapped = self.all_attack()

        raw_obs = self.environment.step([action_mapped])[0]
        return raw_obs

    def move_up(self, idx):
        idx = np.floor(idx)
        try:
            selected = self.marines_list[idx]
            new_coordinates = [selected.x, selected.y - 2]
            return actions.RAW_FUNCTIONS.Move_pt("now", selected.tag, new_coordinates)
        except:
            return actions.RAW_FUNCTIONS.no_op()

    def move_down(self, idx):
        try:
            selected = self.marines_list[idx]
            new_coordinates = [selected.x, selected.y + 2]
            return actions.RAW_FUNCTIONS.Move_pt("now", selected.tag, new_coordinates)
        except:
            return actions.RAW_FUNCTIONS.no_op()

    def move_left(self, idx):
        try:
            selected = self.marines_list[idx]
            new_coordinates = [selected.x - 2, selected.y]
            return actions.RAW_FUNCTIONS.Move_pt("now", selected.tag, new_coordinates)
        except:
            return actions.RAW_FUNCTIONS.no_op()

    def move_right(self, idx):
        try:
            selected = self.marines_list[idx]
            new_coordinates = [selected.x + 2, selected.y]
            return actions.RAW_FUNCTIONS.Move_pt("now", selected.tag, new_coordinates)
        except:
            return actions.RAW_FUNCTIONS.no_op()

    def attack(self, aidx, eidx):
        try:
            selected = self.marines_list[aidx]
            target = self.roaches_list[eidx]
            return actions.RAW_FUNCTIONS.Attack_unit("now", selected.tag, target.tag)
        except:
            return actions.RAW_FUNCTIONS.no_op()

    def all_attack(self):
        marine_tags = []
        marines = self.get_my_units_by_type(self.observation, units.Terran.Marine)
        roaches = self.get_enemy_units_by_type(self.observation, units.Zerg.Roach)
        for marine in range(len(marines)):
            marine_tags.append(marines[marine].tag) 
        target = random.choice(roaches)
        return actions.RAW_FUNCTIONS.Attack_unit("now", marine_tags, target.tag)

    def get_enemy_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.alliance == features.PlayerRelative.ENEMY]

    

    def get_my_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.alliance == features.PlayerRelative.SELF]

    def close(self):

        if self.environment is not None:
            self.environment.close()
        super().close()

    def render(self, mode='human', close=False):
        pass
