from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from base_env import SC2Env
from gym import spaces
import logging
import numpy as np
import random

logger = logging.getLogger(__name__)


class MTBEnv(SC2Env):
    metadata = {'render.modes': ['human']}
    default_settings = { 
        'map_name': "MoveToBeacon",
        'players': [sc2_env.Agent(sc2_env.Race.terran)],
        'agent_interface_format': features.AgentInterfaceFormat(
            action_space=actions.ActionSpace.RAW,
            use_raw_units=True,
            raw_resolution=64),
        'realtime': False
    }

    def __init__(self, **kwargs): 
        super().__init__()
        self.kwargs = kwargs
        self.environment = None
        self.marines = []
        self.beacon = []
        self.obs = None 
        self.step_count = 0
        self.ep_reward = 0
        self._episode = 0

        self.action_space = spaces.Discrete(5) #olası action sayısı 5
        self.observation_space = spaces.Box(
            low=0,
            high=120,
            shape=(120,),
            dtype=np.uint8
        )

    def init_env(self): #environment fonksiyonu oluşturma
        args = {**self.default_settings, **self.kwargs}
        self.environment = sc2_env.SC2Env(**args)

    def reset(self): # When finish episode, mini game is reset 
        if self.environment is None:
            self.init_env()

        self.marines = []
        self.beacon = []

        self._episode += 1
        self._num_step = 0
        self._episode_reward = 0

        raw_obs = self.environment.reset()[0]
        return self.get_derived_obs(raw_obs)

    def get_derived_obs(self, raw_obs): 
        self.obs = raw_obs
        marines = self.get_units_by_type(raw_obs, units.Terran.Marine, 1) #get marines
        
        beacon = self.get_beacon(raw_obs) #get beacon
        self.marines = []
        self.beacon = []
        
        #create an array for all observation 
        obs = np.zeros((40,3), dtype=np.uint8)
        
        for i, m in enumerate(marines):
            self.marines.append(m)
            obs[i] = np.array([m.x, m.y, m[2]]) #m[2] -> number of marines 

        for i, r in enumerate(beacon):
            self.beacon.append(r)
            obs[i + len(marines)] = np.array([r.x, r.y, r[2]])

        return obs.reshape(-1)

    def step(self, action): #step function
        raw_obs = self.take_action(action)
        reward = raw_obs.reward
        obs = self.get_derived_obs(raw_obs)
        self._num_step += 1
        self._episode_reward += reward
        self._total_reward += reward
        done = raw_obs.last()
        info = self.get_info() if done else {}
        # each step will set the dictionary to emtpy
        return obs, reward, done, info

    def take_action(self, action):

        if action == 0:
            action_mapped = self.move_up()
        elif action == 1:
            action_mapped = self.move_down()
        elif action == 2:
            action_mapped = self.move_left()
        elif action == 3:
            action_mapped = self.move_right()
        elif action == 4:
            action_mapped = self.move_beacon()
        else:
            action_mapped = actions.RAW_FUNCTIONS.no_op() #no operation

        raw_obs = self.environment.step([action_mapped])[0] #take action and then step
        return raw_obs

    def get_beacon(self, obs): #getting units byy their type , player relative selects unit's allience parameter
        """
        NONE = 0
        SELF = 1
        ALLY = 2
        NEUTRAL = 3
        ENEMY = 4
        """
        return [unit for unit in obs.observation.raw_units
                if unit.alliance == 3]

    def get_units_by_type(self, obs, unit_type, player_relative=0): #getting units byy their type , player relative selects unit's allience parameter
        """
        NONE = 0
        SELF = 1
        ALLY = 2
        NEUTRAL = 3
        ENEMY = 4
        """
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.alliance == player_relative]

    def move_up(self):
        try:
            selected = self.marines[0]
            new_coordinates = [selected.x, selected.y - 2]
            return actions.RAW_FUNCTIONS.Move_pt("now", selected.tag, new_coordinates)
        except:
            return actions.RAW_FUNCTIONS.no_op()

    def move_down(self):
        try:
            selected = self.marines[0]
            new_coordinates = [selected.x, selected.y + 2]
            return actions.RAW_FUNCTIONS.Move_pt("now", selected.tag, new_coordinates)
        except:
            return actions.RAW_FUNCTIONS.no_op()

    def move_left(self):
        try:
            selected = self.marines[0]
            new_coordinates = [selected.x - 2, selected.y]
            return actions.RAW_FUNCTIONS.Move_pt("now", selected.tag, new_coordinates)
        except:
            return actions.RAW_FUNCTIONS.no_op()

    def move_right(self):
        try:
            selected = self.marines[0]
            new_coordinates = [selected.x + 2, selected.y]
            return actions.RAW_FUNCTIONS.Move_pt("now", selected.tag, new_coordinates)
        except:
            return actions.RAW_FUNCTIONS.no_op()

    def move_beacon(self):
        try:
            selected = self.marines[0]
            beacon = self.beacon[0]
            new_coordinates = [beacon.x, beacon.y ]
            return actions.RAW_FUNCTIONS.Move_pt("now", selected.tag, new_coordinates)
        except:
            return actions.RAW_FUNCTIONS.no_op()

    def close(self):

        if self.environment is not None:
            self.environment.close()
        super().close()

    def render(self, mode='human', close=False):
        pass
