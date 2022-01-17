from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from base_env import SC2Env
from gym import spaces
import logging
import numpy as np
import random

logger = logging.getLogger(__name__)


class BMEnv(SC2Env):
    metadata = {'render.modes': ['human']}
    default_settings = { 
        'map_name': "BuildMarines",
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
        self.env = None
        self.SCVs = []
        self.supply_depot = []
        self.command_center = []
        self.marines = []
        self.obs = None
        self.step_count = 0
        self.ep_reward = 0
        self._episode = 0

        self.action_space = spaces.Discrete(7) #olası action sayısı 7
        # [0: x, 1: y, 2: hp]
        self.observation_space = spaces.Box(     
            low=0,
            high=100000,
            shape=(12 * 1,),
            dtype=np.int64
        )

    def reset(self):
        if self.env is None:
            self.init_env()

        self.SCVs = []
        self.supply_depot = []
        self.command_center = []
        self.marines = []
        self.barrack = []

        self._episode += 1
        self._num_step = 0
        self._episode_reward = 0

        raw_obs = self.env.reset()[0]
        return self.get_derived_obs(raw_obs)

    def init_env(self): #environment fonksiyonu oluşturma
        args = {**self.default_settings, **self.kwargs}
        self.env = sc2_env.SC2Env(**args)

    def get_derived_obs(self, raw_obs): 
        self.obs = raw_obs
        SCVs = self.get_units_by_type(raw_obs, units.Terran.SCV, 1)
        idle_scvs = [scv for scv in SCVs if scv.order_length == 0]
        supply_depot = self.get_units_by_type(raw_obs, units.Terran.SupplyDepot, 1)
        command_center = self.get_units_by_type(raw_obs, units.Terran.CommandCenter, 1)
        marines = self.get_units_by_type(raw_obs, units.Terran.Marine, 1)
        barrack = self.get_units_by_type(raw_obs, units.Terran.Barracks, 1)
        minerals = raw_obs.observation.player.minerals
        free_supply = (raw_obs.observation.player.food_cap -
                       raw_obs.observation.player.food_used)
        can_afford_supply_depot = raw_obs.observation.player.minerals >= 100
        can_afford_barracks = raw_obs.observation.player.minerals >= 150
        can_afford_marine = raw_obs.observation.player.minerals >= 100

        obs = np.zeros((12, 1), dtype=np.uint8)
        obs[0] = len(SCVs)
        obs[1] = len(idle_scvs)
        obs[2] = len(supply_depot)
        obs[3] = len(command_center)
        obs[4] = len(marines)
        obs[5] = len(barrack)
        obs[7] = can_afford_barracks
        obs[8] = can_afford_supply_depot
        obs[9] = can_afford_marine
        obs[10] = minerals
        obs[11] = free_supply

        return obs.reshape(-1)

    def step(self, action):
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
            action_mapped = actions.RAW_FUNCTIONS.no_op()

        elif action == 1:
            action_mapped = self.train_scv()
        elif action == 2:
            action_mapped = self.harvest_minerals()
        elif action == 3:
            action_mapped = self.build_supply_depot()
        elif action == 4:
            action_mapped = self.build_command_center()
        elif action == 5:
            action_mapped = self.build_barracks()
        else:
            action_mapped = self.train_marine()


        raw_obs = self.env.step([action_mapped])[0]
        return raw_obs

    def do_nothing(self):
        return actions.RAW_FUNCTIONS.no_op()

    def get_distances(self, obs, units, xy):
        units_xy = [(unit.x, unit.y) for unit in units]
        return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)

    def get_units_by_type(self, obs, unit_type, player_relative=0):
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

    def get_my_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.alliance == features.PlayerRelative.SELF]

    def get_my_completed_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.build_progress == 100
                and unit.alliance == features.PlayerRelative.SELF]

    def harvest_minerals(self):
        scvs = self.get_my_completed_units_by_type(self.obs, units.Terran.SCV)
        idle_scvs = [scv for scv in scvs if scv.order_length == 0]
        if len(idle_scvs) > 0:
            mineral_patches = [unit for unit in self.obs.observation.raw_units
                               if unit.unit_type in [
                                   units.Neutral.BattleStationMineralField,
                                   units.Neutral.BattleStationMineralField750,
                                   units.Neutral.LabMineralField,
                                   units.Neutral.LabMineralField750,
                                   units.Neutral.MineralField,
                                   units.Neutral.MineralField750,
                                   units.Neutral.PurifierMineralField,
                                   units.Neutral.PurifierMineralField750,
                                   units.Neutral.PurifierRichMineralField,
                                   units.Neutral.PurifierRichMineralField750,
                                   units.Neutral.RichMineralField,
                                   units.Neutral.RichMineralField750
                               ]]
            scv = random.choice(idle_scvs)
            if len(mineral_patches) > 0:
                command_centers = self.get_my_completed_units_by_type(self.obs, units.Terran.CommandCenter)
                cc_distance = self.get_distances(self.obs, command_centers, (scv.x, scv.y))
                command_center = command_centers[np.argmin(cc_distance)]
                distances = self.get_distances(self.obs, mineral_patches, (command_center.x, command_center.y))
                mineral_patch = mineral_patches[np.argmin(distances)]
                return actions.RAW_FUNCTIONS.Harvest_Gather_unit(
                    "now", scv.tag, mineral_patch.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def build_supply_depot(self):
        free_supply = (self.obs.observation.player.food_cap -
                       self.obs.observation.player.food_used)
        scvs = self.get_my_completed_units_by_type(self.obs, units.Terran.SCV)
        if self.obs.observation.player.minerals >= 100 and free_supply < 50:
            x = random.randint(0, 64)
            y = random.randint(0, 64)
            supply_depot_xy = (x, y)

            scv = random.choice(scvs)
            return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt(
                "now", scv.tag, supply_depot_xy)
        return actions.RAW_FUNCTIONS.no_op()

    def train_scv(self):
        scvs = self.get_my_completed_units_by_type(self.obs, units.Terran.SCV)
        completed_command_center = self.get_my_completed_units_by_type(
            self.obs, units.Terran.CommandCenter)
        free_supply = (self.obs.observation.player.food_cap -
                       self.obs.observation.player.food_used)
        if (len(completed_command_center) > 0 and self.obs.observation.player.minerals >= 100
                and free_supply > 0):
            for cc in range(len(completed_command_center)):
                command_center = self.get_my_completed_units_by_type(self.obs, units.Terran.CommandCenter)[cc]

                if command_center.order_length == 1:
                    return actions.RAW_FUNCTIONS.no_op()

                if command_center.order_length < 1:

                    if not (len(scvs) == 16):
                        return actions.RAW_FUNCTIONS.Train_SCV_quick("now", command_center.tag)
                    else:
                        return actions.RAW_FUNCTIONS.no_op()
        return actions.RAW_FUNCTIONS.no_op()

    def build_barracks(self):
        free_supply = (self.obs.observation.player.food_cap -
                       self.obs.observation.player.food_used)
        completed_supply_depots = self.get_my_completed_units_by_type(
            self.obs, units.Terran.SupplyDepot)
        scvs = self.get_my_units_by_type(self.obs, units.Terran.SCV)

        if (len(completed_supply_depots) > 0 and free_supply > 7 ):
            x = random.randint(0, 64)
            y = random.randint(0, 64)
            barracks_xy = (x, y)
            distances = self.get_distances(self.obs, scvs, barracks_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_Barracks_pt("now", scv.tag, barracks_xy)
        return actions.RAW_FUNCTIONS.no_op()

    def train_marine(self):
        completed_barrackses = self.get_my_completed_units_by_type(
            self.obs, units.Terran.Barracks)
        free_supply = (self.obs.observation.player.food_cap -
                       self.obs.observation.player.food_used)
        if (len(completed_barrackses) > 0 and self.obs.observation.player.minerals >= 100
                and free_supply > 0):
            barracks = self.get_my_units_by_type(self.obs, units.Terran.Barracks)[0]
            if barracks.order_length < 5:
                return actions.RAW_FUNCTIONS.Train_Marine_quick("now", barracks.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def build_command_center(self):
        scvs = self.get_my_completed_units_by_type(self.obs, units.Terran.SCV)
        completed_command_center = self.get_my_completed_units_by_type(
            self.obs, units.Terran.CommandCenter)

        if len(completed_command_center) == 1 and self.obs.observation.player.minerals >= 400:
            command_center_xy = [completed_command_center[0].x + 5, completed_command_center[0].y]
            # distances = self.get_distances(obs, scvs, command_center_xy)
            scv = random.choice(scvs)
            return actions.RAW_FUNCTIONS.Build_CommandCenter_pt(
                "now", scv.tag, command_center_xy)
        else:
            return actions.RAW_FUNCTIONS.no_op()

    def select_scv(self): 
        scvs = self.get_my_completed_units_by_type(self.obs, units.Terran.SCV)
        mineral_patches = [unit for unit in self.obs.observation.raw_units
                           if unit.unit_type in [
                               units.Neutral.BattleStationMineralField,
                               units.Neutral.BattleStationMineralField750,
                               units.Neutral.LabMineralField,
                               units.Neutral.LabMineralField750,
                               units.Neutral.MineralField,
                               units.Neutral.MineralField750,
                               units.Neutral.PurifierMineralField,
                               units.Neutral.PurifierMineralField750,
                               units.Neutral.PurifierRichMineralField,
                               units.Neutral.PurifierRichMineralField750,
                               units.Neutral.RichMineralField,
                               units.Neutral.RichMineralField750
                           ]]
        ccs = self.get_my_completed_units_by_type(self.obs, units.Terran.CommandCenter)
        command_center = ccs[0]
        cc_distance = self.get_distances(self.obs, mineral_patches, (command_center.x, command_center.y))
        mineral = mineral_patches[np.argmin[cc_distance]]
        distances = self.get_distances(self.obs, scvs, (mineral.x, mineral.y))
        scv = scvs[np.argmin(distances)]
        return scv

    def close(self):

        if self.env is not None:
            self.env.close()
        super().close()

    def render(self, mode='human', close=False):
        pass
