from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from base_env import SC2Env
from gym import spaces
import logging
import numpy as np
import random

logger = logging.getLogger(__name__)


class CMGEnv(SC2Env):
    metadata = {'render.modes': ['human']}
    default_settings = { 
        'map_name': "CollectMineralsAndGas",
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
        self.SCVs = []
        self.supply_depot = []
        self.command_center = []
        self.refinery = []
        self.obs = None 
        self.step_count = 0
        self.ep_reward = 0
        self._episode = 0
        

        #In the map there is 4 vespene gayseres in static location. in order to collect them SCVs must buld refinery on top of that.
        self.refinery1 = None
        self.refinery2 = None
        self.refinery3 = None
        self.refinery4 = None

        self.refinery_counter1 = 0
        self.refinery_counter2 = 0
        self.refinery_counter3 = 0
        self.refinery_counter4 = 0

        self.action_space = spaces.Discrete(7) #olası action sayısı 7
        self.observation_space = spaces.Box(     
            low=0,
            high=100000,
            shape=(11 * 1,),
            dtype=np.int64
        )
    
    def init_env(self): #environment fonksiyonu oluşturma
        args = {**self.default_settings, **self.kwargs}
        self.environment = sc2_env.SC2Env(**args)

    def reset(self): # When finish episode, mini game is reset 
        if self.environment is None:
            self.init_env()

        self.SCVs = []
        self.supply_depot = []
        self.command_center = []
        self.refinery = []

        
        self._episode += 1
        self._num_step = 0
        self._episode_reward = 0

        raw_obs = self.environment.reset()[0]
        return self.get_derived_obs(raw_obs)

    def get_derived_obs(self, raw_obs): 
        self.obs = raw_obs
        SCVs = self.get_units_by_type(raw_obs, units.Terran.SCV, 1) #get scvs
        idle_scvs = [scv for scv in SCVs if scv.order_length == 0] 
        supply_depot = self.get_units_by_type(raw_obs, units.Terran.SupplyDepot, 1) #get supply-depots
        command_center = self.get_units_by_type(raw_obs, units.Terran.CommandCenter, 1) #get command-centers
        refinery = self.get_units_by_type(raw_obs, units.Terran.Refinery, 1) #get refineries
        minerals = raw_obs.observation.player.minerals #get minerals
        free_supply = (raw_obs.observation.player.food_cap -
                       raw_obs.observation.player.food_used) #get available supplies count

        #checking minerals for building units
        can_afford_supply_depot = raw_obs.observation.player.minerals >= 100
        can_afford_barracks = raw_obs.observation.player.minerals >= 150
        can_afford_marine = raw_obs.observation.player.minerals >= 100
        can_afford_refinery = raw_obs.observation.player.minerals >= 75

        #create an array for all observation 
        obs = np.zeros((11, 1), dtype=np.uint8)
        obs[0] = len(SCVs)
        obs[1] = len(supply_depot)
        obs[2] = len(command_center)
        obs[3] = len(refinery)
        obs[4] = can_afford_barracks
        obs[5] = can_afford_supply_depot
        obs[6] = can_afford_marine
        obs[7] = can_afford_refinery
        obs[8] = minerals
        obs[9] = free_supply
        obs[10] = len(idle_scvs)
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
            action_mapped = self.harvest_gas() #collecting gas
        elif action == 1:
            action_mapped = self.train_scv() #training scvs
        elif action == 2:
            action_mapped = self.harvest_minerals() #collecting gas
        elif action == 3:
            action_mapped = self.build_supply_depot()  #building supply depot
        elif action == 4:
            action_mapped = self.build_refinery() #building refinery
        elif action == 5:
            action_mapped = self.build_command_center() #building command center
        else:
            action_mapped = actions.RAW_FUNCTIONS.no_op() #no operation

        raw_obs = self.environment.step([action_mapped])[0] #take action and then step
        return raw_obs

    def get_distances(self, obs, units, xy): #calculate distance between units
        units_xy = [(unit.x, unit.y) for unit in units]
        return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)

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

    def get_my_units_by_type(self, obs, unit_type): #only get self allience in specific unit
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.alliance == features.PlayerRelative.SELF]

    def get_my_completed_units_by_type(self, obs, unit_type): #only get completed units in specific unit
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.build_progress == 100
                and unit.alliance == features.PlayerRelative.SELF]

    def harvest_minerals(self): #collecting minerals micro-action
        scvs = self.get_my_completed_units_by_type(self.obs, units.Terran.SCV) #get all scvs
        idle_scvs = [scv for scv in scvs if scv.order_length == 0]  #get unbusy scvs
        if len(idle_scvs) > 0:
            mineral_patches = [unit for unit in self.obs.observation.raw_units #get all mineral fields
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
            scv = random.choice(idle_scvs) #select random scv(agent)
            if len(mineral_patches) > 0: #if mineral left in map
                command_centers = self.get_my_completed_units_by_type(self.obs, units.Terran.CommandCenter) #completed command centers
                cc_distance = self.get_distances(self.obs, command_centers, (scv.x, scv.y))  #distances between agent and command center
                command_center = command_centers[np.argmin(cc_distance)] #select closest command center
                distances = self.get_distances(self.obs, mineral_patches, (command_center.x, command_center.y)) #distances between mineral fields and command center
                mineral_patch = mineral_patches[np.argmin(distances)] #select closest mineral field
                return actions.RAW_FUNCTIONS.Harvest_Gather_unit(
                    "now", scv.tag, mineral_patch.tag) #harvest mineral into closest command center
        return actions.RAW_FUNCTIONS.no_op()

    def build_supply_depot(self): #building supply depot for scv production
        scvs = self.get_my_completed_units_by_type(self.obs, units.Terran.SCV) 
        if self.obs.observation.player.minerals >= 100: 
            x = random.randint(0, 64)
            y = random.randint(0, 64)
            supply_depot_xy = (x, y)

            scv = random.choice(scvs)
            return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt(
                "now", scv.tag, supply_depot_xy)  #build supply depot in random location with random selected agent
        return actions.RAW_FUNCTIONS.no_op()

    def train_scv(self): #create new scvs
        completed_command_center = self.get_my_completed_units_by_type(
            self.obs, units.Terran.CommandCenter)
        free_supply = (self.obs.observation.player.food_cap -
                       self.obs.observation.player.food_used)
        if (len(completed_command_center) > 0 and self.obs.observation.player.minerals >= 100
                and free_supply > 0): #if more than one command center exists and minerals and supplies enough for train scv
            for cc in range(len(completed_command_center)): #train scv in available command centers
                command_center = self.get_my_completed_units_by_type(self.obs, units.Terran.CommandCenter)[cc]

                if command_center.order_length == 5: #if command center is busy no op
                    return actions.RAW_FUNCTIONS.no_op()

                if command_center.order_length < 5:  #if command center is available train new scv
                    return actions.RAW_FUNCTIONS.Train_SCV_quick("now", command_center.tag)

        return actions.RAW_FUNCTIONS.no_op()

    def build_refinery(self): #build refinery for harvest gas
        scvs = self.get_my_completed_units_by_type(self.obs, units.Terran.SCV) #get trained scv
        refineries = self.get_my_units_by_type(self.obs, units.Terran.Refinery) #get refineries 
        geysers = [unit for unit in self.obs.observation.raw_units #get geysers
                   if unit.unit_type in [
                       units.Neutral.ProtossVespeneGeyser,
                       units.Neutral.PurifierVespeneGeyser,
                       units.Neutral.RichVespeneGeyser,
                       units.Neutral.ShakurasVespeneGeyser,
                       units.Neutral.VespeneGeyser,
                   ]]
        scv = random.choice(scvs) #select random agent(scv)
        if len(geysers) > 0 and len(refineries) < 4: #if there is a geyser(initial 4 geyser) and not all of them turn into refinery
            ccs = self.get_my_completed_units_by_type(self.obs, units.Terran.CommandCenter)  #get command centers
            command_center_xy = [ccs[0].x, ccs[0].y] #cc locations
            distances = self.get_distances(self.obs, geysers, command_center_xy) #distance between geysers and command center
            if len(refineries) == 0: #no refineries
                geyser = geysers[np.argmin(distances)] #closest geyser
                return actions.RAW_FUNCTIONS.Build_Refinery_pt("now", scv.tag, geyser.tag) #build it

            elif len(refineries) >= 2 and len(ccs) < 2: #if there are too many refineries but not enough command center ->no op
                return actions.RAW_FUNCTIONS.no_op()
            else:
                k = len(refineries)
                geyser = geysers[np.argpartition(distances, k)[k]] #sort distances and select last
                return actions.RAW_FUNCTIONS.Build_Refinery_pt("now", scv.tag, geyser.tag) #build it

        return actions.RAW_FUNCTIONS.no_op()

    def harvest_gas(self): #harvest gas in refineries
        scvs = self.get_my_completed_units_by_type(self.obs, units.Terran.SCV)
        idle_scvs = [scv for scv in scvs if scv.order_length == 0]
        refineries = self.get_my_completed_units_by_type(self.obs, units.Terran.Refinery)
        refinery_tags = []
        for refinery in range(len(refineries)):
            if refinery not in refinery_tags:
                refinery_tags.append(refinery)

        if len(refinery_tags) > 0 and len(idle_scvs) > 0:

            choice = random.randint(0, len(refinery_tags)-1)
            if refinery_tags[choice] is not None:

                if choice == 0 and self.refinery_counter1 < 3:
                    scv = random.choice(idle_scvs)
                    self.refinery_counter1 = self.refinery_counter1 + 1
                    return actions.RAW_FUNCTIONS.Harvest_Gather_unit(
                                "now", scv.tag, refinery_tags[choice])

                elif choice == 1 and self.refinery_counter2 < 3:
                    scv = random.choice(idle_scvs)
                    self.refinery_counter2 = self.refinery_counter2 + 1

                    return actions.RAW_FUNCTIONS.Harvest_Gather_unit(
                        "now", scv.tag, refinery_tags[choice])

                elif choice == 2 and self.refinery_counter3 < 3:
                    scv = random.choice(idle_scvs)
                    self.refinery_counter3 = self.refinery_counter3 + 1

                    return actions.RAW_FUNCTIONS.Harvest_Gather_unit(
                        "now", scv.tag, refinery_tags[choice])

                elif choice == 3 and self.refinery_counter4 < 3:
                    scv = random.choice(idle_scvs)
                    self.refinery_counter4 = self.refinery_counter4 + 1

                    return actions.RAW_FUNCTIONS.Harvest_Gather_unit(
                        "now", scv.tag, refinery_tags[choice])

        return actions.RAW_FUNCTIONS.no_op()

    def build_command_center(self):
        scvs = self.get_my_completed_units_by_type(self.obs, units.Terran.SCV)
        completed_command_center = self.get_my_completed_units_by_type(
            self.obs, units.Terran.CommandCenter)

        if len(completed_command_center) == 1 and self.obs.observation.player.minerals >= 400:
            command_center_xy = [completed_command_center[0].x + 5, completed_command_center[0].y]
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

        if self.environment is not None:
            self.environment.close()
        super().close()

    def render(self, mode='human', close=False):
        pass
