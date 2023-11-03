import numpy as np
import gym
import logging 
from numpy.linalg import norm

from crowd_sim.envs.utils.states import ObservableState
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.info import Timeout, Collision, Danger, ReachGoal, Nothing

class CrowdSim(gym.Env):
    def __init__(self):
        self.time_limit = None
        self.time_step = None
        self.robot = None
        self.humans = None
        self.global_time = None
        self.step_counter = 0

        self.success_reward = None
        self.collision_penalty = None
        self.disconfort_dist = None
        self.disconfort_penalty_factor = None

        self.config = None
        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        self.randomize_attributes = None

        self.circle_radius = None
        self.human_num = None # number of humans in the environment

        self.action_space = None
        self.observation_space = None

        self.robot_fov = None # field of view of the robot from 0 to pi
        self.human_fov = None # field of view of the humans from 0 to pi

        self.thisSeed = None

        self.nenv = None # number of environments

        self.phase = None #train, val test
        self.test_case = None

        self.render_axis = None
        self.humans = []
        self.potential = None
        self.desiredVelocity = [0.0, 0.0]

        self.last_left = 0.
        self.last_right = 0.

    def configure(self, config):
        self.config = config

        self.time_limit = config.env.time_limit
        self.time_step = config.env.time_step
        self.randomize_attributes = config.env.randomize_attributes

        self.success_reward = config.reward.success_reward
        self.collision_penalty = config.reward.collision_penalty
        self.disconfort_dist = config.reward.discomfort_dist
        self.disconfort_penalty_factor = config.reward.discomfort_penalty_factor

        self.case_capacity = {'train': np.iinfo(np.int32).max-2000, 'val': 1000, 'test': 1000}
        self.case_size = {'train': np.iinfo(np.int32).max-2000, 'val': self.config.env.val_size, 'test': self.config.env.test_size}

        self.circle_radius = config.sim.circle_radius
        self.human_num = config.sim.human_num

        self.arena_size = config.sim.arena_size

        self.case_counter = {'train': 0, 'val': 0, 'test': 0}
        logging.info('human number: %d' % self.human_num)
        if self.randomize_attributes:
            logging.info('randomize humans radius and preffered speed')
        else:
            logging.info('fix humans radius and preffered speed')
        
        logging.info('circle radius: %f' % self.circle_radius)

        self.robot_fov = config.robot.fov
        self.human_fov = config.human.fov
        logging.info('robot field of view: %f' % self.robot_fov)
        logging.info('human field of view: %f' % self.human_fov)

        self.r = self.config.human.radius

        self.random_goal_changing = config.human.random_goal_changing
        if self.random_goal_changing:
            self.goal_change_chance = config.human.goal_change_chance
        
        self.end_goal_changing = config.human.end_goal_changing
        if self.end_goal_changing:
            self.end_goal_change_chance = config.human.end_goal_change_chance
        
        self.last_human_states = np.zeros((self.human_num, 5))

        self.predict_steps = config.sim.predict_steps
        self.human_num_range = config.sim.human_num_range
        
        self.max_human_num = self.human_num + self.human_num_range
        self.min_human_num = self.human_num - self.human_num_range

        self.pred_interval = int(config.data.pred_timestep // config.env.time_step)
        self.buffer_len = self.predict_steps * self.pred_interval

        rob_RL = Robot(config, 'robot')
        self.set_robot(rob_RL)

    def generate_human_position(self, human_num):
        for i in range(human_num):
            self.human.append(self.generate_circle_crossing_human())
    
    def generate_circle_crossing_human(self):
        """Generate a human: generate start position on a circle, goal position is at the opposite side"""
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()

        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            v_pref = 1.0 if human.v_pref == 0 else human.v_pref
            px_noise = (np.random.random() - 0.5) * v_pref
            py_noise = (np.random.random() - 0.5) * v_pref
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise
            collide = False

            for i, agent in enumerate([self.robot] + self.humans):
                # keep human at least 3 meters away from robot
                if self.robot.kinematics == 'unicycle' and i == 0:
                    min_dist = self.circle_radius / 2 # Todo: if circle_radius <= 4, it will get stuck here
                else:
                    min_dist = human.radius + agent.radius + self.discomfort_dist
                if norm((px - agent.px, py - agent.py)) < min_dist or \
                        norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break
            if not collide:
                break

        # px = np.random.uniform(-6, 6)
        # py = np.random.uniform(-3, 3.5)
        # human.set(px, py, px, py, 0, 0, 0)
        human.set(px, py, -px, -py, 0, 0, 0)
        return human

    def update_last_human_states(self, human_visibility, reset):
        for i in range(self.human_num):
            if human_visibility[i]:
                humans = np.array(self.humans[i].get_observable_state_list())
                self.last_human_states[i,:] = humans
            else:
                px, py, vx, vy, r = self.last_human_states[i, :]
                px = px + vx * self.time_step
                py = py + vy * self.time_step
                self.last_human_states[i, :] = np.array([px, py, vx, vy, r])
        
    def get_true_human_states(self):
        true_human_states = np.zeros((self.human_num, 2))
        for i in range(self.human_num):
            humans = np.array(self.humans[i].get_observable_state_list())
            true_human_states[i,:] = humans[:2]
        return true_human_states
    
    def generate_robot_human(self, phase, human_num=None):
        if human_num is None:
            human_num = self.human_num

        if self.robot.kinematics == 'unicycle':
            angle = np.random.uniform(0, np.pi * 2)
            px = self.circle_radius * np.cos(angle)
            py = self.circle_radius * np.sin(angle)
            while True:
                gx, gy = np.random.uniform(-self.circle_radius, self.circle_radius, 2)
                if np.linalg.norm([px - gx, py - gy]) >= 6:  # 1 was 6
                    break
            self.robot.set(px, py, gx, gy, 0, 0, np.random.uniform(0, 2*np.pi)) # randomize init orientation

        # randomize starting position and goal position
        else:
            while True:
                px, py, gx, gy = np.random.uniform(-self.circle_radius, self.circle_radius, 4)
                if np.linalg.norm([px - gx, py - gy]) >= 6:
                    break
            self.robot.set(px, py, gx, gy, 0, 0, np.pi/2)

        # generate humans
        self.generate_random_human_position(human_num=human_num)

    def smooth_action(self, action):
        raise NotImplementedError("looks like for sim2real, we don't need this function")
    
    def reset(self):
        if self.phase is not None:
            phase = self.phase
        if self.test_case is not None:
            test_case = self.test_case

        if self.robot is not None:
            raise NotImplementedError("robot is not None")
        if test_case is not None:
            self.case_counter[phase] = test_case
        self.global_time = 0
        self.step_counter = 0

        self.humans = []
        counter_offset = {'train':self.case_capacity['val']+self.case_capacity['test'], 'val':0, 'test':self.case_capacity['val']}
        np.random.seed(counter_offset[phase] + self.case_counter[phase]+self.thisSeed)

        self.generate_robot_human(phase)

        for agent in [self.robot] + self.human:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step
        
        self.case_counter[phase] = (self.case_counter[phase] + int(self.nenv))% self.case_size[phase]

        ob = self.generate_ob(reset = True)
        self.potential = -abs(np.linalg.norm(np.array([self.robot.px, self.robot.py]) - np.array([self.robot.gx, self.robot.gy])))

        return ob

    def update_human_goals_randomly(self):
        for human in self.humans():
            pass
    
    #---helper functions---#
    def calc_offset_angle(self, state1, state2):
        if self.robot.kinematics == 'holonomic':
            real_theta = np.arctan2(state1.vy, state1.vx)
        else:
            real_theta = state1.theta
        
        v_fov = [np.cos(real_theta), np.sin(real_theta)]
        v_12 = [state2.px - state1.px, state2.py - state1.py]
        v_fov = v_fov /norm(v_fov)
        v_12 = v_12 / norm(v_12)
        offset = np.arccos(np.clip(np.dot(v_fov, v_12), -1.0, 1.0))
        return offset
    
    def detect_visible(self, state1, state2, robot1 = False, custom_fov=None, custom_sensor_range = None):
        if self.robot.kinematics == 'holonomic':
            real_theta = np.arctan2(state1.vy, state1.vx)
        else:
            real_theta = state1.theta
        # angle of center line of FOV of agent1
        v_fov = [np.cos(real_theta), np.sin(real_theta)]

        # angle between agent1 and agent2
        v_12 = [state2.px - state1.px, state2.py - state1.py]
        # angle between center of FOV and agent 2

        v_fov = v_fov / np.linalg.norm(v_fov)
        v_12 = v_12 / np.linalg.norm(v_12)

        offset = np.arccos(np.clip(np.dot(v_fov, v_12), a_min=-1, a_max=1))
        if custom_fov:
            fov = custom_fov
        else:
            if robot1:
                fov = self.robot_fov
            else:
                fov = self.human_fov

        if np.abs(offset) <= fov / 2:
            inFov = True
        else:
            inFov = False

        # detect whether state2 is in state1's sensor_range
        dist = np.linalg.norm([state1.px - state2.px, state1.py - state2.py]) - state1.radius - state2.radius
        if custom_sensor_range:
            inSensorRange = dist <= custom_sensor_range
        else:
            if robot1:
                inSensorRange = dist <= self.robot.sensor_range
            else:
                inSensorRange = True

        return (inFov and inSensorRange)
    
    def get_num_human_in_fov(self):
        human_ids = []
        humans_in_view = []
        num_humans_in_view = 0

        for i in range(self.human_num):
            visible = self.detect_visible(self.robot, self.humans[i], robot1=True)
            if visible:
                humans_in_view.append(self.humans[i])
                num_humans_in_view = num_humans_in_view + 1
                human_ids.append(True)
            else:
                human_ids.append(False)

        return humans_in_view, num_humans_in_view, human_ids
    
    def last_human_states_obj(self):
        '''
        convert self.last_human_states to a list of observable state objects for old algorithms to use
        '''
        humans = []
        for i in range(self.human_num):
            h = ObservableState(*self.last_human_states[i])
            humans.append(h)
        return humans

    def calc_reward(self, action):
        # collision detection
        dmin = float('inf')

        danger_dists = []
        collision = False

        for i, human in enumerate(self.humans):
            dx = human.px - self.robot.px
            dy = human.py - self.robot.py
            closest_dist = (dx ** 2 + dy ** 2) ** (1 / 2) - human.radius - self.robot.radius

            if closest_dist < self.discomfort_dist:
                danger_dists.append(closest_dist)
            if closest_dist < 0:
                collision = True
                # logging.debug("Collision: distance between robot and p{} is {:.2E}".format(i, closest_dist))
                break
            elif closest_dist < dmin:
                dmin = closest_dist


        # check if reaching the goal
        reaching_goal = norm(np.array(self.robot.get_position()) - np.array(self.robot.get_goal_position())) < self.robot.radius

        if self.global_time >= self.time_limit - 1:
            reward = 0
            done = True
            episode_info = Timeout()
        elif collision:
            reward = self.collision_penalty
            done = True
            episode_info = Collision()
        elif reaching_goal:
            reward = self.success_reward
            done = True
            episode_info = ReachGoal()

        elif dmin < self.discomfort_dist:
            # only penalize agent for getting too close if it's visible
            # adjust the reward based on FPS
            # print(dmin)
            reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
            done = False
            episode_info = Danger(dmin)

        else:
            # potential reward
            potential_cur = np.linalg.norm(
                np.array([self.robot.px, self.robot.py]) - np.array(self.robot.get_goal_position()))
            reward = 2 * (-abs(potential_cur) - self.potential)
            self.potential = -abs(potential_cur)

            done = False
            episode_info = Nothing()


        # if the robot is near collision/arrival, it should be able to turn a large angle
        if self.robot.kinematics == 'unicycle':
            # add a rotational penalty
            # if action.r is w, factor = -0.02 if w in [-1.5, 1.5], factor = -0.045 if w in [-1, 1];
            # if action.r is delta theta, factor = -2 if r in [-0.15, 0.15], factor = -4.5 if r in [-0.1, 0.1]
            r_spin = -5 * action.r**2

            # add a penalty for going backwards
            if action.v < 0:
                r_back = -2 * abs(action.v)
            else:
                r_back = 0.

            reward = reward + r_spin + r_back

        return reward, done, episode_info
    
    def generate_ob(self, reset):
        visible_human_states, num_visible_humans, human_visibility = self.get_num_human_in_fov()
        self.update_last_human_states(human_visibility, reset=reset)
        if self.robot.policy.name in ['lstm_ppo', 'srnn']:
            ob = [num_visible_humans]
            # append robot's state
            robotS = np.array(self.robot.get_full_state_list())
            ob.extend(list(robotS))

            ob.extend(list(np.ravel(self.last_human_states)))
            ob = np.array(ob)

        else: # for orca and sf
            ob = self.last_human_states_obj()

        return ob
    
    def get_human_actions(self):
        # step all humans
        human_actions = []  # a list of all humans' actions

        for i, human in enumerate(self.humans):
            # observation for humans is always coordinates
            ob = []
            for other_human in self.humans:
                if other_human != human:
                    # Else detectable humans are always observable to each other
                    if self.detect_visible(human, other_human):
                        ob.append(other_human.get_observable_state())
                    else:
                        ob.append(self.dummy_human.get_observable_state())

            if self.robot.visible:
                if self.detect_visible(self.humans[i], self.robot):
                    ob += [self.robot.get_observable_state()]
                else:
                    ob += [self.dummy_robot.get_observable_state()]

            human_actions.append(human.act(ob))

        return human_actions
    
