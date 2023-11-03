from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.states import JointState

class Robot(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)
        self.sensor_range = config.robot.sensor_range
    
    def act(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set before act!')
        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action
    
    def act_joint_state(self, ob):
        action = self.policy.predict(ob)
        return action