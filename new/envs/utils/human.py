from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.states import JointState

class Human(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)
        self.id = None
        self.isObstacle = False
        self.observed_id = -1
    
    def act(self, ob):
        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action

    def act_joint_state(self, ob):
        action = self.policy.predict(ob)
        return action