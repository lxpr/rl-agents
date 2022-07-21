import logging

from gym.utils import seeding

logger = logging.getLogger(__name__)


class IDMAgent():
    """
        An agent that uses IDM.
    """

    def __init__(self,
                 env,
                 config=None):
        """
            A new IDM agent.
        :param env: The environment
        :param config: The agent configuration. Use default if None.
        """
        
        self.env = env
        self.previous_actions = []
        self.remaining_horizon = 0
        self.steps = 0
        self.config = config

    def act(self, state):
        return "IDM"
    
    def set_writer(self, writer):
        """
            Set a tensorboard writer to log the agent internal variables.
        :param SummaryWriter writer: a summary writer
        """
        self.writer = writer
        
    def load(self, filename):
        return False
    
    def seed(self, seed=None):
        """
            Seed the planner randomness source, e.g. for rollout policy
        :param seed: the seed to be used
        :return: the used seed
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self):
        # self.planner.step_by_reset()
        self.remaining_horizon = 0
        self.steps = 0
        
    def plan(self, state):
        """
            Plan an optimal trajectory from an initial state.

        :param state: s, the initial state of the agent
        :return: [a0, a1, a2...], a sequence of actions to perform
        """
        return [self.act(state)]
    
    def record(self, state, action, reward, next_state, done, info):
        pass



