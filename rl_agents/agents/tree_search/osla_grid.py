import logging
import numpy as np
from functools import partial

from rl_agents.agents.common.factory import preprocess_env, safe_deepcopy_env
from rl_agents.agents.tree_search.abstract import Node, AbstractTreeSearchAgent, AbstractPlanner
from rl_agents.agents.tree_search.olop import OLOP
from highway_env.vehicle.behavior import LinearVehicle
from highway_env.envs.common.action import action_factory

logger = logging.getLogger(__name__)


class OSLAGridAgent(AbstractTreeSearchAgent):
    """
        An agent that uses One Step Look Ahead to plan a sequence of action in an MDP.
    """
    def make_planner(self):
        return OSLAGrid(self.env, self.config)

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            "horizon": 10, # 10,
            "episodes": 5,
            "env_preprocessors": []
         })
        return config

    @staticmethod
    def random_policy(state, observation):
        """
            Choose actions from a uniform distribution.

        :param state: the environment state
        :param observation: the corresponding observation
        :return: a tuple containing the actions and their probabilities
        """
        actions = np.arange(state.action_space.n)
        probabilities = np.ones((len(actions))) / len(actions)
        return actions, probabilities

    @staticmethod
    def random_available_policy(state, observation):
        """
            Choose actions from a uniform distribution over currently available actions only.

        :param state: the environment state
        :param observation: the corresponding observation
        :return: a tuple containing the actions and their probabilities
        """
        if hasattr(state, 'get_available_actions'):
            available_actions = state.get_available_actions()
        else:
            available_actions = np.arange(state.action_space.n)
        probabilities = np.ones((len(available_actions))) / len(available_actions)
        return available_actions, probabilities

    @staticmethod
    def idle_policy(state, observation):
        """
            Choose idle action only.

        :param state: the environment state
        :param observation: the corresponding observation
        :return: a tuple containing the actions and their probabilities
        """
        if hasattr(state, 'get_available_actions'):
            available_actions = state.get_available_actions()
        else:
            available_actions = np.arange(state.action_space.n)
        probabilities = np.zeros((len(available_actions)))
        probabilities[1] = 1
        return available_actions, probabilities

    @staticmethod
    def preference_policy(state, observation, action_index, ratio=2):
        """
            Choose actions with a distribution over currently available actions that favors a preferred action.

            The preferred action probability is higher than others with a given ratio, and the distribution is uniform
            over the non-preferred available actions.
        :param state: the environment state
        :param observation: the corresponding observation
        :param action_index: the label of the preferred action
        :param ratio: the ratio between the preferred action probability and the other available actions probabilities
        :return: a tuple containing the actions and their probabilities
        """
        if hasattr(state, 'get_available_actions'):
            available_actions = state.get_available_actions()
        else:
            available_actions = np.arange(state.action_space.n)
        for i in range(len(available_actions)):
            if available_actions[i] == action_index:
                probabilities = np.ones((len(available_actions))) / (len(available_actions) - 1 + ratio)
                probabilities[i] *= ratio
                return available_actions, probabilities
        return OSLAGridAgent.random_available_policy(state, observation)


class OSLAGrid(AbstractPlanner):
    """
       An implementation of One Step Look Ahead.
    """
    def __init__(self, env, config=None):
        """
            New OSLAGrid instance.

        :param config: the OSLAGrid configuration. Use default if None.
        """
        super().__init__(config)
        self.env = env

    @classmethod
    def default_config(cls):
        cfg = super(OSLAGrid, cls).default_config()
        cfg.update({
            "temperature": 0,
            "closed_loop": False
        })
        return cfg

    def reset(self):
        self.root = OSLAGridNode(parent=None, planner=self)

    def run(self, i):
        """
            Run an iteration of One Step Look Ahead, starting from a given state

        :param i: the action.
        """
        depth = 0
        node = self.root
        total_reward = 0
        terminal = False
        action = i
        state = self.mdp.state
        state = self.mdp.transition[state, action]
        total_reward += self.config["gamma"] ** depth * self.mdp.reward[state, action]
        node_observation = observation if self.config["closed_loop"] else None
        node.expand_simple(i)
        if self.mdp.ttc[state] < 1:
            return
        total_reward = self.evaluate(state, total_reward, depth=1)
        node = node.get_child(action, observation=node_observation)

        node.update(total_reward)
        if i == 0 or self.root.value < node.value:
            self.root.update(total_reward)

    def evaluate(self, state, total_reward=0, depth=0):
        """
            Run the rollout policy to yield a sample of the value of being in a given state.

        :param state: the leaf state.
        :param total_reward: the initial total reward accumulated until now
        :param depth: the initial simulation depth
        :return: the total reward of the rollout trajectory
        """

        for h in range(depth, self.config["horizon"]):
            action = np.argmax(self.mdp.ttc[self.mdp.transition[state, range(self.mdp.transition.shape[1])]])
            state = self.mdp.transition[state, action]
            total_reward += self.config["gamma"] ** h * self.mdp.reward[state, action]
            if self.mdp.ttc[state] < 1:
                break
        return total_reward

    def plan(self, state, observation):
        self.reset()
        self.mdp = self.env.unwrapped.to_finite_mdp()
        for i in range(self.config['episodes']):
            self.run(i)
        return self.get_plan()

    def step_planner(self, action):
        if self.config["step_strategy"] == "prior":
            self.step_by_prior(action)
        else:
            super().step_planner(action)

    def step_by_prior(self, action):
        """
            Replace the OSLAGrid tree by its subtree corresponding to the chosen action, but also convert the visit counts
            to prior probabilities and before resetting them.

        :param action: a chosen action from the root node
        """
        self.step_by_subtree(action)
        self.root.convert_visits_to_prior_in_branch()


class OSLAGridNode(Node):
    # K = 1.0
    """ The value function first-order filter gain"""

    def __init__(self, parent, planner, prior=1):
        super(OSLAGridNode, self).__init__(parent, planner)
        self.value = 0
        self.prior = prior

    def selection_rule(self):
        if not self.children:
            return None
        # Tie best counts by best value
        actions = list(self.children.keys())
        counts = Node.all_argmax([self.children[a].count for a in actions])
        return actions[max(counts, key=(lambda i: self.children[actions[i]].get_value()))]

    def expand_simple(self, action):
        """
            Expand a leaf node by creating a new child for given action.

        :param action: action for the desired children node
        """
        self.children[action] = type(self)(self, self.planner)

    def update(self, total_reward):
        """
            Update the visit count and value of this node, given a sample of total reward.

        :param total_reward: the total reward obtained through a trajectory passing by this node
        """
        # self.count += 1
        self.value = total_reward

    def update_branch(self, total_reward):
        """
            Update the whole branch from this node to the root with the total reward of the corresponding trajectory.

        :param total_reward: the total reward obtained through a trajectory passing by this node
        """
        self.update(total_reward)
        if self.parent:
            self.parent.update_branch(total_reward)

    def get_child(self, action, observation=None):
        child = self.children[action]
        if observation is not None:
            if str(observation) not in child.children:
                child.children[str(observation)] = OSLAGridNode(parent=child, planner=self.planner, prior=0)
            child = child.children[str(observation)]
        return child

    def get_value(self):
        return self.value

