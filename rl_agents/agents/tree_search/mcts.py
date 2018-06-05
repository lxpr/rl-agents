import numpy as np
import copy

from gym import logger
from gym.utils import seeding

from rl_agents.agents.abstract import AbstractAgent
from rl_agents.configuration import Configurable


class MCTSAgent(AbstractAgent):
    """
        An agent that uses Monte Carlo Tree Search to plan a sequence of action in an MDP.
    """

    def __init__(self,
                 env,
                 config=None,
                 prior_policy=None,
                 rollout_policy=None,
                 env_preprocessor=None):
        """
            A new MCTS agent.
        :param env: The environment
        :param config: The agent configuration. Use default if None.
        :param prior_policy: The prior distribution over actions given a state
        :param rollout_policy: The distribution over actions used when evaluating leaves
        :param env_preprocessor: a preprocessor function to apply to the environment before planning
        """
        super(MCTSAgent, self).__init__(config)
        self.env = env
        prior_policy = prior_policy or MCTSAgent.random_policy
        rollout_policy = rollout_policy or MCTSAgent.random_policy
        self.mcts = MCTS(prior_policy, rollout_policy, self.config)
        self.env_preprocessor = env_preprocessor
        self.previous_action = None

    @classmethod
    def default_config(cls):
        return dict()

    def plan(self, state):
        """
            Plan an optimal sequence of actions.

            Start by updating the previously found tree with the last action performed.

        :param state: the current state
        :return: the list of actions
        """
        self.mcts.step(self.previous_action)
        try:
            env = self.env_preprocessor(self.env)
        except (AttributeError, TypeError):
            env = self.env
        actions = self.mcts.plan(state=env, observation=state)

        self.previous_action = actions[0]
        return actions

    def reset(self):
        self.mcts.step_by_reset()

    def seed(self, seed=None):
        return self.mcts.seed(seed)

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
        return MCTSAgent.random_available_policy(state, observation)

    def record(self, state, action, reward, next_state, done):
        raise NotImplementedError()

    def act(self, state):
        return self.plan(state)[0]

    def save(self, filename):
        raise NotImplementedError()

    def load(self, filename):
        raise NotImplementedError()


class MCTS(Configurable):
    """
       An implementation of Monte-Carlo Tree Search, with Upper Confidence Tree exploration.
    """
    def __init__(self, prior_policy, rollout_policy, config=None):
        """
            New MCTS instance.

        :param config: the mcts configuration. Use default if None.
                       - iterations: the number of iterations
                       - temperature: the temperature of exploration
                       - max_depth: the maximum depth of the tree
        :param prior_policy: the prior policy used when expanding and selecting nodes
        :param rollout_policy: the rollout policy used to estimate the value of a leaf node

        """
        super(MCTS, self).__init__(config)
        self.root = Node(parent=None)
        self.prior_policy = prior_policy
        self.rollout_policy = rollout_policy
        self.np_random = None
        self.seed()

    @classmethod
    def default_config(cls):
        return dict(iterations=75,
                    temperature=10,
                    max_depth=7)

    def seed(self, seed=None):
        """
            Seed the rollout policy randomness source
        :param seed: the seed to be used
        :return: the used seed
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def run(self, state, observation):
        """
            Run an iteration of Monte-Carlo Tree Search, starting from a given state

        :param state: the initial environment state
        :param observation: the corresponding observation
        """
        node = self.root
        total_reward = 0
        depth = self.config['max_depth']
        terminal = False
        while depth > 0 and node.children and not terminal:
            action = node.select_action(temperature=self.config['temperature'])
            observation, reward, terminal, _ = state.step(action)
            total_reward += reward
            node = node.children[action]
            depth = depth - 1

        if not node.children and \
                (not terminal or node == self.root):
            node.expand(self.prior_policy(state, observation))

        if not terminal:
            total_reward = self.evaluate(state, observation, total_reward, limit=depth)
        node.update_branch(total_reward)

    def evaluate(self, state, observation, total_reward=0, limit=10):
        """
            Run the rollout policy to yield a sample of the value of being in a given state.

        :param state: the leaf state.
        :param observation: the corresponding observation.
        :param total_reward: the initial total reward accumulated until now
        :param limit: the maximum number of simulation steps
        :return: the total reward of the rollout trajectory
        """
        for _ in range(limit):
            actions, probabilities = self.rollout_policy(state, observation)
            action = self.np_random.choice(actions, 1, p=np.array(probabilities))[0]
            observation, reward, terminal, _ = state.step(action)
            total_reward += reward
            if terminal:
                break
        return total_reward

    def plan(self, state, observation):
        """
            Plan an optimal sequence of actions by running several iterations of MCTS.

        :param state: the initial environment state
        :param observation: the corresponding state observation
        :return: the list of actions
        """
        for i in range(self.config['iterations']):
            if (i+1) % 10 == 0:
                logger.debug('{} / {}'.format(i+1, self.config['iterations']))

            # Simplify the environment state if supported
            try:
                state_copy = state.unwrapped.simplified()
            except AttributeError:
                state_copy = state.unwrapped
            # Copy the environment state
            try:
                state_copy = self.custom_deepcopy(state_copy)
            except ValueError:
                state_copy = copy.deepcopy(state_copy)
            self.run(state_copy, observation)
        return self.get_plan()

    @staticmethod
    def custom_deepcopy(obj):
        """
            Perform a deep copy but without copying the environment viewer.
        """
        cls = obj.__class__
        result = cls.__new__(cls)
        memo = {id(obj): result}
        for k, v in obj.__dict__.items():
            if k not in ['viewer', 'automatic_rendering_callback']:
                setattr(result, k, copy.deepcopy(v, memo=memo))
            else:
                setattr(result, k, None)
        return result

    def get_plan(self):
        """
            Get the optimal action sequence of the current tree by recursively selecting the best action within each
            node with no exploration.

        :return: the list of actions
        """
        actions = []
        node = self.root
        while node.children:
            action = node.select_action(temperature=0)
            actions.append(action)
            node = node.children[action]
        return actions

    def step(self, action):
        """
            Update the MCTS tree when the agent performs an action

        :param action: the chosen action from the root node
        """
        self.step_by_reset()

    def step_by_reset(self):
        """
            Reset the MCTS tree to a root node for the new state.
        """
        self.root = Node(None)

    def step_by_subtree(self, action):
        """
            Replace the MCTS tree by its subtree corresponding to the chosen action.

        :param action: a chosen action from the root node
        """
        if action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None
        else:
            # The selected action was never explored, start a new tree.
            self.step_by_reset()

    def step_by_prior(self, action):
        """
            Replace the MCTS tree by its subtree corresponding to the chosen action, but also convert the visit counts
            to prior probabilities and before resetting them.

        :param action: a chosen action from the root node
        """
        self.step_by_subtree(action)
        self.root.convert_visits_to_prior_in_branch()


class Node(object):
    """
        An MCTS tree node, corresponding to a given state.
    """
    K = 1.0
    """ The value function first-order filter gain"""

    def __init__(self, parent, prior=1):
        """
            New node.

        :param parent: its parent node
        :param prior: its prior probability
        """
        self.parent = parent
        self.prior = prior
        self.children = {}
        self.count = 0
        self.value = 0

    def select_action(self, temperature=None):
        """
            Select an action from the node.
            - if exploration is wanted with some temperature, follow the selection strategy.
            - else, select the action with maximum visit count

        :param temperature: the exploration parameter, positive or zero
        :return: the selected action
        """
        if self.children:
            if temperature == 0:
                return max(self.children.keys(), key=(lambda key: self.children[key].count))
            else:
                return max(self.children.keys(), key=(lambda key: self.children[key].selection_strategy(temperature)))
        else:
            return None

    def expand(self, actions_distribution):
        """
            Expand a leaf node by creating a new child for each available action.

        :param actions_distribution: the list of available actions and their prior probabilities
        """
        actions, probabilities = actions_distribution
        for i in range(len(actions)):
            if actions[i] not in self.children:
                self.children[actions[i]] = Node(self, probabilities[i])

    def update(self, total_reward):
        """
            Update the visit count and value of this node, given a sample of total reward.

        :param total_reward: the total reward obtained through a trajectory passing by this node
        """
        self.count += 1
        self.value += self.K / self.count * (total_reward - self.value)

    def max_update(self, total_reward):
        """
            Update the estimated value as a maximum over trajectories with no averaging
        :param total_reward: the total reward obtained through a trajectory passing by this node
        """
        self.count += 1
        self.value = max(total_reward, self.value)

    def update_branch(self, total_reward):
        """
            Update the whole branch from this node to the root with the total reward of the corresponding trajectory.

        :param total_reward: the total reward obtained through a trajectory passing by this node
        """
        self.update(total_reward)
        if self.parent:
            self.parent.update_branch(total_reward)

    def selection_strategy(self, temperature):
        """
            Select an action according to its value, prior probability and visit count.

        :param temperature: the exploration parameter, positive or zero.
        :return: the selected action with maximum value and exploration bonus.
        """
        if not self.parent:
            return self.value

        # return self.value + temperature * self.prior * np.sqrt(np.log(self.parent.count) / self.count)
        return self.value + temperature*self.prior/(self.count+1)

    def convert_visits_to_prior_in_branch(self, regularization=0.5):
        """
            For any node in the subtree, convert the distribution of all children visit counts to prior
            probabilities, and reset the visit counts.

        :param regularization: in [0, 1], used to add some probability mass to all children.
                               when 0, the prior is a Boltzmann distribution of visit counts
                               when 1, the prior is a uniform distribution
        """
        self.count = 0
        total_count = sum([(child.count+1) for child in self.children.values()])
        for child in self.children.values():
            child.prior = regularization*(child.count+1)/total_count + regularization/len(self.children)
            child.convert_visits_to_prior_in_branch()

    def __str__(self, level=0):
        ret = "\t" * level + repr(self.value) + "\n"
        for child in self.children.values():
            ret += child.__str__(level + 1)
        return ret

    def __repr__(self):
        return '<tree node representation>'


