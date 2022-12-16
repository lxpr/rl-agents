import numpy as np
import logging

from rl_agents.agents.common.abstract import AbstractAgent

logger = logging.getLogger(__name__)

# One-step lookahead agent with grid state representation to speed up the simulation

class OSLAGridAgent(AbstractAgent):
    def __init__(self, env, config=None):
        super(OSLAGridAgent, self).__init__(config)
        self.finite_mdp = self.is_finite_mdp(env)
        if self.finite_mdp:
            self.mdp = env.mdp
        elif not self.finite_mdp:
            try:
                self.mdp = env.unwrapped.to_finite_mdp()
            except AttributeError:
                raise TypeError("Environment must be of type finite_mdp.envs.finite_mdp.FiniteMDPEnv or handle a "
                                "conversion method called 'to_finite_mdp' to such a type.")
        self.env = env
        self.reward_for_actions = self.osla_rollout()

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            "horizon": 10,
            "env_preprocessors": []
        })
        return config

    def act(self, state):
        # If the environment is not a finite mdp, it must be converted to one and the state must be recovered.
        self.mdp = self.env.unwrapped.to_finite_mdp()
        state = self.mdp.state
        self.reward_for_actions = self.osla_rollout()
        # print("reward for actions:", self.reward_for_actions, np.argmax(self.reward_for_actions))
        return np.argmax(self.reward_for_actions)

    def osla_rollout(self, depth=0):
        reward_for_actions = np.zeros(self.mdp.transition.shape[1])
        for a in range(self.mdp.transition.shape[1]):
            state = self.mdp.transition[self.mdp.state, a]
            reward_for_actions[a] += self.config["gamma"] ** depth * self.mdp.reward[state, a]
            if self.mdp.terminal[state]:
                    # print(a, step, "terminal")
                    continue
            for step in range(depth + 1, self.config["horizon"]):

                # Select the action with longest time to collision
                # ttc_idle = self.mdp.ttc[self.mdp.transition[state, 1]]
                # best_other_action = np.argmax(self.mdp.ttc[self.mdp.transition[state, [0, 2, 3, 4]]])
                # if self.mdp.ttc[self.mdp.transition[state, best_other_action]] > ttc_idle:
                #     if best_other_action > 0:
                #         best_other_action += 1
                #     action = best_other_action
                # else:
                #     action = 1
                action = np.argmax(self.mdp.ttc[self.mdp.transition[state, range(self.mdp.transition.shape[1])]])
                # action = 1
                state = self.mdp.transition[state, action]
                # print("ttc", self.mdp.ttc[state])
                reward_for_actions[a] += self.config["gamma"] ** step * self.mdp.reward[state, action]
                # # Always choose "IDLE" action in base policy
                # state = self.mdp.transition[state, 1]
                # reward_for_actions[a] += self.config["gamma"] ** step * self.mdp.reward[state, 1]
                # # Select the action with best reward at the next state
                # next_rewards = np.zeros(self.mdp.transition.shape[1])
                # for available_action in range(self.mdp.transition.shape[1]):
                #     next_state = self.mdp.transition[state, available_action]
                #     next_rewards[available_action] = self.mdp.reward[next_state, available_action]
                # action = np.argmax(next_rewards)
                # state = self.mdp.transition[state, action]
                # reward_for_actions[a] += self.config["gamma"] ** step * self.mdp.reward[state, action]
                if self.mdp.terminal[state]:
                    # print(a, step, "terminal")
                    break
        return reward_for_actions

    # def get_state_value(self):
    #     return self.fixed_point_iteration(
    #         lambda v: ValueIterationAgent.best_action_value(self.bellman_expectation(v)),
    #         np.zeros((self.mdp.transition.shape[0],)))
    #
    # def get_state_action_value(self):
    #     return self.fixed_point_iteration(
    #         lambda q: self.bellman_expectation(ValueIterationAgent.best_action_value(q)),
    #         np.zeros((self.mdp.transition.shape[0:2])))
    #
    # @staticmethod
    # def best_action_value(action_values):
    #     return action_values.max(axis=-1)
    #
    # def bellman_expectation(self, value):
    #     if self.mdp.mode == "deterministic":
    #         next_v = value[self.mdp.transition]
    #     elif self.mdp.mode == "stochastic":
    #         next_v = (self.mdp.transition * value.reshape((1, 1, value.size))).sum(axis=-1)
    #     elif self.mdp.mode == "sparse":
    #         # P(s,a,B) * v[B]
    #         next_values = np.take(value, self.mdp.next)
    #         next_v = (self.mdp.transition * next_values).sum(axis=-1)
    #     else:
    #         raise ValueError("Unknown mode")
    #     next_v[self.mdp.terminal] = 0
    #     return self.mdp.reward + self.config["gamma"] * next_v
    #
    # def fixed_point_iteration(self, operator, initial):
    #     value = initial
    #     for iteration in range(self.config["iterations"]):
    #         logger.debug("Value Iteration: {}/{}".format(iteration, self.config["iterations"]))
    #         next_value = operator(value)
    #         if np.allclose(value, next_value):
    #             break
    #         value = next_value
    #     return value

    @staticmethod
    def is_finite_mdp(env):
        try:
            finite_mdp = __import__("finite_mdp.envs.finite_mdp_env")
            if isinstance(env, finite_mdp.envs.finite_mdp_env.FiniteMDPEnv):
                return True
        except (ModuleNotFoundError, TypeError):
            return False

    def plan_trajectory(self, state, horizon=10):
        action_value = self.get_state_action_value()
        states, actions = [], []
        for _ in range(horizon):
            action = np.argmax(action_value[state])
            states.append(state)
            actions.append(action)
            state = self.mdp.next_state(state, action)
            if self.mdp.terminal[state]:
                states.append(state)
                actions.append(None)
                break
        return states, actions

    def record(self, state, action, reward, next_state, done, info):
        pass

    def reset(self):
        pass

    def seed(self, seed=None):
        pass

    def save(self, filename):
        return False

    def load(self, filename):
        return False
