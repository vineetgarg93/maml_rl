import gym
import gym.envs
import gym.spaces
from gym.monitoring import monitor
import os
import os.path as osp
from rllab.envs.base import Env, Step
from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
from rllab.spaces.discrete import Discrete
from rllab.spaces.product import Product
from rllab.misc import logger
import logging


def convert_gym_space(space):
    if isinstance(space, gym.spaces.Box):
        return Box(low=space.low, high=space.high)
    elif isinstance(space, gym.spaces.Discrete):
        return Discrete(n=space.n)
    elif isinstance(space, gym.spaces.Tuple):
        return Product([convert_gym_space(x) for x in space.spaces])
    else:
        raise NotImplementedError


class CappedCubicVideoSchedule(object):
    def __call__(self, count):
        return monitor.capped_cubic_video_schedule(count)


class FixedIntervalVideoSchedule(object):
    def __init__(self, interval):
        self.interval = interval

    def __call__(self, count):
        return count % self.interval == 0


class NoVideoSchedule(object):
    def __call__(self, count):
        return False


class GymEnv(Env, Serializable):
    def __init__(self, env_name, adv_fraction=1.0, record_video=True, video_schedule=None, log_dir=None, record_log=True):
        if log_dir is None:
            if logger.get_snapshot_dir() is None:
                logger.log("Warning: skipping Gym environment monitoring since snapshot_dir not configured.")
            else:
                log_dir = os.path.join(logger.get_snapshot_dir(), "gym_log")
        Serializable.quick_init(self, locals())

        env = gym.envs.make(env_name)
        def_adv = env.adv_action_space.high[0]
        new_adv = def_adv*adv_fraction
        env.update_adversary(new_adv)
        self.env = env
        self.env_id = env.spec.id

        monitor.logger.setLevel(logging.WARNING)

        assert not (not record_log and record_video)

        if log_dir is None or record_log is False:
            self.monitoring = False
        else:
            if not record_video:
                video_schedule = NoVideoSchedule()
            else:
                if video_schedule is None:
                    video_schedule = CappedCubicVideoSchedule()
            self.env.monitor.start(log_dir, video_schedule, force=True)  # add 'force=True' if want overwrite dirs
            self.monitoring = True

        self._observation_space = convert_gym_space(env.observation_space)
        self._pro_action_space = convert_gym_space(env.pro_action_space)
        self._adv_action_space = convert_gym_space(env.adv_action_space)
        self._horizon = env.spec.timestep_limit
        self._log_dir = log_dir

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def pro_action_space(self):
        return self._pro_action_space

    @property
    def adv_action_space(self):
        return self._adv_action_space

    @property
    def horizon(self):
        return self._horizon

    def reset(self,reset_args=None):
        if hasattr(self.env, 'monitor'):
            if hasattr(self.env.monitor, 'stats_recorder'):
                recorder = self.env.monitor.stats_recorder
                if recorder is not None:
                    recorder.done = True
        return self.env.reset() #TODO: add reset_args as argument when using for sample goals

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        return Step(next_obs, reward, done, **info)

    def render(self):
        self.env.render()

    def terminate(self):
        if self.monitoring:
            self.env.monitor.close()
            if self._log_dir is not None:
                print("""
    ***************************

    Training finished! You can upload results to OpenAI Gym by running the following command:

    python scripts/submit_gym.py %s

    ***************************
                """ % self._log_dir)
