
from sandbox.rocky.tf.algos.maml_trpo import MAMLTRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.envs.mujoco.half_cheetah_env_rand import HalfCheetahEnvRand
from rllab.envs.mujoco.half_cheetah_env_rand_direc import HalfCheetahEnvRandDirec
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
#from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.policies.maml_minimal_gauss_mlp_policy import MAMLGaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
import rllab.misc.logger as logger

import tensorflow as tf

#stub(globals())

from rllab.misc.instrument import VariantGenerator, variant


class VG(VariantGenerator):

    @variant
    def fast_lr(self):
        return [0.1]

    @variant
    def meta_step_size(self):
        return [0.01]

    @variant
    def fast_batch_size(self):
        return [20]  # #10, 20, 40

    @variant
    def meta_batch_size(self):
        return [40] # at least a total batch size of 400. (meta batch size*fast batch size)

    @variant
    def seed(self):
        return [1]

    @variant
    def direc(self):  # directionenv vs. goal velocity
        return [False]


# should also code up alternative KL thing

variants = VG().variants()

max_path_length = 200
num_grad_updates = 1
use_maml=True

for v in variants:
    direc = v['direc']
    learning_rate = v['meta_step_size']

    if direc:
        # env = TfEnv(normalize(HalfCheetahEnvRandDirec()))
        env = TfEnv(normalize(GymEnv('HalfCheetahAdv-v1', 1.0)))
    else:
        # env = TfEnv(normalize(HalfCheetahEnvRand()))
        env = TfEnv(normalize(GymEnv('HalfCheetahAdv-v1', 1.0)))

    pro_policy = MAMLGaussianMLPPolicy(
        name="pro_policy",
        env_spec=env.spec,
        grad_step_size=v['fast_lr'],
        hidden_nonlinearity=tf.nn.relu,
        hidden_sizes=(100,100),
        is_protagonist=True
    )

    pro_baseline = LinearFeatureBaseline(env_spec=env.spec)

    adv_policy = MAMLGaussianMLPPolicy(
        name="adv_policy",
        env_spec=env.spec,
        grad_step_size=v['fast_lr'],
        hidden_nonlinearity=tf.nn.relu,
        hidden_sizes=(100,100),
        is_protagonist=False
    )

    adv_baseline = LinearFeatureBaseline(env_spec=env.spec)

    pro_algo = MAMLTRPO(
        env=env,
        pro_policy=pro_policy,
        adv_policy=adv_policy,
        pro_baseline=pro_baseline,
        adv_baseline=adv_baseline,
        batch_size=v['fast_batch_size'], # number of trajs for grad update
        max_path_length=max_path_length,
        meta_batch_size=v['meta_batch_size'],
        num_grad_updates=num_grad_updates,
        n_itr=800,
        use_maml=False,
        step_size=v['meta_step_size'],
        plot=False,
        is_protagonist=True
    )

    adv_algo = MAMLTRPO(
        env=env,
        pro_policy=pro_policy,
        adv_policy=adv_policy,
        pro_baseline=pro_baseline,
        adv_baseline=adv_baseline,
        batch_size=v['fast_batch_size'], # number of trajs for grad update
        max_path_length=max_path_length,
        meta_batch_size=v['meta_batch_size'],
        num_grad_updates=num_grad_updates,
        n_itr=800,
        use_maml=True,
        step_size=v['meta_step_size'],
        plot=False,
        is_protagonist=False
    )

    pro_rews = []
    adv_rews = []
    all_rews = []

    for ni in range(10):
        logger.log('\n\n\n####global itr# {} ####\n\n\n'.format(ni))
        ## Train protagonist
        pro_algo.train()
        # pro_rews += pro_algo.rews; all_rews += pro_algo.rews;
        # logger.log('Protag Reward: {}'.format(np.array(pro_algo.rews).mean()))
        ## Train Adversary
        adv_algo.train()
        # adv_rews += adv_algo.rews; all_rews += adv_algo.rews;
        # logger.log('Advers Reward: {}'.format(np.array(adv_algo.rews).mean()))