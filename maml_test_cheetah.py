
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
from test import test_const_adv, test_learnt_adv
import numpy as np
import pickle
import argparse
import os
import random

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

## Pass arguments ##
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='HalfCheetahTorsoAdv-v1', help='Name of adversarial environment')
parser.add_argument('--max_path_length', type=int, default=200, help='maximum episode length')
parser.add_argument('--layer_size', nargs='+', type=int, default=[100,100], help='layer definition')
parser.add_argument('--if_render', type=int, default=0, help='Should we render?')
parser.add_argument('--after_render', type=int, default=100, help='After how many to animate')
parser.add_argument('--num_grad_updates', type=int, default=1, help='Number of training instances to run')
parser.add_argument('--n_itr', type=int, default=800, help='Number of iterations of the alternating optimization')
parser.add_argument('--n_pro_itr', type=int, default=1, help='Number of iterations for the portagonist')
parser.add_argument('--n_adv_itr', type=int, default=1, help='Number of interations for the adversary')
parser.add_argument('--fast_batch_size', type=int, default=20, help='')
parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every save_every iterations')
parser.add_argument('--meta_batch_size', type=int, default=40, help='')
parser.add_argument('--adv_fraction', type=float, default=1, help='fraction of maximum adversarial force to be applied')
parser.add_argument('--fast_lr', type=float, default=0.1, help='')
parser.add_argument('--meta_step_size', type=float, default=0.01, help='')
parser.add_argument('--gae_lambda', type=float, default=0.97, help='gae_lambda for learner')
parser.add_argument('--folder', type=str, default=os.path.join(os.getcwd(),"results"), help='folder to save result in')


## Parsing Arguments ##
args = parser.parse_args()
env_name = args.env
max_path_length = args.max_path_length
layer_size = tuple(args.layer_size)
ifRender = bool(args.if_render)
afterRender = args.after_render
num_grad_updates = args.num_grad_updates
n_itr = args.n_itr
n_pro_itr = args.n_pro_itr
n_adv_itr = args.n_adv_itr
batch_size = args.fast_batch_size
save_every = args.save_every
meta_batch_size = args.meta_batch_size
adv_fraction = args.adv_fraction
step_size = args.meta_step_size
grad_step_size = args.fast_lr
gae_lambda = args.gae_lambda
save_dir = args.folder


## Initializing summaries for the tests ##
const_test_rew_summary = []
adv_test_rew_summary = []

save_prefix = 'env-{}_Itr{}_BS{}_Adv{}_stp{}_lam{}_{}'.format(env_name, n_itr, batch_size, adv_fraction, step_size, gae_lambda, random.randint(0,1000000))
save_name = save_dir+'/'+save_prefix+'.p'

for v in variants:
#    direc = v['direc']
#    learning_rate = v['meta_step_size']
#    if direc:
#        # env = TfEnv(normalize(HalfCheetahEnvRandDirec()))
#        env = TfEnv(normalize(GymEnv('HalfCheetahAdv-v1', 1.0)))
#    else:
#        # env = TfEnv(normalize(HalfCheetahEnvRand()))
#        env = TfEnv(normalize(GymEnv('HalfCheetahAdv-v1', 1.0)))
    
    env = TfEnv(normalize(GymEnv(env_name, 1.0)))
    env_orig = TfEnv(normalize(GymEnv(env_name, 1.0)))

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
        n_itr=1,
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
        n_itr=1,
        use_maml=True,
        step_size=v['meta_step_size'],
        plot=False,
        is_protagonist=False
    )

    pro_rews = []
    adv_rews = []
    all_rews = []
    const_testing_rews = []
#    const_testing_rews.append(test_const_adv(env_orig, pro_policy, path_length=200))
    adv_testing_rews = []
#    adv_testing_rews.append(test_learnt_adv(env, pro_policy, adv_policy, path_length=200))


    for ni in range(n_itr):
        logger.log('\n\n\n####global itr# {} ####\n\n\n'.format(ni))
        ## Train protagonist
        pro_algo.train()
        pro_rews += pro_algo.rews; all_rews += pro_algo.rews;
        logger.log('Protag Reward: {}'.format(np.array(pro_algo.rews).mean()))
#        ## Train Adversary
        adv_algo.train()
        adv_rews += adv_algo.rews; all_rews += adv_algo.rews;
        logger.log('Advers Reward: {}'.format(np.array(adv_algo.rews).mean()))
#        ## Test the learnt policies
        const_adv_reward = test_const_adv(env, pro_policy, num_tasks = v['meta_batch_size'], path_length=200)
        logger.log('Const Advers Reward: {}'.format(const_adv_reward))
        const_testing_rews.append(const_adv_reward)
        
        learned_adv_reward = test_learnt_adv(env, pro_policy, adv_policy, num_tasks = v['meta_batch_size'], path_length=200)
        logger.log('Learned Advers Reward: {}'.format(learned_adv_reward))
        adv_testing_rews.append(learned_adv_reward)

        if ni!=0 and ni%save_every==0:
            ## SAVING CHECKPOINT INFO ##
            pickle.dump({'args': args,
                         'pro_policy': pro_policy,
                         'adv_policy': adv_policy,
                         'zero_test': const_test_rew_summary,
                         'iter_save': ni,
                         'adv_test': adv_test_rew_summary}, open(save_name+"_"+str(ni)+'.temp','wb'))
#    
#    ## Shutting down the optimizer ##
#    pro_algo.shutdown_worker()
#    adv_algo.shutdown_worker()


logger.log('\n\n\n#### DONE ####\n\n\n')
