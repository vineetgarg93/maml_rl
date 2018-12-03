## Testing ##
from rllab.sampler.utils import rollout
from sandbox.rocky.tf.policies.constant_control_policy import ConstantControlPolicy
import tensorflow as tf

def test_const_adv(env, protag_policy, num_tasks, path_length=100, n_traj=2, render=False):
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "/home/grads/v/vineet/Desktop/RL_Project/maml_rl/checkpoints/pro/pro.ckpt")
        
        const_adv_policy = ConstantControlPolicy(
            env_spec=env.spec,
            is_protagonist=False,
            constant_val = 0.0
        )
        
        paths = []
        sum_rewards = 0.0
        for _ in range(n_traj):
            path = rollout(env, protag_policy, num_tasks, path_length, adv_agent=const_adv_policy, animated=render, test=True, learned_adv = False)
            sum_rewards += path['rewards'].sum()
            paths.append(path)
        avg_rewards = sum_rewards/n_traj
        return avg_rewards

def test_learnt_adv(env, protag_policy, adv_policy, num_tasks, path_length=100, n_traj=2, render=False):
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "/home/grads/v/vineet/Desktop/RL_Project/maml_rl/checkpoints/pro/pro.ckpt")
    
        paths = []
        sum_rewards = 0.0
        for _ in range(n_traj):
            path = rollout(env, protag_policy, num_tasks, path_length, adv_agent=adv_policy, animated=render, test=True)
            sum_rewards += path['rewards'].sum()
            paths.append(path)
        avg_rewards = sum_rewards/n_traj
        return avg_rewards
