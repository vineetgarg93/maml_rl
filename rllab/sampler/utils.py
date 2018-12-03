import numpy as np
from rllab.misc import tensor_utils
import time
import pickle
from sandbox.rocky.tf.envs.vec_env_executor import VecEnvExecutor

def rollout(env, pro_agent, num_tasks, max_path_length=np.inf, animated=False, speedup=1, adv_agent=None, test=False, learned_adv = True):
    
    
    envs = [pickle.loads(pickle.dumps(env)) for _ in range(num_tasks)]
    vec_env = VecEnvExecutor(
        envs=envs,
        max_path_length=200
    )
    
    observations = []
    pro_actions = []
    if adv_agent: adv_actions = []
    rewards = []
    pro_agent_infos = []
    if adv_agent: adv_agent_infos = []
    env_infos = []
    o = vec_env.reset()
    pro_agent.reset()
    if adv_agent: adv_agent.reset()
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        pro_a, pro_agent_info = pro_agent.get_actions(o)
        if test==True: pro_a = pro_agent_info['mean']
        if adv_agent:
            if not learned_adv:
                adv_a, adv_agent_info = adv_agent.get_action(o)
                if test==True and adv_agent_info: adv_a = adv_agent_info['mean']
                adv_a = np.repeat(adv_a[np.newaxis,:],num_tasks, axis=0)
            else:
                adv_a, adv_agent_info = adv_agent.get_actions(o)
                if test==True and adv_agent_info: adv_a = adv_agent_info['mean']
            
            next_o, r, d, env_info = vec_env.step(pro_a, adv_a)
            
#            class temp_action(object): pro=None; adv=None;
#            #from IPython import embed;embed()
#            cum_a = temp_action()
#            cum_a.pro = pro_a; cum_a.adv = adv_a;
#            #print(type(adv_agent))
#            #print('adversary_action = {}'.format(adv_a))
#            next_o, r, d, env_info = env.step(cum_a)
            pro_actions.append(env.pro_action_space.flatten(pro_a))
            adv_actions.append(env.adv_action_space.flatten(adv_a))
            adv_agent_infos.append(adv_agent_info)
        else:
            next_o, r, d, env_info = env.step(pro_a)
            pro_actions.append(env.action_space.flatten(pro_a))

        observations.append(env.observation_space.flatten(o))
        if test:
            rewards.append(np.mean(r))
        else:
            rewards.append(r)
        pro_agent_infos.append(pro_agent_info)
        env_infos.append(env_info)
        path_length += 1
#        if d:
#            break
        o = next_o
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)
    #if animated:
    #    return
    if adv_agent:
        return dict(
            observations=tensor_utils.stack_tensor_list(observations),
            pro_actions=tensor_utils.stack_tensor_list(pro_actions),
            adv_actions=tensor_utils.stack_tensor_list(adv_actions),
            rewards=tensor_utils.stack_tensor_list(rewards),
            pro_agent_infos=tensor_utils.stack_tensor_dict_list(pro_agent_infos),
            adv_agent_infos=tensor_utils.stack_tensor_dict_list(adv_agent_infos),
            env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
        )
    else:
        return dict(
            observations=tensor_utils.stack_tensor_list(observations),
            actions=tensor_utils.stack_tensor_list(pro_actions),
            rewards=tensor_utils.stack_tensor_list(rewards),
            agent_infos=tensor_utils.stack_tensor_dict_list(pro_agent_infos),
            env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
        )


#def rollout(env, agent, max_path_length=np.inf, animated=False, speedup=1, save_video=True, video_filename='sim_out.mp4', reset_arg=None):
#    observations = []
#    actions = []
#    rewards = []
#    agent_infos = []
#    env_infos = []
#    images = []
#    o = env.reset(reset_args=reset_arg)
#    agent.reset()
#    path_length = 0
#    if animated:
#        env.render()
#    while path_length < max_path_length:
#        a, agent_info = agent.get_action(o)
#        next_o, r, d, env_info = env.step(a)
#        observations.append(env.observation_space.flatten(o))
#        rewards.append(r)
#        actions.append(env.action_space.flatten(a))
#        agent_infos.append(agent_info)
#        env_infos.append(env_info)
#        path_length += 1
#        if d: # and not animated:  # TODO testing
#            break
#        o = next_o
#        if animated:
#            env.render()
#            timestep = 0.05
#            time.sleep(timestep / speedup)
#            if save_video:
#                from PIL import Image
#                image = env.wrapped_env.wrapped_env.get_viewer().get_image()
#                pil_image = Image.frombytes('RGB', (image[1], image[2]), image[0])
#                images.append(np.flipud(np.array(pil_image)))
#
#    if animated:
#        if save_video and len(images) >= max_path_length:
#            import moviepy.editor as mpy
#            clip = mpy.ImageSequenceClip(images, fps=20*speedup)
#            if video_filename[-3:] == 'gif':
#                clip.write_gif(video_filename, fps=20*speedup)
#            else:
#                clip.write_videofile(video_filename, fps=20*speedup)
#        #return
#
#    return dict(
#        observations=tensor_utils.stack_tensor_list(observations),
#        actions=tensor_utils.stack_tensor_list(actions),
#        rewards=tensor_utils.stack_tensor_list(rewards),
#        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
#        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
#    )
