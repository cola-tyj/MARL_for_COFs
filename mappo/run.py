import torch
import numpy as np
import argparse
import csv
import sys
import os
import shutil
from reward import RND,predictor,rm_dir,max_pred,average_pred
from env import CustomEnvironment
from replay_buffer import ReplayBuffer
from mappo_mpe import MAPPO_MPE
from normalization import Normalization, RewardScaling
from torch.utils.tensorboard import SummaryWriter

class Runner_MAPPO_MPE:
    def __init__(self, args, env_name, number, seed):
        self.args = args
        self.env_name = env_name
        self.number = number
        self.seed = seed
        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # Create env
        self.env = CustomEnvironment(env_name, self.args, discrete=True) # Discrete action space
        self.args.N = self.env.n  # The number of agents
        self.args.obs_dim_n = [self.env.observation_space for i in range(self.args.N)]  # obs dimensions of N agents
        self.args.action_dim_n = [self.env.action_space for i in range(self.args.N)]  # actions dimensions of N agents
        # Only for homogenous agents environments like Spread in MPE,all agents have the same dimension of observation space and action space
        self.args.obs_dim = self.args.obs_dim_n[0]  # The dimensions of an agent's observation space
        self.args.action_dim = self.args.action_dim_n[0]  # The dimensions of an agent's action space
        self.args.state_dim = np.sum(self.args.obs_dim_n)  # The dimensions of global state space（Sum of the dimensions of the local observation space of all agents）
        # RND reward Net
        self.rnd = RND(self.args.obs_dim,32,256)
        print("obs_dim_n={}".format(self.args.obs_dim_n))
        print("action_space=", self.env.action_space)
        print("action_dim_n={}".format(self.args.action_dim_n))
        print("observation_space=", self.env.observation_space)
        print("state_dim=", self.args.state_dim)
        
        # Create N agents
        self.agent_n = MAPPO_MPE(self.args)
        self.replay_buffer = ReplayBuffer(self.args)

        # Create a tensorboard
        self.writer = SummaryWriter(log_dir='runs/MAPPO/MAPPO_env_{}_number_{}_seed_{}'.format(self.env_name, self.number, self.seed))

        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0
        if self.args.use_reward_norm:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=self.args.N)
        elif self.args.use_reward_scaling:
            print("------use reward scaling------")
            self.reward_scaling = RewardScaling(shape=self.args.N, gamma=self.args.gamma)

    def run(self, ):
        evaluate_num = -1  # Record the number of evaluations
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                #self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps
                evaluate_num += 1

            print('replay_buffer.episode_num:',self.replay_buffer.episode_num)
            _, episode_steps = self.run_episode_mpe(evaluate=False)  # Run an episode
            self.total_steps += episode_steps

            if self.replay_buffer.episode_num == self.args.batch_size:
                print(self.replay_buffer.batch_size)
                resultFilePath = predictor()  # 在train前统一用预测器计算氧气吸附
                self.predictor_reward(resultFilePath)  # 把吸附值添加到 replay_buffer
                try:
                    max_pred(resultFilePath)  # 记录这一轮pre的最大值
                    average_pred(resultFilePath)  # 记录这一轮pre的平均值
                    rm_dir('/home/tianyajun/MARL_for_COFs/cofs')
                    # Save the rewards and models
                except:
                    rm_dir('/home/tianyajun/MARL_for_COFs/cofs')
                self.save_returns()
                print('------Training------')
                self.rnd.update(self.replay_buffer)  # Update RND reward Net
                self.agent_n.train(self.replay_buffer, self.total_steps)  # Training
                self.replay_buffer.reset_buffer()
                #np.save('/home/tianyajun/MARL_for_COFs/mappo/data_train/MAPPO_env_{}_number_{}_seed_{}.npy'.format(self.env_name, self.number, self.seed), np.array(self.evaluate_rewards))
                self.agent_n.save_model(self.env_name, self.number, self.seed, self.total_steps)
        
                if self.total_steps>128000:
                    sys.exit()
        #self.evaluate_policy()
        self.env.close()

    def evaluate_policy(self):
        return  #先不进行评估，后续再说
        evaluate_reward = 0
        for _ in range(self.args.evaluate_times):
            episode_reward, _ = self.run_episode_mpe(evaluate=True)
            evaluate_reward += episode_reward

        evaluate_reward = evaluate_reward / self.args.evaluate_times
        self.evaluate_rewards.append(evaluate_reward)
        print("total_steps:{} \t evaluate_reward:{}".format(self.total_steps, evaluate_reward))
        self.writer.add_scalar('evaluate_step_rewards_{}'.format(self.env_name), evaluate_reward, global_step=self.total_steps)
        rm_dir('/home/tianyajun/MARL_for_COFs/cofs')  # 把evaluate生成的cof删掉
        # Save the rewards and models
        #np.save('./data_train/MAPPO_env_{}_number_{}_seed_{}.npy'.format(self.env_name, self.number, self.seed), np.array(self.evaluate_rewards))
        #self.agent_n.save_model(self.env_name, self.number, self.seed, self.total_steps)

    def run_episode_mpe(self, evaluate=False):
        episode_reward = 0
        intrinsic_rewards = []  # Record the intrinsic rewards during every episode
        obs_n, info, action_mask = self.env.reset()
        if self.args.use_reward_scaling:
            self.reward_scaling.reset()
        if self.args.use_rnn:  # If use RNN, before the beginning of each episode，reset the rnn_hidden of the Q network.
            self.agent_n.actor.rnn_hidden = None
            self.agent_n.critic.rnn_hidden = None
        for episode_step in range(self.args.episode_limit):
            print('--------------')
            a_n, a_logprob_n = self.agent_n.choose_action(obs_n, action_mask, evaluate=evaluate)  # Get actions and the corresponding log probabilities of N agents
            print('动作：', a_n)
            s = obs_n.flatten()  # Global state is the concatenation of all agents' local obs.
            v_n = self.agent_n.get_value(s)  # Get the state values (V(s)) of N agents
            obs_next_n, r_n, done_n, _, action_mask = self.env.step(a_n,self.replay_buffer.episode_num)
            # 计算 RND reward
            rewards_rnd = self.rnd.get_reward(obs_next_n)
            intrinsic_rewards = [b.detach().item() for b in rewards_rnd]
            #r_n = [a + 1.3 * b for a, b in zip(r_n, intrinsic_rewards)]
            print('奖励：', r_n)
            episode_reward += r_n[0]

            if not evaluate:
                '''if self.args.use_reward_norm:
                    r_n = self.reward_norm(r_n)
                elif args.use_reward_scaling:
                    r_n = self.reward_scaling(r_n)'''

                # Store the transition
                self.replay_buffer.store_transition(episode_step, obs_n, s, v_n, a_n, a_logprob_n, r_n, done_n)
                
            obs_n = obs_next_n
            if all(done_n):
                break

        if not evaluate:
            # An episode is over, store v_n in the last step
            s = obs_n.flatten()
            v_n = self.agent_n.get_value(s)
            self.replay_buffer.store_last_value(episode_step + 1, v_n)
            self.save_intrinsic_rewards(intrinsic_rewards)

        return episode_reward, episode_step + 1

    def predictor_reward(self,resultFilePath):
        result = '/home/tianyajun/MARL_for_COFs/result.csv'
        # 初始化两个空列表来存储'cifName'和'pred'列的数据
        cifName_list = []
        pred_list = []

        with open(resultFilePath, newline='', encoding='utf-8') as csvfile:
            # 创建CSV阅读器
            data_reader = csv.reader(csvfile)
            # 读取列标题
            headers = next(data_reader)  # 跳过列标题
            # 找到 'cifName' 和 'pred' 列的索引
            cifName_index = headers.index('cifName')
            pred_index = headers.index('pred')
            
            # 遍历剩余的行，并将'cifName'和'pred'列的值分别添加到对应的列表中
            for row in data_reader:
                cifName_list.append(row[cifName_index])
                pred_list.append(row[pred_index])

        with open('/home/tianyajun/MARL_for_COFs/number.csv', newline='', encoding='utf-8') as csvfile:
            # 创建CSV阅读器
            data_reader = csv.reader(csvfile)        
            for row in data_reader:
                episode_num = int(row[0])
                a_agent = int(row[1])
                b_agent = int(row[2])
                cifname = row[3]
                if cifname in cifName_list:
                    pred = float(pred_list[cifName_list.index(cifname)])
                else:
                    pred = 0.0
                self.replay_buffer.buffer['r_n'][episode_num][args.episode_limit-1][a_agent] += pred
                self.replay_buffer.buffer['r_n'][episode_num][args.episode_limit-1][b_agent] += pred

        # 把预测的数据都存到'result.csv'中
        with open(result, mode='a', newline='', encoding='utf-8') as destination_file:
            writer = csv.writer(destination_file)
            rows_to_append = [(cifName, pred) for cifName, pred in zip(cifName_list, pred_list)]
            writer.writerows(rows_to_append)
            print("数据已追加到目标result.csv文件。")

        os.remove('/home/tianyajun/MARL_for_COFs/number.csv')

    def save_returns(self):
        file_path = '/home/tianyajun/MARL_for_COFs/returns.csv'
        returns = []
        for i in range(self.args.N):
            agent_i_rewards = self.replay_buffer.buffer['r_n'][:, :, i]
            episode_rewards_sum = np.sum(agent_i_rewards, axis=1) 
            returns.append(episode_rewards_sum)
        # 写入数组数据到CSV文件
        with open(file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for j in range(len(episode_rewards_sum)):
                x = []
                for i in range(self.args.N):
                    x += [returns[i][j]]
                writer.writerow(x)
                
    def save_intrinsic_rewards(self,reward):
        file_path = '/home/tianyajun/MARL_for_COFs/intrinsic.csv'
        with open(file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(reward)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MAPPO in MPE environment")
    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help=" Maximum number of training steps")
    parser.add_argument("--episode_limit", type=int, default=4, help="Maximum number of steps per episode")
    parser.add_argument("--evaluate_freq", type=float, default=5000, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=3, help="Evaluate times")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (the number of episodes)")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="Minibatch size (the number of episodes)")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the rnn")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the mlp")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GAE parameter")
    parser.add_argument("--K_epochs", type=int, default=15, help="GAE parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=True, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling. Here, we do not use it.")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_relu", type=float, default=False, help="Whether to use relu, if False, we will use tanh")
    parser.add_argument("--use_rnn", type=bool, default=False, help="Whether to use RNN")
    parser.add_argument("--add_agent_id", type=float, default=False, help="Whether to add agent_id. Here, we do not use it.")
    parser.add_argument("--use_value_clip", type=float, default=False, help="Whether to use value clip.")

    args = parser.parse_args()

    runner = Runner_MAPPO_MPE(args, env_name="simple_spread", number=1, seed=0)
    # 在开始训练之前加载模型
    #runner.agent_n.load_model(env_name="simple_spread", number=1, seed=2, total_steps=53376)
    runner.run()