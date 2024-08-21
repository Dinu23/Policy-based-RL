import argparse
from catch import Catch
import tensorflow as tf
from tensorflow import keras
import numpy as np

from model import Conv_Network, N_Network
from keras.optimizers import Adam

# gpus = tf.config.list_physical_devices("GPU")
# tf.config.set_visible_devices(gpus[0], "GPU")
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def create_loss_function(M, entropy,n):
    def reinforce_loss(y_true,y_pred):
        actions = tf.one_hot(tf.cast(y_true[:,0],tf.int32),n)
        returns = y_true[:,1]
        
        prob = tf.reduce_sum(y_pred * actions,1)
        
        entropies = -tf.reduce_sum(y_pred * tf.math.log(tf.add(y_pred,0.001)),1)

        return -1/M * tf.reduce_sum(returns * tf.math.log(prob) + entropy * entropies )
    return reinforce_loss


class MCPG_Agent:
    def __init__(self,env,policy_network_builder) -> None:
        self.env = env
        self.policy_network_builder = policy_network_builder
        self.policy_network = self.policy_network_builder.create_model(self.env.observation_space.shape, self.env.action_space.n)

    def train(self,
            no_episode = 1000,
            M = 1,
            optimizer = Adam(learning_rate=0.01),
            gamma = 0.9,
            entropy = 0.1,
            verbose = True,
            render = False
    ):
        self.policy_network.compile(optimizer = optimizer, loss = create_loss_function(M,   entropy, self.env.action_space.n))


        episode_rewards = []
        states = []
        actions = []
        rewards = []
        returns = []
        for episode in range(no_episode):
            s =  self.env.reset()
            if(render):
                self.env.render(0.1)
        
            count = 0
            while(True):
                action_probabilities = self.policy_network(np.array([s]))[0]
                if(np.isnan(action_probabilities).any()):
                    print("error")
                    a = np.random.choice(self.env.action_space.n)
                else:
                    a = np.random.choice(self.env.action_space.n,p=np.array(action_probabilities))
                s_next,r,done,_ = self.env.step(a)
                if(render):
                    self.env.render(0.1)
                count +=1
                states.append(np.array(s))
                actions.append(a)
                rewards.append(r) 
                s = s_next
                if(done):
                    break

            R = 0
            episode_returns = []
            for t in range(len(states)-1, len(states)-count-1,-1):
                R = rewards[t] + gamma * R
                episode_returns.append(R)
            episode_returns = episode_returns[::-1]

            returns += episode_returns


            if(episode % M == M-1):
                actions = np.vstack(actions)
                returns = np.vstack(returns)
                y_true = np.hstack([actions,returns])

                self.policy_network.fit(np.array(states), y_true, verbose=False, shuffle=True, batch_size=len(actions))
                states = []
                actions = []
                rewards = []
                returns = []

            episode_reward = self.env.total_reward
            

            if(verbose and episode%5==0):
                print("Episode ",episode+1, " reward: ",episode_reward)
            episode_rewards.append(episode_reward)



        return episode_rewards

    def play(self, render_time = 0.1):
        s =  self.env.reset()
        self.env.render(render_time)
        while(True):
            action_probabilities = self.policy_network(np.array([s]))[0]
            a = np.random.choice(self.env.action_space.n,p=np.array(action_probabilities))
            s_next,r,done,_ = self.env.step(a)
            self.env.render(render_time)
            s = s_next
            if(done):
                break
        
        print(self.env.total_reward)           


    def load_model(self,policy_file="mcpg-policy"):
        self.policy_network.load_weights(policy_file)

    def save_model(self,policy_file="mcpg-policy"):
        self.policy_network.save_weights(policy_file)


def experiment(no_episode):
    rows = 7    
    columns = 7
    speed = 1.0
    max_steps = 250
    max_misses = 10
    observation_type = 'pixel' # 'vector'
    seed = None

    # Initialize environment and Q-array
    env = Catch(rows=rows, columns=columns, speed=speed, max_steps=max_steps,
                max_misses=max_misses, observation_type=observation_type, seed=seed)
    if(observation_type == "vector"):
        builder = N_Network([8,4,4])
    else:
        builder = Conv_Network([8,4,4])

    mcpg_agent = MCPG_Agent(env,builder)

    
    episode_rewards = mcpg_agent.train(no_episode=no_episode)
    print(episode_rewards)





    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Monte Carlo Policy Gradient - REINFORCE')
    parser.add_argument('--episodes', type=int, default=500, help='Number of episodes to run')
    args = parser.parse_args()

    experiment(args.episodes)
