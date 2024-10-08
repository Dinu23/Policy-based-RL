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

def create_reinforce_loss(M, entropy,n, eps):
    def reinforce_loss(y_true,y_pred):
        actions = tf.one_hot(tf.cast(y_true[:,0],tf.int32),n)
        f_values = y_true[:,1]
        probabilities = y_true[:,2]
        entropies = -tf.reduce_sum(y_pred * tf.math.log(tf.add(y_pred,0.001)),1)
        prob = tf.math.divide(tf.reduce_sum(y_pred * actions,1),probabilities)
        
        
        return -1/M * tf.reduce_sum(tf.minimum(f_values * prob , f_values * tf.clip_by_value(prob,1-eps, 1+eps))+ entropy * entropies)
    return reinforce_loss

def value_loss(y_true,y_pred):
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_sum(squared_difference, axis=-1)

class PPO_Agent:
    def __init__(self,env,policy_network_builder,value_network_builder) -> None:
        self.env = env
        self.policy_network_builder = policy_network_builder
        self.policy_network = self.policy_network_builder.create_model(self.env.observation_space.shape, self.env.action_space.n)
        self.value_network_builder = value_network_builder
        self.value_network = self.value_network_builder.create_model(self.env.observation_space.shape, 1)

    def train(self,
            no_episode = 1000,
            M = 1,
            optimizer_policy = Adam(learning_rate=0.01),
            optimizer_value = Adam(learning_rate=0.01),
            gamma = 0.9,
            entropy = 0.1,
            epsilon = 0.1,
            n = 3,
            epochs = 5,
            verbose = True,
            render = False
    ):
        self.policy_network.compile(optimizer = optimizer_policy, loss = create_reinforce_loss(M, entropy, self.env.action_space.n,epsilon))
        self.value_network.compile(optimizer = optimizer_value, loss = value_loss)



        episode_rewards = []
        states = []
        actions = []
        rewards = []
        returns = []
        f_values = []
        probabilities = []
        for episode in range(no_episode):
            s =  self.env.reset()
            if(render):
                self.env.render(0.1)
        
            count = 0
            while(True):
                action_probabilities = self.policy_network(np.array([s]))[0]
                a = np.random.choice(self.env.action_space.n,p=np.array(action_probabilities))
                s_next,r,done,_ = self.env.step(a)
                if(render):
                    self.env.render(0.1)
                count +=1
                states.append(np.array(s))
                actions.append(a)
                rewards.append(r)
                probabilities.append(action_probabilities[a])
                s= s_next
                if(done):
                    break
        
            for t in range(len(states)-count,len(states)):
                    n_max = min(n,len(states)-t)
                    # print(t,t+n_max-1)
                    Q = (gamma**n_max) * self.value_network(np.array([states[t+n_max-1]]))[0]
                    for i in range(n_max):
                        if(t+i>= len(states)):
                            break
                        Q += (gamma**i) *rewards[t+i]
                    returns.append(Q)
                    f_values.append(Q - self.value_network(np.array([states[t]]))[0])
           
            if(episode % M == M-1):
                actions = np.vstack(actions)
                returns = np.vstack(returns)
                f_values = np.vstack(f_values)
                probabilities = np.vstack(probabilities)
                y_true = np.hstack([actions,f_values,probabilities])

                self.value_network.fit(np.array(states), returns, verbose=False, shuffle=True, batch_size=len(actions))
                self.policy_network.fit(np.array(states), y_true, epochs = epochs, verbose=False, shuffle=True, batch_size=len(actions))
                states = []
                actions = []
                rewards = []
                returns = []
                f_values = []
                probabilities = []

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


    def load_model(self,policy_file="ppo-policy",value_file="ppo-file"):
        self.policy_network.load_weights(policy_file)
        self.value_network.load_weights(value_file)

    def save_model(self,policy_file="ppo-policy",value_file="ppo-file"):
        self.policy_network.save_weights(policy_file)
        self.value_network.save_weights(value_file)


def experiment(no_episode):
    rows = 7    
    columns = 7
    speed = 1.0
    max_steps = 250
    max_misses = 10
    observation_type =  'vector'
    seed = None

    env = Catch(rows=rows, columns=columns, speed=speed, max_steps=max_steps,
                max_misses=max_misses, observation_type=observation_type, seed=seed)

    # Initialize environment and Q-array
    if(observation_type == "vector"):
        policy_builder = N_Network()
        value_builder = N_Network(type="value")
    else:
        policy_builder = Conv_Network([8,4,4])
        value_builder = Conv_Network([8,4,4],type="value")

    ppo_agent = PPO_Agent(env,policy_builder,value_builder)


    
    ppo_agent.train(no_episode=no_episode)





    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPO')
    parser.add_argument('--episodes', type=int, default=500, help='Number of episodes to run')
    args = parser.parse_args()

    experiment(args.episodes)




