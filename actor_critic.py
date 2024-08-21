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

def create_reinforce_loss(M, entropy,n):
    def reinforce_loss(y_true,y_pred):
        actions = tf.one_hot(tf.cast(y_true[:,0],tf.int32),n)
        f_values = y_true[:,1]
        
        prob = tf.reduce_sum(y_pred * actions,1)
        
        entropies = -tf.reduce_sum(y_pred * tf.math.log(tf.add(y_pred,0.001)),1)
        
        return -1/M * tf.reduce_sum(f_values * tf.math.log(prob) + entropy * entropies )
    return reinforce_loss

def value_loss(y_true,y_pred):
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_sum(squared_difference, axis=-1)

class AC_Agent:
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
            n = 3,
            type = "both",
            verbose = True,
            render = False
    ):
        # with tf.GradientTape(persistent=True) as tape:
        self.policy_network.compile(optimizer = optimizer_policy, loss = create_reinforce_loss(M, entropy, self.env.action_space.n))
        self.value_network.compile(optimizer = optimizer_value, loss = value_loss)



        episode_rewards = []
        states = []
        actions = []
        rewards = []
        returns = []
        f_values = []
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
                s= s_next
                if(done):
                    break
            
            if(type == "bootstrap"):
                for t in range(len(states)-count,len(states)):
                    n_max = min(n,len(states)-t)
                    Q = (gamma**n_max) * self.value_network(np.array([states[t+n_max-1]]))[0]
                    for i in range(n_max):
                        if(t+i>= len(states)):
                            break
                        Q += (gamma**i) *rewards[t+i]
                    returns.append(Q)
                    f_values.append(Q)
                    
            if(type == "subtraction"):
                Q = 0
                episode_returns = []
                for t in range(len(states)-1, len(states)-count-1,-1):
                    Q = rewards[t] + gamma * Q
                    episode_returns.append(Q)
                episode_returns = episode_returns[::-1]
                returns+= episode_returns
                for t in range(count):
                    f_values.append(episode_returns[t] - self.value_network(np.array([states[len(states)-count + t]]))[0])

            if(type =="both"):
                for t in range(len(states)-count,len(states)):
                    n_max = min(n,len(states)-t)
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
                y_true = np.hstack([actions,f_values])

                self.value_network.fit(np.array(states), returns, verbose=False, shuffle=True, batch_size=len(actions))
                self.policy_network.fit(np.array(states), y_true, verbose=False, shuffle=True, batch_size=len(actions))
                states = []
                actions = []
                rewards = []
                returns = []
                f_values = []

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


    def load_model(self,policy_file="ac-policy",value_file="ac-file"):
        self.policy_network.load_weights(policy_file)
        self.value_network.load_weights(value_file)

    def save_model(self,policy_file="ac-policy",value_file="ac-file"):
        self.policy_network.save_weights(policy_file)
        self.value_network.save_weights(value_file)


def experiment(no_episode,type):
    rows = 7    
    columns = 7
    speed = 1.0
    max_steps = 250
    max_misses = 10
    observation_type = 'pixel' # 'vector'
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

    ac_agent = AC_Agent(env,policy_builder,value_builder)


    
    ac_agent.train(no_episode=no_episode,type= type)





    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Actor Critic')
    parser.add_argument('--type', type=str, default="both", help='Type of run: bootstrap/subtraction/both')
    parser.add_argument('--episodes', type=int, default=500, help='Number of episodes to run')
    args = parser.parse_args()

    experiment(args.episodes,args.type)

