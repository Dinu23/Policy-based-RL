import argparse
import numpy as np
import tensorflow as tf
import scipy
from catch import Catch

from model import Conv_Network, N_Network


class CMAES_Agent:
    def __init__(self,env,policy_network_builder) -> None:
        self.env = env
        self.policy_network_builder = policy_network_builder
        self.policy_network = self.policy_network_builder.create_model(self.env.observation_space.shape, self.env.action_space.n)

    def count_weights(self):
        return np.sum([np.prod(layer.get_shape()) for layer in self.policy_network.trainable_weights])

    def set_weights(self,weights):
        new_weights = []
        i = 0
        for param in self.policy_network.trainable_weights:
            j = i +  np.prod(param.shape)
            new_param = weights[i:j]
            new_param = np.reshape(new_param,param.shape)
            new_weights.append(new_param)
            i = j
        self.policy_network.set_weights(new_weights)
        

    def fitness(self, no_runs):
        fitness = 0
        for _ in range(no_runs):
            s =  self.env.reset()
            while(True):
                action_probabilities = self.policy_network(np.array([s]))[0]
                a = np.random.choice(self.env.action_space.n,p=np.array(action_probabilities))
                s_next,r,done,_ = self.env.step(a)
                s = s_next
                if(done):
                    break
            fitness += self.env.total_reward
        
        fitness = fitness/no_runs
        # print(fitness)     
        return fitness

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


    def load_model(self,policy_file="cmaes-policy"):
        self.policy_network.load_weights(policy_file)

    def save_model(self,policy_file="cmaes-policy"):
        self.policy_network.save_weights(policy_file)
        


def sort_decending(x,y,z):
    order = np.argsort(z)[::-1]

    return x[order], y[order], z[order]

def generate_weights(lamda,miu):
    weights = np.array([np.log((lamda-1)/2) - np.log(i) for i in range(1,miu+1) ])
    weights = weights / np.sum(weights)
    return weights


class CMAES:

    def __init__(self,env,policy_network_builder, sigma = 0.3) -> None:
        self.env = env
        self.agent = CMAES_Agent(env, policy_network_builder)
        self.no_weights = self.agent.count_weights()
        print(self.no_weights)
        self.covariance_matrix = np.identity(self.no_weights)
        # print(self.covariance_matrix.shape)
        self.means = np.zeros(self.no_weights)
        self.sigma = sigma
        self.p_c = np.zeros((self.no_weights,1))
        self.p_sigma = np.zeros((self.no_weights,1))
        # print(self.means.shape)
        

    def train(self,
            no_episodes = 1000,
            lamda=100,
            miu=50,
            weights_gen = generate_weights,
            no_runs = 3,

            verbose = True
            ):
        weights = weights_gen(lamda,miu)
        episode_rewards = []

        miu_weights = np.sum(weights**2) 
        c_c = 4/self.no_weights
        c_sigma = 4/self.no_weights
        c_1 = 2/(self.no_weights**2) 
        c_miu = miu_weights/(self.no_weights**2)
        d_sigma = 1+ np.sqrt(miu_weights/self.no_weights)
        E_n = np.sqrt(self.no_weights) *(1 - 1/(4*self.no_weights) + 1/(21* self.no_weights**2))

        for episode in range(no_episodes):
            ys = np.random.multivariate_normal(np.zeros(self.no_weights), self.covariance_matrix, lamda)
        
            xs =  self.means + self.sigma * ys
            fits = []
            for x in xs:
                self.agent.set_weights(x)
                fitness = self.agent.fitness(no_runs)
                fits.append(fitness)
            fits = np.array(fits)
            xs,ys,fits = sort_decending(xs,ys,fits)
            
            ys_miu = ys[:miu]
            y_weights = np.matrix.transpose(np.matmul(np.array([weights]),ys_miu)[0])
            
            covar_s = scipy.linalg.inv(scipy.linalg.sqrtm(self.covariance_matrix))
            

            self.means = self.means + self.sigma * y_weights

            self.p_c = (1-c_c) * self.p_c + 0 if(np.linalg.norm(self.p_sigma,2) >= 1.5*np.sqrt(self.no_weights)) else np.sqrt(1-(1-c_c)**2)* np.sqrt(miu_weights) * y_weights
            # print(self.p_c.shape)

            self.p_sigma = (1-c_sigma) * self.p_sigma + np.sqrt(1-(1-c_sigma)**2)* np.sqrt(miu_weights) *np.matmul(covar_s , y_weights)
            # print(self.p_sigma.shape)

            val = np.zeros((self.no_weights,self.no_weights))
            for pos in range(miu):
                val += weights[pos] * np.matmul( np.matrix.transpose(ys_miu[pos]) ,ys_miu[pos])

            self.covariance_matrix = (1-c_1 - c_miu) * self.covariance_matrix + c_1 * np.matmul(self.p_c, np.matrix.transpose(self.p_c)) + c_miu * val
            self.sigma = self.sigma * np.exp(c_sigma /d_sigma * (np.linalg.norm(self.p_c,2)/E_n-1))

            episode_reward = fits[0]
            if(verbose and episode % 1 == 0):
                print("Episode ",episode+1, " reward: ",episode_reward)
            episode_rewards.append(episode_reward)

        return episode_rewards





def experiment(no_episode):
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
    else:
        policy_builder = Conv_Network([8,4,4])
        
    lamda = 20
    miu = 10
    workers = 10
    cmaes = CMAES(env,policy_builder)
    rewards = cmaes.train(no_episode,lamda=lamda, miu = miu)# workers=workers)
    print(rewards)





    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPO')
    parser.add_argument('--episodes', type=int, default=500, help='Number of episodes to run')
    args = parser.parse_args()

    experiment(args.episodes)


