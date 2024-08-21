from CMAES import CMAES
from actor_critic import AC_Agent
from catch import Catch
from model import Conv_Network, N_Network
from monte_carlo_policy_gradient import MCPG_Agent
from keras.optimizers import Adam
import numpy as np

from ppo import PPO_Agent



def run_mcpg(no_repetitions, no_episodes):
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

    models = [[8,4,4],[4]]
    learning_rates = [0.1,0.01,0.001]
    entropies = [0.01, 0.1, 0.5]

    verbose = True
    render = False

    for model in models:
        for learning_rate in learning_rates:
            for entropy in entropies:
                results = []
                for rep in range(no_repetitions):
                    builder = Conv_Network(model)
                    mcpg_agent = MCPG_Agent(env, builder)
                    optimizer = Adam(learning_rate=learning_rate)
                    episode_rewards = np.array(mcpg_agent.train(no_episodes,optimizer=optimizer,entropy=entropy,verbose = verbose, render=render))
                    results.append(episode_rewards)
                results = np.array(results)
                np.save(f"mcpg_model_{len(model)}_lr_{learning_rate}_entropy_{entropy}",results)






def run_ac_bootstrap(no_repetitions, no_episodes):
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

    models = [[8,4,4],[4]]
    learning_rates = [0.1,0.01,0.001]
    entropies = [0.01, 0.1, 0.5]
    type= "bootstrap"
    verbose = True
    render = False

    for model in models:
        for learning_rate in learning_rates:
            for entropy in entropies:
                results = []
                for rep in range(no_repetitions):
                    policy_builder = Conv_Network(model)
                    value_builder = Conv_Network(model,type="value")
                    ac_agent = AC_Agent(env, policy_builder, value_builder)
                    optimizer_policy = Adam(learning_rate=learning_rate)
                    optimizer_value = Adam(learning_rate=learning_rate)
                    episode_rewards = np.array(ac_agent.train(no_episodes,type=type,optimizer_policy=optimizer_policy,optimizer_value=optimizer_value,entropy=entropy,verbose = verbose, render=render))
                    results.append(episode_rewards)
                results = np.array(results)
                np.save(f"ac_bootstrap_model_{len(model)}_lr_{learning_rate}_entropy_{entropy}",results)




def run_ac_subtraction(no_repetitions, no_episodes):
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

    models = [[8,4,4],[4]]
    learning_rates = [0.1,0.01,0.001]
    entropies = [0.01, 0.1, 0.5]
    type= "subtraction"
    verbose = True
    render = False

    for model in models:
        for learning_rate in learning_rates:
            for entropy in entropies:
                results = []
                for rep in range(no_repetitions):
                    policy_builder = Conv_Network(model)
                    value_builder = Conv_Network(model,type="value")
                    ac_agent = AC_Agent(env, policy_builder, value_builder)
                    optimizer_policy = Adam(learning_rate=learning_rate)
                    optimizer_value = Adam(learning_rate=learning_rate)
                    episode_rewards = np.array(ac_agent.train(no_episodes,type=type,optimizer_policy=optimizer_policy,optimizer_value=optimizer_value,entropy=entropy,verbose = verbose, render=render))
                    results.append(episode_rewards)
                results = np.array(results)
                np.save(f"ac_subtraction_model_{len(model)}_lr_{learning_rate}_entropy_{entropy}",results)


def run_ac_both(no_repetitions, no_episodes):
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

    models = [[8,4,4]]
    learning_rates = [0.1,0.01,0.001]
    entropies = [0.01, 0.1, 0.5]
    type="both"
    verbose = True
    render = False

    for model in models:
        for learning_rate in learning_rates:
            for entropy in entropies:
                results = []
                for rep in range(no_repetitions):
                    policy_builder = Conv_Network(model)
                    value_builder = Conv_Network(model,type="value")
                    ac_agent = AC_Agent(env, policy_builder, value_builder)
                    optimizer_policy = Adam(learning_rate=learning_rate)
                    optimizer_value = Adam(learning_rate=learning_rate)
                    episode_rewards = np.array(ac_agent.train(no_episodes,type=type,optimizer_policy=optimizer_policy,optimizer_value=optimizer_value,entropy=entropy,verbose = verbose, render=render))
                    results.append(episode_rewards)
                results = np.array(results)
                np.save(f"ac_both_model_{len(model)}_lr_{learning_rate}_entropy_{entropy}",results)



def run_different_sizes(no_repetitions, no_episodes):
    rows = 5    
    columns = 5
    speed = 1.0
    max_steps = 250
    max_misses = 10
    observation_type = 'pixel' # 'vector'
    seed = None

    # Initialize environment and Q-array
    env = Catch(rows=rows, columns=columns, speed=speed, max_steps=max_steps,
                max_misses=max_misses, observation_type=observation_type, seed=seed)

    model = [8,4,4]
    learning_rate = 0.01
    entropy = 0.1
    type="both"
    verbose = True
    render = False

    results = []
    for rep in range(no_repetitions):
        
        policy_builder = Conv_Network(model)
        value_builder = Conv_Network(model,type="value")
        ac_agent = AC_Agent(env, policy_builder, value_builder)
        optimizer_policy = Adam(learning_rate=learning_rate)
        optimizer_value = Adam(learning_rate=learning_rate)
        episode_rewards = np.array(ac_agent.train(no_episodes,type=type,optimizer_policy=optimizer_policy,optimizer_value=optimizer_value,entropy=entropy,verbose = verbose, render=render))
        results.append(episode_rewards)

    results = np.array(results)
    np.save(f"ac_both_env_5x5",results)

    rows = 11    
    columns = 11
    speed = 1.0
    max_steps = 250
    max_misses = 10
    observation_type = 'pixel' # 'vector'
    seed = None

    # Initialize environment and Q-array
    env = Catch(rows=rows, columns=columns, speed=speed, max_steps=max_steps,
                max_misses=max_misses, observation_type=observation_type, seed=seed)

    results = []
    for rep in range(no_repetitions):
        policy_builder = Conv_Network(model)
        value_builder = Conv_Network(model,type="value")
        ac_agent = AC_Agent(env, policy_builder, value_builder)
        optimizer_policy = Adam(learning_rate=learning_rate)
        optimizer_value = Adam(learning_rate=learning_rate)
        episode_rewards = np.array(ac_agent.train(no_episodes,type=type,optimizer_policy=optimizer_policy,optimizer_value=optimizer_value,entropy=entropy,verbose = verbose, render=render))
        results.append(episode_rewards)
    results = np.array(results)
    np.save(f"ac_both_env_11x11",results)



def run_different_speed(no_repetitions, no_episodes):
    rows = 7    
    columns = 7
    speed = 0.5
    max_steps = 250
    max_misses = 10
    observation_type = 'pixel' # 'vector'
    seed = None

    # Initialize environment and Q-array
    env = Catch(rows=rows, columns=columns, speed=speed, max_steps=max_steps,
                max_misses=max_misses, observation_type=observation_type, seed=seed)

    model = [8,4,4]
    learning_rate = 0.01
    entropy = 0.1
    type="both"
    verbose = True
    render = False

    results = []
    for rep in range(no_repetitions):
        policy_builder = Conv_Network(model)
        value_builder = Conv_Network(model,type="value")
        ac_agent = AC_Agent(env, policy_builder, value_builder)
        optimizer_policy = Adam(learning_rate=learning_rate)
        optimizer_value = Adam(learning_rate=learning_rate)
        episode_rewards = np.array(ac_agent.train(no_episodes,type=type,optimizer_policy=optimizer_policy,optimizer_value=optimizer_value,entropy=entropy,verbose = verbose, render=render))
        results.append(episode_rewards)
    results = np.array(results)
    np.save(f"ac_both_speed_0.5",results)

    rows = 7    
    columns = 7
    speed = 1.5
    max_steps = 250
    max_misses = 10
    observation_type = 'pixel' # 'vector'
    seed = None

    # Initialize environment and Q-array
    env = Catch(rows=rows, columns=columns, speed=speed, max_steps=max_steps,
                max_misses=max_misses, observation_type=observation_type, seed=seed)

    results = []
    for rep in range(no_repetitions):
        policy_builder = Conv_Network(model)
        value_builder = Conv_Network(model,type="value")
        ac_agent = AC_Agent(env, policy_builder, value_builder)
        optimizer_policy = Adam(learning_rate=learning_rate)
        optimizer_value = Adam(learning_rate=learning_rate)
        episode_rewards = np.array(ac_agent.train(no_episodes,type=type,optimizer_policy=optimizer_policy,optimizer_value=optimizer_value,entropy=entropy,verbose = verbose, render=render))
        results.append(episode_rewards)
    results = np.array(results)
    np.save(f"ac_both_speed_1.5",results)

def run_vector(no_repetitions, no_episodes):
    rows = 7    
    columns = 7
    speed = 1.0
    max_steps = 250
    max_misses = 10
    observation_type = 'vector' # 'vector'
    seed = None

    # Initialize environment and Q-array
    env = Catch(rows=rows, columns=columns, speed=speed, max_steps=max_steps,
                max_misses=max_misses, observation_type=observation_type, seed=seed)

    model = [32,16,16]
    learning_rate = 0.01
    entropy = 0.1
    type="both"
    verbose = True
    render = False

    results = []
    for rep in range(no_repetitions):
        policy_builder = N_Network(model)
        value_builder = N_Network(model,type="value")
        ac_agent = AC_Agent(env, policy_builder, value_builder)
        optimizer_policy = Adam(learning_rate=learning_rate)
        optimizer_value = Adam(learning_rate=learning_rate)
        episode_rewards = np.array(ac_agent.train(no_episodes,type=type,optimizer_policy=optimizer_policy,optimizer_value=optimizer_value,entropy=entropy,verbose = verbose, render=render))
        results.append(episode_rewards)
    results = np.array(results)
    np.save(f"ac_both_vector_speed_1.0",results)

    
    # rows = 7    
    # columns = 7
    # speed = 1.5
    # max_steps = 250
    # max_misses = 10
    # observation_type = 'vector' # 'vector'
    # seed = None

    # # Initialize environment and Q-array
    # env = Catch(rows=rows, columns=columns, speed=speed, max_steps=max_steps,
    #             max_misses=max_misses, observation_type=observation_type, seed=seed)

    # model = [8,4,4]
    # learning_rate = 0.01
    # entropy = 0.1
    # type="both"
    # verbose = True
    # render = False

    # results = []
    # for rep in range(no_repetitions):
    #     policy_builder = N_Network(model)
    #     value_builder = N_Network(model,type="value")
    #     ac_agent = AC_Agent(env, policy_builder, value_builder)
    #     optimizer_policy = Adam(learning_rate=learning_rate)
    #     optimizer_value = Adam(learning_rate=learning_rate)
    #     episode_rewards = np.array(ac_agent.train(no_episodes,type=type,optimizer_policy=optimizer_policy,optimizer_value=optimizer_value,entropy=entropy,verbose = verbose, render=render))
    #     results.append(episode_rewards)
    # results = np.array(results)
    # np.save(f"ac_both_vector_speed_1.5",results)

def run_combined_env(no_repetitions, no_episodes):
    rows = 7    
    columns = 14
    speed = 0.5
    max_steps = 250
    max_misses = 10
    observation_type = 'pixel' # 'vector'
    seed = None

    # Initialize environment and Q-array
    env = Catch(rows=rows, columns=columns, speed=speed, max_steps=max_steps,
                max_misses=max_misses, observation_type=observation_type, seed=seed)

    model = [8,4,4]
    learning_rate = 0.01
    entropy = 0.1
    type="both"
    verbose = True
    render = False

    results = []
    for rep in range(no_repetitions):
        policy_builder = Conv_Network(model)
        value_builder = Conv_Network(model,type="value")
        ac_agent = AC_Agent(env, policy_builder, value_builder)
        optimizer_policy = Adam(learning_rate=learning_rate)
        optimizer_value = Adam(learning_rate=learning_rate)
        episode_rewards = np.array(ac_agent.train(no_episodes,type=type,optimizer_policy=optimizer_policy,optimizer_value=optimizer_value,entropy=entropy,verbose = verbose, render=render))
        results.append(episode_rewards)
    results = np.array(results)
    np.save(f"ac_both_env_14x7_speed_0.5",results)

def run_ppo(no_repetitions, no_episodes):
    rows = 7    
    columns = 7
    speed = 1
    max_steps = 250
    max_misses = 10
    observation_type = 'pixel' # 'vector'
    seed = None

    # Initialize environment and Q-array
    env = Catch(rows=rows, columns=columns, speed=speed, max_steps=max_steps,
                max_misses=max_misses, observation_type=observation_type, seed=seed)

    model = [8,4,4]
    learning_rate = 0.01
    entropy = 0.1
    epoch = 7
    epsilon = 0.2
    verbose = True
    render = False

    results = []
    for rep in range(no_repetitions):
        print(rep)
        policy_builder = Conv_Network(model)
        value_builder = Conv_Network(model,type="value")
        ac_agent = PPO_Agent(env, policy_builder, value_builder)
        optimizer_policy = Adam(learning_rate=learning_rate)
        optimizer_value = Adam(learning_rate=learning_rate)
        episode_rewards = np.array(ac_agent.train(no_episodes,optimizer_policy=optimizer_policy,optimizer_value=optimizer_value,entropy=entropy,epochs= epoch,epsilon = epsilon,verbose = verbose, render=render))
        results.append(episode_rewards)
    results = np.array(results)
    np.save(f"ppo_2",results)


def run_cmaes(no_repetitions,no_episodes):
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
    workers = 20

    for rep in range(no_repetitions):
        print(rep)
        rep = 4
        cmaes = CMAES(env,policy_builder)
        rewards = cmaes.train(no_episodes,lamda=lamda, miu = miu)#, workers=workers)
        rewards = np.array(rewards)
        np.save(f"CMAES_rep{rep}",rewards)
        break



def run_expeiemnts():
    no_repetitions = 5
    no_episodes = 1000

    # run_mcpg(no_repetitions,no_episodes)
    
    # run_ac_bootstrap(no_repetitions,no_episodes)

    # run_ac_subtraction(no_repetitions,no_episodes)
    
    # run_ac_both(no_repetitions,no_episodes)

    # run_ppo(no_repetitions,no_episodes)
    run_vector(5,1000)
    # no_repetitions = 5
    # no_episodes = 200
    # run_cmaes(no_repetitions,no_episodes)
    pass


run_expeiemnts()

