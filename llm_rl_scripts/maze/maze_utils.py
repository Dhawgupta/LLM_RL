from llm_rl_scripts.maze.env import MazeEnv, describe_observation, describe_observation_give_position, illegal_penalty_reward, illegal_penalty_diff_scale, manhatten_actions, standard_reward, describe_observation_only_walls
from llm_rl_scripts.maze.mazes import maze2d_umaze, double_t_maze
import numpy as np

def setup_maze_env(maze_name, describe_function, reward_function=None):
    # setup environment
    if maze_name == 'umaze':
        maze = maze2d_umaze()
        valid_goals = np.array([[3, 3]])
        start_position = (3, 1)
    elif maze_name == "double_t_maze":
        maze = double_t_maze()
        valid_goals = np.array([[8, 6]])
        start_position = (1, 1)
    else:
        raise ValueError(f'unknown maze name: {maze_name}')
    
    # valid_goals = np.where(maze == 0)
    # valid_goals = np.array(list(zip(valid_goals[0], valid_goals[1])), dtype=np.int32)
    if describe_function == "describe_observation":
        describe_function = describe_observation
    elif describe_function == "describe_observation_give_position":
        describe_function = describe_observation_give_position
    elif describe_function == "describe_observation_only_walls":
        describe_function = describe_observation_only_walls
    else:
        raise ValueError(f'unknown describe function: {describe_function}')
    
    if reward_function is None or reward_function == "standard_reward":
        reward_function = standard_reward
    elif reward_function == "illegal_penalty_reward":
        reward_function = illegal_penalty_reward
    elif reward_function == "illegal_penalty_diff_scale":
        reward_function = illegal_penalty_diff_scale
    else:
        raise ValueError(f'unknown reward function: {reward_function}')
    
    env = MazeEnv(
        maze=maze, 
        valid_goals=valid_goals, 
        actions=manhatten_actions, 
        max_steps=100, 
        display_initial_position=True,
        describe_function=describe_function,
        reward_function=reward_function,
        last_k=1,
    )
    return env

def pick_start_position(maze_name):
    if maze_name == 'umaze':
        return (3, 1)
    elif maze_name == "double_t_maze":
        return (1, 1)
    else:
        raise ValueError(f'unknown maze name: {maze_name}')