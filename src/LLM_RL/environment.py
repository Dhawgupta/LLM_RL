from __future__ import annotations
from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple
from transformers.tokenization_utils import PreTrainedTokenizer
import numpy as np

# define text objects

@dataclass
class Text:
    text: str
    is_action: bool

TextHistory = List[Text]
text_history_to_str = lambda text_history: ''.join(map(lambda x: x.text, text_history))

# text trajectory should fit into a single context window, otherwise is truncated

class TextTrajectory(NamedTuple):
    text_history: TextHistory
    reward: List[float]
    done: bool

    def __post_init__(self):
        assert len(self.reward) == len(self.text_history) # reward for each text
        assert all([r == 0.0 for r, t in zip(self.reward, self.text_history) if not t.is_action]) # reward for non-actions texts is 0.0

# text trajectory chain is a linked list of text trajectories

class TextTrajectoryChain(NamedTuple):
    text_trajectory: TextTrajectory
    next: Optional[TextTrajectoryChain]

# text environment

class TextEnv(ABC):
    @abstractmethod
    def step(self, text_history: TextHistory) -> Tuple[TextHistory, float, bool]:
        pass

    @abstractmethod
    def reset(self, seed: Optional[int]=None, options: Optional[Dict]=None) -> TextHistory:
        pass

    def close(self) -> None:
        pass

# text policy

class TextPolicy(ABC):
    @abstractmethod
    def act(self, text_history: TextHistory) -> TextHistory:
        pass

# interact with the environment

def interact_environment(
    env: TextEnv, 
    policy: TextPolicy, 
    text_history: Optional[TextHistory]=None, 
    env_seed: Optional[int]=None, 
    env_options: Optional[Dict]=None, 
) -> List[Tuple[TextHistory, TextHistory, TextHistory, float, bool]]:
    
    if text_history is None:
        text_history = env.reset(env_seed, env_options)
    
    transitions = []
    done = False
    while not done:
        pre_action_history = text_history
        text_history = policy.act(text_history)
        post_action_history = text_history

        text_history, r, done = env.step(text_history)
        post_transition_history = text_history
        
        transitions.append((pre_action_history, post_action_history, post_transition_history, r, done,))
    return transitions

# user policy

class UserPolicy(TextPolicy):    
    def __init__(
        self, 
        initial_str: str, 
        postproc_print_f: Optional[Callable[[str], str]]=None, 
        postproc_action_f: Optional[Callable[[str], str]]=None, 
    ):
        self.initial_str = initial_str
        self.postproc_print_f = postproc_print_f if postproc_print_f is not None else lambda x: x
        self.postproc_action_f = postproc_action_f if postproc_action_f is not None else lambda x: x

    def act(self, text_history: TextHistory) -> TextHistory:
        print('='*25)
        print(self.postproc_print_f(text_history_to_str(text_history)))
        print('='*25)
        response = input(self.initial_str)
        response = self.initial_str + response
        return text_history+[Text(self.postproc_action_f(response), True)]


"""tokenize environment objects"""


class TokenHistory(NamedTuple):
    tokens: np.ndarray # 1d int32 array
    is_action: np.ndarray # 1d bool array

    def __post_init__(self):
        assert len(self.tokens.shape) == 1 and len(self.is_action.shape) == 1, '(tokens, is_action) must be 1 dimensional'
        assert self.tokens.shape == self.is_action.shape, '(tokens, is_action) must have the same shape'

class TokenTrajectory(NamedTuple):
    tokens: np.ndarray # 1d int32 array
    is_action: np.ndarray # 1d bool array
    reward: np.ndarray # 1d float32 array
    done: np.ndarray # 1d bool array

    def __post_init__(self):
        assert all(len(item.shape) == 1 for item in self), '(tokens, is_action, reward, done) must all be 1 dimensional'
        assert all([item.shape == self[0].shape for item in self[1:]]), '(tokens, is_action, reward, done) must all have the same shape'
        assert not np.any(((1 - self.is_action.astype(np.float32)) * self.reward) != 0.0), 'reward must be 0.0 if not an action'
        assert not np.any(self.done[:-1]), 'done can only be true at the last token in the sequence'

class TokenTrajectoryChain(NamedTuple):
    token_trajectory: TokenTrajectory
    next: Optional[TokenTrajectoryChain]


def text_history_to_token_history(text_history: TextHistory, tokenizer: PreTrainedTokenizer) -> TokenHistory:
    tokens = []
    is_action = []
    
    for item in text_history:
        
        # tokenize
        new_tokens = tokenizer.encode(item.text)
        
        tokens.extend(new_tokens)
        is_action.extend([item.is_action]*len(new_tokens))
    
    return TokenHistory(
        np.array(tokens, dtype=np.int32), 
        np.array(is_action, dtype=np.bool_), 
    )

def text_trajectory_to_token_trajectory(text_trajectory: TextTrajectory, tokenizer: PreTrainedTokenizer) -> TokenTrajectory:
    
    tokens = []
    is_action = []
    reward = []
    done = []

    for i, item in enumerate(text_trajectory.text_history):
        
        # tokenize
        new_tokens = tokenizer.encode(item.text)
        
        tokens.extend(new_tokens)
        is_action.extend([item.is_action]*len(new_tokens))
        
        # add reward at the last token in the text
        reward.extend(([0.0]*(len(new_tokens)-1))+[text_trajectory.reward[i]])
        done.extend([False]*len(new_tokens))
    
    # add done at last token in trajectory
    done[-1] = text_trajectory.done

    return TokenTrajectory(
        np.array(tokens, dtype=np.int32), 
        np.array(is_action, dtype=np.bool_), 
        np.array(reward, dtype=np.float32), 
        np.array(done, dtype=np.bool_), 
    )

def text_trajectory_chain_to_token_trajectory_chain(text_trajectory_chain: TextTrajectoryChain, tokenizer: PreTrainedTokenizer) -> TokenTrajectoryChain:
    return TokenTrajectoryChain(
        text_trajectory_to_token_trajectory(text_trajectory_chain.text_trajectory, tokenizer), 
        text_trajectory_chain_to_token_trajectory_chain(text_trajectory_chain.next, tokenizer) if text_trajectory_chain.next is not None else None, 
    )
