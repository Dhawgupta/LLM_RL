from __future__ import annotations
from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union, Any, Iterator
from transformers.tokenization_utils import PreTrainedTokenizer
import numpy as np
from LLM_RL.utils import get_tensor_stats
from tqdm.auto import tqdm

# define text objects

@dataclass
class Text:
    text: str
    is_action: bool

TextHistory = Tuple[Text]
text_history_to_str = lambda text_history: ''.join(map(lambda x: x.text, text_history))

# text trajectory should fit into a single context window, otherwise is truncated

class TextTrajectory(NamedTuple):
    text_history: TextHistory
    reward: Tuple[float]
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

class InteractionTransition(NamedTuple):
    pre_action_history: TextHistory
    post_action_history: TextHistory
    post_transition_history: TextHistory
    reward: float
    done: bool

def interact_environment(
    env: TextEnv, 
    policy: TextPolicy, 
    initial_text_history: Optional[TextHistory]=None, 
    env_seed: Optional[int]=None, 
    env_options: Optional[Dict]=None, 
) -> List[InteractionTransition]:
    
    text_history = initial_text_history
    if text_history is None:
        text_history = env.reset(env_seed, env_options)
    
    transitions = []
    done = False
    while not done:
        pre_action_history = text_history
        text_history = policy.act(text_history)
        post_action_history = text_history

        text_history, reward, done = env.step(text_history)
        post_transition_history = text_history
        
        transitions.append(
            InteractionTransition(
                pre_action_history=pre_action_history, 
                post_action_history=post_action_history, 
                post_transition_history=post_transition_history, 
                reward=reward, 
                done=done, 
            )
        )
    return transitions

def text_env_eval(
    env: TextEnv, 
    policy: TextPolicy, 
    n_rounds: int, 
    initial_text_history: Optional[TextHistory]=None, 
    seed_generator: Optional[Iterator[int]]=None, 
    env_options: Optional[Dict]=None, 
    interaction_callback: Optional[Callable[[List[Tuple[TextHistory, TextHistory, TextHistory, float, bool]]], None]]=None, 
    verbose: bool=True, 
) -> Tuple[List[List[InteractionTransition]], Dict[str, Any]]:
    
    interactions, rewards, dones = [], [], []
    for _ in tqdm(range(n_rounds), disable=not verbose):
        interaction = interact_environment(
            env, 
            policy, 
            initial_text_history=initial_text_history, 
            env_seed=None if seed_generator is None else next(seed_generator), 
            env_options=env_options, 
        )
        
        interactions.append(interaction)
        rewards.append(sum(map(lambda x: x.reward, interaction)))
        dones.append(interaction[-1].done)
        if interaction_callback is not None:
            interaction_callback(interaction)
    
    rewards = np.asarray(rewards, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    results_summary = dict(
        reward=dict(
            mean=np.mean(rewards), 
            std=np.std(rewards), 
            min=np.min(rewards), 
            max=np.max(rewards), 
        ), 
        done=dict(
            mean=np.mean(dones), 
            std=np.std(dones), 
            min=np.min(dones), 
            max=np.max(dones), 
        ), 
    )
    
    return interactions, results_summary

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
    
    @classmethod
    def from_text_history(
        cls, 
        text_history: TextHistory, 
        tokenizer: PreTrainedTokenizer, 
        token_process: Optional[Callable[[List[int]], List[int]]]=None, 
    ) -> TokenHistory:
        if token_process is None:
            token_process = lambda x: x

        tokens = []
        is_action = []
        
        for item in text_history:
            
            # tokenize
            new_tokens = token_process(tokenizer.encode(item.text))
            
            tokens.extend(new_tokens)
            is_action.extend([item.is_action]*len(new_tokens))
        
        return cls(
            np.array(tokens, dtype=np.int32), 
            np.array(is_action, dtype=np.bool_), 
        )

class TokenTrajectory(NamedTuple):
    tokens: np.ndarray # 1d int32 array
    is_action: np.ndarray # 1d bool array
    reward: np.ndarray # 1d float32 array
    done: Union[bool, np.ndarray] # bool scalar

    def __post_init__(self):
        assert all(len(item.shape) == 1 for item in self), '(tokens, is_action, reward, done) must all be 1 dimensional'
        assert all([item.shape == self[0].shape for item in self[1:]]), '(tokens, is_action, reward, done) must all have the same shape'
        assert not np.any(((1 - self.is_action.astype(np.float32)) * self.reward) != 0.0), 'reward must be 0.0 if not an action'
    
    @classmethod
    def from_text_trajectory(
        cls, 
        text_trajectory: TextTrajectory, 
        tokenizer: PreTrainedTokenizer, 
        token_process: Optional[Callable[[List[int]], List[int]]]=None, 
    ) -> TokenTrajectory:
        if token_process is None:
            token_process = lambda x: x
        
        tokens = []
        is_action = []
        reward = []
        done = []

        for i, item in enumerate(text_trajectory.text_history):
            
            # tokenize
            new_tokens = token_process(tokenizer.encode(item.text))
            
            tokens.extend(new_tokens)
            is_action.extend([item.is_action]*len(new_tokens))
            
            # add reward at the last token in the text
            reward.extend(([0.0]*(len(new_tokens)-1))+[text_trajectory.reward[i]])
            done.extend([False]*len(new_tokens))
        
        # get done
        done = text_trajectory.done

        return cls(
            np.array(tokens, dtype=np.int32), 
            np.array(is_action, dtype=np.bool_), 
            np.array(reward, dtype=np.float32), 
            np.array(done, dtype=np.bool_), 
        )

class TokenTrajectoryChain(NamedTuple):
    token_trajectory: TokenTrajectory
    next: Optional[TokenTrajectoryChain]

    def __post_init__(self):
        curr, dones = self, []
        while curr.next is not None:
            dones.append(curr.token_trajectory.done)
            curr = curr.next
        assert not np.any(dones[:-1]), 'token trajectory chain can only be done at the end'
    
    def to_list(self) -> List[TokenTrajectory]:
        curr, l = self, []
        while curr is not None:
            l.append(curr.token_trajectory)
            curr = curr.next
        return l

    @classmethod
    def from_text_trajectory_chain(
        cls, 
        text_trajectory_chain: TextTrajectoryChain, 
        tokenizer: PreTrainedTokenizer, 
        token_process: Optional[Callable[[List[int]], List[int]]]=None, 
    ) -> TokenTrajectoryChain:
        return TokenTrajectoryChain(
            TokenTrajectory.from_text_trajectory(
                text_trajectory_chain.text_trajectory, 
                tokenizer, 
                token_process=token_process, 
            ), 
            cls.from_text_trajectory_chain(
                text_trajectory_chain.next, 
                tokenizer, 
                token_process=token_process, 
            ) if text_trajectory_chain.next is not None else None, 
        )
