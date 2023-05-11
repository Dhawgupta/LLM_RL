from __future__ import annotations
from collections import defaultdict
from copy import deepcopy
import random
from enum import Enum
from typing import List, Optional
from dataclasses import dataclass
from termcolor import colored
from llm_rl_scripts.wordle.utils import Cache

IDX2CHAR = list('abcdefghijklmnopqrstuvwxyz')
CHAR2IDX = {c: i for i, c in enumerate(IDX2CHAR)}
ALPHA_SIZE = len(IDX2CHAR)
N_CHARS = 5
N_TRIES = 6

class CharKnowledge(Enum):
    NOT_HERE = 0
    POSSIBLE = 1
    HERE = 2

@dataclass
class CharState:
    position_knowledge: List[CharKnowledge]

    def __post_init__(self):
        assert len(self.position_knowledge) == N_CHARS
    
    def __eq__(self, other: CharState):
        return self.position_knowledge == other.position_knowledge

    def __hash__(self) -> int:
        return hash(tuple(self.position_knowledge))
    
    @classmethod
    def unknown(cls):
        return cls([CharKnowledge.POSSIBLE for _ in range(N_CHARS)])
    
    @classmethod
    def not_used(cls):
        return cls([CharKnowledge.NOT_HERE for _ in range(N_CHARS)])
    
    def correct_pos(self, pos):
        new_mask = deepcopy(self.position_knowledge)
        new_mask[pos] = CharKnowledge.HERE
        return CharState(new_mask)
    
    def wrong_pos(self, pos):
        new_mask = deepcopy(self.position_knowledge)
        new_mask[pos] = CharKnowledge.NOT_HERE
        return CharState(new_mask)
    
    def word_satisfies(self, c, word):
        if all([k == CharKnowledge.POSSIBLE for k in self.position_knowledge]):
            return True
        if all([k == CharKnowledge.NOT_HERE for k in self.position_knowledge]):
            return c not in word
        for i in range(N_CHARS):
            if self.position_knowledge[i] == CharKnowledge.HERE:
                if c != word[i]:
                    return False
            if self.position_knowledge[i] == CharKnowledge.NOT_HERE:
                if c == word[i]:
                    return False
        return c in word

class WordleState:
    def __init__(self, state: List[CharState]):
        assert len(state) == ALPHA_SIZE
        self.state = state
    
    @classmethod
    def initial_state(cls):
        return cls([CharState.unknown() for i in range(ALPHA_SIZE)])
    
    def word_in_state(self, word: str) -> bool:
        for i in range(len(self.state)):
            if not self.state[i].word_satisfies(IDX2CHAR[i], word):
                return False
        return True
    
    def transition_state(self, selected_word: str, target_word: str):
        # assert self.word_in_state(target_word)
        new_state = deepcopy(self.state)
        for i, c in enumerate(selected_word):
            if c == target_word[i]:
                new_state[CHAR2IDX[c]] = new_state[CHAR2IDX[c]].correct_pos(i)
            elif c in target_word:
                new_state[CHAR2IDX[c]] = new_state[CHAR2IDX[c]].wrong_pos(i)
            else:
                new_state[CHAR2IDX[c]] = new_state[CHAR2IDX[c]].not_used()
        return WordleState(new_state)
    
    def transition_from_str(self, selected_word: str, transition_str: str):
        transition_chars = [transition_str[i:(i+3)] for i in range(0, len(transition_str), 3)]
        new_state = deepcopy(self.state)
        for i, c in enumerate(selected_word):
            transition_c = transition_chars[i]
            if transition_c == '<g>':
                new_state[CHAR2IDX[c]] = new_state[CHAR2IDX[c]].correct_pos(i)
            elif transition_c == '<y>':
                new_state[CHAR2IDX[c]] = new_state[CHAR2IDX[c]].wrong_pos(i)
            elif transition_c == '<b>':
                new_state[CHAR2IDX[c]] = new_state[CHAR2IDX[c]].not_used()
            else:
                raise NotImplementedError
        return WordleState(new_state)

    def __eq__(self, other: WordleState) -> bool:
        return self.state == other.state
    
    def __hash__(self) -> int:
        return hash(tuple(self.state))
    
    def __str__(self):
        str_items = defaultdict(list)
        for i in range(len(self.state)):
            if all([item == CharKnowledge.POSSIBLE for item in self.state[i].position_knowledge]):
                continue
            if all([item == CharKnowledge.NOT_HERE for item in self.state[i].position_knowledge]):
                str_items['none'].append(IDX2CHAR[i])
                continue
            char_values = []
            k = 'wrong_pos'
            for x in range(N_CHARS):
                if self.state[i].position_knowledge[x] == CharKnowledge.HERE:
                    k = 'known'
                    char_values.append(colored(str(x), 'green'))
                elif self.state[i].position_knowledge[x] == CharKnowledge.NOT_HERE:
                    char_values.append(colored(str(x), 'yellow'))
            str_items[k].append(f'{IDX2CHAR[i]}: {",".join(char_values)}')
        return '\n'.join(['\n'.join(str_items[k]) for k in ['known', 'wrong_pos', 'none'] if len(str_items[k]) > 0])

class Vocabulary:
    def __init__(
        self, 
        all_vocab: List[str], 
        wordle_state: Optional[WordleState], 
        cache: Optional[Cache]=None, 
        fill_cache: bool=True, 
        rng: Optional[random.Random]=None, 
    ):
        # assert all([len(w) == N_CHARS for w in filtered_vocab])
        self.fill_cache = fill_cache
        self.cache = cache
        if self.cache is None:
            self.cache = Cache()
        self.all_vocab = all_vocab
        self.all_vocab_set = set(self.all_vocab)
        if wordle_state is not None:
            if wordle_state in self.cache:
                self.filtered_vocab = self.cache[wordle_state]
            else:
                self.filtered_vocab = list(filter(lambda x: wordle_state.word_in_state(x), self.all_vocab))
                if self.fill_cache:
                    self.cache[wordle_state] = self.filtered_vocab
        else:
            self.filtered_vocab = list(self.all_vocab)
        if rng is None:
            rng = random.Random()
        self.rng = rng
    
    @classmethod
    def from_file(cls, vocab_file: str, fill_cache: bool=True, rng: Optional[random.Random]=None):
        vocab = []
        for item in open(vocab_file, 'r'):
            item = item.strip()
            if len(item) == N_CHARS:
                vocab.append(item)
        return cls(vocab, None, None, fill_cache, rng)
    
    def filtered_vocab_size(self):
        return len(self.filtered_vocab)
    
    def all_vocab_size(self):
        return len(self.all_vocab)
    
    def get_random_word_filtered(self):
        return self.rng.choice(self.filtered_vocab)
    
    def get_random_word_all(self):
        return self.rng.choice(self.all_vocab)
    
    def update_vocab(self, wordle_state: WordleState):
        return Vocabulary(self.all_vocab, wordle_state, cache=self.cache, fill_cache=self.fill_cache, rng=self.rng)
    
    def __contains__(self, item: str) -> bool:
        return item in self.all_vocab_set

    def __str__(self) -> str:
        return '\n'.join(self.filtered_vocab)

class WordleGame:
    def __init__(
        self, 
        state: WordleState, 
        vocab: Vocabulary, 
        action_history: List[str], 
        require_words_in_vocab: bool=True, 
    ):
        self.state = state
        self.vocab = vocab
        self.action_history = action_history
        self.require_words_in_vocab = require_words_in_vocab
    
    @classmethod
    def initialize(cls, vocab: Vocabulary, require_words_in_vocab: bool=True):
        init_wordle = WordleState.initial_state()
        return cls(init_wordle, vocab.update_vocab(init_wordle), action_history=[], require_words_in_vocab=require_words_in_vocab)
    
    def next(self, action: str):
        if (len(action) != N_CHARS) or (not all([c in CHAR2IDX for c in action])) or (self.require_words_in_vocab and action not in self.vocab):
            new_mdp = WordleGame(self.state, self.vocab, action_history=self.action_history+[action])
            return new_mdp, new_mdp.reward(), new_mdp.is_terminal()
        if self.is_terminal():
            return None
        word = self.vocab.get_random_word_filtered()
        new_state = self.state.transition_state(action, word)
        new_mdp = WordleGame(new_state, self.vocab.update_vocab(new_state), action_history=self.action_history+[action])
        return new_mdp, new_mdp.reward(), new_mdp.is_terminal()
    
    def all_next(self, action: str):
        if (len(action) != N_CHARS) or (not all([c in CHAR2IDX for c in action])) or (self.require_words_in_vocab and action not in self.vocab):
            return [(WordleGame(self.state, self.vocab, action_history=self.action_history+[action]), 1,)]
        if self.is_terminal():
            return []
        new_states = defaultdict(list)
        for word in self.vocab.filtered_vocab:
            new_states[self.state.transition_state(action, word)].append(word)
        return [(WordleGame(new_state, self.vocab.update_vocab(new_state), action_history=self.action_history+[words[0]]), len(words)) for new_state, words in new_states.items()]
    
    def __str__(self):
        all_action_strs = []
        for action in self.action_history:
            if (len(action) != N_CHARS) or (not all([c in CHAR2IDX for c in action])) or (self.require_words_in_vocab and action not in self.vocab):
                all_action_strs.append(action)
                continue
            action_str = ''
            for i, c in enumerate(action):
                if self.state.state[CHAR2IDX[c]].position_knowledge[i] == CharKnowledge.HERE:
                    action_str += colored(c, 'green')
                elif all([k == CharKnowledge.NOT_HERE for k in self.state.state[CHAR2IDX[c]].position_knowledge]):
                    action_str += c
                elif self.state.state[CHAR2IDX[c]].position_knowledge[i] == CharKnowledge.NOT_HERE:
                    action_str += colored(c, 'yellow')
            all_action_strs.append(action_str)
        return '\n'.join(all_action_strs)
    
    @classmethod
    def from_str(cls, game_str: str, vocab: Optional[Vocabulary], require_words_in_vocab: bool=True):
        action_transitions = game_str.split('\n')
        if len(game_str) == 0:
            action_transitions = []
        assert len(action_transitions) <= N_TRIES * 2, 'Too many transitions'
        actions = action_transitions[::2]
        transitions = action_transitions[1::2]
        assert len(actions) == len(transitions), 'number of actions and transitions do not match'
        wordle_state = WordleState.initial_state()
        filtered_actions = set()
        for i in range(len(actions)):
            action, transition = actions[i], transitions[i]
            if (len(action) != N_CHARS) or (not all([c in CHAR2IDX for c in action])) or (require_words_in_vocab and action not in vocab):
                continue
            wordle_state = wordle_state.transition_from_str(action, transition)
            filtered_actions.add(action)
        if vocab is None:
            vocab = Vocabulary(list(filtered_actions), wordle_state)
        vocab = vocab.update_vocab(wordle_state)
        return cls(state=wordle_state, vocab=vocab, action_history=list(actions), require_words_in_vocab=require_words_in_vocab)

    def transition_sequence(self):
        transition_strs = []
        for action in self.action_history:
            if (len(action) != N_CHARS) or (not all([c in CHAR2IDX for c in action])) or (self.require_words_in_vocab and action not in self.vocab):
                transition_strs.append('')
                continue
            state_str = ''
            for i, c in enumerate(action):
                if self.state.state[CHAR2IDX[c]].position_knowledge[i] == CharKnowledge.HERE:
                    state_str += '<g>'
                elif all([k == CharKnowledge.NOT_HERE for k in self.state.state[CHAR2IDX[c]].position_knowledge]):
                    state_str += '<b>'
                elif self.state.state[CHAR2IDX[c]].position_knowledge[i] == CharKnowledge.NOT_HERE:
                    state_str += '<y>'
            transition_strs.append(state_str)
        return transition_strs
    
    def reward(self):
        if (len(self.action_history) > 0) and ((len(self.action_history[-1]) != N_CHARS) or (self.action_history[-1] not in self.vocab)):
            return -1.0
        return int(self.vocab.filtered_vocab_size() == 1 and self.vocab.filtered_vocab[0] in self.action_history) - 1
    
    def is_terminal(self):
        return len(self.action_history) == N_TRIES or self.reward() == 0
