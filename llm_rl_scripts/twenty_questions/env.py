from dataclasses import replace
from typing import Dict, List, Optional, Tuple
import random
from LLM_RL.environment import Text, TextEnv, BatchedTextEnv, TextHistory, TextPolicy
from .data import INVALID_QUESTION, INITIAL_STR, WordVariants, create_trajectory_from_history, rtg_to_token_str, token_str_to_rtg
from .oracle import TwentyQuestionsOracle


class TwentyQuestionsPolicyEnvironment(TextEnv):
    def __init__(
        self, 
        oracle: TwentyQuestionsOracle,
        word_list: List[WordVariants],  
        max_conversation_length: int=20,
    ):
        self.oracle = oracle
        self.word_list = word_list
        self.max_conversation_length = max_conversation_length

        self.random = random.Random(None)

        self.curr_word: Optional[WordVariants] = None

    def step(self, text_history: TextHistory) -> Tuple[TextHistory, float, bool]:
        assert text_history[-1].is_action
        assert self.curr_word is not None, "call env.reset() first."

        question = text_history[-1].text.strip()
        answer = self.oracle.generate_answer(self.curr_word, question)
        # print(f"step: question={question}, answer={answer}")
        answer_text = Text(answer + "\n", is_action=False)

        trajectory = create_trajectory_from_history(self.curr_word, text_history + [answer_text], self.max_conversation_length)

        return trajectory.text_history, trajectory.reward[-2], trajectory.done
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> TextHistory:
        if seed is not None:
            self.random = random.Random(seed)

        deterministic = options.get("deterministic", False)
        if deterministic:
            assert seed is not None, "In deterministic mode, the seed specifies which word to use."
            word_ind = seed % len(self.word_list)
            self.curr_word = self.word_list[word_ind]
        else:
            self.curr_word = self.random.choice(self.word_list)

        # print(f"reset: word={self.curr_word}")
        return [Text(INITIAL_STR, is_action=False)]

    def copy(self):
        return TwentyQuestionsPolicyEnvironment(
            oracle=self.oracle,
            word_list=self.word_list,
            max_conversation_length=self.max_conversation_length,
        )


class TwentyQuestionsDTPolicyEnvironment(TwentyQuestionsPolicyEnvironment):
    
    def step(self, text_history: TextHistory) -> Tuple[TextHistory, float, bool]:
        next_text_history, reward, done = super().step(text_history)

        next_answer = next_text_history[-1].text
        last_answer = next_text_history[-3].text
        
        last_rtg = token_str_to_rtg(last_answer, max_conversation_len=self.max_conversation_length)
        next_rtg = last_rtg - reward
        
        new_next_answer = rtg_to_token_str(next_rtg) + next_answer
        new_next_text_history = next_text_history[:-1] + [Text(new_next_answer, is_action=False)]

        return new_next_text_history, reward, done

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> TextHistory:
        _ = super().reset(seed=seed, options=options)
        target_rtg = options["target_rtg"]
        target_rtg_str = rtg_to_token_str(target_rtg, max_conversation_len=self.max_conversation_length)
        return [Text(target_rtg_str + INITIAL_STR, is_action=False)]

