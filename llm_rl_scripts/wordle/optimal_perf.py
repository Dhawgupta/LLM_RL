from llm_rl_scripts.wordle.game import Vocabulary
from llm_rl_scripts.wordle.scripted_policies import OptimalPolicy
from LLM_RL.utils import convert_path
from llm_rl_scripts.wordle.env import WordleEnvironment
from LLM_RL.environment import interact_environment

if __name__ == '__main__':
    vocab_path = 'llm_rl_scripts/wordle/vocab/wordle_official_400.txt'
    vocab = Vocabulary.from_file(vocab_file=convert_path(vocab_path), fill_cache=False, rng=None)
    policy = OptimalPolicy(vocab=vocab, progress_bar=True)
    env = WordleEnvironment(vocab, require_words_in_vocab=True)

    transitions = interact_environment(
        env, 
        policy, 
        env_seed=None,
    )[0]
    
    history = transitions[-1].post_transition_history
    rewards = sum([[transition.reward, 0.0] for transition in transitions], [])
    done = transitions[-1].done

    print(rewards)


