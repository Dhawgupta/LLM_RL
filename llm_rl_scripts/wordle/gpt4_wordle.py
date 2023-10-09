import openai
from llm_rl_scripts.wordle.env import WordleEnvironment, ReformatWordleEnvironment
from llm_rl_scripts.wordle.game import Vocabulary

SYSTEM_PROMPT = "You are an expert wordle player. You only respond in json."

MAIN_PROMPT = """\
Welcome to the game of Wordle. Your objective is to guess a hidden 5 letter word. You have 6 attempts to guess it correctly and you should try to guess it in as few attempts as possible. When guessing the word, you should format your word as a space separated sequence of letters, like "s h i r e" for example. After guessing the word, you will receive feedback from the game environment in the form of a sequence of 5 space separated letters like "b y g g b", where each letter indicates some information about the hidden word. The environment will return one of three letters – "b", "g", or "y" – for each letter in the word you guessed. We describe the meaning of each letter below:

"b": If the environment returns a "b", it means that the letter at that position in your guessed word is not in the hidden word.
"y": If the environment returns a "y", it means that the letter at that position in your guessed word is in the hidden word but is not in the correct position.
"g": If the environment returns a "g", it means that the letter at that position in your guessed word is in the hidden word and is in the correct position.

If you guess an invalid word (e.g. not a 5 letter word), the environment will respond with nothing. You should use this information returned by the environment to update your belief about what the hidden word might be and adjust your next guess accordingly.

Here is an example. If the current status of the game is given as:
```
guess 1: p a n i c
feedback 1: b b y b b
guess 2: f e l o n
feedback 2: g b b y g
```
You might guess the next word to be:
{"guess": "f r o w n"}

Now let's start a new game. Return your word as a space separated sequence of 5 letters in a json array with key "guess", like in the above example. Now, guess the next word given the current game state:

```
{{game_content}}
```
""".strip()

VOCAB_FILE = "llm_rl_scripts/wordle/vocab/wordle_official_400.txt"

class GPT4WordlePolicy():
    pass

if __name__ == "__main__":
    vocab = Vocabulary.from_file(
        vocab_file=VOCAB_FILE, 
        fill_cache=False, 
    )
    env = ReformatWordleEnvironment(WordleEnvironment(vocab))


