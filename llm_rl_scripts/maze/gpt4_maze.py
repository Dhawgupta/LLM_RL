import os
import json
import time
import openai
import jax
import pickle as pkl
from llm_rl_scripts.maze.maze_utils import setup_maze_env
from llm_rl_scripts.wordle.env import WordleEnvironment, ReformatWordleEnvironment
from llm_rl_scripts.wordle.game import Vocabulary
from LLM_RL.environment import TextPolicy, TextHistory, Text, text_env_eval
from llm_rl_scripts.wordle.game import Vocabulary
from JaxSeq.bucket_manager import open_with_bucket as open
from JaxSeq.utils import create_path
from LLM_RL.utils import convert_path
from IPython import embed

openai.api_key = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = "You are an expert maze solver. You only respond in json."

MAIN_PROMPT = """\
Welcome to the maze solver task. Your objective is to reach the goal in as few steps as possible. At each step you will be given information about where the goal is your current position,
and the walls that surround you. 

Here is an example. 
```
environment: The goal is at position 8, 6. Your current position is at position 1, 7. There are walls to your left, above you, below you.\n
action: move up\n

environment: The goal is at position 8, 6. Your current position is at position 1, 2. There are walls above you, below you.\n
action: move left\n
```

Your possible actions are ["move up\n", "move up\n", "move left\n", "move right\n"].

Based on the feedback from the environment, you know that the goal is at position 8, 6, and your current position is at position 1, 2. You also know that there are walls above you and below you, but not to your left or right. Knowing this, you might guess the next action to be:
{"thought": "I know the goal is at position 8, 6 and my current position is at position 1, 2. I also know that there are walls above me and below me, but not to my left or right. A good action to try might therefore be \"move right\n\", since it is in the list of possible actions, and it will get me closer to the goal. Therefore this is a good action to try next.", "action": "move right\n"}


Now let's start a new game. Return your action in a json array with key "thought" followed by key "action", like in the example above. Now, guess the next word given the current game state:

```
{{game_content}}
```
""".strip()

VOCAB_FILE = "llm_rl_scripts/wordle/vocab/wordle_official_400.txt"

class GPT4MazePolicy(TextPolicy):
    
    def __init__(self):
        self.prompt = MAIN_PROMPT

    def act(self, text_history: TextHistory) -> TextHistory:
        game_content = ""
        # for i, item in enumerate(text_history[1:]):
        #     if i % 2 == 0:
        #         game_content += f"action: {item.text}"
        #     else:
        #         game_content += f"environment: {item.text}"
        game_content = f"environment: {text_history[-1].text}"
        game_content = game_content.strip()
        prompt = self.prompt.replace('{{game_content}}', game_content)
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    temperature=0.0,
                    max_tokens=1024,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
            except openai.OpenAIError as e:
                print(e)
                time.sleep(10)
                continue
            break
        response_text = response.choices[0].message.content
        response_json = json.loads(response_text)
        return text_history+(Text(response_json['action'].strip() + "\n", True),)

if __name__ == "__main__":
    N_INTERACTIONS = 1
    OUTPUTS_PATH = "data/outputs/gpt4_maze/"

    def text_history_to_str(text_history: TextHistory) -> str:
        return '\n'.join(map(lambda x: x.text, text_history))

    env = setup_maze_env(maze_name="double_t_maze", describe_function="describe_observation_give_position", reward_function="standard_reward", last_k=1)

    policy = GPT4MazePolicy()

    def print_interaction(interaction):
        print('='*25)
        print(text_history_to_str(interaction[-1].post_transition_history))
        print('='*25)

    interation_raw_results, interaction_summary_results = text_env_eval(
        env=env,
        policy=policy,
        n_rollouts=N_INTERACTIONS,
        interaction_callback=print_interaction,
    )

    print(interaction_summary_results)

    create_path(OUTPUTS_PATH)
    with open(os.path.join(OUTPUTS_PATH, 'interactions.pkl'), 'wb') as f:
        pkl.dump(interation_raw_results, f)
    with open(os.path.join(OUTPUTS_PATH, 'interactions_summary.json'), 'w') as f:
        json.dump(jax.tree_util.tree_map(lambda x: float(x), interaction_summary_results), f)



