import re
from typing import Optional
from LLM_RL.environment import Text, TextEnv, TextHistory
from LLM_RL.text_nav.make_game import build_and_compile_game


class TextNavEnv(TextEnv):
    """
    Environment for textual navigation game.
    """

    def __init__(self,
                 display_location: bool = False,
                 display_inventory: bool = False):
        
        self.infos = EnvInfos(description=True,
                              admissible_commands=True,
                              location=display_location,
                              inventory=display_inventory)
        self.reset()

    def _reset(self, seed: Optional[int] = None):
        _, self.game_file = build_and_compile_game(not self.infos.location)
        self.env = textworld.start(game_file, self.infos)

        self.state = env.reset()
        self.state.feedback = re.sub("-=.*=-\n", "", self.state.feedback)

    def _step(self, command: str):
        command = command.strip()
        self.state, _, _ = self.env.step(command)        
        
        if self.infos.inventory:
            self.state["inventory"], _, _, _ = self.env.step("inventory")
            self.state.feedback += "\nInventory:{}\n".format(self.state["inventory"])

        redundant = ["examine", "look", "inventory"]
        self.state["admissible_commands"] = list(
            c for c in self.state["admissible_commands"] if not any(a in c for a in redundant))
        self.state.feedback += "\nAdmissible commands:{}\n".format(
            ",".join(self.state["admissible_commands"]))
        
        self.state.feedback = re.sub("-=.*=-\n", "", self.state.feedback)
    
    def reset(self) -> TextHistory:
        self._reset()
        return tuple(self.state.feedback,)

    def step(self, text_history: TextHistory) -> Tuple[TextHistory, float, bool]:
        assert text_history[-1].is_action

        command = text_history[-1].text
        self._step(command)
        return (
            text_history + (Text(command, true), Text(self.state.feedback, False)),
            self.state["score"],
            self.state["done"] 
        )
