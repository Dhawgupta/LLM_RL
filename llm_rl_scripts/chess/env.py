import chess
from stockfish import Stockfish
from LLM_RL.utils import convert_path
import os
import numpy as np
from LLM_RL.environment import TextEnv, Text, TextHistory
from typing import Optional, Dict
import chess

CHESS_ENGINE_PATH = os.environ.get("CHESS_ENGINE_PATH", convert_path("stockfish/stockfish-ubuntu-20.04-x86-64-avx2"))

def preprocess_move(move: str):
    return " ".join(move) + "\n"

def postprocess_move(move: str):
    return move.replace(" ", "").strip()

def preprocess_state(state: str):
    return " ".join(state) + "\n"

def preprocess_state_og(state: str):
    return " ".join(state)

def postprocess_state(state: str):
    return state.replace("  ", "__temp__").replace(" ", "").replace("__temp__", " ").strip()

class ChessEnv():
    metadata = {'render.modes': ['human']}

    def __init__(self, side="w", fen=True, from_position=None):
        """Initialize the underlying chess environment.

        Args:
            side (str, optional): String representing which side the agent is playing. Defaults to "w". (for white pieces)
            fen (bool, optional): Whether or not to play with "fen" states or "move history" states. Defaults to True.
            from_position (_type_, optional): Fen string describing the position to start from. Defaults to None.
        """
        if from_position is None:
            self.board = chess.Board() 
            self.starting_position = self.board.fen()
        else:
            self.board = chess.Board(fen=from_position)
            self.starting_position = from_position

        self.fen = fen

        # set the side of the board which this agent is playing
        if side == "w":
            self.side = chess.WHITE
        else:
            self.side = chess.BLACK
        
        self.stockfish_params = {
            "Threads": 1, # More threads will make the engine stronger, but < # of logical processors
            "UCI_Elo": 1200,
        }

        

        self.stockfish = Stockfish(path=CHESS_ENGINE_PATH, parameters=self.stockfish_params)
    
    def reset(self):
        self.board = chess.Board(fen=self.starting_position)

        self.stockfish.set_fen_position(self.starting_position)
        return self.starting_position, {}

    def sample_valid_action(self):
        """Plays a random legal move. For debugging purposes. """
        move = np.random.choice(self.board.legal_moves, 1)[0]
        return self.board.san(move)

    def step(self, action: str, opponent_move: bool = True):
        """
        Reward is 0 for legal moves, -1 for illegal moves and defeat, 
        1 for victory and 0.5 for stalemate.

        action: a move in san format for the chess environment.
        opponent_move: a boolean which controls whether or not to fetch an opponent move
        from the engine 

        returns: state, reward, done, {}
        state: either in FEN notation or the move history depending on the status of 
        the flag self.fen
        reward: 0 for non-terminal states and draws, 1 for victory, -1 for illegal moves and loss
        """
        try:
            move : chess.Move = self.board.push_san(action)
        except Exception as e:
            reward, done = -1, 0
            opponent = None
        else:
            self.stockfish.make_moves_from_current_position([move.uci()])
            if self.board.is_game_over():
                reward = 1 if self.board.is_checkmate() else 0 # it's agent's turn so the agent wins
                done = 1
                opponent = None
            elif opponent_move:
                opponent = self._make_engine_move()
                reward = -1 if self.board.is_checkmate() else 0
                done = self.board.is_game_over()
        finally:
            state = self._get_state()
            return state, reward, done, {"opponent move": opponent}

    def _get_state(self):
        """
        Return either the FEN state or the move history depending on 
        the initial parameters of the environment.
        """
        if self.fen:
            return self.board.fen()
    
    def get_board(self):
        """
        Boiler plate getter function for getting access to board which we are
        playing on.
        """
        return self.board

    def _make_engine_move(self):
        """
        return: a move in san notation representing the move the opponent made.
        """
        move : str = self.stockfish.get_best_move_time(100)
        self.stockfish.make_moves_from_current_position([move])
        ch_move : chess.Move = chess.Move.from_uci(move)
        san_move = self.board.san(ch_move)
        # self._update_move_history(self.board.san(ch_move))
        self.board.push(ch_move)
       
        return san_move
    
    def san_make_move(self, move):
        """
        Get the opponents move and track this in the board state.
        move: a string representing the move either in uci or san notation
        """
        self.board.push_san(move)
        self._update_move_history(move)
        # don't we also want the opponents move to go the model afterward?
        return self._get_state()
    
    def make_move(self, move: chess.Move):
        """
        Get the opponents move, but move is a chess.Move object.
        """
        str_move : str = move.uci()
        self.board.push(move)
        self._update_move_history(str_move)

        done = self.board.is_game_over()

        return self._get_state(), done

    def render(self, mode='human', close=False):
        print(self.board)
        return self.board

class FenChessHistoryEnvSingleTurn(TextEnv):
    def __init__(self, initial_history: TextHistory, max_moves=400, from_position=None):
        super().__init__()
        self.chess_env = ChessEnv(fen=True, from_position=from_position)
        self.from_position = from_position
        self.max_moves = max_moves
        self.from_position = from_position
        self.initial_history = initial_history

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        self.init_state, _ = self.chess_env.reset()
        self.num_moves_made = 0
        return self.initial_history + (Text(preprocess_state(self.init_state), False),)

    def step(self, text_history: TextHistory):
        assert text_history[-1].is_action
        action = text_history[-1].text
        action = postprocess_move(action)
        st, reward, done, opp_mv = self.chess_env.step(action) 
        new_state = Text(preprocess_state(st), False)
        self.num_moves_made += 1
        if self.num_moves_made > self.max_moves:
            done = 1
        return self.initial_history + (new_state,), reward, done
    
    def copy(self):
        return FenChessHistoryEnvSingleTurn(self.initial_history, self.max_moves, self.from_position)

class FenChessHistoryEnv(TextEnv):
    def __init__(self, max_moves=400, from_position=None):
        super().__init__()
        self.chess_env = ChessEnv(fen=True, from_position=from_position)
        self.from_position = from_position
        self.max_moves = max_moves
        self.from_position = from_position
        # self.initial_history = initial_history

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        self.init_state, _ = self.chess_env.reset()
        self.num_moves_made = 0
        return (Text(preprocess_state_og(self.init_state), False),)

    def step(self, text_history: TextHistory):
        assert text_history[-1].is_action
        action = text_history[-1].text
        action = postprocess_move(action)
        st, reward, done, opp_mv = self.chess_env.step(action) 
        new_state = Text(preprocess_state_og(st), False)
        self.num_moves_made += 1
        if self.num_moves_made > self.max_moves:
            done = 1
        return (new_state,), reward, done
    
    def copy(self):
        return FenChessHistoryEnv( self.max_moves, self.from_position)

def large_piece_random_endgame(pieces:str):
    """Provide a string like 'kQK' to represent a black king, white queen, 
    and white king on the board"""
    board = chess.Board()
    while True:
        board.clear()
        possible_squares = np.arange(0, 64)
        for piece in pieces:
            p = chess.Piece.from_symbol(piece)
            square = np.random.choice(possible_squares)
            board.set_piece_at(square, p)
            possible_squares = possible_squares[possible_squares != square]
        fen = board.fen()
        if board.is_valid() and not board.is_check():
            return fen