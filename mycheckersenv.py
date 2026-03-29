import functools
import numpy as np
from gymnasium.spaces import Box, Discrete, Dict
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

EMPTY = 0
P1 = 1
P2 = -1
P1_KING = 2
P2_KING = -2

def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    """
    internal_env = raw_env(render_mode=render_mode)
    internal_env = wrappers.TerminateIllegalWrapper(internal_env, illegal_reward=-1)
    internal_env = wrappers.AssertOutOfBoundsWrapper(internal_env)
    internal_env = wrappers.OrderEnforcingWrapper(internal_env)
    return internal_env

class raw_env(AECEnv):
    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "checkers_6x6_v0",
        "is_parallelizable": False,
    }

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.board_size = 6
        self.agents = ["player_0", "player_1"]
        self.possible_agents = self.agents[:]
        self.action_spaces = {i: Discrete(1296) for i in self.agents}
        self.observation_spaces = {
            i: Dict({
                "observation": Box(low=-2, high=2, shape=(36,), dtype=np.int8),
                "action_mask": Box(low=0, high=1, shape=(1296,), dtype=np.int8)
            }) for i in self.agents
        }

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def render(self):
        if self.render_mode is None:
            return
        
        board_str = ""
        for r in range(self.board_size):
            row_str = ""
            for c in range(self.board_size):
                val = self.board[r * self.board_size + c]
                if val == P1: row_str += "x "
                elif val == P1_KING: row_str += "X "
                elif val == P2: row_str += "o "
                elif val == P2_KING: row_str += "O "
                else: row_str += ". "
            board_str += row_str + "\n"
        
        if self.render_mode == "human":
            print(board_str)
        return board_str

    def close(self):
        pass

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.rewards = {i: 0 for i in self.agents}
        self._cumulative_rewards = {i: 0 for i in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}
        self.num_moves = 0
        self.last_capture_move = 0
        
        # 6x6 Checkers Setup
        # P1 (x) is at the top (rows 0, 1)
        # P2 (o) is at the bottom (rows 4, 5)
        self.board = np.zeros(36, dtype=np.int8)
        for r in range(2):
            for c in range(6):
                if (r + c) % 2 == 1:
                    self.board[r * 6 + c] = P1
        for r in range(4, 6):
            for c in range(6):
                if (r + c) % 2 == 1:
                    self.board[r * 6 + c] = P2

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.mandatory_jumper = None # If multi-jump is required, stores the square

    def observe(self, agent):
        board = self.board.copy()
        # Flip perspective for player_1 so model always plays as positive pieces
        if agent == "player_1":
            board = -board[::-1]
            
        action_mask = np.zeros(1296, dtype=np.int8)
        legal_moves = self._get_legal_moves(agent)
        for move in legal_moves:
            if agent == "player_1":
                from_sq = 35 - (move // 36)
                to_sq = 35 - (move % 36)
                flipped_move = from_sq * 36 + to_sq
                action_mask[flipped_move] = 1
            else:
                action_mask[move] = 1
            
        return {"observation": board, "action_mask": action_mask}

    def _get_legal_moves(self, agent):
        moves = []
        jumps = []
        
        is_p1 = (agent == "player_0")
        pieces = [P1, P1_KING] if is_p1 else [P2, P2_KING]
        forward_dir = 1 if is_p1 else -1
        
        for i in range(36):
            if self.mandatory_jumper is not None and self.board[self.mandatory_jumper] in pieces:
                if i != self.mandatory_jumper:
                    continue
                
            if self.board[i] in pieces:
                r, c = i // 6, i % 6
                is_king = abs(self.board[i]) == 2
                
                dirs = [(forward_dir, -1), (forward_dir, 1)]
                if is_king:
                    dirs += [(-forward_dir, -1), (-forward_dir, 1)]
                    
                for dr, dc in dirs:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 6 and 0 <= nc < 6:
                        dest = nr * 6 + nc
                        if self.board[dest] == EMPTY:
                            if self.mandatory_jumper is None:
                                moves.append(i * 36 + dest)
                        elif self.board[dest] not in pieces and self.board[dest] != EMPTY:
                            # Jump
                            jr, jc = nr + dr, nc + dc
                            if 0 <= jr < 6 and 0 <= jc < 6:
                                jdest = jr * 6 + jc
                                if self.board[jdest] == EMPTY:
                                    jumps.append(i * 36 + jdest)
                                    
        # Mandatory jumps!
        if len(jumps) > 0:
            return jumps
            
        # If we are in a mandatory jump chain, but no jumps are available, we cannot do a normal move.
        if self.mandatory_jumper is not None:
            return []
            
        return moves

    def step(self, action):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return

        # CLEAR instantaneous step rewards so they don't compound
        self.rewards = {i: 0 for i in self.agents}
        
        # Small step penalty to encourage faster completion
        self.rewards[self.agent_selection] -= 0.01

        agent = self.agent_selection
        mask = self.observe(agent)["action_mask"]
        if mask[action] == 0:
            # Illegal move handling handled by PettingZoo wrapper, but just in case
            self.terminations = {i: True for i in self.agents}
            self.rewards[agent] = -1
            self._accumulate_rewards()
            return

        # Decode action
        is_p1 = (agent == "player_0")
        
        # If player_1, actions are from flipped perspective, so we must map them back
        # The observe() function flipped the board `board[::-1]`.
        # So cell i becomes 35 - i.
        if not is_p1:
            from_sq = 35 - (action // 36)
            to_sq = 35 - (action % 36)
        else:
            from_sq = action // 36
            to_sq = action % 36

        # Move piece
        piece = self.board[from_sq]
        self.board[from_sq] = EMPTY
        self.board[to_sq] = piece
        
        # Check promotion
        to_row = to_sq // 6
        if is_p1 and to_row == 5 and piece == P1:
            self.board[to_sq] = P1_KING
        elif not is_p1 and to_row == 0 and piece == P2:
            self.board[to_sq] = P2_KING

        # Check if it was a jump
        jumped = False
        if abs(from_sq // 6 - to_sq // 6) == 2:
            jumped = True
            mid_r = (from_sq // 6 + to_sq // 6) // 2
            mid_c = (from_sq % 6 + to_sq % 6) // 2
            self.board[mid_r * 6 + mid_c] = EMPTY
            self.rewards[agent] += 1.0  # Reward for capture
            self.rewards[self.agents[1 - self.agents.index(agent)]] -= 1.0
            self.last_capture_move = self.num_moves

        # Check multi-jump
        self.mandatory_jumper = None
        if jumped:
            # See if this piece can jump again
            self.mandatory_jumper = to_sq
            legal_next = self._get_legal_moves(agent)
            if len(legal_next) == 0:
                self.mandatory_jumper = None
                self.agent_selection = self._agent_selector.next()
            else:
                # Same player goes again
                pass
        else:
            self.agent_selection = self._agent_selector.next()

        self.num_moves += 1
        # Truncate if too many moves or if no capture for a long time (Draw rule)
        if self.num_moves >= 200 or (self.num_moves - self.last_capture_move) >= 60:
            self.truncations = {i: True for i in self.agents}

        # Check win/loss (no pieces or no legal moves)
        p1_pieces = sum(1 for p in self.board if p in [P1, P1_KING])
        p2_pieces = sum(1 for p in self.board if p in [P2, P2_KING])
        
        p1_moves = self._get_legal_moves("player_0")
        p2_moves = self._get_legal_moves("player_1")
        
        p1_lost = p1_pieces == 0 or len(p1_moves) == 0
        p2_lost = p2_pieces == 0 or len(p2_moves) == 0
        
        if p1_lost or p2_lost:
            self.terminations = {i: True for i in self.agents}
            if p1_lost and not p2_lost:
                self.rewards["player_1"] += 10
                self.rewards["player_0"] -= 10
            elif p2_lost and not p1_lost:
                self.rewards["player_0"] += 10
                self.rewards["player_1"] -= 10
            
        self._accumulate_rewards()
