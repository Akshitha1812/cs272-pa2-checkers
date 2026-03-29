# 6x6 Checkers Custom PettingZoo Environment

This repository implements a 6x6 Checkers environment using the PettingZoo AEC API and an Actor-Critic agent for self-play.

## Environment Documentation (`mycheckersenv.py`)

### Observation Space
The observation space is a PettingZoo `Dict` consisting of:
- `observation`: A `Box(low=-2, high=2, shape=(36,), dtype=np.int8)`. The 6x6 board is flattened into a 36-element array. `1` represents Player 1's man, `2` for Player 1's King. `-1` represents Player 2's man, `-2` for Player 2's King. `0` represents an empty square.
- `action_mask`: A `Box(low=0, high=1, shape=(1296,), dtype=np.int8)` mapping available legal actions.

### Action Space
The action space is a `Discrete(1296)` space. Actions represent moves from any of the 36 squares to any other of the 36 squares (36 x 36 = 1296). Only valid moves satisfying the game rules (moving diagonally, jumping over opponent pieces) have an `action_mask` value of `1`.

### Rewards
- **+1** for capturing an opponent's piece.
- **-1** for having a piece captured by the opponent. 
- **+10** for winning the game (capturing all opponent pieces or blocking all their legal moves).
- **-10** for losing the game.

### Termination and Truncation
- **Termination:** The game terminates when one player loses all their pieces or has no remaining legal moves available.
- **Truncation:** The environment prevents illegal moves by strictly applying the `action_mask`. If the agent attempts a move outside the mask, standard wrappers will catch it or heavily penalize the agent.


