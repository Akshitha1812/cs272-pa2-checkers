# CS272-26SP-PA2 Submission: Multi-Agent Reinforcement Learning

## 1. GitHub Repository Link
[Insert Link to your Public GitHub Repository Here]

## 2. Agent Logic and Function Approximation Design
**Agent Logic & Function Approximation (118 words):**
The Actor-Critic agent uses a shared Multi-Layer Perceptron (MLP) with two hidden layers (128 units each) to extract board features. `torch` (2.8.0) is used for the function approximation. The shared trunk feeds into two independent heads: 
1. **Actor Head:** Outputs a 1296-dimensional tensor representing the logit probabilities for each possible move. Invalid moves are masked by replacing their logits with a large negative number before applying Softmax, generating a valid Categorical distribution for action sampling.
2. **Critic Head:** Outputs a single scalar evaluating the state's value. 
During self-play, agents optimize the network after each episode by maximizing the log probability of actions scaled by the advantage (Return - Critic Value), while the Critic is optimized using the Smooth L1 Loss against the empirical return.

## 3. Training & Final Cumulative Reward
**Final Cumulative Reward of Sample Run:**
Sample Run Cumulative Reward -> Player 1 (x): 124, Player 2 (o): -137

## 4. Sample Run
```text
=== Sample Run with Trained Agent ===

=== Sample Run with Trained Agent ===

player_0's turn:
. x . x . x 
x . x . x . 
. . . . . . 
. . . . . . 
. o . o . o 
o . o . o . 


player_1's turn:
. x . x . x 
x . . . x . 
. . . x . . 
. . . . . . 
. o . o . o 
o . o . o . 


player_0's turn:
. x . x . x 
x . . . x . 
. . . x . . 
. . . . o . 
. o . o . . 
o . o . o . 


player_1's turn:
. x . x . x 
x . . . x . 
. . . . . . 
. . . . . . 
. o . o . x 
o . o . o . 


... [190+ intermediate turns omitted for brevity] ...

player_0's turn:
. . . . . . 
. . O . . . 
. X . . . . 
. . x . . . 
. . . X . X 
X . . . . . 


Final Board State:
. . . X . . 
. . . . . . 
. . . . . . 
. . x . . . 
. . . X . X 
X . . . . . 

Run completed!
```
