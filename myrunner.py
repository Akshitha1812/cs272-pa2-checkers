import torch
from mycheckersenv import env as checkers_env
from myagent import ACAgent

def main():
    env = checkers_env()
    agent = ACAgent()
    
    episodes = 200
    print("Starting Self-Play Training...")
    
    for episode in range(episodes):
        env.reset()
        
        states = {"player_0": [], "player_1": []}
        actions = {"player_0": [], "player_1": []}
        log_probs = {"player_0": [], "player_1": []}
        rewards = {"player_0": [], "player_1": []}
        values = {"player_0": [], "player_1": []}
        dones = {"player_0": [], "player_1": []}
        
        for agent_name in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            
            # Store reward from previous action
            if len(states[agent_name]) > 0:
                rewards[agent_name].append(reward)
                dones[agent_name].append(termination or truncation)
            
            if termination or truncation:
                env.step(None)
                continue
                
            obs = observation["observation"]
            mask = observation["action_mask"]
            
            # Forward pass to get value
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            _, value = agent.model(obs_tensor)
            
            action, log_prob = agent.get_action(obs, mask)
            
            states[agent_name].append(obs)
            actions[agent_name].append(action)
            log_probs[agent_name].append(log_prob)
            values[agent_name].append(value)
            
            env.step(action)
            
        # Update agent at end of episode using trajectories
        final_reward = 0
        all_rewards = []
        all_log_probs = []
        all_values = []
        all_next_values = []
        all_dones = []
        
        for a_name in ["player_0", "player_1"]:
            if len(rewards[a_name]) > 0:
                final_reward += sum(rewards[a_name])
                all_rewards.extend(rewards[a_name])
                all_log_probs.extend(log_probs[a_name])
                all_values.extend(values[a_name])
                
                next_vals = [v for v in values[a_name][1:]] + [torch.tensor([[0.0]])]
                all_next_values.extend(next_vals)
                all_dones.extend(dones[a_name])
                
        if len(all_rewards) > 0:
            agent.update(all_rewards, all_log_probs, all_values, all_next_values, all_dones)
                
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1} finished. Last episode cumulative reward approx: {final_reward}")

    # Show a sample run in human mode
    print("\n=== Sample Run with Trained Agent ===")
    env = checkers_env(render_mode="human")
    env.reset()
    sample_run_rewards = {"player_0": 0, "player_1": 0}
    
    for agent_name in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        
        # Only add reward if it's not the initial reward=0 from reset
        if termination or truncation:
            sample_run_rewards[agent_name] += reward
            env.step(None)
            continue
            
        print(f"\n{agent_name}'s turn:")
        env.render()
        
        action, _ = agent.get_action(observation["observation"], observation["action_mask"])
        env.step(action)
        sample_run_rewards[agent_name] += reward
        
    print("\nFinal Board State:")
    env.render()
    print("Run completed!")
    print(f"Sample Run Cumulative Reward -> Player 1 (x): {sample_run_rewards['player_0']}, Player 2 (o): {sample_run_rewards['player_1']}")

if __name__ == "__main__":
    main()
