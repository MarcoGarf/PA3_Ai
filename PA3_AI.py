import networkx as nx
import matplotlib.pyplot as plt


import random

class MDP:
    def __init__(self, states, graph_structure):
        self.states = states
        self.graph_structure = graph_structure

class MonteCarloAgent:
    def __init__(self, mdp, alpha=0.1):
        self.mdp = mdp
        self.alpha = alpha
        self.state_values = {state: 0 for state in mdp.states}
        self.state_visit_counts = {state: 0 for state in mdp.states}

    def run_episode(self):
        episode_states = []
        episode_actions = []
        episode_rewards = []

        current_state = 'RU8P'
        while current_state != 'CLASS':
            possible_actions = list(self.mdp.graph_structure[current_state].keys())
            action = random.choice(possible_actions)

            next_states = self.mdp.graph_structure[current_state][action]['next_states']
            rewards = self.mdp.graph_structure[current_state][action]['rewards']

            # Choose a next state based on probabilities
            next_state, probability = random.choices(next_states, weights=[p for s, p in next_states])[0]

            # Record the experience
            episode_states.append(current_state)
            episode_actions.append(action)
            episode_rewards.append(rewards[next_states.index((next_state, probability))])

            # Move to the next state
            current_state = next_state

        return episode_states, episode_actions, episode_rewards

    def monte_carlo(self, num_episodes):
        for episode in range(num_episodes):
            episode_states, episode_actions, episode_rewards = self.run_episode()

            returns = 0
            for t in range(len(episode_states) - 1, -1, -1):
                returns += episode_rewards[t]
                state = episode_states[t]
                if state not in episode_states[:t]:
                    self.state_visit_counts[state] += 1
                    self.state_values[state] += self.alpha * (returns - self.state_values[state])

            # Print episode information
            print(f"Episode {episode + 1}:")
            print("States:", episode_states)
            print("Actions:", episode_actions)
            print("Rewards:", episode_rewards)
            print("Total Reward:", sum(episode_rewards))
            print()

        # Print final state values and average rewards
        print("Final State Values:")
        for state in self.mdp.states:
            print(f"{state}: {self.state_values[state]}")

        print("\nAverage Rewards:")
        for state in self.mdp.states:
            avg_reward = self.state_values[state] / max(1, self.state_visit_counts[state])
            print(f"{state}: {avg_reward}")

class ValueIterationMDP:
    def __init__(self, states, graph_structure, discount_factor=0.99, tolerance=0.001):
        self.states = states
        self.graph_structure = graph_structure
        self.discount_factor = discount_factor
        self.tolerance = tolerance
        self.values = {state: 0 for state in states}

    def bellman_equation(self, state, actions_for_state):
        expected_values = [
            sum(
                probability * (reward + self.discount_factor * self.values[next_state])
                for (next_state, probability), reward in zip(
                    self.graph_structure[state][action]['next_states'],
                    self.graph_structure[state][action]['rewards']
                )
            )
            for action in actions_for_state
        ]
        return max(expected_values)

    def get_optimal_action(self, state, actions_for_state):
        action_values = {
            action: self.bellman_equation(state, actions_for_state)
            for action in actions_for_state
        }
        optimal_action = max(action_values, key=action_values.get)
        return optimal_action

    def value_iteration(self):
        iteration = 0
        max_change = float('inf')

        while max_change > self.tolerance:
            max_change = 0

            for state in self.states:
                actions_for_state = list(self.graph_structure[state].keys())
                previous_value = self.values[state]
                self.values[state] = self.bellman_equation(state, actions_for_state)
                max_change = max(max_change, abs(self.values[state] - previous_value))

            print(f"\nIteration {iteration + 1}:")
            for state in self.states:
                print(f"{state}: {self.values[state]}")

            iteration += 1

        print(f"\nNumber of iterations: {iteration}")
        print("Final values:")
        for state in self.states:
            print(f"{state}: {self.values[state]}")

        optimal_policy = {
            state: self.get_optimal_action(state, list(self.graph_structure[state].keys()))
            for state in self.states
        }
        print("\nOptimal Policy:")
        for state, action in optimal_policy.items():
            print(f"{state}: {action}")



# Define your MDP parameters
states = ['RU8P', 'TU10P', 'RU10P', 'RD10P', 'RU8A', 'RD8A', 'TU10A', 'RU10A', 'RD10A', 'TD10A', 'CLASS']
# Define the graph structure with rewards, transition probabilities, and actions
graph_structure = {
    'RU8P': {
        'P': {'next_states': [('TU10P', 1.0)], 'rewards': [+2]},
        'R': {'next_states': [('RU10P', 1.0)], 'rewards': [0]},
        'S': {'next_states': [('RD10P', 1.0)], 'rewards': [-1]}
    },

    'TU10P': {
        'P': {'next_states': [('RU10A', 1.0)], 'rewards': [+2]},
        'R': {'next_states': [('RU8A', 1.0)], 'rewards': [0]}
    },
    'RU10P': {
        'P': {'next_states': [('RU8A', 0.5), ('RU10A', 0.5)], 'rewards': [+2, +2]},
        'R': {'next_states': [('RU8A', 1.0)], 'rewards': [0]},
        'S': {'next_states': [('RD8A', 1.0)], 'rewards': [-1]}
    },
    'RD10P': {
        'P': {'next_states': [('RD8A', 0.5), ('RD10A', 0.5)], 'rewards': [+2, +2]},
        'R': {'next_states': [('RD8A', 1.0)], 'rewards': [0]}
        
        
    },

    'RU8A': {
        'P': {'next_states': [('TU10A', 1.0)], 'rewards': [+2]},
        'R': {'next_states': [('RU10A', 1.0)], 'rewards': [0]},
        'S': {'next_states': [('RD10A', 1.0)], 'rewards': [-1]}
    },

    'RD8A': {
        'P': {'next_states': [('TD10A', 1.0)], 'rewards': [+2]},
        'R': {'next_states': [('RD10A', 1.0)], 'rewards': [0]},
        
    },

    'TU10A': {
        'P': {'next_states': [('CLASS', 1.0)], 'rewards': [-1]},
        'R': {'next_states': [('CLASS', 1.0)], 'rewards': [-1]},
        'S': {'next_states': [('CLASS', 1.0)], 'rewards': [-1]}
    },
    'RU10A': {
        'P': {'next_states': [('CLASS', 1.0)], 'rewards': [0]},
        'R': {'next_states': [('CLASS', 1.0)], 'rewards': [0]},
        'S': {'next_states': [('CLASS', 1.0)], 'rewards': [0]}
    },
    'RD10A': {
        'P': {'next_states': [('CLASS', 1.0)], 'rewards': [+4]},
        'R': {'next_states': [('CLASS', 1.0)], 'rewards': [+4]},
        'S': {'next_states': [('CLASS', 1.0)], 'rewards': [+4]}
    },
    'TD10A': {
        'P': {'next_states': [('CLASS', 1.0)], 'rewards': [+3]},
        'R': {'next_states': [('CLASS', 1.0)], 'rewards': [+3]},
        'S': {'next_states': [('CLASS', 1.0)], 'rewards': [+3]}
    }

    
}

# Create a directed graph from the defined structure
G = nx.DiGraph()

# Create MDP instance
mdp = MDP(states, graph_structure)

# Create Monte Carlo agent
agent = MonteCarloAgent(mdp)

print("MONTE CARLO SIMULATION")
# Run Monte Carlo simulation
agent.monte_carlo(num_episodes=50)

print("VALUE ITERATION")
actions = ['P', 'R', 'S']
# Create MDP object
mdp = ValueIterationMDP(states, actions, graph_structure)

# Run Value Iteration
mdp.value_iteration()


for state, actions in graph_structure.items():
    G.add_node(state)
    for action, action_data in actions.items():
        for i, (next_state, probability) in enumerate(action_data['next_states']):
            reward = action_data['rewards'][i]
            G.add_edge(state, next_state, label=f"{action}\n{reward}\n{probability}")

# Plot the graph
"""
pos = nx.spring_layout(G)
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw(G, pos, with_labels=True, node_size=2000, node_color='skyblue', font_size=10)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title("MDP Graph with Rewards, Probabilities, and Actions")
plt.show()
"""
