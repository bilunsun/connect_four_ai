import copy
import numpy as np
import random
import time
from typing import List

from connect_four import ConnectFour


class Node:
    def __init__(self, parent_node: "Node", move: int, state: ConnectFour) -> None:
        self.parent_node = parent_node
        self.child_nodes = []

        self.state = copy.deepcopy(state)            
        self.move = move
        self.untried_moves = self.state.legal_moves

        self.visit_count = 0
        self.total_action_value = 0
        self.mean_action_value = 0
        self.prior_probability = 0

        if self.parent_node is not None:
            self.prior_probability = self.parent_node.prior_probability
        
        if self.move is not None:
            self.state.make_move(move)
    
    @property
    def upper_confidence(self) -> float:
        return self.prior_probability * np.sqrt(self.parent_node.visit_count) / (1 + self.visit_count)
    
    @property
    def ucb1(self) -> float:
        if self.visit_count == 0:
            return np.inf

        return self.mean_action_value + 2 * np.sqrt(np.log(self.parent_node.visit_count) / self.visit_count)

    def select_child_node(self) -> "Node":
        selected_child_node = sorted(self.child_nodes, 
            key=lambda child_node: child_node.ucb1)[-1]  # Biggest score
        return selected_child_node
    
    def add_child(self, move: int) -> "Node":
        child_node = Node(move=move, parent_node=self, state=self.state)
        self.child_nodes.append(child_node)

        return child_node

    def update(self, action_value: float) -> None:
        self.visit_count += 1
        self.total_action_value += action_value
        self.mean_action_value = self.total_action_value / self.visit_count


class MCTS:
    """
    Monte Carlo Tree Search algorithm implementation
    Based on https://www.youtube.com/watch?v=UXW2yZndl7U
    """
    def __init__(self, root_state: ConnectFour, player: int, itermax: int = 100, timeout_s: float = 1.0):
        self.root_node = Node(parent_node=None, move=None, state=root_state)
        self.player = player
        self.itermax = itermax
        self.timeout_s = timeout_s
    
    def select(self) -> Node:
        """
        Get a leaf node by selecting child nodes that maximizes UCB1
        """
        current_node = self.root_node
        
        while len(current_node.child_nodes) != 0:
            current_node = current_node.select_child_node()
        
        return current_node
    
    def expand(self, current_node: Node) -> Node:
        """
        Create a child node for each available action/move
        """
        current_node.visit_count += 1

        if current_node.visit_count == 1:  # If this is the first time the node is visited, expand
            for move in current_node.untried_moves:  # For each possible move, create a child node
                current_node.add_child(move=move)
            
            current_node.untried_moves = []  # Clear the untried moves
            
            return current_node.child_nodes[0]  # Set the current node to the first child node
        else:  # Otherwise, proceed to roll-out
            return current_node
    
    def rollout(self, current_node: Node) -> None:
        """
        Perform rollout by playing random moves until the game ends
        """
        while not current_node.state.game_over:
            random_move = random.choice(current_node.state.legal_moves)
            current_node.state.make_move(random_move)
    
    def backpropagate(self, current_node: Node) -> None:
        """
        Assign a score to the result, then propagate it to all the ancestor nodes
        """
        # First get a score
        if self.player == ConnectFour.WHITE:
            if current_node.state.winner == "white":
                final_value = 1
            elif current_node.state.winner == "black":
                final_value = -1
            else:
                final_value = 0
        else:
            if current_node.state.winner == "black":
                final_value = 1
            elif current_node.state.winner == "white":
                final_value = -1
            else:
                final_value = 0

        # Then, propagate it to all the ancestors
        while current_node is not None:
            current_node.update(action_value=final_value)
            current_node = current_node.parent_node

    def get_best_move(self) -> int:
        iteration_index = 0
        start_time_s = time.time()

        while iteration_index < self.itermax and time.time() - start_time_s < self.timeout_s:
            # Select
            leaf_node = self.select()
            
            # Expand
            expanded_child_node = self.expand(leaf_node)
            
            # Rollout
            self.rollout(expanded_child_node)
            
            # Backpropagate
            self.backpropagate(expanded_child_node)
            
            iteration_index += 1
        
        # When the max iteration count has been reached, return the best move
        best_move = sorted(self.root_node.child_nodes, key=lambda node: node.visit_count)[-1].move
        return best_move


def main():
    results = {
        "white": 0,
        "black": 0,
        "drawn": 0
    }

    for i in range(10):
        sample_state = ConnectFour()

        while not sample_state.game_over:
            if sample_state.turn == ConnectFour.WHITE:
                move = random.choice(sample_state.legal_moves)
                # mcts = MCTS(root_state=sample_state, player=ConnectFour.WHITE, itermax=100)
                # move = mcts.get_best_move()
            else:
                # move = random.choice(sample_state.legal_moves)
                mcts = MCTS(root_state=sample_state, player=ConnectFour.BLACK, itermax=100)
                move = mcts.get_best_move()

            sample_state.make_move(move)
            # sample_state.print_board()

        results[sample_state.winner] += 1
        
    print(results)


if __name__ == "__main__":
    main()