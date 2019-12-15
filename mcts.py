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

        if self.move is not None:
            assert move in self.state.legal_moves
            self.state.make_move(move)

        # self.untried_moves = self.state.legal_moves

        self.visit_count = 0
        self.total_action_value = 0
        self.mean_action_value = 0
        self.prior_probability = 0

        if self.parent_node is not None:
            self.prior_probability = self.parent_node.prior_probability

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
        if current_node.visit_count == 0:
            return current_node
        
        if not current_node.state.legal_moves:
            current_node.state.print_board()
            print("No more legal moves???")
            exit()
        for move in current_node.state.legal_moves:  # For each possible move, create a child node
            current_node.add_child(move=move)
        
        return current_node.child_nodes[0]  # Set the current node to the first child node
    
    def rollout(self, current_node: Node) -> int:
        """
        Perform rollout by playing random moves until the game ends, then return the final value
        """
        # Make a copy of the state, to not modify the current node
        copied_state = copy.deepcopy(current_node.state)

        while not copied_state.game_over:
            random_move = random.choice(copied_state.legal_moves)
            copied_state.make_move(random_move)
        
        # Get the final value
        if copied_state.winner == "drawn":
            return 0

        if self.player == ConnectFour.WHITE:
            if copied_state.winner == "white":
                final_value = 1
            elif copied_state.winner == "black":
                final_value = -1
        else:
            if copied_state.winner == "black":
                final_value = 1
            elif copied_state.winner == "white":
                final_value = -1
        
        return final_value
    
    def backpropagate(self, current_node: Node, value: int) -> None:
        """
        Propagate the score to all the ancestor nodes
        """
        while current_node is not None:
            current_node.update(action_value=value)
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
            final_value = self.rollout(expanded_child_node)
            
            # Backpropagate
            self.backpropagate(expanded_child_node, final_value)
            
            iteration_index += 1
        
        print(f"Iterations: {iteration_index}")
        scores = [child_node.visit_count for child_node in self.root_node.child_nodes]
        print("Scores: ", scores)
        
        # When the max iteration count has been reached, update the tree, and return the best move
        best_node = sorted(self.root_node.child_nodes, key=lambda node: node.visit_count)[-1]

        self.root_node = best_node
        return best_node.move
    
    def make_opponent_move(self, opponent_move) -> None:
        if len(self.root_node.child_nodes) == 0:  # If this is the first move
            self.root_node.state.make_move(opponent_move)
            return

        changed = False
        for child_node in self.root_node.child_nodes:
            if child_node.move == opponent_move:
                changed = True
                self.root_node = child_node
                break


def main():
    results = {
        "white": 0,
        "black": 0,
        "drawn": 0
    }

    for i in range(10):
        print(i)
        sample_state = ConnectFour()

        white_mcts = MCTS(root_state=sample_state, player=ConnectFour.WHITE, itermax=2000, timeout_s=4)
        black_mcts = MCTS(root_state=sample_state, player=ConnectFour.BLACK, itermax=2000, timeout_s=4)

        while not sample_state.game_over:
            sample_state.print_board()

            if sample_state.turn == ConnectFour.WHITE:
                # print("Before: ", white_mcts.root_node)
                # move = random.choice(sample_state.legal_moves)
                # move = int(input("Your turn: "))
                # print("After : ", white_mcts.root_node)
                move = white_mcts.get_best_move()

                # Update the MCTS for the opponent
                black_mcts.make_opponent_move(move)
            else:
                # move = random.choice(sample_state.legal_moves)
                move = int(input("Your Move: "))
                # move = black_mcts.get_best_move()

                white_mcts.make_opponent_move(move)

            sample_state.make_move(move)

        results[sample_state.winner] += 1
        
    print(results)


if __name__ == "__main__":
    main()