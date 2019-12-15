import copy
import numpy as np
import random
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
        return self.mean_action_value + 2 * np.sqrt(np.log(self.parent_node.visit_count) / self.visit_count)

    def select_child_node(self) -> "Node":
        # selected_child_node = sorted(self.child_nodes, 
        #     key=lambda child_node: child_node.mean_action_value + child_node.upper_confidence)[-1]  # Biggest score

        selected_child_node = sorted(self.child_nodes, 
            key=lambda child_node: child_node.ucb1)[-1]  # Biggest score
        return selected_child_node
    
    def add_child(self, move: int, state: ConnectFour) -> "Node":
        child_node = Node(move=move, parent_node=self, state=state)

        self.untried_moves.remove(move)
        self.child_nodes.append(child_node)

        return child_node

    def update(self, action_value: float) -> None:
        self.visit_count += 1
        self.total_action_value += action_value
        self.mean_action_value = self.total_action_value / self.visit_count


class MCTS:
    def __init__(self, root_state: ConnectFour, player: int, itermax: int = 100):
        self.root_state = copy.deepcopy(root_state)
        self.root_node = Node(parent_node=None, move=None, state=root_state)
        self.player = player
        self.itermax = itermax

    def get_best_move(self) -> int:
        for iteration_index in range(self.itermax):
            current_node = self.root_node
            current_state = copy.deepcopy(self.root_state)

            # Select
            while current_node.child_nodes and not current_node.untried_moves:  # Repeat until node is fully expanded and non-terminal
                current_node = current_node.select_child_node()  # Select a child based on the PUCT algorithm
                current_state.make_move(current_node.move)  # Make the move

            # Expand
            # if current_node.untried_moves:  # If expansion is possible
            #     current_move = random.choice(current_node.untried_moves)  # Randomly select a move
            #     current_state.make_move(current_move)

            #     # Create a child node, and update the current_node to be the child node
            #     current_node = current_node.add_child(move=current_move, state=current_state)
            
            # Expand
            if current_node.visit_count == 0:  # If the leaf node has never been visited/calculated before
                for move in current_node.untried_moves:  # For each possible move, create a child node
                    current_node.add_child(move=move, state=current_state)
                
                current_node = current_node.child_nodes[0]  # Set the current node to the first child node       

            # Rollout / Simulation
            while not current_state.game_over:
                random_move = random.choice(current_state.legal_moves)
                current_state.make_move(random_move)
            
            # Backpropagate
            if self.player == ConnectFour.WHITE:
                if current_state.winner == "white":
                    final_value = 1
                elif current_state.winner == "black":
                    final_value = -1
                else:
                    final_value = 0
            else:
                if current_state.winner == "black":
                    final_value = 1
                elif current_state.winner == "white":
                    final_value = -1
                else:
                    final_value = 0

            while current_node is not None:
                current_node.update(action_value=final_value)
                current_node = current_node.parent_node
        
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
        print(i)
        sample_state = ConnectFour()

        # white_mcts = MCTS(root_state=sample_state, player=ConnectFour.WHITE)

        while not sample_state.game_over:
            if sample_state.turn == ConnectFour.WHITE:
                mcts = MCTS(root_state=sample_state, player=ConnectFour.WHITE, itermax=100)
                move = mcts.get_best_move()
            else:
                mcts = MCTS(root_state=sample_state, player=ConnectFour.BLACK, itermax=1)
                move = mcts.get_best_move()

            sample_state.make_move(move)
            # sample_state.print_board()

        results[sample_state.winner] += 1
        
    print(results)


if __name__ == "__main__":
    main()