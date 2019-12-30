import copy
import numpy as np
import random
import time
from typing import List

from connect_four import ConnectFour


class Node:
    def __init__(self, parent_node: "Node", move: int, state: ConnectFour = None) -> None:
        self.parent_node = parent_node
        self.child_nodes = []

        if self.parent_node is None:
            self.state = copy.deepcopy(state)
        else:
            self.state = copy.deepcopy(self.parent_node.state)

        self.move = move

        if self.move is not None:
            assert move in self.state.legal_moves
            self.state.make_move(move)

        self.untried_moves = self.state.legal_moves[::]  # Is this list copy needed?

        self.visit_count = 0
        self.total_action_value = 0
        self.mean_action_value = 0
        self.prior_probability = 0

        if self.parent_node is not None:
            self.prior_probability = self.parent_node.prior_probability

    @property
    def is_expandable(self) -> bool:
        return not self.state.game_over and len(self.untried_moves) > 0

    @property
    def upper_confidence(self) -> float:
        return self.prior_probability * np.sqrt(self.parent_node.visit_count) / (1 + self.visit_count)

    @property
    def ucb1(self) -> float:
        if self.visit_count == 0:
            return np.inf

        return self.mean_action_value + np.sqrt(2) * np.sqrt(np.log(self.parent_node.visit_count) / self.visit_count)

    def select_child_node(self) -> "Node":
        selected_child_node = sorted(self.child_nodes,
            key=lambda child_node: child_node.ucb1)[-1]  # Biggest score
        return selected_child_node

    def add_child_node(self) -> "Node":
        move = self.untried_moves.pop()
        child_node = Node(move=move, parent_node=self)
        self.child_nodes.append(child_node)

        return child_node

    def add_children_nodes(self) -> "Node":
        for move in self.untried_moves:
            child_node = Node(move=move, parent_node=self)
            self.child_nodes.append(child_node)

        self.untried_moves = []

        return self.child_nodes[0]


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
        Stop when one or more of the legal moves in a node does not have a corresponding child node
        In the implementation where expansion creates all possible children, that simply means checking
        that there are no child nodes
        """
        current_node = self.root_node

        while len(current_node.child_nodes) != 0:
            current_node = current_node.select_child_node()

        return current_node

    def expand(self, current_node: Node) -> Node:
        """
        Create a child node if possible
        """
        if current_node.is_expandable:
            current_node = current_node.add_children_nodes()

        return current_node

    def rollout(self, current_node: Node) -> str:
        """
        Perform rollout by playing random moves until the game ends, then return the final value
        """
        # Make a copy of the state, to not modify the current node
        copied_state = copy.deepcopy(current_node.state)

        while not copied_state.game_over:
            random_move = random.choice(copied_state.legal_moves)
            copied_state.make_move(random_move)

        return copied_state.winner

    def backpropagate(self, current_node: Node, winner: str) -> None:
        """
        Propagate the score to all the ancestor nodes of the same turn
        """
        while current_node.parent_node:
            current_node.visit_count += 1

            # Propagate to the ancestor nodes
            # Note that only the nodes with the opposite turn than the current_node's turn are updated
            # Indeed, if one player wins the roll-out, then only the nodes that are *about* to make
            #   the player's move ought to be rewarded
            if winner != "drawn":
                winner_turn = ConnectFour.WHITE if winner == "white" else ConnectFour.BLACK

                if current_node.state.turn != winner_turn:
                    current_node.total_action_value += 1

                current_node.mean_action_value = current_node.total_action_value / current_node.visit_count

            current_node = current_node.parent_node

        # Note that the root node also needs to have its visit count updated
        # Otherwise, the UCB1 score will not work
        self.root_node.visit_count += 1

    def get_best_move(self) -> int:
        iteration_index = 0
        start_time_s = time.time()

        while iteration_index < self.itermax and time.time() - start_time_s < self.timeout_s:
            # Select
            leaf_node = self.select()

            # Expand
            expanded_child_node = self.expand(leaf_node)

            # Rollout
            winner = self.rollout(expanded_child_node)

            # Backpropagate
            self.backpropagate(expanded_child_node, winner)

            iteration_index += 1

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
        assert changed == True


def main():
    results = {
        "white": 0,
        "black": 0,
        "drawn": 0
    }

    for i in range(1):
        print(i)
        sample_state = ConnectFour()

        # white_mcts = MCTS(root_state=sample_state, player=ConnectFour.WHITE, itermax=1000, timeout_s=40000000000)
        # black_mcts = MCTS(root_state=sample_state, player=ConnectFour.BLACK, itermax=1000, timeout_s=40000000000)

        while not sample_state.game_over:
            sample_state.print_board()

            if sample_state.turn == ConnectFour.WHITE:
                # print("Before: ", white_mcts.root_node)
                move = random.choice(sample_state.legal_moves)
                # move = int(input("Your turn: "))
                # print("After : ", white_mcts.root_node)
                # white_mcts = MCTS(root_state=sample_state, player=ConnectFour.WHITE, itermax=10, timeout_s=40000000000)
                # move = white_mcts.get_best_move()

                # Update the MCTS for the opponent
                black_mcts.make_opponent_move(move)
            else:
                black_mcts = MCTS(root_state=sample_state, player=ConnectFour.BLACK, itermax=1000, timeout_s=40000000000)
                # move = random.choice(sample_state.legal_moves)
                # move = int(input("Your Move: "))
                move = black_mcts.get_best_move()

                # white_mcts.make_opponent_move(move)

            sample_state.make_move(move)

        print("FINAL STATE")
        sample_state.print_board()

        results[sample_state.winner] += 1

    print(results)


if __name__ == "__main__":
    main()