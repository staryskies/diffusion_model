"""
Memory:
All Python containers have a significant amount of overhead, so the bot packs all the information it needs into a 780-bit integer.
The ghost has 1MB of memory, so it can use common containers.

Execution is split into two phases:
Steps 1..300:
* The ghost runs a DFS to find the slot machine with the highest alpha value.
* The bot runs a DFS to the first slot machine it finds and collects it.
Steps 301..2000:
* The ghost goes back and forth between the two slot machines. It visits the lower-numbered node when step % 2 == 0.
* The bot runs a DFS through the graph to collect all the coins from phase 1. Then it runs another DFS to find the pair that the ghost is using.
"""

import logging
import random
import sys
from typing import Any
from dataclasses import dataclass

def SubmissionBot(step: int, total_steps: int, pos: int, last_pos: int, neighbors: list[int], has_slot: bool, slot_coins: int, data: int | None) -> tuple[int, int]:
    SIZE = 100
    UNINITIALIZED = 124
    CLEARING_GRAPH = 125
    SEARCHING_SLOT = 126
    FOUND_SLOT = 127

    def pack(state: int, node: int | None, visited: set[int], stack: list[int], backtrack: bool = False) -> int:
        """
        Pack the following into a single integer:
        - Node (7 bits): for Phase 2, the node that was chosen by the ghost, or a state value
        - Backtrack flag (1 bit): whether the last move was a backtrack
        - Visited mask (100 bits): whether each node has been visited in the current DFS
        - Stack (remaining bits): the stack of nodes that were previously visited
        """
        node_int = node if node is not None else state
        backtrack_int = 1 if backtrack else 0
        visited_int = sum(1 << node for node in visited)
        stack_int = sum((idx + 1) * (SIZE + 1) ** i for i, idx in enumerate(stack))
        return node_int | (backtrack_int << 7) | (visited_int << 8) | (stack_int << 108)
    
    def unpack(data: int | None) -> tuple[int, int | None, set[int], list[int], bool]:
        if data is None:
            return UNINITIALIZED, None, {pos}, [], False

        node_int = data & 127
        node = None if node_int >= SIZE else node_int
        data >>= 7

        state = node_int if node_int >= SIZE else FOUND_SLOT
        
        backtrack = (data & 1) == 1
        data >>= 1

        visited_int = data & ((1 << SIZE) - 1)
        visited = set()
        for i in range(SIZE):
            if (visited_int >> i) & 1:
                visited.add(i)
        data >>= SIZE

        stack_int = data
        stack = []
        while stack_int > 0:
            idx = stack_int % (SIZE + 1)
            stack.append(idx - 1)
            stack_int //= (SIZE + 1)
        return state, node, visited, stack, backtrack
    
    def do_dfs() -> tuple[int, int]:
        for v in neighbors:
            if v not in visited:
                visited.add(v)
                return v, pack(state, node, visited, stack)
        if not stack:
            return -1, pack(state, node, visited, stack)
        prev_node = stack.pop()
        return prev_node, pack(state, node, visited, stack, backtrack=True)
    
    state, node, visited, stack, is_backtracking = unpack(data)
        
    logging.info(f"Step {step}/{total_steps}: state={state}, node={node}, visited={visited}, stack={stack}, backtrack={is_backtracking}")

    if step > 1 and pos != last_pos and not is_backtracking:
        stack.append(last_pos)
    if (data_size := sys.getsizeof(pack(state, node, visited, stack))) > 128:
        # Handle case where stack has grown too large. This should be fairly rare.
        logging.warning("Step %d: Stack overflow with %d bytes! Reached %d edges, visited %d nodes", step, data_size, len(stack), len(visited))
        stack.pop()
        return last_pos, pack(state, node, visited, stack)

    if step <= 250:  # Phase 1
        if has_slot:  # Continue spinning while waiting for Phase 1 to end
            return -1, pack(state, node, visited, stack)
        return do_dfs()

    else: # Phase 2
        if state == UNINITIALIZED:
            state = CLEARING_GRAPH
            visited.clear()
            visited.add(pos)
            stack.clear()
            # Fall through

        if state == CLEARING_GRAPH:
            if len(visited) == SIZE:
                state = SEARCHING_SLOT
                visited.clear()
                visited.add(pos)
                stack.clear()
                slot_coins = 0  # Simulate the bot picking these up
                # Fall through
            else:
                return do_dfs()

        if state == SEARCHING_SLOT:
            if slot_coins > 0:
                node = pos
                state = FOUND_SLOT
                visited.clear()
                visited.add(pos)
                stack.clear()
                # Fall through
            else:
                return do_dfs()

        if state == FOUND_SLOT:
            # TODO: for maps with low alpha, try moving between slot machines
            return -1, pack(state, node, visited, stack)

        logging.error("Step %d: Unexpected state %d", step, state)
        return -1, pack(state, node, visited, stack)


def SubmissionGhost(step: int, total_steps: int, pos: int, last_pos: int, neighbors: list[int], has_slot: bool, slot_coins: int, data: dict[str, Any] | None) -> tuple[int, dict[str, Any]]:
    COIN_CAP = 50
    SIZE = 100
    SAMPLES = 4

    def best_node(graph: list[set[int]], slots: dict[int, list[int]]) -> int:
        scores = [(slot_score[::-1], u) for u, slot_score in slots.items()]
        return max(scores)[1]

    def node_to(target: int) -> int:
        distances = {target: 0}
        queue = [target]
        while queue:
            u = queue.pop(0)
            for v in data['graph'][u]:
                if v not in distances:
                    distances[v] = distances[u] + 1
                    queue.append(v)
        return min(neighbors, key=lambda v: distances[v])

    if data is None:
        data = {
            'visited': {pos},
            'stack': [],
            'graph': [set() for _ in range(SIZE)],
            'slots': dict(),
            'best_node': None,
        }
    logging.info(f"Step {step}/{total_steps}: data={data}")

    for v in neighbors:
        data['graph'][pos].add(v)
        data['graph'][v].add(pos)

    if step <= 250:  # Phase 1
        
        if has_slot:
            if pos in data['slots']:
                data['slots'][pos] += [slot_coins]
            else:  # The first time we see the slot machine it should never have coins
                data['slots'][pos] = []

        for v in neighbors:
            if v not in data['visited']:
                data['visited'].add(v)
                data['stack'].append(pos)
                return v, data

        if has_slot and len(data['slots'][pos]) < SAMPLES and max(data['slots'][pos], default=0) < COIN_CAP:
            return -1, data

        if data['stack']:
            idx = data['stack'].pop()
            return idx, data
        else:
            return -1, data
    else:  # Phase 2
        if data['best_node'] is None:
            data['best_node'] = best_node(data['graph'], data['slots'])
        
        target = data['best_node']
        if pos == target:
            return -1, data  # TODO: once the bot arrives, spin a different machine if it's profitable
        return node_to(target), data

@dataclass
class Vertice:
    slotmachine: bool
    coins: int
    neighbors: list[int]    

class Tester:
    def __init__(self):
        self.vertex_dict = {
            0:  Vertice(True, 0, [1, 3, 5, 7]),
            1:  Vertice(True, 0, [0, 2, 4, 6]),
            2:  Vertice(False, 0, [1, 3, 5, 7]),
            3:  Vertice(True, 0, [0, 2, 4, 6]),
            4:  Vertice(True, 0, [1, 3, 5, 7]),
            5:  Vertice(False, 0, [0, 2, 4, 6]),
            6:  Vertice(True, 0, [1, 3, 5, 7]),
            7:  Vertice(False, 0, [0, 2, 4, 6]),
            8:  Vertice(False, 0, [9, 11, 13, 15]),
            9:  Vertice(False, 0, [8, 10, 12, 14]),
            10: Vertice(True, 0, [9, 11, 13, 15]),
            11: Vertice(True, 0, [8, 10, 12, 14]),
            12: Vertice(False, 0, [9, 11, 13, 15]),
            13: Vertice(True, 0, [8, 10, 12, 14]),
            14: Vertice(False, 0, [9, 11, 13, 15]),
            15: Vertice(False, 0, [8, 10, 12, 14]),
            16: Vertice(False, 0, [17, 18, 19, 0]),
            17: Vertice(True, 0, [16, 18, 19, 1]),
            18: Vertice(True, 0, [16, 17, 19, 2]),
            19: Vertice(True, 0, [16, 17, 18, 3])
        }

    
        self.total_coins = 0
        self.cycles = 2000

    def generatecoins(self, vertice: int):
        self.vertex_dict[vertice].coins  += random.randint(0, 50)

    def collectcoins(self, vertice: int):
        self.total_coins += self.vertex_dict[vertice].coins 
        self.vertex_dict[vertice].coins  = 0

    def test(self):

        next_node = 0
        next_node_ghost = 0

        temp_bot = None
        temp_bot_ghost = None

        for i in range(self.cycles):
            before_node = next_node
            before_node_ghost = next_node_ghost



            next_node, temp_bot = SubmissionBot(i, 
                                                self.cycles, 
                                                next_node, 
                                                before_node, 
                                                self.vertex_dict[next_node].neighbors, 
                                                self.vertex_dict[next_node].coins, 
                                                self.vertex_dict[next_node].slotmachine, 
                                                temp_bot)
            next_node_ghost, temp_bot_ghost = SubmissionGhost(i, 
                                                            self.cycles, 
                                                            next_node_ghost, 
                                                            before_node_ghost, 
                                                            self.vertex_dict[next_node_ghost].neighbors, 
                                                            self.vertex_dict[next_node_ghost].coins,  
                                                            self.vertex_dict[next_node_ghost].slotmachine, 
                                                            temp_bot_ghost)


            if next_node == -1:
                next_node = before_node
            if next_node_ghost == -1:
                next_node_ghost = before_node_ghost


            if before_node == before_node_ghost and self.vertex_dict[before_node].slotmachine == True:
                self.generatecoins(before_node)
                self.collectcoins(before_node)
            else:
                if  self.vertex_dict[before_node].slotmachine == True:
                    self.generatecoins(before_node)
                    self.collectcoins(before_node)
                if self.vertex_dict[before_node_ghost].slotmachine == True:
                    self.generatecoins(before_node_ghost)

        print(self.total_coins)
        self.total_coins = 0
            
        

        
test = Tester()
test.test()
test.test()