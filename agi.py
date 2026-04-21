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
import sys
from typing import Any


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