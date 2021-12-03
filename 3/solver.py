from typing import Generator
from copy import deepcopy
from random import randint, shuffle


class Connections:
    def __init__(self, size: int) -> None:
        self.size = size
        self.__connections: list[list[list[list[int]]]] = []

        for cell_idx in range(self.size ** 2):
            cell_states: list[list[list[int]]] = []

            for state in range(self.size):
                state_connections: list[list[int]] = []

                for other_cell in range(self.size ** 2):
                    state_connections.append([])

                cell_states.append(state_connections)

            self.__connections.append(cell_states)

    def add_connection(self, cell1: int, state1: int, cell2: int, state2: int) -> None:
        self.__connections[cell1][state1][cell2].append(state2)
        self.__connections[cell2][state2][cell1].append(state1)

    def remove_connection(self, cell1: int, state1: int, cell2: int, state2: int) -> None:
        self.__connections[cell1][state1][cell2].remove(state2)
        self.__connections[cell2][state2][cell1].remove(state1)

    def connection_exists(self, cell1: int, state1: int, cell2: int, state2: int) -> bool:
        return state2 in self.__connections[cell1][state1][cell2]

    def is_state_possible(self, cell: int, state: int) -> bool:
        return all(self.__connections[cell][state][:cell] + self.__connections[cell][state][cell+1:])

    def remove_state(self, cell: int, state: int) -> list[tuple[int, int]]:
        states_to_check: list[tuple[int, int]] = []

        for other_cell in range(self.size ** 2):
            for other_state in range(self.size):
                if self.connection_exists(cell, state, other_cell, other_state):
                    states_to_check.append((other_cell, other_state))
                    self.remove_connection(cell, state, other_cell, other_state)

        return states_to_check

    def get_connections(self, cell: int, state: int,
                        neighbour_cells: list[int] = None, neighbour_states: list[int] = None) -> list[tuple[int, int]]:
        connections: list[tuple[int, int]] = []

        if neighbour_cells is None:
            neighbour_cells = range(self.size ** 2)

        if neighbour_states is None:
            neighbour_states = range(self.size)

        for other_cell in neighbour_cells:
            for other_state in neighbour_states:
                if self.connection_exists(cell, state, other_cell, other_state):
                    connections.append((other_cell, other_state))

        return connections

    def cell_possible_states(self, cell: int) -> list[int]:
        possible_states: list[int] = []

        for state in range(self.size):
            if self.is_state_possible(cell, state):
                possible_states.append(state)

        return possible_states


class SudokuSolver:
    def __init__(self, init_state: list[list[int]]) -> None:
        self.size: int = len(init_state)
        self.init_state = init_state
        self.__connections = self.__init_cells()

    def __init_cells(self) -> Connections:
        connections = Connections(self.size)
        flatten = [n for row in self.init_state for n in row]

        for i in range(self.size ** 2):
            if flatten[i] != 0:
                potential_states1 = [flatten[i] - 1]
            else:
                potential_states1 = list(range(self.size))

            row1, col1 = self.__unravel_index(i, self.size)
            for j in range(i + 1, self.size ** 2):
                if flatten[j] != 0:
                    potential_states2 = [flatten[j] - 1]
                else:
                    potential_states2 = list(range(self.size))

                row2, col2 = self.__unravel_index(j, self.size)

                for s1 in potential_states1:
                    for s2 in potential_states2:
                        if s1 == s2 and (row1 == row2 or
                                         col1 == col2 or
                                         self.__locate_subcell(i) == self.__locate_subcell(j)):
                            pass
                        else:
                            connections.add_connection(i, s1, j, s2)

        return connections

    def __locate_subcell(self, idx: int) -> int:
        row, column = self.__unravel_index(idx, self.size)

        subcell_row = row // int(self.size ** .5)
        subcell_column = column // int(self.size ** .5)

        return self.__ravel_index(subcell_row, subcell_column, int(self.size ** .5))

    def __reshape_to_2d(self, flatten: list[int]) -> list[list[int]]:
        return [[flatten[self.size * row + col] for col in range(self.size)] for row in range(self.size)]

    def __ravel_index(self, row: int, column: int, size: int) -> int:
        return row * size + column

    def __unravel_index(self, idx: int, size: int) -> tuple[int, int]:
        row = idx // size
        column = idx % size

        return row, column

    def __traverse_solution_tree(self, cell: int = 0, picked_values: list[int] = None) -> Generator[list[int], None, None]:
        """
        Return generator of found solutions.

        :param cell: number of current cell (initially equal to 0,
                     which means, that we always start from first
                     cell)
        :param picked_values: values of each cell, that have been picked
                              up on current solution
        :return: generator of solutions.
        """
        if picked_values is None:
            picked_values = []

        if cell == self.size ** 2:
            yield []
        else:
            for state in self.__connections.cell_possible_states(cell):
                state_consistent = True
                for other_cell in range(cell):
                    if not self.__connections.connection_exists(cell, state, other_cell, picked_values[other_cell]):
                        state_consistent = False
                        break

                if state_consistent:
                    for path in self.__traverse_solution_tree(cell+1, picked_values + [state]):
                        yield [state] + path

    def __generate_solutions(self) -> Generator[list[list[int]], None, None]:
        found_solutions = self.__traverse_solution_tree()
        for solution in found_solutions:
            solution = [val + 1 for val in solution]
            yield self.__reshape_to_2d(solution)

    def __remove_inconsistent_connections(self) -> None:
        queue = []

        for cell in range(self.size ** 2):
            for state in range(self.size):
                if not self.__connections.is_state_possible(cell, state):
                    queue.append((cell, state))

        while queue:
            cell, state = queue.pop(0)
            changed_states = self.__connections.remove_state(cell, state)

            if not self.__connections.cell_possible_states(cell):
                raise ValueError("Sudoku could not be solved.")

            for changed_state in changed_states:
                if not self.__connections.is_state_possible(*changed_state) and changed_state not in queue:
                    queue.append(changed_state)

    def solve(self) -> list[list[int]]:
        self.__remove_inconsistent_connections()

        old_connections = deepcopy(self.__connections)
        cells = list(range(self.size ** 2))
        shuffle(cells)
        for cell in cells:
            poss_states = self.__connections.cell_possible_states(cell)
            if len(poss_states) > 1:
                picked_state = randint(0, len(poss_states))

                for state in poss_states[:picked_state] + poss_states[picked_state+1:]:
                    self.__connections.remove_state(cell, state)

                try:
                    return self.solve()
                except Exception:
                    self.__connections = deepcopy(old_connections)
                    continue

        if not all([len(self.__connections.cell_possible_states(c)) == 1 for c in range(self.size ** 2)]):
            raise ValueError("Sudoku could not be solved.")
        else:
            flatten = []
            for cell in range(self.size ** 2):
                flatten.append(self.__connections.cell_possible_states(cell)[0] + 1)

            return self.__reshape_to_2d(flatten)

    def find_all_solutions(self) -> Generator[list[list[int]], None, None]:
        self.__remove_inconsistent_connections()

        return self.__generate_solutions()
