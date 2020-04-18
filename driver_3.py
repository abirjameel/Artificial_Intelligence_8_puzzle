import queue as Q
import time
import resource
from heapq import heappush, heappop, heapify
import sys
import math

#### SKELETON CODE ####

## The Class that Represents the Puzzle

class PuzzleState(object):

    """docstring for PuzzleState"""

    def __init__(self, config, n, parent=None, action="Initial", cost=0, key=None):

        if n*n != len(config) or n < 2:

            raise Exception("the length of config is not correct!")

        self.n = n

        self.cost = cost

        self.parent = parent

        self.action = action

        self.dimension = n

        self.config = config

        self.children = []
        
        self.key = key
        

        for i, item in enumerate(self.config):

            if item == 0:

                self.blank_row = int(i / self.n)

                self.blank_col = int(i % self.n)

                break

    def display(self):

        for i in range(self.n):

            line = []

            offset = i * self.n

            for j in range(self.n):

                line.append(self.config[offset + j])

            print (line)

    def move_left(self):

        if self.blank_col == 0:

            return None

        else:

            blank_index = self.blank_row * self.n + self.blank_col

            target = blank_index - 1

            new_config = list(self.config)

            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]
            

            return PuzzleState(tuple(new_config), self.n, parent=self, action="Left", cost=self.cost + 1)

    def move_right(self):

        if self.blank_col == self.n - 1:

            return None

        else:

            blank_index = self.blank_row * self.n + self.blank_col

            target = blank_index + 1

            new_config = list(self.config)

            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]
            
            

            return PuzzleState(tuple(new_config), self.n, parent=self, action="Right", cost=self.cost + 1)

    def move_up(self):

        if self.blank_row == 0:

            return None

        else:

            blank_index = self.blank_row * self.n + self.blank_col

            target = blank_index - self.n

            new_config = list(self.config)

            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]
            

            return PuzzleState(tuple(new_config), self.n, parent=self, action="Up", cost=self.cost + 1)

    def move_down(self):

        if self.blank_row == self.n - 1:

            return None

        else:

            blank_index = self.blank_row * self.n + self.blank_col

            target = blank_index + self.n

            new_config = list(self.config)

            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]

            return PuzzleState(tuple(new_config), self.n, parent=self, action="Down", cost=self.cost + 1)

    def expand(self):

        """expand the node"""

        # add child nodes in order of UDLR

        if len(self.children) == 0:

            up_child = self.move_up()

            if up_child is not None:

                self.children.append(up_child)

            down_child = self.move_down()

            if down_child is not None:

                self.children.append(down_child)

            left_child = self.move_left()

            if left_child is not None:

                self.children.append(left_child)

            right_child = self.move_right()

            if right_child is not None:

                self.children.append(right_child)

        return self.children


class SearchAlgo:
    
    def __init__(self, initial_state):
        self.initial_state = initial_state
        
    
    def test_goal(self, config):
        if config==(0,1,2,3,4,5,6,7,8):
            return True
        else :
            return False
    
    def get_path(self, final_state):
        path = []
        cost = 0
        current_state = final_state
        while current_state.parent:
            path.append(current_state.action)
            cost += 1
            current_state = current_state.parent
            path.reverse()
        return path, len(path), cost
    
    def writeOutput(self, path, nodesExpand, search_depth, max_search_depth, cost, run_time, maxram):
        with open('output.txt', 'w') as output_file:
            output_file.write('path_to_goal: ' + str(path) + '\n')
            output_file.write('cost_of_path: ' + str(cost) + '\n')
            output_file.write('nodes_expanded: ' + str(nodesExpand) + '\n')
            output_file.write('search_depth: ' + str(search_depth) + '\n')
            output_file.write('running_time: ' + str(run_time) + '\n')
            output_file.write('max_ram_usage: ' + str(maxram) + '\n')
            output_file.write('max_search_depth: ' + str(max_search_depth))
    
            
    def calculate_manhattan_dist(self, config):
        """calculate the manhattan distance of a tile"""
        goal_state = [0,1,2,3,4,5,6,7,8]
        score = 0
        for i in config:
            xg, yg = int(goal_state.index(i)/3), int(goal_state.index(i)%3)
            xs, ys = int(config.index(i)/3), int(config.index(i)%3)
            scr = abs(xs-xg) + abs(ys-yg)
            score+=scr
        return score
    
    def heuristic(self, state):
        _, _, ini2curr_cost = self.get_path(state)
        h_n = self.calculate_manhattan_dist(state.config)

        return ini2curr_cost+h_n
    
    
    def bfs_search(self):
        frontier = Q.Queue()
        frontier.put(self.initial_state)
        fronexplored = set()
        # define and initialize variables
        nodesExpand = 0
        start_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        max_memory = 0
        delta_mem = 0
        start_time = time.time()
        max_depth = 0
        
        while frontier:
            board = frontier.get()
            fronexplored.add(board.config)
        
            if self.test_goal(board.config):
                path, depth, cost = self.get_path(board)
                final_time = time.time()
                total_time = final_time - start_time
                self.writeOutput(path=path, nodesExpand=nodesExpand, search_depth=depth, max_search_depth=max_depth, cost=cost, run_time=total_time, maxram=max_memory/1000000)
                return True
            
            for neighbour in list(board.expand()):
            
                if neighbour.config not in fronexplored:
                
                    frontier.put(neighbour)
                    fronexplored.add(neighbour.config)
                    _, dep, _ = self.get_path(neighbour)
                
                    if dep>max_depth:
                        max_depth = dep
            nodesExpand +=1
            delta_mem = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) - start_mem
            if delta_mem > max_memory:
                max_memory = delta_mem
            
        return True
    
    
    def dfs_search(self):
        frontier = [self.initial_state]
        fronexplored = set()
        nodesExpand = 0
        start_time = time.time()
        max_depth = 0
        start_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        
        max_memory = 0
        while frontier:
            board = frontier.pop()
            fronexplored.add(board.config)
            
            if self.test_goal(board.config):
                path, depth, cost = self.get_path(board)
                final_time = time.time()
                total_time = final_time - start_time
                self.writeOutput(path=path, nodesExpand=nodesExpand, search_depth=depth, max_search_depth=max_depth, cost=cost, run_time=total_time, maxram=max_memory/1000000)
                return True
                
            neighbours = list(board.expand())
            for neighbour in neighbours:
                
                if neighbour.config not in fronexplored:
                    
                    frontier.append(neighbour)
                    fronexplored.add(neighbour.config)
                    _, dep, _ = self.get_path(neighbour)
                    
                    if dep>max_depth:
                        max_depth = dep
            nodesExpand +=1
            delta_mem = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) - start_mem
            if delta_mem > max_memory:
                max_memory = delta_mem
        
        return True
    
    
    def A_star_search(self):
        initial_state = self.initial_state
        #calculate Heuristic and set initial node
        
        key = self.heuristic(initial_state)
        
        initial_state.key = key
        fronexplored= set()
        Queue = []
        Queue.append(initial_state)
        fronexplored.add(initial_state.config)
    
        nodesExpand = 0
        max_search_depth = 0
        start_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        max_memory = 0
        delta_mem = 0
        start_time = time.time()
    
        while Queue:
            Queue.sort(key=lambda o: o.key) 
            curr_state = Queue.pop(0)
            if self.test_goal(curr_state.config):
                path, depth, cost = self.get_path(curr_state)
                final_time = time.time() - start_time
                self.writeOutput(path=path, nodesExpand=nodesExpand, search_depth=depth,max_search_depth=max_search_depth, cost=cost, run_time=final_time, maxram=max_memory / 1000000)
                return True
        
            neighbours = curr_state.expand()
        
            for neighbour in neighbours:      
            
                if neighbour.config not in fronexplored:
                    
                    key = self.heuristic(neighbour)
                    neighbour.key = key
                    Queue.append(neighbour)               
                    fronexplored.add(neighbour.config)
                
                    _, dep, _ = self.get_path(neighbour)
                    if dep>max_search_depth:
                        max_search_depth = dep
            nodesExpand +=1
            delta_mem = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) - start_mem
            if delta_mem > max_memory:
                max_memory = delta_mem
        return True
    
    
# Main Function that reads in Input and Runs corresponding Algorithm


def main():

    sm = sys.argv[1].lower()

    begin_state = sys.argv[2].split(",")

    begin_state = tuple(map(int, begin_state))

    size = int(math.sqrt(len(begin_state)))

    hard_state = PuzzleState(begin_state, size)
    print(hard_state.config)
    if sm == "bfs":
        bfs = SearchAlgo(initial_state = hard_state)
        bfs.bfs_search()

    elif sm == "dfs":
        dfs = SearchAlgo(initial_state = hard_state)
        dfs.dfs_search()

    elif sm == "ast":

        ast = SearchAlgo(initial_state = hard_state)
        ast.A_star_search()

    else:

        print("Enter valid command arguments !")

if __name__ == '__main__':

    main()
    
    
