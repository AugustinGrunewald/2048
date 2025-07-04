import pygame as pyg
from pygame.locals import K_RIGHT, K_LEFT, K_UP, K_DOWN

import numpy as np
from numpy.typing import NDArray
import random as rd

# Key dictionary used in the update of the playground
keys_dict = {
    K_RIGHT : [range(2,-1,-1),0,1],
    K_LEFT : [range(1,4,1),0,-1],
    K_UP : [range(1,4,1),-1,0],
    K_DOWN : [range(2,-1,-1),1,0]
}

class Playground():
    def __init__(self) -> None:
        self.__matrices = np.zeros(shape=(4,4), dtype=np.int64)
        self.__score = 0
        self.__number_movements = 0
        self.__memory = []

        # Initializing the merged matrices used in the update method
        self.__movement_matrices = np.zeros((4,4), dtype=bool) # A matrices that keep in memory if a number is already the output of a merge with another one 

        # Initializing with 2 randoms tiles 
        self.__adding_tile()
        self.__adding_tile()

    def __adding_tile(self) -> None:
        # Finding empty spaces
        empty_coord = [[int(x),int(y)] for x,y in zip(np.where(self.__matrices == 0)[0], np.where(self.__matrices == 0)[1])]

        if not empty_coord:
            pass
        else:
            # Choosing randomly coord and if we add a 2 (70% of chance) or a 4 (30% of chance)
            rand_coord = rd.choice(empty_coord)
            value = rd.binomialvariate(1,0.7) * 2 + 2 

            self.__matrices[rand_coord[0], rand_coord[1]] = value

    def update(self, key_pressed : int) -> None:
        # Reset the merged matrices
        self.__movement_matrices = np.zeros((4,4), dtype=bool) # False = movement possible, no merged already occurred
        self.__number_movements += 1

        for _ in range(4): # At maximum 4 movements are possible for one key pressed
            for i in range(4):
                for j in keys_dict[key_pressed][0]: # Going throw each number reversed, and ignoring the first one that won't move
                    # Inverse if needed i,j for the following, using a dictionary
                    var_order = {
                        K_RIGHT : [i,j],
                        K_LEFT : [i,j],
                        K_UP : [j,i],
                        K_DOWN : [j,i]
                    }

                    x : int = var_order[key_pressed][0] 
                    y : int = var_order[key_pressed][1]

                    x_bis : int = keys_dict[key_pressed][1]
                    y_bis : int = keys_dict[key_pressed][2]

                    # Moving the blocks
                    if self.__possible_movement([x,y], [x+x_bis, y+y_bis]):
                        # If destination is 0
                        if self.__matrices[x+x_bis, y+y_bis] == 0:
                            self.__matrices[x+x_bis, y+y_bis] = self.__matrices[x,y]
                            self.__matrices[x,y] = 0

                        # Else need for merging the two numbers
                        else:
                            self.__score += 2 * self.__matrices[x,y]
                            self.__matrices[x+x_bis, y+y_bis] = 2 * self.__matrices[x,y]
                            self.__matrices[x,y] = 0
                            self.__movement_matrices[x+x_bis, y+y_bis] = True

        # Adding a random tile 
        self.__adding_tile()

    def __possible_movement(self, start_coord : list, dest_coord : list) -> bool:
        # No need to move the 0, so return False if that's the case
        start_value = self.__matrices[start_coord[0], start_coord[1]]
        dest_value = self.__matrices[dest_coord[0], dest_coord[1]]

        if start_value != 0:
            # If the destination value is 0 or, the same value as the starting value and that hasn't be merged yet
            if dest_value == 0 or (start_value == dest_value and self.__movement_matrices[start_coord[0], start_coord[1]] == False and self.__movement_matrices[dest_coord[0], dest_coord[1]] == False): 
                return True
        return False
    
    def game_lost(self) -> bool:
        # Only checking if there's no zeros any more and in that case if no similar numbers are next to each other
        if not np.any(self.__matrices == 0):
            # If no 0 values anymore, second check
            if self.__non_similar_values():
                return True
        
        return False
    
    def frozen_playground(self, limit : int = 10) -> bool:
        if len(self.__memory) == limit:
            # Updating the memory
            self.__memory[1:] = self.__memory[:-1]
            self.__memory.append(self.__matrices)
            return all(np.array_equal(tab, self.__memory[0]) for tab in self.__memory[1:])

        else:
            self.__memory.append(self.__matrices)
            return False
    
    def __non_similar_values(self) -> bool: # If no similar values return False
        similar_values_column = np.array([[x == y for x,y in zip(self.__matrices[:3,i], self.__matrices[1:,i])] for i in range(4)], dtype=bool)
        similar_values_row = np.array([[x == y for x,y in zip(self.__matrices[i,:3], self.__matrices[i,1:])] for i in range(4)], dtype=bool)

        return bool(np.all(similar_values_row == False, axis=None)) and bool(np.all(similar_values_column == False, axis=None))
    
    @property
    def get_matrices(self) -> NDArray[np.int64]:
        return self.__matrices
    
    @property
    def score(self) -> int:
        return self.__score
    
    @property
    def number_movements(self) -> int:
        return self.__number_movements
    