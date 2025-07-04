from modules import tile
from data import graphics

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
    def __init__(self, complete_window_width : int, complete_window_height : int, screen_width_val : int, screen_height_val : int, outline_tile_size : int, inner_tile_size : int) -> None:
        self.__matrices = np.zeros(shape=(4,4), dtype=np.int64)
        self.score = 0

        # Initializing the merged matrices used in the update method
        self.__movement_matrices = np.zeros((4,4), dtype=bool) # A matrices that keep in memory if a number is already the output of a merge with another one 

        # Constant attributes 
        self.__COMPLETE_WINDOW_WIDTH = complete_window_width
        self.__COMPLETE_WINDOW_HEIGHT = complete_window_height
        self.__SCREEN_WIDTH = screen_width_val
        self.__SCREEN_HEIGHT = screen_height_val
        self.__OUTLINE_TILE_SIZE = outline_tile_size
        self.__INNER_TILE_SIZE = inner_tile_size
        self.__DELTA = (self.__OUTLINE_TILE_SIZE - self.__INNER_TILE_SIZE)//2

        temp = (self.__COMPLETE_WINDOW_WIDTH - self.__SCREEN_WIDTH - 2*self.__DELTA)//2
        self.__TOP_LEFT_BACKGROUND_COORD = (temp, self.__COMPLETE_WINDOW_HEIGHT - self.__SCREEN_HEIGHT - self.__DELTA - temp)
        self.__ORIGIN_GAME_COORD = [self.__TOP_LEFT_BACKGROUND_COORD[0] + self.__DELTA, self.__TOP_LEFT_BACKGROUND_COORD[1] + self.__DELTA]

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

    def draw(self, surface : pyg.Surface) -> None:
        # Drawing the background
        surface.fill(graphics.colors_rectangle["background"])
        playground_background = pyg.Rect(self.__TOP_LEFT_BACKGROUND_COORD, (self.__SCREEN_WIDTH + 2 * self.__DELTA, self.__SCREEN_HEIGHT + 2 * self.__DELTA))
        pyg.draw.rect(surface, graphics.colors_rectangle["background_grid"], playground_background, border_radius=4)

        # Drawing the top of the screen 
        text_surface = graphics.fonts["title"].render("2048", True, graphics.number_colors[2])
        text_rect = text_surface.get_rect(center=(self.__COMPLETE_WINDOW_WIDTH//2 - 120, self.__TOP_LEFT_BACKGROUND_COORD[1]//2))
        surface.blit(text_surface, text_rect)

        # Drawing the score counter 
        score_rect_back = pyg.Rect((self.__COMPLETE_WINDOW_WIDTH//2 + 50, self.__TOP_LEFT_BACKGROUND_COORD[1]//2 - 35), (100, 70))
        pyg.draw.rect(surface, graphics.colors_rectangle["background_grid"], score_rect_back, border_radius=4)

        text_score_name_surface = graphics.fonts["font_30"].render("SCORE", True, (255,255,255))
        text_score_name_rect = text_score_name_surface.get_rect(center=(self.__COMPLETE_WINDOW_WIDTH//2 + 50 + 50, self.__TOP_LEFT_BACKGROUND_COORD[1]//2 - 15))
        text_score_value_surface = graphics.fonts["font_40"].render(f"{self.score}", True, (255,255,255))
        text_score_value_rect = text_score_value_surface.get_rect(center=(self.__COMPLETE_WINDOW_WIDTH//2 + 50 + 50, self.__TOP_LEFT_BACKGROUND_COORD[1]//2 + 15))

        surface.blit(text_score_name_surface, text_score_name_rect)
        surface.blit(text_score_value_surface, text_score_value_rect)

        # Drawing the grid
        for i in range(4):
            for j in range(4):
                current_tile = tile.Tile(self.__matrices[i,j], [i,j], self.__OUTLINE_TILE_SIZE, self.__INNER_TILE_SIZE, self.__ORIGIN_GAME_COORD)
                current_tile.draw(surface)

    def update(self, key_pressed : int, display_surface : pyg.Surface, timer : pyg.time.Clock, fps : int) -> None:
        # Reset the merged matrices
        self.__movement_matrices = np.zeros((4,4), dtype=bool) # False = movement possible, no merged already occurred

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
                            self.score += 2 * self.__matrices[x,y]
                            self.__matrices[x+x_bis, y+y_bis] = 2 * self.__matrices[x,y]
                            self.__matrices[x,y] = 0
                            self.__movement_matrices[x+x_bis, y+y_bis] = True

            # Updating the screen for each move
            self.draw(display_surface)
            pyg.display.update()
            timer.tick(fps)

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
    
    def __non_similar_values(self) -> bool: # If no similar values return False
        similar_values_column = np.array([[x == y for x,y in zip(self.__matrices[:3,i], self.__matrices[1:,i])] for i in range(4)], dtype=bool)
        similar_values_row = np.array([[x == y for x,y in zip(self.__matrices[i,:3], self.__matrices[i,1:])] for i in range(4)], dtype=bool)

        return bool(np.all(similar_values_row == False, axis=None)) and bool(np.all(similar_values_column == False, axis=None))
    
    @property
    def get_matrices(self) -> NDArray[np.int64]:
        return self.__matrices