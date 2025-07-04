from data import graphics

import pygame as pyg

class Tile(pyg.sprite.Sprite):
    def __init__(self, value : int, coord : list, outline_tile_size : int, inner_tile_size : int, origin_coord : list[int]) -> None:
        super().__init__()
        self.__value = value
        self.__coord = coord
        self.__outline_tile_size = outline_tile_size
        self.__inner_tile_size = inner_tile_size
        self.__origin_coord = origin_coord

        # Building the inner and outer rectangles
        top_left_outer = (self.__coord[1] * self.__outline_tile_size + self.__origin_coord[0], self.__coord[0] * self.__outline_tile_size + self.__origin_coord[1])
        delta = (self.__outline_tile_size - self.__inner_tile_size)//2
        top_left_inner = (top_left_outer[0] + delta, top_left_outer[1] + delta)

        self.outer_square = pyg.Rect(top_left_outer, (self.__outline_tile_size, self.__outline_tile_size))
        self.inner_square = pyg.Rect(top_left_inner, (self.__inner_tile_size, self.__inner_tile_size))

    def draw(self, surface : pyg.Surface) -> None:
        # Drawing the rectangles
        pyg.draw.rect(surface, graphics.colors_rectangle["background_grid"], self.outer_square, border_radius=4)
        pyg.draw.rect(surface, graphics.colors_rectangle[self.__value], self.inner_square, border_radius=4)

        if self.__value != 0:
            # Building the text of the number, putting it on the center and plotting it
            text_surface = graphics.fonts["font_40"].render(str(self.__value), True, graphics.number_colors[self.__value])
            center = (self.outer_square.topleft[0] + self.__outline_tile_size//2, self.outer_square.topleft[1] + self.__outline_tile_size//2)
            text_centered = text_surface.get_rect(center=center)
            surface.blit(text_surface, text_centered)