import pygame as pyg

# Colors
colors_rectangle = {
    "background" : pyg.Color(250, 248, 240),
    "background_grid" : pyg.Color(186, 175, 163),
    0 : pyg.Color(204, 194, 183), # Corresponds to the empty tile
    2 : pyg.Color(237, 229, 220),
    4 : pyg.Color(235, 225, 204),
    8 : pyg.Color(233, 181, 130),
    16 : pyg.Color(232, 155, 109),
    32 : pyg.Color(231, 131, 103),
    64 : pyg.Color(217, 128, 94),
    128 : pyg.Color(233, 215, 151),
    256 : pyg.Color(217, 190, 111),
    512 : pyg.Color(232, 209, 127),
    1024 : pyg.Color(215, 183, 89),
    2048 : pyg.Color(217, 181, 86),
}

number_colors = {
    2 : pyg.Color(118, 111, 102),
    4 : pyg.Color(118, 111, 102),
    8 : pyg.Color(248, 245, 242),
    16 : pyg.Color(248, 245, 242),
    32 : pyg.Color(248, 245, 242),
    64 : pyg.Color(248, 245, 242),
    128 : pyg.Color(248, 245, 242),
    256 : pyg.Color(248, 245, 242),
    512 : pyg.Color(248, 245, 242),
    1024 : pyg.Color(248, 245, 242),
    2048 : pyg.Color(248, 245, 242),
}  

pyg.font.init()

# Fonts 
fonts = {
    # Creating a text police for the numbers
    "font_40" : pyg.font.Font(None, 40),
    "font_35" : pyg.font.Font(None, 35),
    "font_30" : pyg.font.Font(None, 30),
    "title": pyg.font.Font(None, 90)
}