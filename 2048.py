from data import graphics, bots_name
from modules import playground

import pygame as pyg
from pygame.locals import K_RIGHT, K_LEFT, K_UP, K_DOWN, K_ESCAPE, K_q, K_r, K_y, K_n, QUIT

import sys

# Always initializing the Pygame process, and the clock
pyg.init()

fps = 45
frame_per_sec = pyg.time.Clock()


# Screen expansion factor, not used yet
alpha = 1

# Screen basic information
outline_tile_size = int(50 * 2 * alpha)
inner_tile_size = int(46 * 2 * alpha)

screen_width = 4 * outline_tile_size
screen_height = 4 * outline_tile_size

complete_window_width = int(screen_width * 1.2)
complete_window_height = int(screen_height * 1.5)

display_surface = pyg.display.set_mode((complete_window_width, complete_window_height)) # Adding more space on the window for some extra features

pyg.display.set_caption("2048")


# Initializing the bot
playing_with_bot = False


# Initializing the base objects
playground_obj = playground.Playground(complete_window_width,complete_window_height, screen_width, screen_height, outline_tile_size, inner_tile_size)

if not playing_with_bot:
    # Main game loop for humans
    while True:
        # Initializing a boolean value that store if the user ask for a reset
        reset_asking = False

        # Checking if we're quitting the game, reset the game or moving 
        for event in pyg.event.get():
            if event.type == QUIT or (event.type == pyg.KEYDOWN and event.key == K_q) or (event.type == pyg.KEYDOWN and event.key == K_ESCAPE):
                pyg.quit()
                sys.exit()

            if event.type == pyg.KEYDOWN and event.key == K_r:
                reset_asking = True

            # Updating playground if arrow has been pressed
            if event.type == pyg.KEYDOWN and (event.key == K_LEFT or event.key == K_RIGHT or event.key == K_UP or event.key == K_DOWN):
                playground_obj.update(event.key, display_surface, frame_per_sec, fps)

        # Refreshing the screen
        playground_obj.draw(display_surface)

        if playground_obj.game_lost():
            # Plotting a window with the score and creating a transparent surface
            transparent_surface = pyg.Surface((complete_window_width, complete_window_height), pyg.SRCALPHA)
            transparent_surface.fill((216, 194, 124, 128))
            display_surface.blit(transparent_surface, (0,0))  

            text_surface = graphics.fonts["title"].render("Game over !", True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=(complete_window_width//2, complete_window_height//2 - 60))
            display_surface.blit(text_surface, text_rect) 

            your_score_surface = graphics.fonts["font_40"].render(f"Your final score : {playground_obj.score}", True, (255, 255, 255))      
            your_score_rectangle = your_score_surface.get_rect(center=(complete_window_width//2, complete_window_height//2 - 20))
            display_surface.blit(your_score_surface, your_score_rectangle) 

            new_game_surface = graphics.fonts["font_30"].render("Press r to play again", True, (255, 255, 255))      
            new_game_rectangle = new_game_surface.get_rect(center=(complete_window_width//2, complete_window_height//2 + 20))
            display_surface.blit(new_game_surface, new_game_rectangle)

            user_react = False

            while not user_react:
                for event in pyg.event.get():
                    if event.type == pyg.KEYDOWN:
                        if event.key == K_r:
                            playground_obj = playground.Playground(complete_window_width,complete_window_height, screen_width, screen_height, outline_tile_size, inner_tile_size)
                            user_react = True

                        elif event.key == K_q or event.key == K_ESCAPE:
                            # Case quitting
                            pyg.quit()
                            sys.exit()

                pyg.display.update()
                frame_per_sec.tick(fps)

        # Checking for reset
        if reset_asking:
            # Plotting a window asking if we really want to reset and creating a transparent surface
            transparent_surface = pyg.Surface((complete_window_width, complete_window_height), pyg.SRCALPHA)
            transparent_surface.fill((216, 194, 124, 128))
            display_surface.blit(transparent_surface, (0,0))

            text_surface = graphics.fonts["font_35"].render("Do you want to reset the game ?", True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=(complete_window_width//2, complete_window_height//2 - 60))
            display_surface.blit(text_surface, text_rect)  

            text_yes = graphics.fonts["font_30"].render("YES (press y)", True, (255, 255, 255))
            text_yes_rect = text_yes.get_rect(center=(complete_window_width//2 - 80, complete_window_height//2))
            display_surface.blit(text_yes, text_yes_rect)

            text_no = graphics.fonts["font_30"].render("NO (press n)", True, (255, 255, 255))
            text_no_rect = text_no.get_rect(center=(complete_window_width//2 + 80, complete_window_height//2))
            display_surface.blit(text_no, text_no_rect)

            user_react = False

            while not user_react:
                for event in pyg.event.get():
                    if event.type == pyg.KEYDOWN:
                        if event.key == K_y:
                            # Case yes
                            playground_obj = playground.Playground(complete_window_width,complete_window_height, screen_width, screen_height, outline_tile_size, inner_tile_size)
                            user_react = True

                        elif event.key == K_n:
                            # Case no
                            user_react = True
                        
                        elif event.key == K_q or event.key == K_ESCAPE:
                            # Case quitting
                            pyg.quit()
                            sys.exit()

                pyg.display.update()
                frame_per_sec.tick(fps)

        # Displaying and making sure to update it each FPS
        pyg.display.update()
        frame_per_sec.tick(fps)

# We're playing with the bot
else:
    # Initializing time values to have bot movement each certain amount of time
    bot = bots_name.bots["evolutionary_MLP"](200)

    while True:
        # Checking if we're quitting the game
        for event in pyg.event.get():
            if event.type == QUIT or (event.type == pyg.KEYDOWN and event.key == K_q) or (event.type == pyg.KEYDOWN and event.key == K_ESCAPE):
                pyg.quit()
                sys.exit()

        # The bot plays each half second
        current_time = pyg.time.get_ticks()
        if (current_time - bot.last_move_time) > bot.get_bot_delay: 
            bot.last_move_time = current_time

            # Updating the game
            playground_obj.update(bot.play_move(playground_obj.get_matrices), display_surface, frame_per_sec, fps)     

            # Refreshing the screen
            playground_obj.draw(display_surface)

        # Freeze the screen if we've lost
        if playground_obj.game_lost():
            user_react = False
            print("GAME LOST\n")

            while not user_react:
                for event in pyg.event.get():
                    if event.type == pyg.KEYDOWN:
                        if event.key == K_q or event.key == K_ESCAPE:
                            # Case quitting
                            pyg.quit()
                            sys.exit()

                pyg.display.update()
                frame_per_sec.tick(fps)            

        # Displaying and making sure to update it each FPS
        pyg.display.update()
        frame_per_sec.tick(fps)