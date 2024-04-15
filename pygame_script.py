import pygame
import sys
import random

# Constants
TILE_SIZE = 60
GRID_SIZE = 8
WINDOW_SIZE = TILE_SIZE * GRID_SIZE
FPS = 30

# Colors
WHITE = (255, 255, 255)
NORMAL_FLOOR_COLOR = (200, 200, 200)  # Light grey for the normal floor

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption('Box Pushing Game')
clock = pygame.time.Clock()

# Load textures
try:
    carpet_texture = pygame.image.load('assets/carpet_texture.jpg').convert()
    carpet_texture = pygame.transform.scale(carpet_texture, (TILE_SIZE, TILE_SIZE))
    box_texture = pygame.image.load('assets/box_texture.jpg').convert()
    box_texture = pygame.transform.scale(box_texture, (TILE_SIZE, TILE_SIZE))
    player_icon = pygame.image.load('assets/player_icon.jpg').convert_alpha()
    player_icon = pygame.transform.scale(player_icon, (TILE_SIZE, TILE_SIZE))
except pygame.error as e:
    print(f"Error loading the textures: {e}")
    sys.exit()

# Define carpet rectangle dimensions and position
carpet_width, carpet_height = random.randint(2, 5), random.randint(2, 5)
carpet_x = random.randint(0, GRID_SIZE - carpet_width)
carpet_y = random.randint(0, GRID_SIZE - carpet_height)

# Player and Box positions
player_pos = [4, 4]
box_pos = [3, 3]

# Game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            dx, dy = 0, 0
            if event.key == pygame.K_LEFT:
                dx = -1
            elif event.key == pygame.K_RIGHT:
                dx = 1
            elif event.key == pygame.K_UP:
                dy = -1
            elif event.key == pygame.K_DOWN:
                dy = 1

            # Calculate new player position
            new_player_x = max(0, min(GRID_SIZE - 1, player_pos[0] + dx))
            new_player_y = max(0, min(GRID_SIZE - 1, player_pos[1] + dy))

            # Check if the new position collides with the box
            if [new_player_x, new_player_y] == box_pos:
                # Calculate new box position
                new_box_x = box_pos[0] + dx
                new_box_y = box_pos[1] + dy
                # Ensure the box stays within bounds
                if 0 <= new_box_x < GRID_SIZE and 0 <= new_box_y < GRID_SIZE:
                    box_pos = [new_box_x, new_box_y]
                    player_pos = [new_player_x, new_player_y]
            else:
                player_pos = [new_player_x, new_player_y]

    # Drawing
    screen.fill(WHITE)
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            if carpet_x <= x < carpet_x + carpet_width and carpet_y <= y < carpet_y + carpet_height:
                screen.blit(carpet_texture, (x * TILE_SIZE, y * TILE_SIZE))
            else:
                pygame.draw.rect(screen, NORMAL_FLOOR_COLOR, (x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE))

    screen.blit(player_icon, (player_pos[0] * TILE_SIZE, player_pos[1] * TILE_SIZE))
    screen.blit(box_texture, (box_pos[0] * TILE_SIZE, box_pos[1] * TILE_SIZE))

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
