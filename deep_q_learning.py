import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import os

# Initialize Pygame
pygame.init()

# Set up display
WIDTH, HEIGHT = 800, 600
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Car Game with Deep Q-learning")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Define car properties
car_width, car_height = 40, 60
car_speed = 10
car_position = [WIDTH // 2, HEIGHT - 100]  # Initial car position
DEFAULT_CAR_POSITION = car_position.copy()
OBSTACLES = 10

# Load images or default shapes
car_image_path = "car.png"
goal_image_path = "goal.png"
obstacle_image_path = "obstacle.png"

def load_image(path, default_color, size):
    if os.path.exists(path):
        return pygame.transform.scale(pygame.image.load(path), size)
    else:
        surface = pygame.Surface(size)
        surface.fill(default_color)
        return surface

car_image = load_image(car_image_path, BLACK, (car_width, car_height))
goal_image = load_image(goal_image_path, GREEN, (car_width, car_height))
obstacle_image = load_image(obstacle_image_path, RED, (50, 50))

# Goal and obstacles
goal_position = [WIDTH // 2, 50]
obstacles = [[random.randint(0, WIDTH - 50), random.randint(100, HEIGHT - 200)] for _ in range(OBSTACLES)]

# Define deep Q-network model
class DQN(nn.Module): # inheretance from nn.Module
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(2, 128) # 2 dimensional vector to 128 dimensional vector (hidden layer)
        self.fc2 = nn.Linear(128, 128) # hidden layer
        self.fc3 = nn.Linear(128, 4)  # 4 actions's Q values

    def forward(self, x):
        x = torch.relu(self.fc1(x)) #ReLU -> Rectified Linear Unit sets - values to 0, and keeps positive ones
        x = torch.relu(self.fc2(x))
        return self.fc3(x) # Q values

# Initialize networks and optimizer
policy_net = DQN() # Main net, constatly updates, DQN -> gives estimation for the best action
target_net = DQN()  # copy of policy_net, stabilizing the studying process via staying unedited for a period of time
target_net.load_state_dict(policy_net.state_dict()) # copy of policy_net (weights and biases - torzítások)
target_net.eval() # értékesítési mód, won't change it's value in studying process nor change its value.

optimizer = optim.Adam(policy_net.parameters(), lr=0.001) # adaptively modifies the learning rate
# arg1: policy_net.parameters() -> update all its values and weights
# arg2: learning rate, how big steps do we update the net's parameters in each iteration
memory = deque(maxlen=10000)  # Experience replay memory
# from collections
# stores the so called "experience samples"
# arg1: max 10.000 samples
#   -> if the memory is full, remove the oldest one

# DQN parameters
actions = ["LEFT", "RIGHT", "UP", "DOWN"]
batch_size = 64 # how many patterns at once to upgrade the weights
gamma = 0.99 # discount factor, future rewards
epsilon = 1.0  # 1.0 means random action, 0.0 means Q action, now at start its 1.0
epsilon_decay = 0.995 # how much we decrease it per step
min_epsilon = 0.01
target_update = 10  # Target network update frequency, 10 = after 10 iteration
font = pygame.font.Font(None, 36)

episode = 0
good_try = 0
bad_try = 0

# Function to draw objects on the screen
def draw_objects():
    global good_try, bad_try
    window.fill(WHITE)

    window.blit(goal_image, (goal_position[0], goal_position[1]))
    window.blit(car_image, (car_position[0], car_position[1]))
    for obs in obstacles:
        window.blit(obstacle_image, (obs[0], obs[1]))

    episode_text = font.render(f"Episode: {episode}", True, BLACK)
    good_try_text = font.render(f"Good try: {good_try}", True, GREEN)
    bad_try_text = font.render(f"Bad try: {bad_try}", True, RED)
    window.blit(episode_text, (10, 10))
    window.blit(good_try_text, (10, 50))
    window.blit(bad_try_text, (10, 80))

    pygame.display.update()

# Helper functions
def move_car(action):
    if action == "LEFT" and car_position[0] - car_speed >= 0:
        car_position[0] -= car_speed
    elif action == "RIGHT" and car_position[0] + car_speed + car_width <= WIDTH:
        car_position[0] += car_speed
    elif action == "UP" and car_position[1] - car_speed >= 0:
        car_position[1] -= car_speed
    elif action == "DOWN" and car_position[1] + car_speed + car_height <= HEIGHT:
        car_position[1] += car_speed

def check_collision():
    car_rect = pygame.Rect(car_position[0], car_position[1], car_width, car_height)
    for obs in obstacles:
        obstacle_rect = pygame.Rect(obs[0], obs[1], 50, 50)
        if car_rect.colliderect(obstacle_rect):
            return -100  # Obstacle collision penalty
    goal_rect = pygame.Rect(goal_position[0], goal_position[1], car_width, car_height)
    if car_rect.colliderect(goal_rect):
        return 100  # Reward for reaching the goal
    return -1  # Small negative reward for each step

# Experience replay and training function
# updates policy_net according to experience replay
def optimize_model():
    if len(memory) < batch_size: # makes sure to have enough sample in order to update
        return

    batch = random.sample(memory, batch_size) # randomly samples from memory (chooses batch_size samples)
    state_batch, action_batch, reward_batch, next_state_batch = zip(*batch) # orders into lists

    # Eredeti lassabb változat:
    #state_batch = torch.tensor(state_batch, dtype=torch.float32)

    # PyTorch tensores
    # -> Arrays, that the GPU can effectively handle, simultaneous process
    state_batch = torch.tensor(np.array(state_batch), dtype=torch.float32) # random batch from replay memory
    action_batch = torch.tensor(np.array(action_batch), dtype=torch.long) # actions made in state_batch, long needed for gather()
    reward_batch = torch.tensor(np.array(reward_batch), dtype=torch.float32) # rewards got from state_action, float needed for loss()
    next_state_batch = torch.tensor(np.array(next_state_batch), dtype=torch.float32) # states that the agent got to from previous actions

    current_q_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
    # policy_net(state_batch) -> calculates Q values for each state in state_batch
    # .gather(1, action_batch.unsqueeze(1)) -> takes the Q values for the actions in action_batch
    # squeeze and unsqeeze helps to handle the size
    next_q_values = target_net(next_state_batch).max(1)[0] # chooses next HIGHEST q value for all the incoming states
    target_q_values = reward_batch + (gamma * next_q_values) # take into consideration the future possible wins

    loss = nn.functional.mse_loss(current_q_values, target_q_values) # Mean Squared Error: difference between actual and target_q
    optimizer.zero_grad() # zeroes the previous gradiens' values, to not stack up throught backpropagation
    loss.backward() # Visszaterjesztés (backpropagation) -> calculates the gradients for net's every weight and bias
    optimizer.step() # updates net's weight and bias using the calculated gradiants

# Main game loop
def game_loop():
    global epsilon, episode, good_try, bad_try
    running = True
    frame_count = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        state = np.array(car_position, dtype=np.float32)

        # exploration vs exploitation (action choose)
        if random.random() < epsilon:
            action = random.choice(actions) # so the agent won't stick to one strategy
        else:
            with torch.no_grad(): # use gathered knowledge, best possible reward (exploitation), with: context manager
                # no update of weights and biases -> save memory and time
                action = actions[policy_net(torch.tensor(state)).argmax().item()] # calc best action for current state
                # torch.no_grad helps here that while we calc best action, dont update parameters (no backward gradient calc)

        move_car(action)
        reward = check_collision()
        next_state = np.array(car_position, dtype=np.float32)

        memory.append((state, actions.index(action), reward, next_state))

        if reward == -100:
            bad_try += 1
            car_position[0], car_position[1] = DEFAULT_CAR_POSITION
            episode += 1
        elif reward == 100:
            good_try += 1
            car_position[0], car_position[1] = DEFAULT_CAR_POSITION
            episode += 1

        optimize_model()

        if frame_count % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if epsilon > min_epsilon:
            epsilon *= epsilon_decay

        draw_objects()
        frame_count += 1
        pygame.time.delay(1)

    pygame.quit()

game_loop()
