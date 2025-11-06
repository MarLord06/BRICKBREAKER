import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt

# --- Parámetros del entorno ---
WIDTH, HEIGHT = 400, 400
BALL_RADIUS = 8
PADDLE_WIDTH, PADDLE_HEIGHT = 60, 10
PADDLE_Y = HEIGHT - 40

# --- Hiperparámetros RL ---
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995
ALPHA = 0.001
GAMMA = 0.95
EPISODES = 500
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE = 10

# --- Inicialización ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Deep Q-Learning Brick Breaker")
clock = pygame.time.Clock()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Red neuronal DQN ---
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
    def forward(self, x):
        return self.model(x)

# --- Funciones auxiliares ---
def get_state(ball_x, ball_y, dx, dy, paddle_x):
    return np.array([ball_x / WIDTH, ball_y / HEIGHT, dx / 5.0, dy / 5.0, paddle_x / WIDTH], dtype=np.float32)

def choose_action(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(3)
    with torch.no_grad():
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = policy_net(state_t)
        return torch.argmax(q_values).item()

# --- Experiencia Replay ---
memory = deque(maxlen=MEMORY_SIZE)

def remember(s, a, r, s2, done):
    memory.append((s, a, r, s2, done))

def replay():
    if len(memory) < BATCH_SIZE:
        return
    batch = random.sample(memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

    q_values = policy_net(states).gather(1, actions)
    next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
    expected_q = rewards + (1 - dones) * GAMMA * next_q_values

    loss = nn.MSELoss()(q_values, expected_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# --- Inicialización de redes ---
state_size = 5
action_size = 3
policy_net = DQN(state_size, action_size).to(device)
target_net = DQN(state_size, action_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=ALPHA)

# --- Visualización del aprendizaje ---
rewards_history = []
avg_window = deque(maxlen=20)

plt.ion()
fig, ax = plt.subplots()
ax.set_xlabel("Episodio")
ax.set_ylabel("Recompensa promedio")
line, = ax.plot([], [])
plt.title("Evolución del Aprendizaje (DQN)")

# --- Loop de entrenamiento ---
epsilon = EPSILON_START
for episode in range(EPISODES):
    ball_x, ball_y = WIDTH / 2, HEIGHT / 2
    dx, dy = 4 * random.choice([-1, 1]), 4
    paddle_x = WIDTH / 2 - PADDLE_WIDTH / 2
    total_reward = 0

    for step in range(1000):
        state = get_state(ball_x, ball_y, dx, dy, paddle_x)
        action = choose_action(state, epsilon)

        # --- Acción ---
        if action == 0:
            paddle_x -= 10
        elif action == 2:
            paddle_x += 10
        paddle_x = np.clip(paddle_x, 0, WIDTH - PADDLE_WIDTH)

        # --- Movimiento de la bola ---
        ball_x += dx
        ball_y += dy
        reward = 0
        done = False

        if ball_x <= 0 or ball_x >= WIDTH - BALL_RADIUS:
            dx *= -1
        if ball_y <= 0:
            dy *= -1
        if ball_y >= HEIGHT:
            reward = -1
            done = True

        if (PADDLE_Y - BALL_RADIUS <= ball_y <= PADDLE_Y + PADDLE_HEIGHT) and \
           (paddle_x <= ball_x <= paddle_x + PADDLE_WIDTH):
            dy *= -1
            reward = +1

        next_state = get_state(ball_x, ball_y, dx, dy, paddle_x)
        remember(state, action, reward, next_state, done)
        total_reward += reward

        # --- Entrenamiento del agente ---
        replay()

        # --- Renderización ---
        screen.fill((0, 0, 0))
        pygame.draw.circle(screen, (255, 0, 0), (int(ball_x), int(ball_y)), BALL_RADIUS)
        pygame.draw.rect(screen, (0, 150, 255), (int(paddle_x), PADDLE_Y, PADDLE_WIDTH, PADDLE_HEIGHT))
        pygame.display.flip()
        clock.tick(240)

        if done:
            break

    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    avg_window.append(total_reward)
    rewards_history.append(np.mean(avg_window))

    # Actualizar red objetivo cada cierto tiempo
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Actualizar gráfico
    line.set_data(range(len(rewards_history)), rewards_history)
    ax.set_xlim(0, len(rewards_history))
    ax.set_ylim(-1, 1.5)
    plt.pause(0.001)

    print(f"🎮 Episodio {episode+1}/{EPISODES} | Epsilon: {epsilon:.3f} | Recompensa media: {np.mean(avg_window):.2f}")

pygame.quit()
#plt.ioff()
#plt.show()
