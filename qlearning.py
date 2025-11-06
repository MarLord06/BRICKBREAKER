import pygame
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random

# --- Parámetros del entorno ---
WIDTH, HEIGHT = 400, 400
BALL_RADIUS = 8
PADDLE_WIDTH, PADDLE_HEIGHT = 60, 10
PADDLE_Y = HEIGHT - 40

# --- Hiperparámetros RL ---
EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.95
EPISODES = 300

# --- Inicialización ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("RL Brick Breaker - Aprendizaje Visualizado")
clock = pygame.time.Clock()

# --- Q-learning básico ---
def discretize_state(ball_x, ball_y, dx, dy, paddle_x):
    return (int(ball_x / 20), int(ball_y / 20), int(dx), int(dy), int(paddle_x / 20))

Q = {}  # Tabla Q dinámica

def get_Q(state):
    if state not in Q:
        Q[state] = np.zeros(3)  # acciones: izq, quieto, der
    return Q[state]

def choose_action(state):
    if np.random.rand() < EPSILON:
        return np.random.randint(3)
    return np.argmax(get_Q(state))

# --- Visualización del aprendizaje ---
rewards_history = []
avg_window = deque(maxlen=20)

plt.ion()
fig, ax = plt.subplots()
ax.set_xlabel("Episodio")
ax.set_ylabel("Recompensa promedio")
line, = ax.plot([], [])
plt.title("Evolución del Aprendizaje (Recompensa Promedio)")

# --- Loop de entrenamiento ---
for episode in range(EPISODES):
    ball_x, ball_y = WIDTH/2, HEIGHT/2
    dx, dy = 4 * random.choice([-1, 1]), 4
    paddle_x = WIDTH/2 - PADDLE_WIDTH/2
    total_reward = 0

    for step in range(1000):
        state = discretize_state(ball_x, ball_y, dx, dy, paddle_x)
        action = choose_action(state)

        # --- Acción ---
        if action == 0: paddle_x -= 5
        elif action == 2: paddle_x += 5
        paddle_x = np.clip(paddle_x, 0, WIDTH - PADDLE_WIDTH)

        # --- Movimiento bola ---
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

        next_state = discretize_state(ball_x, ball_y, dx, dy, paddle_x)
        total_reward += reward

        # --- Actualización Q-Learning ---
        Q_s = get_Q(state)
        Q_next = get_Q(next_state)
        Q_s[action] = Q_s[action] + ALPHA * (reward + GAMMA * np.max(Q_next) - Q_s[action])

        # --- Render gráfico ---
        screen.fill((0, 0, 0))
        pygame.draw.circle(screen, (255, 0, 0), (int(ball_x), int(ball_y)), BALL_RADIUS)
        pygame.draw.rect(screen, (0, 150, 255), (int(paddle_x), PADDLE_Y, PADDLE_WIDTH, PADDLE_HEIGHT))
        pygame.display.flip()
        clock.tick(120)

        if done:
            break

    avg_window.append(total_reward)
    rewards_history.append(np.mean(avg_window))

    # Actualizar gráfico
    line.set_data(range(len(rewards_history)), rewards_history)
    ax.set_xlim(0, len(rewards_history))
    ax.set_ylim(-1, 1.5)
    plt.pause(0.001)

    print(f"🎮 Episodio {episode+1}/{EPISODES} - Recompensa media: {np.mean(avg_window):.2f}")

pygame.quit()
plt.ioff()
plt.show()
