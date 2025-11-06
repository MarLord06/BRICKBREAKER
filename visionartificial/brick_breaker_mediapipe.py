import cv2
import numpy as np
import mediapipe as mp


def game():
    """
    Juego Brick Breaker controlado con la mano usando MediaPipe
    """
    BALL_SPEED = 6  # Aumentado de 5 a 6 para más velocidad
    dx, dy = 5, -4  # Velocidades iniciales más altas
    x1, y1, x2, y2 = 90, 150, 100, 160
    y6 = 410
    f = 0
    x11 = []
    
    # Sistema de incremento de velocidad
    speed_multiplier = 1.0
    next_speed_threshold = 10  # Primer incremento a los 10 bloques
    speed_stage = 1  # Etapa actual de velocidad

    # Inicializar MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(static_image_mode=False,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5,
                           max_num_hands=1)

    cap = cv2.VideoCapture(1)
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo acceder a la cámara")
        return

    frame_height, frame_width, _ = frame.shape
    prev_x3 = frame_width // 2
    x3 = prev_x3
    smoothing_factor = 0.2

    # Crear ladrillos que ocupen todo el ancho
    brick_height = 10
    brick_spacing = 5
    margin = 10  # Margen pequeño en los bordes
    
    # Calcular para que ocupen todo el ancho
    available_width = frame_width - (2 * margin)
    num_bricks_per_row = 25  # Más ladrillos
    total_spacing = brick_spacing * (num_bricks_per_row - 1)
    brick_width = (available_width - total_spacing) // num_bricks_per_row
    
    x7 = margin
    y7 = 50
    for i in range(6):  # 6 filas en lugar de 4
        x11.append([])
        for j in range(num_bricks_per_row):
            x9 = x7 + (brick_width + brick_spacing) * j
            y9 = y7 + (brick_height + brick_spacing) * i
            x11[i].append(str(x9) + "_" + str(y9))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Actualizar posición de la pelota primero
        x1 = int(x1 + dx)
        y1 = int(y1 + dy)
        x2 = int(x2 + dx)
        y2 = int(y2 + dy)
        
        # Control manual con la mano
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Dibujar los landmarks de la mano
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Calcular posición de control
                wrist_x = hand_landmarks.landmark[0].x
                index_base_x = hand_landmarks.landmark[5].x
                middle_base_x = hand_landmarks.landmark[9].x
                pinky_base_x = hand_landmarks.landmark[17].x
                avg_x = (wrist_x + index_base_x + middle_base_x + pinky_base_x) / 4
                target_x3 = int(avg_x * w)
                x3 = int(prev_x3 * smoothing_factor + target_x3 * (1 - smoothing_factor))
                x3 = max(25, min(x3, w - 25))
                prev_x3 = x3
        else:
            x3 = prev_x3

        # Dibujar paleta
        cv2.rectangle(frame, (x3 - 25, y6), (x3 + 25, y6 + 10), (255, 255, 255), -1)

        # Dibujar pelota (ya actualizada arriba)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1)

        # Dibujar ladrillos
        for i in range(6):  # 6 filas
            for j in range(num_bricks_per_row):
                rec = x11[i][j]
                if rec != []:
                    x12, y12 = map(int, rec.split("_"))
                    # Colores degradados por fila
                    brick_color = (0, 220 - i * 30, i * 20)  # Verde a amarillo
                    cv2.rectangle(frame, (x12, y12),
                                  (x12 + brick_width, y12 + brick_height),
                                  brick_color, -1)

        # Colisiones con paredes
        if x2 >= w:
            dx = -abs(dx)
        if x1 <= 0:
            dx = abs(dx)
        if y1 <= 50:
            dy = abs(dy)

        # Colisiones con ladrillos
        for i in range(6):  # 6 filas
            for j in range(num_bricks_per_row):
                rec = x11[i][j]
                if rec != []:
                    x12, y12 = map(int, rec.split("_"))
                    if not (x2 < x12 or x1 > x12 + brick_width or y2 < y12 or y1 > y12 + brick_height):
                        dy = abs(dy)
                        x11[i][j] = []
                        f += 1
                        
                        # Sistema de incremento de velocidad
                        if f >= next_speed_threshold:
                            speed_multiplier += 0.05  # Incremento del 5%
                            
                            # Aplicar incremento a las velocidades actuales
                            dx_sign = 1 if dx > 0 else -1
                            dy_sign = 1 if dy > 0 else -1
                            dx_magnitude = abs(dx) * 1.05
                            dy_magnitude = abs(dy) * 1.05
                            dx = dx_sign * dx_magnitude
                            dy = dy_sign * dy_magnitude
                            
                            # Calcular siguiente umbral
                            if speed_stage == 1:
                                next_speed_threshold += 10  # Cada 10 bloques (10, 20, 30...)
                            elif speed_stage == 2:
                                next_speed_threshold += 15  # Cada 15 bloques (35, 50, 65...)
                            else:
                                next_speed_threshold += 20  # Cada 20 bloques (85, 105, 125...)
                            
                            # Cambiar etapa según bloques rotos
                            if f >= 30 and speed_stage == 1:
                                speed_stage = 2
                            elif f >= 60 and speed_stage == 2:
                                speed_stage = 3
                        
                        # Agregar un poco de velocidad horizontal aleatoria al romper ladrillos
                        if abs(dx) < 2:
                            dx += np.random.choice([-1, 1]) * 0.8
                        break

        # Rebote con la paleta
        if dy > 0 and y2 >= y6 and y1 <= y6 + 10:
            if (x3 - 25) <= (x1 + x2) / 2 <= (x3 + 25):
                dy = -abs(dy)
                offset = ((x1 + x2) / 2 - x3) / 25
                dx += offset * 3  # Aumentado de 2 a 3 para más efecto
                dx = np.clip(dx, -6, 6)
        
        # PROTECCIÓN GLOBAL: Forzar velocidad horizontal mínima SIEMPRE
        MIN_HORIZONTAL_SPEED = 1.0
        if abs(dx) < MIN_HORIZONTAL_SPEED:
            # Mantener dirección pero forzar velocidad mínima
            dx = MIN_HORIZONTAL_SPEED if dx >= 0 else -MIN_HORIZONTAL_SPEED

        # Game Over
        if y2 > y6 + 15:
            cv2.putText(frame, "GAME OVER", (w // 2 - 100, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.putText(frame, "Press R to Restart or Q to Quit", (w // 2 - 200, h // 2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.imshow("Brick Breaker", frame)
            key = cv2.waitKey(0)
            if key in [ord('r'), ord('R')]:
                return 'restart'
            else:
                cap.release()
                cv2.destroyAllWindows()
                return 'quit'

        # Mostrar puntaje
        cv2.putText(frame, f"SCORE: {f}", (w // 2 - 60, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (210, 120, 120), 2)
        
        # Mostrar multiplicador de velocidad
        if speed_multiplier > 1.0:
            speed_text = f"SPEED: x{speed_multiplier:.2f}"
            cv2.putText(frame, speed_text, (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Mostrar instrucciones
        cv2.putText(frame, "Hand Control - ESC or Q to Quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Brick Breaker", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or k in [ord('q'), ord('Q')]:  # ESC o Q para salir
            break

    hands.close()
    cap.release()
    cv2.destroyAllWindows()
    return 'quit'


# Bucle principal
if __name__ == "__main__":
    print("🎮 Brick Breaker - Control con Mano")
    print("=" * 60)
    print("🖐️  Mueve tu mano para controlar la paleta")
    print("\nControles:")
    print("  - Mano: Mover la paleta")
    print("  - R: Reiniciar (en Game Over)")
    print("  - Q o ESC: Salir")
    print("=" * 60)
    print("\n🎯 ¡Destruye todos los ladrillos!")
    print()
    
    while True:
        result = game()
        if result != 'restart':
            break
    
    print("\n👋 ¡Gracias por jugar!")
