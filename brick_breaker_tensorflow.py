import cv2
import numpy as np
import mediapipe as mp
from random import random
from ai_player import BrickBreakerAI, HeuristicAI, record_training_data_with_heuristic


def game(ai_mode='manual', collect_data=False):
    """
    ai_mode: 'manual', 'heuristic', 'neural'
    collect_data: True para guardar datos durante el juego
    """
    BALL_SPEED = 5
    dx, dy = 4, -3
    x1, y1, x2, y2 = 90, 150, 100, 160
    y6 = 410
    f = 0
    x11 = []
    training_data = []  # Para recolectar datos durante el juego

    # Inicializar MediaPipe Hands
    mp_hands = mp.solutions.hands
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

    # Inicializar IA según el modo
    neural_ai = None
    heuristic_ai = None
    
    if ai_mode == 'neural':
        print("🧠 Cargando IA Neural (GRU)...")
        neural_ai = BrickBreakerAI(frame_width, frame_height, load_pretrained=True)
        if neural_ai.training_mode:
            print("⚠️ No hay modelo entrenado. Cambiando a heurística.")
            ai_mode = 'heuristic'
            heuristic_ai = HeuristicAI(frame_width, frame_height)
        else:
            print("✅ Modelo neural cargado correctamente")
            # Cargar heurística también para comparación
            heuristic_ai = HeuristicAI(frame_width, frame_height)
            neural_ai.reset_history()  # Limpiar historial al inicio
    
    elif ai_mode == 'heuristic':
        print("🎯 Iniciando IA Heurística (perfecta)...")
        heuristic_ai = HeuristicAI(frame_width, frame_height)
        print("✅ Heurística lista (ideal para recolección de datos)")

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
        
        ball_center_x = (x1 + x2) / 2
        ball_center_y = (y1 + y2) / 2
        
        # Control del jugador
        if ai_mode in ['neural', 'heuristic']:
            # Decidir qué IA usar
            if ai_mode == 'neural' and neural_ai:
                target_x3 = neural_ai.get_target_position(ball_center_x, ball_center_y, dx, dy)
            elif heuristic_ai:
                target_x3 = heuristic_ai.get_target_position(ball_center_x, ball_center_y, dx, dy)
            else:
                target_x3 = x3  # Fallback
            
            # Recolectar datos si está activado (usar heurística como ground truth)
            # IMPORTANTE: Solo guardamos cuando la pelota va hacia abajo (dy > 0)
            if collect_data and dy > 0:  # Solo cuando la pelota cae hacia la paleta
                if heuristic_ai:
                    ground_truth_x = heuristic_ai.get_target_position(ball_center_x, ball_center_y, dx, dy)
                else:
                    ground_truth_x = target_x3
                training_data.append([ball_center_x, ball_center_y, dx, dy, ground_truth_x])

            # IA se mueve muy rápido para no fallar
            ai_speed = 20  # Aumentado de 12 a 20 para reacción más rápida
            if abs(target_x3 - x3) > ai_speed:
                x3 += ai_speed if target_x3 > x3 else -ai_speed
            else:
                x3 = int(target_x3)
            x3 = max(25, min(x3, w - 25))
            prev_x3 = x3

        elif ai_mode == 'manual':
            # Control manual con la mano
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
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
                        break

        # Rebote con la paleta
        if dy > 0 and y2 >= y6 and y1 <= y6 + 10:
            if (x3 - 25) <= (x1 + x2) / 2 <= (x3 + 25):
                dy = -abs(dy)
                offset = ((x1 + x2) / 2 - x3) / 25
                dx += offset * 2
                dx = np.clip(dx, -6, 6)

        # Game Over
        if y2 > y6 + 15:
            cv2.putText(frame, "GAME OVER", (w // 2 - 100, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.imshow("Brick Breaker", frame)
            key = cv2.waitKey(0)
            if key in [ord('r'), ord('R')]:
                np.save("data/training_data.npy", np.array(training_data))
                return
            elif key in [ord('q'), ord('Q'), 27]:
                cap.release()
                cv2.destroyAllWindows()
                return

        # Mostrar puntaje
        cv2.putText(frame, f"SCORE: {f}", (w // 2 - 60, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (210, 120, 120), 2)

        # Mostrar modo
        if ai_mode == 'neural':
            mode_text = "NEURAL AI - Press H:Heuristic M:Manual"
            color = (255, 0, 255)  # Magenta
            # Mostrar predicción de la red vs heurística (debug)
            if neural_ai and heuristic_ai:
                heuristic_target = heuristic_ai.get_target_position(ball_center_x, ball_center_y, dx, dy)
                neural_target = target_x3
                error = abs(neural_target - heuristic_target)
                distance_to_target = abs(x3 - neural_target)
                
                # Líneas de objetivo
                cv2.line(frame, (int(heuristic_target), y6 + 20), (int(heuristic_target), y6 + 40), (0, 255, 0), 3)  # Verde = correcto
                cv2.line(frame, (int(neural_target), y6 + 20), (int(neural_target), y6 + 40), (255, 0, 255), 3)  # Magenta = predicción
                cv2.circle(frame, (int(x3), y6 + 30), 5, (255, 255, 0), -1)  # Cyan = posición actual de paleta
                
                # Info de debug
                cv2.putText(frame, f"Pred Error: {error:.0f}px", (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Distance: {distance_to_target:.0f}px", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        elif ai_mode == 'heuristic':
            mode_text = "HEURISTIC AI - Press N:Neural M:Manual"
            color = (0, 255, 255)  # Amarillo
        else:
            mode_text = "MANUAL - Press H:Heuristic N:Neural"
            color = (0, 255, 0)  # Verde
        
        cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        if collect_data:
            cv2.putText(frame, f"Recording: {len(training_data)} samples", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        cv2.imshow("Brick Breaker", frame)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:  # ESC
            break
        elif k in [ord('q'), ord('Q')]:  # Q para salir y guardar
            hands.close()
            cap.release()
            cv2.destroyAllWindows()
            if collect_data and training_data:
                np.save("data/training_data.npy", np.array(training_data))
                print(f"\n💾 {len(training_data)} muestras guardadas en data/training_data.npy")
            print("👋 Cerrando juego...")
            return None
        elif k in [ord('m'), ord('M')]:
            hands.close()
            cap.release()
            cv2.destroyAllWindows()
            if collect_data and training_data:
                np.save("data/training_data.npy", np.array(training_data))
                print(f"\n💾 {len(training_data)} muestras guardadas en data/training_data.npy")
            return 'manual'
        elif k in [ord('h'), ord('H')]:
            hands.close()
            cap.release()
            cv2.destroyAllWindows()
            if collect_data and training_data:
                np.save("data/training_data.npy", np.array(training_data))
                print(f"\n💾 {len(training_data)} muestras guardadas en data/training_data.npy")
            return 'heuristic'
        elif k in [ord('n'), ord('N')]:
            hands.close()
            cap.release()
            cv2.destroyAllWindows()
            if collect_data and training_data:
                np.save("data/training_data.npy", np.array(training_data))
                print(f"\n💾 {len(training_data)} muestras guardadas en data/training_data.npy")
            return 'neural'
    
    # Si salió del loop sin presionar tecla específica (ESC)
    if collect_data and training_data:
        np.save("data/training_data.npy", np.array(training_data))
        print(f"\n💾 {len(training_data)} muestras guardadas en data/training_data.npy")

    hands.close()
    cap.release()
    cv2.destroyAllWindows()


# Bucle principal
print("🎮 Brick Breaker - 3 Modos de Juego")
print("=" * 60)
print("Modos:")
print("  🖐️  MANUAL     - Control con la mano (MediaPipe)")
print("  🎯 HEURISTIC  - IA perfecta (ideal para datos)")
print("  🧠 NEURAL     - Red neuronal GRU (experimental)")
print("\nControles:")
print("  - M: Modo Manual")
print("  - H: Modo Heurística")  
print("  - N: Modo Neural")
print("  - Q: Salir y guardar datos")
print("  - R: Reiniciar (en Game Over)")
print("  - ESC: Salir")
print("=" * 60)
print()

# Preguntar si quiere recolectar datos
collect_choice = input("¿Recolectar datos de entrenamiento? (s/n): ").strip().lower()
collect_data = collect_choice in ['s', 'si', 'yes', 'y']

if collect_data:
    print("✅ Recolección de datos ACTIVADA - Los datos se guardarán al presionar Q")
else:
    print("⚠️ Recolección de datos DESACTIVADA")
print()

current_mode = 'neural'  # Empezar con heurística para demo
while True:
    result = game(ai_mode=current_mode, collect_data=collect_data)
    if result in ['manual', 'heuristic', 'neural']:
        current_mode = result
    else:
        break
