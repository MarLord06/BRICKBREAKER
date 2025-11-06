import cv2
import numpy as np
from random import random, randint
import mediapipe as mp
from ai_player import SimpleAI, BrickBreakerAI



def game(ai_mode=True):
    """
    ai_mode: True = IA juega automáticamente, False = control manual con la mano
    """
    # Velocidad constante de la pelota (magnitud del vector)
    BALL_SPEED = 5
    
    dx = 4 #values with which the ball's pixel x coord increases
    dy = -3 #values with which the ball's pixel y coord increases (negativo = arriba)
    dx1 =4 
    dy1 =4
    x1 = 90 #initial x coord values for ball's top left corner
    x2 = 100 #initial x coord values for ball's bottom right corner
    y1 = 150 #initial y coord values for ball's top left corner
    y2 = 160 #initial y coord values for ball's bottom right corner
    x3 = 0  # Se actualizará con las dimensiones reales de la cámara
    y3 = 150
    x4 = 150
    x5 = 10
    x6 = 60
    y5 = 420
    y6 = 410
    y7 = 50
    x8 = 60
    y8 = 60
    f=0
    x11 = []
    
    # Inicializar MediaPipe Hands con configuración optimizada
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,  # Video mode para mejor tracking
        min_detection_confidence=0.5,  # Reducido para mejor detección rápida
        min_tracking_confidence=0.5,   # Reducido para seguir movimientos rápidos
        max_num_hands=1,
        model_complexity=1  # Mayor complejidad para mejor precisión
    )
    
    cap = cv2.VideoCapture(1)
    
    # Leer un frame para obtener las dimensiones reales de la cámara
    ret, test_frame = cap.read()
    if ret:
        frame_height, frame_width, _ = test_frame.shape
        print(f"Dimensiones de la ventana: {frame_width}x{frame_height}")
    else:
        # Valores por defecto si falla
        frame_width, frame_height = 640, 480
    
    # Variables para suavizado del movimiento (usar centro real de la ventana)
    prev_x3 = frame_width // 2  # Posición inicial centrada en el ancho real
    x3 = prev_x3  # Inicializar posición de la paleta
    smoothing_factor = 0.2  # Factor de suavizado (menor = más responsivo, mayor = más suave)
    
    # Inicializar IA si está en modo AI
    ai = None
    use_neural = True  # CAMBIAR A True PARA USAR RED NEURONAL (requiere entrenamiento)
    
    if ai_mode:
        if use_neural:
            print("🧠 Iniciando IA con Red Neuronal...")
            ai = BrickBreakerAI(frame_width, frame_height, load_pretrained=True)
            
            if ai.training_mode:
                print("⚠️ No hay modelo entrenado. Usando heurística.")
                print("   Para entrenar: python3 train_neural_ai.py")
                print("   O cambia use_neural=False para usar heurística perfecta")
            else:
                print("✅ Red Neuronal cargada - La IA jugará automáticamente")
        else:
            ai = SimpleAI(frame_width, frame_height)
            print("🤖 IA Simple (Heurística) activada")
    else:
        print("🖐️ Modo manual - Usa tu mano para controlar")
    
    # Generar ladrillos centrados dinámicamente
    num_bricks_per_row = 18
    brick_width = 50
    brick_height = 10
    brick_spacing = 10  # Espacio entre ladrillos
    total_bricks_width = num_bricks_per_row * (brick_width + brick_spacing)
    
    # Calcular offset para centrar los ladrillos
    x7 = (frame_width - total_bricks_width) // 2
    
    for i in range(4):
        x11.append([])
        for j in range(num_bricks_per_row):
            x11[i].append([])
    
    for i in range(4):
        for j in range(num_bricks_per_row):
            x9 = x7 + (brick_width + brick_spacing) * j
            y9 = y7 + (brick_height + brick_spacing) * i
            x11[i][j] = str(x9) + "_" + str(y9)
    
    while( 1 ):
        _, frame = cap.read( )
        frame = cv2.flip(frame, 1)  # Espejo horizontal para mejor experiencia
        h, w, c = frame.shape
        
        # Determinar posición de la paleta según el modo
        if ai_mode:
            # Modo IA: la computadora controla la paleta
            # Calcular posición del centro de la pelota
            ball_center_x = (x1 + x2) / 2
            ball_center_y = (y1 + y2) / 2
            
            # Obtener predicción según tipo de IA
            if use_neural and isinstance(ai, BrickBreakerAI):
                # Red Neuronal (o heurística si no está entrenada)
                target_x3 = ai.get_target_position(ball_center_x, ball_center_y, dx, dy)
            else:
                # IA Simple (Heurística)
                target_x3 = ai.predict_target(ball_center_x, ball_center_y, dx, dy, x3)
            
            # Movimiento suave hacia el objetivo
            ai_speed = 12  # Velocidad de la IA (aumentada para respuesta más rápida)
            if abs(target_x3 - x3) > ai_speed:
                if target_x3 > x3:
                    x3 += ai_speed
                else:
                    x3 -= ai_speed
            else:
                x3 = int(target_x3)
            
            # Limitar x3 para que la paleta no salga de la pantalla
            x3 = max(25, min(x3, w - 25))
            prev_x3 = x3
        else:
            # Modo manual: control con la mano
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            # Detectar la mano y obtener la posición X
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Usar promedio de múltiples puntos para mayor estabilidad
                    wrist_x = hand_landmarks.landmark[0].x
                    index_base_x = hand_landmarks.landmark[5].x
                    middle_base_x = hand_landmarks.landmark[9].x
                    pinky_base_x = hand_landmarks.landmark[17].x
                    
                    # Promedio de múltiples puntos para mayor precisión
                    avg_x = (wrist_x + index_base_x + middle_base_x + pinky_base_x) / 4
                    
                    # Convertir coordenadas normalizadas a píxeles
                    target_x3 = int(avg_x * w)
                    
                    # Aplicar suavizado para movimientos más fluidos
                    x3 = int(prev_x3 * smoothing_factor + target_x3 * (1 - smoothing_factor))
                    
                    # Limitar x3 para que la paleta no salga de la pantalla
                    x3 = max(25, min(x3, w - 25))
                    
                    # Actualizar posición anterior para el siguiente frame
                    prev_x3 = x3
            else:
                # Si no se detecta mano, mantener última posición conocida
                x3 = prev_x3
        
        # Dibujar la paleta controlada por la mano
        img1 = cv2.rectangle( frame,( x3-25 ,y6 ), ( x3+25 ,y6+10 ), ( 255 ,255 ,255 ), -1 )
        
        # Dibujar bordes de la ventana (después de la paleta para que no la tape)
        border_thickness = 5
        border_color = (255, 255, 255)  # Color blanco
        
        # Borde izquierdo
        cv2.rectangle(frame, (0, 0), (border_thickness, h), border_color, -1)
        # Borde derecho
        cv2.rectangle(frame, (w - border_thickness, 0), (w, h), border_color, -1)
        # Borde superior (misma amplitud que los laterales)
        cv2.rectangle(frame, (0, 0), (w, border_thickness), border_color, -1)    
        
        # Actualizar posición de la pelota (asegurar que sean enteros)
        x1 = int(x1 + dx)
        y1 = int(y1 + dy)
        y2 = int(y2 + dy)
        x2 = int(x2 + dx)
        img1 = cv2.rectangle( frame, ( x1 ,y1 ), ( x2 ,y2 ), ( 255 ,255 ,255 ), -1 )
        a = random()
        for i in range(4):
            for j in range(18):
                
                rec = x11[i][j]
                
                    
                if rec != []:
                    rec1 = str(rec)

                    rec_1 = rec1.split("_")
    
                    x12 = int(rec_1[0])
                    y12 = int(rec_1[1])
               
                
                    
                    
                    
                
                
                # Dibujar ladrillo con colores verdes tipo juego retro
                # Diferentes tonos de verde según la fila
                if i == 0:
                    brick_color = (0, 255, 0)  # Verde brillante
                elif i == 1:
                    brick_color = (0, 200, 0)  # Verde medio
                elif i == 2:
                    brick_color = (0, 150, 50)  # Verde oscuro
                else:
                    brick_color = (0, 100, 100)  # Verde muy oscuro
                
                # Dibujar ladrillo con borde para efecto retro
                cv2.rectangle( frame, ( x12 , y12 ), ( x12+brick_width , y12+brick_height ), brick_color, -1 )
                cv2.rectangle( frame, ( x12 , y12 ), ( x12+brick_width , y12+brick_height ), (0, 50, 0), 2 )  # Borde oscuro
        
        # Rebote en pared derecha - mantener velocidad constante
        if ( x2 >= w ):
            dx = -abs(dx)  # Invertir dirección sin cambiar magnitud
            
            
        # Colisión con bricks - mantener velocidad constante
        brick_collision = False
        for i in range(4):
            if brick_collision:
                break
            for j in range(num_bricks_per_row):
                ree = x11[i][j]
                if ree != []:
                    ree1 = str(ree)
                    ree_1 = ree1.split("_")
                    x13 = int(ree_1[0])
                    y13 = int(ree_1[1])
                    
                    # Brick boundaries
                    brick_left = x13
                    brick_right = x13 + brick_width
                    brick_top = y13
                    brick_bottom = y13 + brick_height
                    
                    # Verificar colisión AABB (Axis-Aligned Bounding Box)
                    # Solo si hay superposición en ambos ejes
                    if not (x2 < brick_left or x1 > brick_right or y2 < brick_top or y1 > brick_bottom):
                        dy = abs(dy)  # Rebotar hacia abajo manteniendo velocidad
                        x11[i][j] = []
                        f = f + 1
                        brick_collision = True
                        break
        
        # Rebote en techo
        if (y1<=50):
            dy = abs(dy)  # Rebotar hacia abajo                       
                       
        score = "SCORE : "+str(f)
        font = cv2.FONT_HERSHEY_SIMPLEX
            
        bottomLeftCornerOfText = ( w//2 - 80 ,25 )  # Centrado según el ancho real
        fontScale              = 1
        fontColor              = ( 210 ,120 ,120 )
        lineType               = 2
        cv2.putText( img1 ,score,bottomLeftCornerOfText ,font ,fontScale ,fontColor ,lineType )
        
        # Mostrar dimensiones de la ventana (para debug)
        dim_text = f"{w}x{h}"
        cv2.putText( img1 ,dim_text, (10, h-10) ,font ,0.5 ,(100, 255, 100) ,1 )
                         
        # Rebote en pared izquierda - mantener velocidad constante
        if ( x1 <= 0 ):
            dx = abs(dx)  # Invertir dirección sin cambiar magnitud
        
        # Variable para controlar si la pelota tocó la paleta
        paddle_hit = False
        
        # Rebote con la paleta - COLISIÓN CENTRADA
        # Solo detectar colisión si la pelota viene de arriba (dy > 0)
        if dy > 0 and y2 >= y6 and y1 <= y6+10:
            # Calcular centros para detección más precisa
            ball_center_x = (x1 + x2) / 2
            ball_center_y = (y1 + y2) / 2
            paddle_center_x = x3
            paddle_center_y = y6 + 5  # Centro vertical de la paleta
            
            # Zona de colisión más centrada (20px desde el centro en lugar de 25px desde bordes)
            # Esto previene golpes con las esquinas y fuerza golpes más centrados
            paddle_half_width = 20
            paddle_left = paddle_center_x - paddle_half_width
            paddle_right = paddle_center_x + paddle_half_width
            
            # Colisión solo si el CENTRO de la pelota está dentro de la zona válida
            if paddle_left <= ball_center_x <= paddle_right:
                paddle_hit = True
                
                # Calcular desplazamiento del centro de la pelota respecto al centro de la paleta
                offset = (ball_center_x - paddle_center_x) / paddle_half_width  # Normalizado entre -1 y 1
                
                # Ajustar dx basado en dónde golpea la pelota en la paleta
                dx = dx + offset * 2  # Añadir efecto angular
                
                # Limitar dx para que no sea demasiado horizontal
                if abs(dx) < 2:
                    dx = 2 if dx > 0 else -2
                if abs(dx) > 6:
                    dx = 6 if dx > 0 else -6
                
                # Invertir dy manteniendo velocidad
                dy = -abs(dy)
        
        # Game Over - solo si la pelota pasó la línea Y NO tocó la paleta
        if y2 > y6 + 15 and not paddle_hit:
            # Crear overlay semi-transparente
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # GAME OVER principal
            gameOverPos = ( w//2 - 150 , h//2 - 50 )
            cv2.putText( frame ,'GAME OVER!' ,gameOverPos ,font ,1.5 ,(255, 255, 255) ,3 )
            
            # Score final
            finalScore = f"Final Score: {f}"
            scorePos = ( w//2 - 100 , h//2 + 10 )
            cv2.putText( frame ,finalScore ,scorePos ,font ,0.8 ,(200, 200, 200) ,2 )
            
            # Mostrar info de IA si aplica
            if ai_mode:
                if use_neural and isinstance(ai, BrickBreakerAI):
                    ai_type = "Neural AI" if not ai.training_mode else "Heuristic AI"
                else:
                    ai_type = "Simple AI"
                aiInfo = f"AI Type: {ai_type}"
                aiInfoPos = ( w//2 - 120 , h//2 + 40 )
                cv2.putText( frame ,aiInfo ,aiInfoPos ,font ,0.5 ,(150, 150, 255) ,1 )
            
            # Instrucciones
            restartText = "Press 'R' to Restart"
            restartPos = ( w//2 - 120 , h//2 + 70 )
            cv2.putText( frame ,restartText ,restartPos ,font ,0.6 ,(100, 255, 100) ,2 )
            
            quitText = "Press 'Q' to Quit"
            quitPos = ( w//2 - 100 , h//2 + 110 )
            cv2.putText( frame ,quitText ,quitPos ,font ,0.6 ,(100, 100, 255) ,2 )
            
            cv2.imshow('Brick Breaker - Hand Control',frame)
            
            # Esperar input del usuario
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('r') or key == ord('R'):
                    # Reiniciar el juego
                    return  # Sale de la función game() y el while externo la vuelve a llamar
                elif key == ord('q') or key == ord('Q') or key == 27:  # Q o ESC
                    # Salir completamente
                    hands.close()
                    cv2.destroyAllWindows()
                    cap.release()
                    exit()  # Terminar el programa completamente
                    
            break
        
        # Dibujar indicador según el modo
        if ai_mode:
            # Indicador de modo IA
            if use_neural and isinstance(ai, BrickBreakerAI):
                color = (255, 0, 255) if ai.training_mode else (0, 255, 0)  # Magenta si usa heurística, verde si usa red
                label = 'HEURISTIC' if ai.training_mode else 'NEURAL AI'
                cv2.circle(frame, (x3, y6 - 30), 5, color, -1)
                cv2.putText(frame, label, (x3 - 35, y6 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Mostrar modo
                mode_text = 'Neural Network Mode' if not ai.training_mode else 'Heuristic Mode (No model trained)'
                cv2.putText(frame, mode_text + ' - Press M for Manual', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            else:
                cv2.circle(frame, (x3, y6 - 30), 5, (0, 255, 255), -1)
                cv2.putText(frame, 'SIMPLE AI', (x3 - 30, y6 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                cv2.putText(frame, 'Simple Heuristic AI - Press M for Manual', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            # Indicador de tracking manual
            if results.multi_hand_landmarks:
                cv2.circle(frame, (x3, y6 - 30), 5, (0, 255, 0), -1)  # Punto verde arriba de la paleta
                cv2.putText(frame, 'HAND', (x3 - 20, y6 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Mostrar modo en la esquina
            cv2.putText(frame, 'MANUAL MODE - Press A for AI', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        window_title = 'Brick Breaker - AI Mode' if ai_mode else 'Brick Breaker - Hand Control'
        cv2.imshow(window_title, frame)
        k = cv2.waitKey( 5 ) & 0xFF
        if k == 27:  # ESC para salir durante el juego
            hands.close()
            cv2.destroyAllWindows()
            cap.release()
            exit()
        elif k == ord('m') or k == ord('M'):  # Cambiar a modo manual
            if ai_mode:
                print("🖐️ Cambiando a modo manual...")
                hands.close()
                cap.release()
                cv2.destroyAllWindows()
                return 'manual'
        elif k == ord('a') or k == ord('A'):  # Cambiar a modo AI
            if not ai_mode:
                print("🤖 Cambiando a modo IA...")
                hands.close()
                cap.release()
                cv2.destroyAllWindows()
                return 'ai'
    
    # Cerrar recursos si sale del loop sin game over
    hands.close()
    cap.release()
    cv2.destroyAllWindows()

# Loop principal del juego
print("🎮 Brick Breaker con IA")
print("Controles:")
print("  - ESC: Salir del juego")
print("  - A: Cambiar a modo IA")
print("  - M: Cambiar a modo Manual")
print("  - R: Reiniciar (en Game Over)")
print("  - Q: Salir (en Game Over)")
print()

# Modo inicial
current_mode = 'ai'  # Comenzar en modo IA

while True:
    result = game(ai_mode=(current_mode == 'ai'))
    
    # Cambiar modo si se presionó una tecla para hacerlo
    if result == 'manual':
        current_mode = 'manual'
    elif result == 'ai':
        current_mode = 'ai'
