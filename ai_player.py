import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from collections import deque


class HeuristicAI:
    """
    IA heurística PERFECTA que simula la física del juego.
    Calcula exactamente dónde caerá la pelota considerando rebotes.
    IDEAL para recolectar datos de entrenamiento rápidamente.
    """
    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.paddle_y = 410

    def get_target_position(self, ball_x, ball_y, dx, dy):
        """
        Simula la física del juego para predecir dónde caerá la pelota.
        Considera rebotes en paredes laterales y techo.
        """
        # Si la pelota va hacia arriba, esperar en el centro
        if dy <= 0:
            return self.frame_width / 2
        
        # Si dx es casi 0 (vertical), quedarse bajo la pelota
        if abs(dx) < 0.3:
            return np.clip(ball_x, 25, self.frame_width - 25)
        
        # Simular trayectoria
        sim_x = float(ball_x)
        sim_y = float(ball_y)
        sim_dx = float(dx)
        sim_dy = float(dy)
        
        max_iterations = 500
        iterations = 0
        
        # Simular hasta que llegue a la altura de la paleta
        while sim_y < self.paddle_y and iterations < max_iterations:
            sim_x += sim_dx
            sim_y += sim_dy
            
            # Rebote en pared izquierda
            if sim_x <= 0:
                sim_x = 0
                sim_dx = abs(sim_dx)
            
            # Rebote en pared derecha
            if sim_x >= self.frame_width:
                sim_x = self.frame_width
                sim_dx = -abs(sim_dx)
            
            # Rebote en techo
            if sim_y <= 50:
                sim_y = 50
                sim_dy = abs(sim_dy)
            
            iterations += 1
        
        # Limitar a zona segura
        target_x = np.clip(sim_x, 25, self.frame_width - 25)
        return target_x


class BrickBreakerAI:
    """
    Red neuronal recurrente (GRU) que aprende de la heurística perfecta.
    Mantiene un historial de 5 frames para capturar la dinámica temporal.
    """
    def __init__(self, frame_width, frame_height, load_pretrained=True):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.model_path = "checkpoints/brickbreaker_model.keras"
        self.training_mode = True
        
        # Configuración para secuencias temporales
        self.sequence_length = 5
        self.feature_size = 4
        self.state_history = deque(maxlen=self.sequence_length)
        self.history_initialized = False
        
        # Construir modelo GRU
        self.model = self._build_model()
        
        # Cargar modelo si existe
        if load_pretrained and os.path.exists(self.model_path):
            try:
                self.model = tf.keras.models.load_model(self.model_path)
                self.training_mode = False
                self.reset_history()
                print(f"✅ Modelo neural cargado desde {self.model_path}")
            except Exception as e:
                print(f"⚠️ No se pudo cargar el modelo: {e}")
        
        # Heurística de respaldo
        self.heuristic = HeuristicAI(frame_width, frame_height)

    def _build_model(self):
        """Red neuronal SIMPLE - a veces menos es más"""
        inputs = layers.Input(shape=(self.sequence_length, self.feature_size))
        # UN solo GRU pero con más unidades
        x = layers.GRU(256, return_sequences=False)(inputs)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Learning rate más agresivo
            loss='huber',  # Huber loss es más robusto que MSE
            metrics=['mae']
        )
        return model
    
    def reset_history(self):
        """Limpia el historial de estados"""
        self.state_history.clear()
        self.history_initialized = False
    
    def _normalize_state(self, ball_x, ball_y, dx, dy):
        """Normaliza un estado para la red neuronal"""
        return np.array([
            ball_x / self.frame_width,
            ball_y / self.frame_height,
            dx / 10.0,
            dy / 10.0
        ], dtype=np.float32)
    
    def _ensure_history_initialized(self, state_vector):
        """Inicializa el historial si está vacío"""
        if not self.history_initialized or len(self.state_history) < self.sequence_length:
            self.state_history.clear()
            for _ in range(self.sequence_length):
                self.state_history.append(state_vector.copy())
            self.history_initialized = True
    
    def _update_history(self, state_vector):
        """Actualiza el historial con el nuevo estado"""
        if not self.history_initialized:
            self._ensure_history_initialized(state_vector)
        else:
            self.state_history.append(state_vector.copy())
    
    def _get_sequence_input(self):
        """Obtiene la secuencia lista para el modelo"""
        if not self.history_initialized or len(self.state_history) < self.sequence_length:
            state_vector = self.state_history[-1] if self.state_history else np.zeros(self.feature_size, dtype=np.float32)
            self._ensure_history_initialized(state_vector)
        sequence = np.array(self.state_history, dtype=np.float32)
        return sequence.reshape(1, self.sequence_length, self.feature_size)

    def get_target_position(self, ball_x, ball_y, dx, dy):
        """
        Retorna la posición objetivo de la paleta.
        Usa red neuronal si está entrenada, heurística si no.
        """
        current_state = self._normalize_state(ball_x, ball_y, dx, dy)
        self._update_history(current_state)
        
        if self.training_mode:
            # Usar heurística perfecta
            return self.heuristic.get_target_position(ball_x, ball_y, dx, dy)
        else:
            # Si la pelota va hacia arriba, usar heurística directamente
            if dy < 0:
                return self.heuristic.get_target_position(ball_x, ball_y, dx, dy)
            
            # Usar red neuronal solo cuando la pelota baja
            sequence_input = self._get_sequence_input()
            prediction = self.model.predict(sequence_input, verbose=0)[0][0]
            neural_prediction = float(prediction * self.frame_width)
            
            # Sanity check: si la predicción es muy extrema, usar heurística
            if neural_prediction < 25 or neural_prediction > self.frame_width - 25:
                return self.heuristic.get_target_position(ball_x, ball_y, dx, dy)
            
            return neural_prediction

    def train_from_heuristic(self, num_samples=30000):
        """
        Entrena la red neuronal usando secuencias generadas por la heurística.
        Genera datos sintéticos con casos realistas de rebotes y ángulos extremos.
        """
        print(f"🎓 Generando {num_samples} secuencias de entrenamiento...")
        print("   Incluyendo rebotes múltiples, ángulos extremos y zonas críticas")
        
        sequences = []
        targets = []
        
        for i in range(num_samples):
            # Generar casos desafiantes
            if i % 10 < 4:
                # 40% cerca de paredes
                ball_x = np.random.uniform(0, 80) if np.random.rand() < 0.5 else np.random.uniform(self.frame_width - 80, self.frame_width)
            else:
                ball_x = np.random.uniform(40, self.frame_width - 40)
            
            if i % 10 < 3:
                # 30% ángulos extremos
                dx = np.random.choice([-6, -5.5, 5.5, 6])
            else:
                dx = np.random.uniform(-6, 6)
            
            ball_y = np.random.uniform(50, 380)
            dy = np.random.uniform(1.5, 6.0)
            
            # 20% casos con múltiples rebotes
            if i % 10 < 2:
                ball_y = np.random.uniform(50, 200)
            
            # Simular secuencia de estados
            seq_states = []
            sim_x, sim_y = float(ball_x), float(ball_y)
            sim_dx, sim_dy = float(dx), float(dy)
            
            for _ in range(self.sequence_length):
                seq_states.append(self._normalize_state(sim_x, sim_y, sim_dx, sim_dy))
                
                # Avanzar simulación
                sim_x += sim_dx
                sim_y += sim_dy
                
                # Rebotes
                if sim_x <= 0:
                    sim_x = 0
                    sim_dx = abs(sim_dx)
                elif sim_x >= self.frame_width:
                    sim_x = self.frame_width
                    sim_dx = -abs(sim_dx)
                
                if sim_y <= 50:
                    sim_y = 50
                    sim_dy = abs(sim_dy)
                elif sim_y >= self.frame_height - 80:
                    sim_y = self.frame_height - 80
                    sim_dy = -abs(sim_dy)
            
            # Obtener etiqueta de la heurística
            last_state = seq_states[-1]
            last_ball_x = last_state[0] * self.frame_width
            last_ball_y = last_state[1] * self.frame_height
            last_dx = last_state[2] * 10.0
            last_dy = abs(last_state[3] * 10.0) + 0.1
            
            target_x = self.heuristic.get_target_position(last_ball_x, last_ball_y, last_dx, last_dy)
            
            sequences.append(seq_states)
            targets.append(target_x / self.frame_width)
            
            if (i + 1) % 3000 == 0:
                print(f"  {i + 1}/{num_samples} secuencias generadas")
        
        X = np.array(sequences, dtype=np.float32)
        y = np.array(targets, dtype=np.float32)
        
        print("🧠 Entrenando red neuronal GRU...")
        history = self.model.fit(
            X, y,
            epochs=120,
            batch_size=64,
            verbose=1,
            validation_split=0.15,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=12,
                    restore_best_weights=True
                )
            ]
        )
        
        print("✅ Entrenamiento completado!")
        self.training_mode = False
        self.reset_history()
    
    def train_from_collected_data(self, collected_data):
        """
        Entrena la red neuronal usando datos recolectados del juego real.
        collected_data: array de (ball_x, ball_y, dx, dy, paddle_x)
        """
        print(f"🎓 Procesando {len(collected_data)} muestras recolectadas...")
        
        sequences = []
        targets = []
        
        # Necesitamos crear secuencias de 5 frames
        for i in range(len(collected_data) - self.sequence_length):
            sequence_frames = collected_data[i:i + self.sequence_length]
            
            # Normalizar cada frame de la secuencia
            seq_states = []
            for frame in sequence_frames:
                ball_x, ball_y, dx, dy, _ = frame
                normalized = self._normalize_state(ball_x, ball_y, dx, dy)
                seq_states.append(normalized)
            
            # El target es la posición de la paleta del último frame
            target_paddle = sequence_frames[-1][4]  # paddle_x
            
            sequences.append(seq_states)
            targets.append(target_paddle / self.frame_width)  # Normalizar
            
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(collected_data) - self.sequence_length} secuencias procesadas")
        
        if len(sequences) == 0:
            print("❌ ERROR: No hay suficientes datos para crear secuencias")
            print(f"   Se necesitan al menos {self.sequence_length + 1} frames")
            return
        
        X = np.array(sequences, dtype=np.float32)
        y = np.array(targets, dtype=np.float32)
        
        print(f"\n🧠 Entrenando red neuronal GRU con {len(X)} secuencias...")
        print(f"   Shape de entrada: {X.shape}")
        print(f"   Shape de salida: {y.shape}")
        
        # Data augmentation - agregar pequeñas perturbaciones
        print("🔄 Aplicando data augmentation...")
        X_aug = []
        y_aug = []
        for i in range(len(X)):
            # Original
            X_aug.append(X[i])
            y_aug.append(y[i])
            
            # Variación 1: pequeño ruido en posición
            noise = np.random.normal(0, 0.01, X[i].shape)
            X_aug.append(X[i] + noise)
            y_aug.append(y[i])
            
            # Variación 2: espejo horizontal (invertir x)
            mirrored = X[i].copy()
            mirrored[:, 0] = 1.0 - mirrored[:, 0]  # Invertir ball_x
            mirrored[:, 2] = -mirrored[:, 2]  # Invertir dx
            X_aug.append(mirrored)
            y_aug.append(1.0 - y[i])  # Invertir target también
        
        X_aug = np.array(X_aug, dtype=np.float32)
        y_aug = np.array(y_aug, dtype=np.float32)
        print(f"   Datos aumentados: {len(X_aug)} secuencias (3x)")
        
        history = self.model.fit(
            X_aug, y_aug,
            epochs=200,  # Más épocas con más datos
            batch_size=64,
            verbose=1,
            validation_split=0.15,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=20,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=8,
                    min_lr=0.00001
                )
            ]
        )
        
        print("✅ Entrenamiento completado!")
        print(f"   Loss final: {history.history['loss'][-1]:.6f}")
        print(f"   Val loss final: {history.history['val_loss'][-1]:.6f}")
        self.training_mode = False
        self.reset_history()
    
    def save_model(self, filepath=None):
        """Guarda el modelo entrenado"""
        if filepath is None:
            filepath = self.model_path
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"💾 Modelo guardado en {filepath}")


# ----------------------------------------------------
# 🧩 Modo para recolectar datos automáticamente
# ----------------------------------------------------
def record_training_data_with_heuristic(frame_width, frame_height, output_path="data/training_data.npy", num_samples=10000):
    """
    Genera datos de entrenamiento usando la heurística perfecta.
    Mucho más rápido que jugar manualmente.
    Guarda: [ball_x, ball_y, dx, dy, paddle_target_x]
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    heuristic = HeuristicAI(frame_width, frame_height)
    data = []
    
    print(f"🎮 Generando {num_samples} muestras con heurística perfecta...")
    
    for i in range(num_samples):
        # Generar estados aleatorios realistas
        ball_x = np.random.uniform(50, frame_width - 50)
        ball_y = np.random.uniform(100, 400)
        dx = np.random.uniform(-6, 6)
        dy = np.random.uniform(1, 6)
        
        # Calcular posición objetivo con heurística
        target_x = heuristic.get_target_position(ball_x, ball_y, dx, dy)
        
        # Guardar muestra
        data.append([ball_x, ball_y, dx, dy, target_x])
        
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{num_samples} muestras generadas")
    
    np.save(output_path, np.array(data))
    print(f"💾 Datos guardados en {output_path} ({len(data)} muestras)")
    return np.array(data)
