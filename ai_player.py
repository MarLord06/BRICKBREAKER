"""
Red Neuronal SIMPLE para jugar Brick Breaker automáticamente
Predice dónde caerá la pelota y mueve la paleta ahí
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from collections import deque


class BrickBreakerAI:
    def __init__(self, frame_width, frame_height, load_pretrained=True):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.training_mode = False
        
        # Configuración para historial de estados
        self.sequence_length = 5
        self.feature_size = 4
        self.state_history = deque(maxlen=self.sequence_length)
        self.history_initialized = False

        # Construir modelo recurrente para capturar dinámica temporal
        self.model = self._build_model()
        
        # Intentar cargar modelo pre-entrenado
        if load_pretrained and os.path.exists('brick_breaker_model.keras'):
            try:
                self.model = keras.models.load_model('brick_breaker_model.keras')
                print("✅ Modelo cargado exitosamente")
                self.training_mode = False
                self.reset_history()
            except:
                print("⚠️ No se pudo cargar el modelo. Usando heurística.")
                self.training_mode = True
        else:
            print("⚠️ No hay modelo entrenado. Usando heurística.")
            self.training_mode = True
    
    def _build_model(self):
        """Crea la red neuronal recurrente utilizada para predecir la trayectoria."""
        inputs = keras.Input(shape=(self.sequence_length, self.feature_size))
        x = keras.layers.GRU(64, return_sequences=False)(inputs)
        x = keras.layers.Dense(64, activation='relu')(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(32, activation='relu')(x)
        outputs = keras.layers.Dense(1, activation='sigmoid')(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='mse',
            metrics=['mae']
        )
        return model

    def reset_history(self):
        """Vacía el historial de estados para iniciar un nuevo episodio."""
        self.state_history.clear()
        self.history_initialized = False

    def _normalize_state(self, ball_x, ball_y, ball_dx, ball_dy):
        """Convierte un estado crudo en valores normalizados listos para la red."""
        return np.array([
            ball_x / self.frame_width,
            ball_y / self.frame_height,
            ball_dx / 10.0,
            ball_dy / 10.0
        ], dtype=np.float32)

    def _ensure_history_initialized(self, state_vector):
        """Rellena el historial con el estado actual si aún no existe."""
        if not self.history_initialized or len(self.state_history) < self.sequence_length:
            self.state_history.clear()
            for _ in range(self.sequence_length):
                self.state_history.append(state_vector.copy())
            self.history_initialized = True

    def _update_history(self, state_vector):
        """Actualiza la secuencia de estados usada por la red neuronal."""
        if not self.history_initialized:
            self._ensure_history_initialized(state_vector)
        else:
            self.state_history.append(state_vector.copy())

    def _get_sequence_input(self):
        """Obtiene la secuencia actual con la forma esperada por el modelo."""
        if not self.history_initialized or len(self.state_history) < self.sequence_length:
            state_vector = self.state_history[-1] if self.state_history else np.zeros(self.feature_size, dtype=np.float32)
            self._ensure_history_initialized(state_vector)
        sequence = np.array(self.state_history, dtype=np.float32)
        return sequence.reshape(1, self.sequence_length, self.feature_size)

    def get_state(self, ball_x, ball_y, ball_dx, ball_dy):
        """
        Normalizar el estado del juego (sin posición de paleta, solo pelota)
        Devuelve forma (1, 4) por compatibilidad con código existente.
        """
        state = self._normalize_state(ball_x, ball_y, ball_dx, ball_dy)
        return state.reshape(1, -1)
    
    def predict_paddle_position(self, ball_x, ball_y, ball_dx, ball_dy):
        """
        HEURÍSTICA MEJORADA: Predecir dónde caerá la pelota
        Simula el movimiento exacto incluyendo todos los rebotes
        """
        # Solo predecir si la pelota va hacia abajo
        if ball_dy <= 0:
            # Si va hacia arriba, esperar en el centro
            return self.frame_width / 2
        
        # Simular movimiento
        sim_x = float(ball_x)
        sim_y = float(ball_y)
        sim_dx = float(ball_dx)
        sim_dy = float(ball_dy)
        
        paddle_y = 430  # Altura de la paleta
        max_iterations = 500  # Prevenir loops infinitos
        iterations = 0
        
        # Simular hasta que llegue a la altura de la paleta
        while sim_y < paddle_y and iterations < max_iterations:
            # Mover pelota
            sim_x += sim_dx
            sim_y += sim_dy
            
            # Rebote en pared izquierda
            if sim_x <= 10:
                sim_x = 10
                sim_dx = abs(sim_dx)  # Asegurar que vaya a la derecha
            
            # Rebote en pared derecha
            if sim_x >= self.frame_width - 10:
                sim_x = self.frame_width - 10
                sim_dx = -abs(sim_dx)  # Asegurar que vaya a la izquierda
            
            # Rebote en techo
            if sim_y <= 10:
                sim_y = 10
                sim_dy = abs(sim_dy)  # Asegurar que vaya hacia abajo
            
            iterations += 1
        
        # Limitar a zona segura de la paleta
        target_x = max(30, min(self.frame_width - 30, sim_x))
        return target_x
    
    def get_target_position(self, ball_x, ball_y, ball_dx, ball_dy):
        """
        Obtener posición objetivo de la paleta
        """
        current_state = self._normalize_state(ball_x, ball_y, ball_dx, ball_dy)
        self._update_history(current_state)

        if self.training_mode:
            # Usar heurística
            return self.predict_paddle_position(ball_x, ball_y, ball_dx, ball_dy)
        else:
            # Usar red neuronal con contexto temporal
            sequence_input = self._get_sequence_input()
            prediction = self.model.predict(sequence_input, verbose=0)[0][0]
            return prediction * self.frame_width
    
    def train_from_heuristic(self, num_samples=30000):
        """
        Entrena la red neuronal usando secuencias de estados derivados de la heurística.
        Cada ejemplo contiene varios frames consecutivos para capturar la dinámica.
        """
        print(f"🎓 Generando {num_samples} secuencias realistas para entrenamiento...")
        print("   Incluyendo rebotes múltiples, ángulos pronunciados y zonas cercanas a las paredes")

        sequences = []  # Forma: (num_samples, sequence_length, feature_size)
        targets = []    # Posición normalizada donde debe estar la paleta

        # Reutilizar la heurística como profesor para etiquetar los ejemplos
        for i in range(num_samples):
            # Inicializar pelota en escenarios desafiantes
            if i % 10 < 4:
                # Cerca de las paredes para forzar rebotes laterales
                ball_x = np.random.uniform(10, 80) if np.random.rand() < 0.5 else np.random.uniform(self.frame_width - 80, self.frame_width - 10)
            else:
                ball_x = np.random.uniform(40, self.frame_width - 40)

            if i % 10 < 3:
                # Ángulos extremos
                ball_dx = np.random.choice([-6, -5.5, 5.5, 6])
            else:
                ball_dx = np.random.uniform(-6, 6)

            ball_y = np.random.uniform(40, 380)
            ball_dy = np.random.uniform(1.5, 6.0)  # Siempre hacia abajo

            # Aumentar probabilidad de múltiples rebotes
            if i % 10 < 2:
                ball_y = np.random.uniform(40, 200)

            seq_states = []

            sim_ball_x = float(ball_x)
            sim_ball_y = float(ball_y)
            sim_dx = float(ball_dx)
            sim_dy = float(ball_dy)

            for _ in range(self.sequence_length):
                # Registrar estado actual
                seq_states.append(self._normalize_state(sim_ball_x, sim_ball_y, sim_dx, sim_dy))

                # Avanzar un frame en la simulación
                sim_ball_x += sim_dx
                sim_ball_y += sim_dy

                # Rebote lateral
                if sim_ball_x <= 10:
                    sim_ball_x = 10
                    sim_dx = abs(sim_dx)
                elif sim_ball_x >= self.frame_width - 10:
                    sim_ball_x = self.frame_width - 10
                    sim_dx = -abs(sim_dx)

                # Rebote en techo
                if sim_ball_y <= 10:
                    sim_ball_y = 10
                    sim_dy = abs(sim_dy)

                # Evitar que baje demasiado (simular antes de llegar a la paleta real)
                if sim_ball_y >= self.frame_height - 80:
                    sim_ball_y = self.frame_height - 80
                    sim_dy = -abs(sim_dy)  # Rebote hacia arriba para generar historial variado

            # Usar el último estado de la secuencia para obtener la etiqueta del profesor
            last_state = seq_states[-1]
            last_ball_x = last_state[0] * self.frame_width
            last_ball_y = last_state[1] * self.frame_height
            last_ball_dx = last_state[2] * 10.0
            last_ball_dy = abs(last_state[3] * 10.0) + 0.1  # Asegurar movimiento hacia abajo

            target_x = self.predict_paddle_position(last_ball_x, last_ball_y, last_ball_dx, last_ball_dy)

            sequences.append(seq_states)
            targets.append(target_x / self.frame_width)

            if (i + 1) % 3000 == 0:
                print(f"  {i + 1}/{num_samples} secuencias generadas")

        X = np.array(sequences, dtype=np.float32)
        y = np.array(targets, dtype=np.float32)

        print("🧠 Entrenando red neuronal (puede tardar varios minutos)...")
        history = self.model.fit(
            X,
            y,
            epochs=120,
            batch_size=64,
            verbose=1,
            validation_split=0.15,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=12,
                    restore_best_weights=True
                )
            ]
        )

        print("✅ Entrenamiento completado!")
        self.training_mode = False
        self.reset_history()
    
    def save_model(self, filepath='brick_breaker_model.keras'):
        """Guardar modelo entrenado"""
        self.model.save(filepath)
        print(f"💾 Modelo guardado: {filepath}")
    
    def load_model(self, filepath='brick_breaker_model.keras'):
        """Cargar modelo entrenado"""
        if os.path.exists(filepath):
            self.model = keras.models.load_model(filepath)
            print(f"✅ Modelo cargado: {filepath}")
            self.training_mode = False
            self.reset_history()
            return True
        return False


class SimpleAI:
    """
    IA simple basada en heurística (sin red neuronal)
    Predice dónde caerá la pelota y mueve la paleta ahí
    """
    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.paddle_y = 410  # Posición Y de la paleta
    
    def predict_target(self, ball_x, ball_y, ball_dx, ball_dy, paddle_x):
        """
        Predecir dónde debe estar la paleta
        """
        # Si la pelota va hacia arriba, seguirla suavemente
        if ball_dy < 0:
            return ball_x
        
        # Si la pelota va hacia abajo, predecir dónde caerá
        frames_to_paddle = (self.paddle_y - ball_y) / abs(ball_dy) if ball_dy > 0 else 999
        
        if frames_to_paddle > 0 and frames_to_paddle < 100:
            # Predecir posición X futura
            predicted_x = ball_x + (ball_dx * frames_to_paddle)
            
            # Manejar rebotes en paredes
            bounces = 0
            while (predicted_x < 0 or predicted_x > self.frame_width) and bounces < 5:
                if predicted_x < 0:
                    predicted_x = abs(predicted_x)
                    ball_dx = abs(ball_dx)
                elif predicted_x > self.frame_width:
                    predicted_x = 2 * self.frame_width - predicted_x
                    ball_dx = -abs(ball_dx)
                bounces += 1
            
            return predicted_x
        else:
            # Si está muy lejos, mantener posición actual
            return paddle_x
