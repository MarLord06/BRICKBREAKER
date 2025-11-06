# 🎮 Brick Breaker con IA - Sistema Completo

## 📋 Descripción

Sistema de Brick Breaker con **3 modos de control** y recolección automática de datos:

1. **🖐️ Manual** - Control con la mano usando MediaPipe (lento para datos)
2. **🎯 Heurística** - IA perfecta que simula física (IDEAL para generar datos)
3. **🧠 Neural** - Red neuronal GRU que aprende de la heurística

## 🚀 Uso Rápido

### 1. Jugar con Heurística (Recomendado para empezar)

```bash
python3 brick_breaker_tensorflow.py
```

**Controles en el juego:**
- `H` - Cambiar a modo Heurística
- `M` - Cambiar a modo Manual
- `N` - Cambiar a modo Neural
- `ESC` - Salir

### 2. Entrenar la Red Neuronal

```bash
python3 train_neural_ai.py
```

Selecciona opción **3** (30,000 ejemplos) - tarda ~8 minutos.

La red neuronal aprenderá automáticamente de la heurística perfecta.

### 3. Usar la Red Neuronal

Después de entrenar, presiona `N` en el juego para cambiar a modo Neural.

## 📊 Comparación de Modos

| Modo | Precisión | Velocidad | Uso Principal |
|------|-----------|-----------|---------------|
| **Heurística** | 100% perfecta | Instantánea | Generación de datos, benchmark |
| **Neural** | ~95% (con entrenamiento) | Rápida | Experimentación con ML |
| **Manual** | Depende del jugador | Variable | Diversión, validación |

## 🎯 Flujo de Trabajo Recomendado

### Para Recolección de Datos (RÁPIDO):

La heurística genera datos automáticamente durante el entrenamiento. **No necesitas jugar manualmente**.

```bash
# Entrenar directamente (genera datos internamente)
python3 train_neural_ai.py
# Selecciona opción 3 o 4
```

### Para Experimentar:

```bash
# 1. Ver la heurística en acción
python3 brick_breaker_tensorflow.py
# Presiona H

# 2. Entrenar la red neuronal
python3 train_neural_ai.py

# 3. Probar la red neuronal
python3 brick_breaker_tensorflow.py
# Presiona N
```

## 🧠 Arquitectura de la Red Neuronal

**Tipo:** Red Recurrente (GRU)

**Entrada:** Secuencia de 5 frames con 4 features cada uno
- ball_x (normalizado 0-1)
- ball_y (normalizado 0-1)  
- dx (velocidad X, normalizado -1 a 1)
- dy (velocidad Y, normalizado -1 a 1)

**Arquitectura:**
```
Input(5, 4) → GRU(64) → Dense(64,relu) → Dropout(0.2) → Dense(32,relu) → Dense(1,sigmoid)
```

**Salida:** Posición X objetivo de la paleta (0-1, desnormalizado a píxeles)

## 📈 Ventajas de la Heurística

✅ **Perfecta:** Calcula exactamente dónde caerá la pelota
✅ **Rápida:** Genera miles de muestras en segundos
✅ **Consistente:** No falla nunca
✅ **Educativa:** Muestra la solución óptima
✅ **Benchmark:** Para comparar el rendimiento de la red neuronal

## 🔧 Archivos

- `ai_player.py` - Clases HeuristicAI y BrickBreakerAI (GRU)
- `train_neural_ai.py` - Script de entrenamiento
- `brick_breaker_tensorflow.py` - Juego principal con 3 modos
- `checkpoints/brickbreaker_model.keras` - Modelo entrenado (después de entrenar)

## 💡 Tips

1. **Usa la heurística primero** para entender cómo funciona el juego perfectamente
2. **La red neuronal es experimental** - puede fallar algunos golpes
3. **No necesitas jugar manualmente** para recolectar datos
4. **Más datos = mejor red neuronal** (prueba con 50,000 ejemplos)

## ⚙️ Requisitos

```bash
pip install tensorflow opencv-python mediapipe numpy
```

## 🎓 Conceptos de IA

**Heurística:** Algoritmo basado en reglas y física. Siempre perfecto pero no "aprende".

**Red Neuronal:** Aprende patrones de los datos. Puede generalizar pero tiene error de aproximación.

**Por qué usar ambos:** La heurística es el "profesor perfecto" que enseña a la red neuronal.

---

**¡Disfruta jugando y experimentando con IA!** 🚀
