"""
Script de entrenamiento para BrickBreakerAI
Entrena una red neuronal GRU usando datos de la heurística perfecta
"""

import numpy as np
from ai_player import BrickBreakerAI, record_training_data_with_heuristic


def train_model(use_collected_data=True):
    """Entrenar modelo solo con datos recolectados del juego"""
    print("=" * 60)
    print("🧠 ENTRENAMIENTO RED NEURONAL GRU - BRICK BREAKER AI")
    print("=" * 60)
    
    # Cargar datos recolectados
    print("\n📂 Cargando datos recolectados...")
    try:
        data = np.load("data/training_data.npy")
        print(f"✅ {len(data)} muestras encontradas")
    except FileNotFoundError:
        print("❌ ERROR: No se encontró 'data/training_data.npy'")
        print("   Primero debes jugar y recolectar datos:")
        print("   1. Ejecuta: python3 brick_breaker_tensorflow.py")
        print("   2. Responde 's' para recolectar datos")
        print("   3. Juega en modo heurístico (tecla H)")
        print("   4. Presiona Q para guardar")
        return
    
    if len(data) < 1000:
        print(f"⚠️  ADVERTENCIA: Solo tienes {len(data)} muestras")
        print("   Se recomienda mínimo 1000-2000 para buen entrenamiento")
        resp = input("   ¿Continuar de todos modos? (s/n): ")
        if resp.lower() not in ['s', 'si', 'yes', 'y']:
            print("❌ Entrenamiento cancelado")
            return
    
    # Crear IA
    print("\n📦 Creando modelo GRU...")
    ai = BrickBreakerAI(1280, 720, load_pretrained=False)
    ai.training_mode = True
    
    # Entrenar con datos reales
    print(f"\n🎓 Entrenando con {len(data)} ejemplos REALES...")
    ai.train_from_collected_data(data)
    
    # Guardar
    print("\n💾 Guardando modelo...")
    ai.save_model()
    
    # Evaluar
    print("\n📊 Evaluando modelo...")
    evaluate_model(ai)
    
    print("\n" + "=" * 60)
    print("✅ LISTO! Ejecuta: python3 brick_breaker_tensorflow.py")
    print("   Cambia use_neural = True en el juego para usar la red")
    print("=" * 60)


def evaluate_model(ai, num_tests=100):
    """Evaluar qué tan bien predice"""
    print(f"Probando {num_tests} casos...")
    
    errors = []
    
    for _ in range(num_tests):
        # Caso aleatorio
        ball_x = np.random.uniform(50, ai.frame_width - 50)
        ball_y = np.random.uniform(100, 400)
        dx = np.random.uniform(-6, 6)
        dy = np.random.uniform(1, 6)
        
        # Reiniciar historial
        ai.reset_history()
        
        # Comparar heurística vs red neuronal
        heuristic = ai.heuristic.get_target_position(ball_x, ball_y, dx, dy)
        neural = ai.get_target_position(ball_x, ball_y, dx, dy)
        
        error = abs(neural - heuristic)
        errors.append(error)
    
    avg_error = np.mean(errors)
    max_error = np.max(errors)
    
    print(f"\n📈 Resultados:")
    print(f"  Error promedio: {avg_error:.1f} píxeles")
    print(f"  Error máximo:   {max_error:.1f} píxeles")
    
    if avg_error < 50:
        print(f"  ✅ Excelente!")
    elif avg_error < 100:
        print(f"  ⚠️ Aceptable")
    else:
        print(f"  ❌ Necesita más entrenamiento")


if __name__ == "__main__":
    print("\n🎮 ENTRENAMIENTO - BRICK BREAKER AI")
    print("\n📊 Este script entrena la red neuronal con los datos")
    print("   que recolectaste jugando en brick_breaker_tensorflow.py\n")
    
    input("Presiona ENTER para entrenar con tus datos recolectados...")
    train_model()
