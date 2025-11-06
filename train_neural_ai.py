"""
Script SIMPLE de entrenamiento para la Red Neuronal
"""

import numpy as np
from ai_player import BrickBreakerAI

def train_model(num_samples=10000):
    """Entrenar modelo desde cero"""
    print("=" * 60)
    print("🧠 ENTRENAMIENTO SIMPLE - BRICK BREAKER AI")
    print("=" * 60)
    
    # Crear IA
    print("\n📦 Creando modelo...")
    ai = BrickBreakerAI(1280, 720, load_pretrained=False)
    ai.training_mode = True
    
    # Entrenar
    print(f"\n🎓 Entrenando con {num_samples} ejemplos...")
    ai.train_from_heuristic(num_samples=num_samples)
    
    # Guardar
    print("\n💾 Guardando modelo...")
    ai.save_model('brick_breaker_model.keras')
    
    # Evaluar modelo
    print("\n📊 Evaluando modelo...")
    evaluate_model(ai, num_tests=100)
    
    # Evaluar
    print("\n📊 Evaluando modelo...")
    evaluate_model(ai)
    
    print("\n" + "=" * 60)
    print("✅ LISTO! Ejecuta: python3 brick_breaker_tensorflow.py")
    print("=" * 60)

def evaluate_model(ai, num_tests=100):
    """Evaluar qué tan bien predice"""
    print(f"Probando {num_tests} casos...")
    
    errors = []
    
    for _ in range(num_tests):
        # Caso aleatorio
        ball_x = np.random.uniform(50, ai.frame_width - 50)
        ball_y = np.random.uniform(100, 400)
        ball_dx = np.random.uniform(-6, 6)
        ball_dy = np.random.uniform(1, 6)
        
        # Reiniciar historial para independizar cada evaluación
        if hasattr(ai, 'reset_history'):
            ai.reset_history()
        
        # Comparar heurística vs red neuronal
        heuristic = ai.predict_paddle_position(ball_x, ball_y, ball_dx, ball_dy)
        neural = ai.get_target_position(ball_x, ball_y, ball_dx, ball_dy)
        
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
    print("\n🎮 ENTRENAMIENTO MEJORADO - BRICK BREAKER AI")
    print("\n⚠️ NOTA: La heurística ya funciona perfectamente sin entrenamiento!")
    print("   Solo entrena la red neuronal si quieres experimentar.\n")
    print("Opciones:")
    print("  1. Rápido (10,000 ejemplos - ~2 min)")
    print("  2. Normal (20,000 ejemplos - ~4 min)")
    print("  3. Completo (30,000 ejemplos - ~6 min) ⭐ RECOMENDADO")
    print("  4. Máximo (50,000 ejemplos - ~10 min)")
    
    choice = input("\nSelecciona (1-4): ").strip()
    
    if choice == "1":
        train_model(10000)
    elif choice == "2":
        train_model(20000)
    elif choice == "3":
        train_model(30000)
    elif choice == "4":
        train_model(50000)
    else:
        print("❌ Opción inválida")
