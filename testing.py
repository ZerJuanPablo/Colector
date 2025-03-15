import pygame
import environment
import time

def main():
    env = environment.CollectorEnv(render=True)
    state = env.reset()
    acc_reward = 0
    
    running = True
    clock = pygame.time.Clock()
    
    while running:
        # Procesar eventos de Pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Leer el estado de las teclas
        keys = pygame.key.get_pressed()
        action = 0  # Acción por defecto: no moverse
        if keys[pygame.K_LEFT]:
            action = 1  # Izquierda
        elif keys[pygame.K_RIGHT]:
            action = 2  # Derecha
        elif keys[pygame.K_UP]:
            action = 3  # Arriba
        elif keys[pygame.K_DOWN]:
            action = 4  # Abajo

        # Ejecutar un paso en el environment
        state, reward, done, info = env.step(action)
        acc_reward += reward
        print(f"Reward: {reward} | Total reward: {acc_reward}")
        print(f"Total reward: {acc_reward}")

        # Renderizar el environment
        env.render()
        
        # Limitar la velocidad de actualización (60 FPS)
        clock.tick(60)
        
        # Si el episodio termina, reiniciar el environment
        if done:
            state = env.reset()
    
    env.close()

if __name__ == '__main__':
    main()
