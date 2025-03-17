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
        # PyGame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Key inputs
        keys = pygame.key.get_pressed()
        action = 0  
        if keys[pygame.K_LEFT]:
            action = 1  
        elif keys[pygame.K_RIGHT]:
            action = 2  
        elif keys[pygame.K_UP]:
            action = 3  
        elif keys[pygame.K_DOWN]:
            action = 4  

        state, reward, done, info = env.step(action)
        acc_reward += reward
        print(f"Reward: {reward} | Total reward: {acc_reward}")
        print(f"Total reward: {acc_reward}")

        env.render()
        
        # (60 FPS)
        clock.tick(60)
        
        # reset environment
        if done:
            state = env.reset()
    
    env.close()

if __name__ == '__main__':
    main()
