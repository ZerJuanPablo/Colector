import pygame
import numpy as np
import math
import random

class CollectorEnv:
    def __init__(self, render=False):
        # Configuration
        self.WIDTH = 800
        self.HEIGHT = 600
        self.render_mode = render
        self.agent_size = 40
        self.agent_speed = 5
        self.BALL_RADIUS = 10
        self.NUM_BALLS = 10
        self.TRAP_RADIUS = 15
        self.NUM_TRAPS = 3
        self.SENSOR_RANGE = 600
        self.NUM_BALL_SENSORS = 8  # Directions: N, NE, E, SE, S, SW, W, NW
        self.NUM_TRAP_SENSORS = 4  # Cardinal directions: N, E, S, W
        
        # PyGame render
        if self.render_mode:
            pygame.init()
            self.window = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("RL Environment")
        
        # Action and observation definition
        self.action_space = 5  # nothing + directions
        self.observation_shape = (self.NUM_BALL_SENSORS + self.NUM_TRAP_SENSORS + 1,) # position of balls, position of traps, own position
        
        # Ball position (Fixed to make training faster)
        self.FIXED_BALLS = np.array([
            [120, 180], [110, 520],  
            [430, 90], [400, 330],   
            [750, 360], [85, 500],   
            [310, 280], [150, 430],   
            [650, 70], [580, 100]   
        ])[:self.NUM_BALLS]  
        
        self.FIXED_TRAPS = np.array([
            [240, 390],  
            [655, 100], [390, 550] 
        ])[:self.NUM_TRAPS]

        self.reset()

    def reset(self, variability=0):
        self.agent_pos = np.array([
            self.WIDTH//2 + random.randint(-50, 50),
            self.HEIGHT//2 + random.randint(-50, 50)
        ]).clip([0, 0], [self.WIDTH - self.agent_size, self.HEIGHT - self.agent_size])
        
        if variability > 0:
            offset_balls = np.random.uniform(-variability, variability, size=self.FIXED_BALLS.shape)
            offset_traps = np.random.uniform(-variability, variability, size=self.FIXED_TRAPS.shape)
            self.balls = (self.FIXED_BALLS + offset_balls).clip([self.BALL_RADIUS]*2,[self.WIDTH-self.BALL_RADIUS, self.HEIGHT-self.BALL_RADIUS])
            self.traps = (self.FIXED_TRAPS + offset_traps).clip([self.TRAP_RADIUS]*2,[self.WIDTH-self.TRAP_RADIUS, self.HEIGHT-self.TRAP_RADIUS])
        else:
            self.balls = self.FIXED_BALLS.copy()
            self.traps = self.FIXED_TRAPS.copy()

        
        self.active_balls = np.ones(self.NUM_BALLS, dtype=bool)
        
        self.score = 0
        self.hit_cooldown = 0
        
        return self._get_observation()

    def _get_ball_sensors(self):
        sensor_readings = np.zeros(self.NUM_BALL_SENSORS)
        angles = [0, 45, 90, 135, 180, 225, 270, 315]
        
        for i, angle in enumerate(angles):
            min_distance = self.SENSOR_RANGE
            rad = math.radians(angle)
            
            # Check active balls only
            for ball_idx in np.where(self.active_balls)[0]:
                bx, by = self.balls[ball_idx]
                
                # Calculate angle to ball
                dx = bx - self.agent_pos[0]
                dy = by - self.agent_pos[1]
                ball_angle = math.degrees(math.atan2(dy, dx)) % 360
                
                # Find angular difference
                angle_diff = min(abs(ball_angle - angle), 360 - abs(ball_angle - angle))
                
                if angle_diff < 22.5:  # 45 degree coverage per sensor
                    distance = math.hypot(dx, dy)
                    if distance < min_distance:
                        min_distance = distance
            
            # Normalize and invert (1 = close, 0 = beyond sensor range)
            sensor_readings[i] = 1.0 - min(min_distance / self.SENSOR_RANGE, 1.0)
        
        return sensor_readings

    def _get_trap_sensors(self):
        sensor_readings = np.zeros(self.NUM_TRAP_SENSORS)
        directions = [
            (0, -1),   # North
            (1, 0),    # East
            (0, 1),    # South
            (-1, 0)    # West
        ]
        
        for i, (dx_dir, dy_dir) in enumerate(directions):
            min_distance = self.SENSOR_RANGE
            agent_x, agent_y = self.agent_pos
            
            for step in range(1, self.SENSOR_RANGE // self.agent_speed):
                x = agent_x + dx_dir * step * self.agent_speed
                y = agent_y + dy_dir * step * self.agent_speed
                
                # Check against all traps
                for trap in self.traps:
                    tx, ty = trap
                    if abs(x - tx) < self.TRAP_RADIUS and abs(y - ty) < self.TRAP_RADIUS:
                        distance = math.hypot(x - agent_x, y - agent_y)
                        if distance < min_distance:
                            min_distance = distance
                            break
            
            sensor_readings[i] = 1.0 - (min_distance / self.SENSOR_RANGE)
        
        return sensor_readings

    def _get_observation(self):
        # Get sensor readings
        ball_sensors = self._get_ball_sensors()
        trap_sensors = self._get_trap_sensors()
        
        # Progress indicator
        progress = 1.0 - (np.sum(self.active_balls) / self.NUM_BALLS)
        
        return np.concatenate([
            ball_sensors,
            trap_sensors,
            [progress]
        ])

    def _check_collision(self, circle_pos, radius):
        closest_x = max(self.agent_pos[0], min(circle_pos[0], self.agent_pos[0] + self.agent_size))
        closest_y = max(self.agent_pos[1], min(circle_pos[1], self.agent_pos[1] + self.agent_size))
        distance = math.hypot(closest_x - circle_pos[0], closest_y - circle_pos[1])
        return distance <= radius

    def step(self, action):
        # Record state before moving
        prev_agent_pos = np.copy(self.agent_pos)
        
        # Calculate minimum distance to active balls before moving
        if np.any(self.active_balls):
            active_ball_positions = self.balls[self.active_balls]
            prev_distance = np.min(np.linalg.norm(active_ball_positions - prev_agent_pos, axis=1))
        else:
            prev_distance = 0
        
        # Apply small cost per step to encourage efficiency
        reward = -0.5  # Small penalty for each step
        done = False
        
        # Process movement based on action
        if action == 1:  # left
            self.agent_pos[0] = max(0, self.agent_pos[0] - self.agent_speed)
        elif action == 2:  # right
            self.agent_pos[0] = min(self.WIDTH - self.agent_size, self.agent_pos[0] + self.agent_speed)
        elif action == 3:  # up
            self.agent_pos[1] = max(0, self.agent_pos[1] - self.agent_speed)
        elif action == 4:  # down
            self.agent_pos[1] = min(self.HEIGHT - self.agent_size, self.agent_pos[1] + self.agent_speed)
        
        # Add a small penalty if the agent didn't move (discourage staying still)
        if np.array_equal(self.agent_pos, prev_agent_pos) and action != 0:
            reward -= 0.3  # Penalty for trying to move but hitting a wall
        
        # Calculate new distances after moving
        if np.any(self.active_balls):
            active_ball_positions = self.balls[self.active_balls]
            new_distance = np.min(np.linalg.norm(active_ball_positions - self.agent_pos, axis=1))
            
            # Distance-based shaping reward
            k = 0.08  # Reasonably strong shaping factor
            distance_improvement = prev_distance - new_distance
            
            # Only give positive reward for actual improvement
            if distance_improvement > 0:
                reward += distance_improvement * k
        else:
            new_distance = 0
            
        # Count balls collected this step
        balls_collected_this_step = 0
        
        # Reward for collecting balls with increasing returns
        for i in np.where(self.active_balls)[0]:
            if self._check_collision(self.balls[i], self.BALL_RADIUS):
                self.active_balls[i] = False
                balls_collected_this_step += 1
                self.score += 1
        
        # Progressive reward: more points for each successive ball collected
        # This encourages collecting all balls, not just the closest ones
        if balls_collected_this_step > 0:
            # Calculate how many balls were already collected before this step
            total_balls = len(self.active_balls)
            balls_collected_before = total_balls - np.sum(self.active_balls) - balls_collected_this_step
            
            # Progressive reward formula: base_reward * (1 + 0.5 * balls_already_collected)
            for i in range(balls_collected_this_step):
                collected_ball_number = balls_collected_before + i + 1
                ball_reward = 12.0 * (1 + 0.8 * (collected_ball_number - 1))
                reward += ball_reward
        
        # Big bonus for collecting all balls
        if np.sum(self.active_balls) == 0 and balls_collected_this_step > 0:
            reward += 200.0  # Significant bonus for collecting all balls
        
        # Penalty for hitting traps
        if self.hit_cooldown <= 0:
            for trap in self.traps:
                if self._check_collision(trap, self.TRAP_RADIUS):
                    reward -= 50
                    print("trapped")
                    self.hit_cooldown = 30
                    break
        else:
            self.hit_cooldown -= 1
            
        return self._get_observation(), float(reward), done, {}

    def render(self):
        self.window.fill((255, 255, 255))
        
        # Balls
        for i, ball in enumerate(self.balls):
            if self.active_balls[i]: 
                pygame.draw.circle(self.window, (0, 255, 0), ball, self.BALL_RADIUS)
        
        # Traps
        for trap in self.traps:
            pygame.draw.circle(self.window, (128, 0, 128), trap, self.TRAP_RADIUS)
        
        # Agent
        agent_color = (139, 0, 0) if self.hit_cooldown > 0 else (255, 0, 0)
        pygame.draw.rect(self.window, agent_color, 
                        (self.agent_pos[0], self.agent_pos[1], self.agent_size, self.agent_size))
        
        # Score (grabbed balls)
        font = pygame.font.Font(None, 36)
        text = font.render(f'Score: {self.score}', True, (0, 0, 0))
        self.window.blit(text, (10, 10))
        
        pygame.display.update()
        pygame.time.Clock().tick(60)

    def close(self):
        if self.render_mode:
            pygame.quit()