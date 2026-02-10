import random
import math
import numpy as np
import tkinter as tk
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# --- Configuration & Hyperparameters ---
MAP_WIDTH = 1200
MAP_HEIGHT = 900
GRID_SIZE = 50  # Spatial hashing grid size

# Ecosystem Constants
INITIAL_HERBIVORES = 50
INITIAL_CARNIVORES = 10
INITIAL_FOOD = 100
FOOD_RESPAWN_RATE = 0.3
FOOD_ENERGY_VALUE = 50.0

# Pheromone System
PHEROMONE_GRID_SCALE = 20
PHEROMONE_DECAY = 0.985
PHEROMONE_EMIT_COST = 0.15

# Physics & Metabolism
MAX_LIFESPAN = 2500
BASAL_METABOLISM = 0.05

# --- Advanced Math Helpers (Toroidal) ---
def get_toroidal_delta(p1, p2, w, h):
    """Calculates vector from p1 to p2 on a torus (shortest path)."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx > w / 2: dx -= w
    elif dx < -w / 2: dx += w
    if dy > h / 2: dy -= h
    elif dy < -h / 2: dy += h
    return np.array([dx, dy])

def normalize_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

# --- Spatial Partitioning (High Performance) ---
class SpatialGrid:
    def __init__(self, width, height, cell_size):
        self.cell_size = cell_size
        self.cols = width // cell_size + 1
        self.rows = height // cell_size + 1
        self.grid = {}

    def clear(self):
        self.grid = {}

    def add(self, obj, x, y):
        cx, cy = int(x // self.cell_size), int(y // self.cell_size)
        if (cx, cy) not in self.grid: self.grid[(cx, cy)] = []
        self.grid[(cx, cy)].append(obj)

    def get_nearby(self, x, y, radius_cells=1):
        cx, cy = int(x // self.cell_size), int(y // self.cell_size)
        nearby = []
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                nx, ny = (cx + dx) % self.cols, (cy + dy) % self.rows
                if (nx, ny) in self.grid:
                    nearby.extend(self.grid[(nx, ny)])
        return nearby

# --- The Brain: Recurrent & Hebbian Learning ---
class Brain:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Memory state (Recurrent Loop)
        self.memory = np.zeros(output_size)

        # Weights: Input+Memory -> Hidden
        self.w_ih = np.random.uniform(-0.5, 0.5, (input_size + output_size, hidden_size))
        # Weights: Hidden -> Output
        self.w_ho = np.random.uniform(-0.5, 0.5, (hidden_size, output_size))
        
        # Traces for Hebbian learning
        self.last_full_input = None 
        self.last_hidden = None
        self.last_output = None

    def forward(self, sensor_inputs):
        # Concatenate Sensors + Previous Memory
        self.last_full_input = np.concatenate((sensor_inputs, self.memory)).reshape(1, -1)
        
        # Layer 1: Input -> Hidden (Tanh)
        self.last_hidden = np.tanh(np.dot(self.last_full_input, self.w_ih))
        
        # Layer 2: Hidden -> Output
        raw_out = np.dot(self.last_hidden, self.w_ho)
        
        # Activation Functions
        # 0: Target Speed (Sigmoid 0-1)
        # 1: Turn Rate (Tanh -1 to 1)
        # 2: Pheromone Emit (Sigmoid 0-1)
        o0 = 1.0 / (1.0 + np.exp(-raw_out[0, 0])) 
        o1 = np.tanh(raw_out[0, 1])
        o2 = 1.0 / (1.0 + np.exp(-raw_out[0, 2]))
        
        self.last_output = np.array([[o0, o1, o2]])
        
        # Update Memory for next frame
        self.memory = self.last_output[0]
        
        return self.memory

    def learn(self, reward, learning_rate):
        """
        Modified Hebbian Learning.
        Adjusts weights based on the success (reward) of the last action.
        """
        if self.last_full_input is None: return

        # Clip reward for stability
        r = np.clip(reward, -2.0, 2.0)
        
        # 1. Output Layer Update
        delta_ho = learning_rate * r * (self.last_hidden.T @ self.last_output)
        self.w_ho += delta_ho

        # 2. Input Layer Update (Credit Assignment)
        delta_ih = learning_rate * r * (self.last_full_input.T @ self.last_hidden)
        self.w_ih += delta_ih
        
        # 3. Weight Decay (Prevent explosion)
        self.w_ho *= 0.9995
        self.w_ih *= 0.9995

    def clone(self):
        """Lamarckian Inheritance: Pass on the TRAINED weights"""
        new_b = Brain(self.input_size, self.hidden_size, self.output_size)
        new_b.w_ih = self.w_ih.copy() 
        new_b.w_ho = self.w_ho.copy()
        new_b.memory = np.zeros(self.output_size) 
        return new_b

    def mutate(self, rate, magnitude):
        """Genetic drift/mutation"""
        mask_ih = np.random.rand(*self.w_ih.shape) < rate
        self.w_ih[mask_ih] += np.random.normal(0, magnitude, np.count_nonzero(mask_ih))
        
        mask_ho = np.random.rand(*self.w_ho.shape) < rate
        self.w_ho[mask_ho] += np.random.normal(0, magnitude, np.count_nonzero(mask_ho))

# --- Genetics ---
@dataclass
class Genome:
    is_carnivore: bool
    learning_rate: float = 0.1
    detect_radius: float = 100.0
    max_speed: float = 3.0
    size: float = 5.0
    r: int = 100
    g: int = 100
    b: int = 100
    generation: int = 0

    def update_color(self):
        """
        Update color based on species and learning rate (intelligence).
        Higher Learning Rate = Brighter/More Intense Colors.
        """
        # Map learning rate (0.001 - 0.2) to brightness factor (0 - 200)
        brightness = int(np.clip(self.learning_rate * 1000, 0, 200))

        if self.is_carnivore:
            # PREDATOR: Red Base. Smart ones get Orange/Yellowish centers.
            self.r = 255
            self.g = brightness // 2 
            self.b = brightness // 2 
        else:
            # PREY: Cyan/Blue Base. Smart ones get brighter Cyan/White.
            # Distinct from Green Food.
            self.r = brightness // 2
            self.g = 100 + (brightness // 2)
            self.b = 255

    def mutate(self):
        # Mutate Learning Rate (Meta-Learning)
        if random.random() < 0.25:
            self.learning_rate *= random.uniform(0.8, 1.25)
            self.learning_rate = np.clip(self.learning_rate, 0.001, 0.3)

        # Mutate Physical Stats
        if random.random() < 0.1: self.detect_radius = np.clip(self.detect_radius + random.uniform(-10, 10), 30, 250)
        if random.random() < 0.1: self.max_speed = np.clip(self.max_speed + random.uniform(-0.5, 0.5), 1.0, 9.0)
        
        self.update_color()

# --- Organism Entity ---
class Organism:
    def __init__(self, genome: Genome, x, y, brain=None):
        self.genome = genome
        self.genome.update_color() # Ensure color is correct on init
        self.pos = np.array([float(x), float(y)])
        self.angle = random.uniform(-math.pi, math.pi)
        self.energy = 100.0
        self.age = 0
        self.dead = False
        
        # Inputs: 
        # Bias, Energy, Smell, TargetDist, TargetAngle, ThreatDist, ThreatAngle
        SENSOR_COUNT = 7 
        self.brain = brain if brain else Brain(SENSOR_COUNT, 12, 3)
        
        # Tracking for Reward System
        self.prev_target_dist = 9999.0

    def get_sensors(self, food_grid, agent_grid, pheromone_map):
        # 1. Pheromone Sensor
        px = int(np.clip(self.pos[0] // PHEROMONE_GRID_SCALE, 0, pheromone_map.shape[0]-1))
        py = int(np.clip(self.pos[1] // PHEROMONE_GRID_SCALE, 0, pheromone_map.shape[1]-1))
        smell = pheromone_map[px, py]

        # 2. Vision Scanning
        closest_target_dist = self.genome.detect_radius
        closest_target_angle = 0
        found_target = False

        closest_threat_dist = self.genome.detect_radius
        closest_threat_angle = 0
        
        # Scan Targets
        if not self.genome.is_carnivore:
            # Herbivore: Look for Food (Green blocks)
            nearby_food = food_grid.get_nearby(self.pos[0], self.pos[1], radius_cells=2)
            for fx, fy in nearby_food:
                delta = get_toroidal_delta(self.pos, [fx, fy], MAP_WIDTH, MAP_HEIGHT)
                dist = np.linalg.norm(delta)
                if dist < closest_target_dist:
                    closest_target_dist = dist
                    closest_target_angle = normalize_angle(math.atan2(delta[1], delta[0]) - self.angle)
                    found_target = True
        else:
            # Carnivore: Look for Herbivores
            nearby_agents = agent_grid.get_nearby(self.pos[0], self.pos[1], radius_cells=2)
            for ag in nearby_agents:
                if ag is self or ag.genome.is_carnivore: continue 
                delta = get_toroidal_delta(self.pos, ag.pos, MAP_WIDTH, MAP_HEIGHT)
                dist = np.linalg.norm(delta)
                if dist < closest_target_dist:
                    closest_target_dist = dist
                    closest_target_angle = normalize_angle(math.atan2(delta[1], delta[0]) - self.angle)
                    found_target = True

        # Scan Threats
        nearby_agents = agent_grid.get_nearby(self.pos[0], self.pos[1], radius_cells=2)
        for ag in nearby_agents:
            if ag is self: continue
            is_threat = False
            # Herbs fear Carns
            if not self.genome.is_carnivore and ag.genome.is_carnivore: is_threat = True
            # Carns fear Carns (Competition)
            if self.genome.is_carnivore and ag.genome.is_carnivore: is_threat = True 
            
            if is_threat:
                delta = get_toroidal_delta(self.pos, ag.pos, MAP_WIDTH, MAP_HEIGHT)
                dist = np.linalg.norm(delta)
                if dist < closest_threat_dist:
                    closest_threat_dist = dist
                    closest_threat_angle = normalize_angle(math.atan2(delta[1], delta[0]) - self.angle)

        # Normalize Inputs
        inputs = np.array([
            1.0, # Bias
            self.energy / 100.0,
            smell,
            (closest_target_dist / self.genome.detect_radius) if found_target else 1.0,
            closest_target_angle / math.pi,
            closest_threat_dist / self.genome.detect_radius,
            closest_threat_angle / math.pi
        ])
        
        return inputs, closest_target_dist, found_target

    def update(self, food_grid, agent_grid, pheromone_map):
        self.age += 1
        
        # Sense
        inputs, dist_to_target, found_target = self.get_sensors(food_grid, agent_grid, pheromone_map)
        
        # Think
        outputs = self.brain.forward(inputs)
        
        target_speed = (outputs[0] + 1) / 2 * self.genome.max_speed
        turn_rate = outputs[1] * 0.35
        emit_pheromone = outputs[2]
        
        # Act
        self.angle += turn_rate
        vel = np.array([math.cos(self.angle), math.sin(self.angle)]) * target_speed
        self.pos += vel
        
        # Toroidal Wrap
        self.pos[0] %= MAP_WIDTH
        self.pos[1] %= MAP_HEIGHT

        # Pheromone Emission
        if emit_pheromone > 0.5:
            self.energy -= PHEROMONE_EMIT_COST
            px, py = int(self.pos[0] // PHEROMONE_GRID_SCALE), int(self.pos[1] // PHEROMONE_GRID_SCALE)
            if 0 <= px < pheromone_map.shape[0] and 0 <= py < pheromone_map.shape[1]:
                # Herbivores emit different smell intensity/value conceptually, 
                # but for simplicity here we use one map. 
                # Smart agents learn if this smell means "food nearby" or "danger".
                pheromone_map[px, py] = min(1.0, pheromone_map[px, py] + 0.1)

        # Metabolism
        move_cost = (target_speed**2) * 0.004
        basal_cost = (self.genome.size**3) * 0.0004 + BASAL_METABOLISM
        self.energy -= (move_cost + basal_cost)

        # --- REINFORCEMENT LEARNING ---
        reward = -0.01 # Time penalty
        
        # Shaping Reward
        if found_target:
            if dist_to_target < self.prev_target_dist:
                reward += 0.03 # Closer to goal
            else:
                reward -= 0.03 # Moving away
            self.prev_target_dist = dist_to_target
        else:
            reward -= 0.05 # Lost

        self.brain.learn(reward, self.genome.learning_rate)

        if self.energy <= 0 or self.age > MAX_LIFESPAN:
            self.dead = True

    def interact(self, food_list, nearby_agents):
        eaten_count = 0
        if not self.genome.is_carnivore:
            # Herbivore Logic
            for i in range(len(food_list) - 1, -1, -1):
                delta = get_toroidal_delta(self.pos, food_list[i], MAP_WIDTH, MAP_HEIGHT)
                if np.dot(delta, delta) < (self.genome.size + 4)**2:
                    self.energy += FOOD_ENERGY_VALUE
                    food_list.pop(i)
                    eaten_count += 1
                    self.brain.learn(3.0, self.genome.learning_rate) # Big Reward
        else:
            # Carnivore Logic
            for prey in nearby_agents:
                if prey is self or prey.genome.is_carnivore or prey.dead: continue
                delta = get_toroidal_delta(self.pos, prey.pos, MAP_WIDTH, MAP_HEIGHT)
                if np.dot(delta, delta) < (self.genome.size + prey.genome.size)**2:
                    self.energy += 90.0
                    prey.dead = True 
                    eaten_count += 1
                    self.brain.learn(5.0, self.genome.learning_rate) # Huge Reward
        
        self.energy = min(self.energy, 250.0)
        return eaten_count

    def reproduce(self):
        threshold = 160 if self.genome.is_carnivore else 140
        cost = 80 if self.genome.is_carnivore else 60
        
        if self.energy > threshold:
            self.energy -= cost
            
            # Create Genome
            child_g = Genome(
                is_carnivore=self.genome.is_carnivore,
                learning_rate=self.genome.learning_rate,
                detect_radius=self.genome.detect_radius,
                max_speed=self.genome.max_speed,
                generation=self.genome.generation + 1
            )
            child_g.mutate()
            
            # Create Brain (Lamarckian Clone)
            child_b = self.brain.clone()
            child_b.mutate(0.05, 0.1)
            
            offset = np.random.uniform(-10, 10, 2)
            nx = (self.pos[0] + offset[0]) % MAP_WIDTH
            ny = (self.pos[1] + offset[1]) % MAP_HEIGHT
            
            return Organism(child_g, nx, ny, child_b)
        return None

# --- Main Simulation Logic ---
class World:
    def __init__(self):
        self.organisms = []
        self.food = []
        self.pheromones = np.zeros((MAP_WIDTH // PHEROMONE_GRID_SCALE, MAP_HEIGHT // PHEROMONE_GRID_SCALE))
        
        # Spawning Herbivores
        for _ in range(INITIAL_HERBIVORES):
            g = Genome(is_carnivore=False, learning_rate=0.01)
            self.organisms.append(Organism(g, random.uniform(0, MAP_WIDTH), random.uniform(0, MAP_HEIGHT)))
            
        # Spawning Carnivores
        for _ in range(INITIAL_CARNIVORES):
            g = Genome(is_carnivore=True, learning_rate=0.01, max_speed=3.5, size=8.0)
            self.organisms.append(Organism(g, random.uniform(0, MAP_WIDTH), random.uniform(0, MAP_HEIGHT)))
            
        self.food = [[random.uniform(0, MAP_WIDTH), random.uniform(0, MAP_HEIGHT)] for _ in range(INITIAL_FOOD)]
        
        self.spatial_agents = SpatialGrid(MAP_WIDTH, MAP_HEIGHT, GRID_SIZE)
        self.spatial_food = SpatialGrid(MAP_WIDTH, MAP_HEIGHT, GRID_SIZE)
        
        self.stats = {"herb": 0, "carn": 0, "gen": 0}

    def update(self):
        # 1. Pheromone Decay
        self.pheromones *= PHEROMONE_DECAY
        
        # 2. Rebuild Grids
        self.spatial_agents.clear()
        self.spatial_food.clear()
        for org in self.organisms:
            self.spatial_agents.add(org, org.pos[0], org.pos[1])
        for fx, fy in self.food:
            self.spatial_food.add([fx, fy], fx, fy) 
            
        # 3. Update Organisms
        new_babies = []
        survivors = []
        random.shuffle(self.organisms)
        
        for org in self.organisms:
            if org.dead: continue
            
            org.update(self.spatial_food, self.spatial_agents, self.pheromones)
            
            # Interaction
            nearby_agents = self.spatial_agents.get_nearby(org.pos[0], org.pos[1], radius_cells=1)
            org.interact(self.food, nearby_agents)
            
            if not org.dead:
                baby = org.reproduce()
                if baby: new_babies.append(baby)
                survivors.append(org)
        
        self.organisms = survivors + new_babies
        
        # 4. Food Respawn
        if len(self.food) < INITIAL_FOOD * 1.5 and random.random() < FOOD_RESPAWN_RATE:
            self.food.append([random.uniform(0, MAP_WIDTH), random.uniform(0, MAP_HEIGHT)])
            
        # 5. The Ark (Repopulation)
        herbs = [o for o in self.organisms if not o.genome.is_carnivore]
        carns = [o for o in self.organisms if o.genome.is_carnivore]
        
        if len(herbs) < 8:
            self.organisms.append(Organism(Genome(is_carnivore=False), random.uniform(0, MAP_WIDTH), random.uniform(0, MAP_HEIGHT)))
        if len(carns) < 3 and random.random() < 0.05:
            self.organisms.append(Organism(Genome(is_carnivore=True, max_speed=3.5), random.uniform(0, MAP_WIDTH), random.uniform(0, MAP_HEIGHT)))

        self.stats["herb"] = len(herbs)
        self.stats["carn"] = len(carns)
        if self.organisms:
            self.stats["gen"] = max(o.genome.generation for o in self.organisms)

# --- Visualization ---
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Ecosystem: Red(Carnivore) vs Cyan(Herbivore) vs Green(Food)")
        self.canvas = tk.Canvas(root, width=MAP_WIDTH, height=MAP_HEIGHT, bg="#050505")
        self.canvas.pack()
        self.label = tk.Label(root, text="Initializing...", fg="white", bg="#111", font=("Consolas", 10))
        self.label.pack(fill="x")
        
        self.world = World()
        self.is_running = True
        self.loop()

    def draw(self):
        self.canvas.delete("all")
        
        # Draw Pheromones (Purple Heatmap)
        rows, cols = np.where(self.world.pheromones > 0.15)
        scale = PHEROMONE_GRID_SCALE
        for r, c in zip(rows, cols):
            alpha = int(self.world.pheromones[r, c] * 120)
            if alpha > 0:
                # Purple: #aa00aa
                color = f'#{alpha:02x}00{alpha:02x}' 
                self.canvas.create_rectangle(r*scale, c*scale, (r+1)*scale, (c+1)*scale, fill=color, outline="")

        # Draw Food (Bright Green)
        for fx, fy in self.world.food:
            self.canvas.create_rectangle(fx-3, fy-3, fx+3, fy+3, fill="#00FF00", outline="")

        # Draw Organisms
        for org in self.world.organisms:
            r = org.genome.size
            # Get pre-calculated distinct color
            color = "#%02x%02x%02x" % (org.genome.r, org.genome.g, org.genome.b)
            
            # Draw Body
            self.canvas.create_oval(org.pos[0]-r, org.pos[1]-r, org.pos[0]+r, org.pos[1]+r, fill=color, outline="")
            
            # Draw Intelligence Aura (White border for smart ones)
            if org.genome.learning_rate > 0.1:
                 self.canvas.create_oval(org.pos[0]-r-1, org.pos[1]-r-1, org.pos[0]+r+1, org.pos[1]+r+1, outline="white")

            # Eye Direction
            ex = org.pos[0] + math.cos(org.angle) * r
            ey = org.pos[1] + math.sin(org.angle) * r
            self.canvas.create_line(org.pos[0], org.pos[1], ex, ey, fill="white", width=2)

        self.label.config(text=f"Prey (Cyan): {self.world.stats['herb']} | Pred (Red): {self.world.stats['carn']} | Gen: {self.world.stats['gen']}")

    def loop(self):
        if self.is_running:
            self.world.update()
            self.draw()
            self.root.after(20, self.loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()