import tkinter as tk
import random
import math
import numpy as np
from math import sin, cos, pi, atan2, sqrt

# ==========================================
# 1. 全局配置 (Configuration)
# ==========================================
CONFIG = {
    "Width": 1200,
    "Height": 800,
    "Tick": 0.2,             # 物理模拟的时间步长
    "Grid_Size": 40,         # 空间哈希网格大小
    "FPS": 60,               # 目标帧率

    # --- 进化参数 ---
    "Mutation_Rate": 0.15,
    "Mutation_Strength": 0.3,
    "Instinct_Strength": 3.0, # 初始本能强度

    # --- 植物 ---
    "Plant_Spawn_Rate": 10,
    "Plant_Energy": 150.0,
    "Plant_Max": 700,

    # --- 食草动物 (Herbivore) ---
    "Herb_Init": 80,
    "Herb_Energy_Init": 500,
    "Herb_Energy_Max": 1200,
    "Herb_Base_Cost": 0.8,
    "Herb_Repro_Thresh": 900,
    "Herb_Repro_Cost": 500,
    "Herb_Repro_Cool": 5.0,
    "Herb_Max_Age": 1500,

    # --- 食肉动物 (Carnivore) ---
    "Carn_Init": 15,
    "Carn_Energy_Init": 1000,
    "Carn_Energy_Max": 2500,
    "Carn_Base_Cost": 1.5,
    "Carn_Repro_Thresh": 1800,
    "Carn_Repro_Cost": 900,
    "Carn_Repro_Cool": 8.0,
    "Carn_Max_Age": 2000,
    "Carn_Eat_Gain": 0.7,     # 进食转化率
}

NN_INPUTS = 10
NN_HIDDEN = 12
NN_OUTPUTS = 2

# ==========================================
# 2. 核心数学工具 (Math Utils)
# ==========================================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

# ==========================================
# 3. 基因组 (Genome)
# ==========================================

class Genome:
    __slots__ = ('stats', 'w1', 'b1', 'w2', 'b2')

    def __init__(self, parent=None, instinct_type=None):
        # 初始化空白基因
        self.stats = np.random.rand(6)
        self.w1 = np.random.randn(NN_INPUTS, NN_HIDDEN)
        self.b1 = np.random.randn(NN_HIDDEN)
        self.w2 = np.random.randn(NN_HIDDEN, NN_OUTPUTS)
        self.b2 = np.random.randn(NN_OUTPUTS)

        if parent:
            self.inherit_from(parent)
        elif instinct_type:
            self.apply_instincts(instinct_type)

    def inherit_from(self, parent):
        rate = CONFIG["Mutation_Rate"]
        power = CONFIG["Mutation_Strength"]
        
        # 生理变异
        self.stats = np.clip(parent.stats + np.random.normal(0, 0.05, size=6), 0.01, 1.0)
        
        # 神经网络变异 (稀疏变异)
        mask_w1 = np.random.rand(*self.w1.shape) < rate
        self.w1 = parent.w1 + mask_w1 * np.random.normal(0, power, self.w1.shape)
        
        mask_w2 = np.random.rand(*self.w2.shape) < rate
        self.w2 = parent.w2 + mask_w2 * np.random.normal(0, power, self.w2.shape)
        
        self.b1 = parent.b1.copy()
        self.b2 = parent.b2.copy()

    def apply_instincts(self, type_id):
        """为初始生物注入本能，避免完全随机的愚蠢行为"""
        s = CONFIG["Instinct_Strength"]
        # 输入: 0:FoodDist, 1:FoodAngle, 2:FoeDist, 3:FoeAngle
        # 输出: 0:Thrust, 1:Turn
        
        if type_id == 1: # Herbivore
            # 看到食物(Angle) -> 转向(Turn)
            self.w1[1, 0] = s
            self.w2[0, 1] = s
            # 看到天敌(Angle) -> 反向转向
            self.w1[3, 1] = s
            self.w2[1, 1] = -s
            # 看到天敌(Dist) -> 加速逃跑 (Dist越小值越小，负负得正)
            self.w1[2, 2] = -s * 2
            self.w2[2, 0] = s * 2

        elif type_id == 2: # Carnivore
            # 看到猎物(Angle) -> 转向
            self.w1[1, 0] = s
            self.w2[0, 1] = s
            # 看到猎物(Dist) -> 加速冲刺
            self.w1[0, 2] = -s
            self.w2[2, 0] = s

# ==========================================
# 4. 生物实体 (Agent)
# ==========================================

class Agent:
    __slots__ = ('x', 'y', 'type', 'heading', 'alive', 'age', 'cooldown', 
                 'genome', 'size', 'size_gene', 'max_size', 'color', 
                 'max_speed', 'sense_radius', 'energy', 'max_energy', 'id')

    def __init__(self, x, y, type_id, genome=None):
        self.x, self.y = x, y
        self.type = type_id 
        self.heading = random.uniform(-pi, pi)
        self.alive = True
        self.age = 0
        self.cooldown = 0
        self.id = id(self) # 用于渲染缓存的唯一Key
        
        # 基因处理
        self.genome = genome if genome else Genome(instinct_type=type_id)
        
        # 表现型解码
        if self.type == 0: # Plant
            self.size = 3
            self.color = "#2ecc71"
            self.energy = CONFIG["Plant_Energy"]
            # 填充哑数据防止报错
            self.max_speed = 0
            self.sense_radius = 0
            self.max_energy = 0
            self.max_size = 3
            self.size_gene = 0
        else:
            base_size = 4 if type_id == 1 else 6
            self.size_gene = 0.5 + self.genome.stats[0] 
            self.max_size = base_size * self.size_gene * 2.0
            self.size = base_size * self.size_gene # 出生时较小
            
            base_speed = 3.0 if type_id == 1 else 3.5
            self.max_speed = base_speed + self.genome.stats[1] * 4.0
            
            self.sense_radius = 40.0 + self.genome.stats[2] * 120.0
            
            self.max_energy = CONFIG["Herb_Energy_Max"] if type_id==1 else CONFIG["Carn_Energy_Max"]
            self.energy = CONFIG["Herb_Energy_Init"] if type_id==1 else CONFIG["Carn_Energy_Init"]
            
            # 颜色生成
            if type_id == 1:
                r, g, b = int(50*self.genome.stats[3]), int(150+100*self.genome.stats[4]), int(200+55*self.genome.stats[5])
                self.color = f"#{r:02x}{g:02x}{b:02x}"
            else:
                r, g, b = int(200+55*self.genome.stats[3]), int(50*self.genome.stats[4]), int(50*self.genome.stats[5])
                self.color = f"#{r:02x}{g:02x}{b:02x}"

    def update(self, env, nearby):
        if not self.alive: return
        self.age += 1
        self.cooldown -= CONFIG["Tick"]
        
        if self.type == 0:
            if self.age > 2000: self.alive = False
            return

        # 生长
        if self.size < self.max_size:
            self.size += 0.01 * CONFIG["Tick"]

        # --- 感知 (优化: 内联计算) ---
        inputs = np.zeros(NN_INPUTS)
        w, h = env.width, env.height
        
        # 寻找最近物体
        closest_food = self._find_closest(nearby['plants'] if self.type==1 else nearby['herbs'], w, h)
        closest_foe = self._find_closest(nearby['carns'] if self.type==1 else [], w, h)
        closest_mate = self._find_closest(nearby['herbs'] if self.type==1 else nearby['carns'], w, h, exclude_self=True)

        # 填充输入向量
        if closest_food:
            d = sqrt(closest_food[0])
            inputs[0] = min(1.0, d / self.sense_radius)
            inputs[1] = closest_food[1] / pi
        else: inputs[0] = 1.0

        if closest_foe:
            d = sqrt(closest_foe[0])
            inputs[2] = min(1.0, d / self.sense_radius)
            inputs[3] = closest_foe[1] / pi
        else: inputs[2] = 1.0

        if closest_mate:
            d = sqrt(closest_mate[0])
            inputs[4] = min(1.0, d / self.sense_radius)
            inputs[5] = closest_mate[1] / pi
        else: inputs[4] = 1.0

        inputs[6] = self.energy / self.max_energy
        inputs[7] = 0 
        inputs[8] = self.age / 1000.0
        inputs[9] = random.random()

        # --- 思考 ---
        h_layer = tanh(np.dot(inputs, self.genome.w1) + self.genome.b1)
        out = np.dot(h_layer, self.genome.w2) + self.genome.b2
        
        thrust = sigmoid(out[0]) 
        turn = tanh(out[1])      
        
        # --- 运动 ---
        self.heading += turn * 0.2
        speed = thrust * self.max_speed
        tick = CONFIG["Tick"]
        
        dx = cos(self.heading) * speed * tick
        dy = sin(self.heading) * speed * tick
        
        self.x = (self.x + dx) % w
        self.y = (self.y + dy) % h

        # --- 代谢 ---
        base_cost = CONFIG["Herb_Base_Cost"] if self.type == 1 else CONFIG["Carn_Base_Cost"]
        cost = (base_cost + (speed**2 * self.size * 0.005) + (self.sense_radius * 0.002) + (self.size**2 * 0.01)) * tick
        self.energy -= cost
        
        max_age = CONFIG["Herb_Max_Age"] if self.type == 1 else CONFIG["Carn_Max_Age"]
        if self.energy <= 0 or self.age > max_age:
            self.alive = False

    def _find_closest(self, agents, w, h, exclude_self=False):
        # 性能关键路径：使用距离平方，减少开方运算
        min_dist_sq = self.sense_radius ** 2
        res = None
        my_x, my_y, my_h = self.x, self.y, self.heading
        
        for other in agents:
            if exclude_self and (other is self): continue
            
            dx = other.x - my_x
            dy = other.y - my_y
            
            # 环形世界最短路径修正
            if abs(dx) > w * 0.5: dx -= w if dx > 0 else -w
            if abs(dy) > h * 0.5: dy -= h if dy > 0 else -h
                
            d_sq = dx*dx + dy*dy
            
            if d_sq < min_dist_sq:
                min_dist_sq = d_sq
                # 只有在确定是最近的时候才计算角度
                abs_angle = atan2(dy, dx)
                rel_angle = abs_angle - my_h
                if rel_angle > pi: rel_angle -= 2*pi
                elif rel_angle < -pi: rel_angle += 2*pi
                res = (d_sq, rel_angle)
        return res

    def can_reproduce(self):
        if self.cooldown > 0: return False
        thresh = CONFIG["Herb_Repro_Thresh"] if self.type == 1 else CONFIG["Carn_Repro_Thresh"]
        return self.energy > thresh

# ==========================================
# 5. 世界与物理引擎 (World)
# ==========================================

class World:
    def __init__(self, w, h):
        self.width = w
        self.height = h
        self.plants = []
        self.herbs = []
        self.carns = []
        self.kill_events = [] 
        
        # 名人堂 (Hall of Fame) - 存储 (age, genome)
        self.hof_herbs = []
        self.hof_carns = []
        
        self.spawn_random(0, CONFIG["Plant_Max"] // 2)
        self.spawn_random(1, CONFIG["Herb_Init"])
        self.spawn_random(2, CONFIG["Carn_Init"])

    def spawn_random(self, type_id, count):
        for _ in range(count):
            a = Agent(random.uniform(0, self.width), random.uniform(0, self.height), type_id)
            if type_id == 0: self.plants.append(a)
            elif type_id == 1: self.herbs.append(a)
            elif type_id == 2: self.carns.append(a)

    def spawn_child(self, parent):
        if parent.type == 1: cost = CONFIG["Herb_Repro_Cost"]
        else: cost = CONFIG["Carn_Repro_Cost"]
        
        parent.energy -= cost
        parent.cooldown = CONFIG["Herb_Repro_Cool"] if parent.type == 1 else CONFIG["Carn_Repro_Cool"]
        
        child = Agent(parent.x, parent.y, parent.type, Genome(parent.genome))
        child.x += random.uniform(-5, 5)
        child.y += random.uniform(-5, 5)
        
        if parent.type == 1: self.herbs.append(child)
        else: self.carns.append(child)

    def update(self):
        self.kill_events.clear()
        
        # 1. 植物生长
        if len(self.plants) < CONFIG["Plant_Max"]:
            if random.random() < 0.9: 
                for _ in range(CONFIG["Plant_Spawn_Rate"]):
                    self.spawn_random(0, 1)

        # 2. 空间哈希网格构建
        grid = {}
        gs = CONFIG["Grid_Size"]
        w_grid = self.width // gs + 1
        h_grid = self.height // gs + 1
        
        all_active = self.herbs + self.carns + self.plants
        for a in all_active:
            if not a.alive: continue
            gx, gy = int(a.x // gs), int(a.y // gs)
            key = (gx, gy)
            if key not in grid: grid[key] = []
            grid[key].append(a)

        # 3. 动态生物更新
        movers = self.herbs + self.carns
        for agent in movers:
            if not agent.alive: continue
            
            # 从网格获取邻居
            gx, gy = int(agent.x // gs), int(agent.y // gs)
            nearby = {'plants':[], 'herbs':[], 'carns':[]}
            search_radius = int(agent.sense_radius // gs) + 1
            
            for i in range(-search_radius, search_radius + 1):
                for j in range(-search_radius, search_radius + 1):
                    ngx = (gx + i) % w_grid
                    ngy = (gy + j) % h_grid
                    cell = grid.get((ngx, ngy))
                    if cell:
                        for neighbor in cell:
                            if neighbor is agent: continue
                            if neighbor.type == 0: nearby['plants'].append(neighbor)
                            elif neighbor.type == 1: nearby['herbs'].append(neighbor)
                            elif neighbor.type == 2: nearby['carns'].append(neighbor)
            
            agent.update(self, nearby)
            
            # 物理碰撞 (距离平方检测)
            if agent.type == 1: # Herbivore
                for p in nearby['plants']:
                    if p.alive:
                        dx = p.x - agent.x
                        dy = p.y - agent.y
                        if abs(dx) > self.width * 0.5: dx -= self.width if dx > 0 else -self.width
                        if abs(dy) > self.height * 0.5: dy -= self.height if dy > 0 else -self.height
                        
                        dist_sq = dx*dx + dy*dy
                        r_sum = agent.size + p.size
                        if dist_sq < r_sum * r_sum:
                            agent.energy = min(agent.energy + p.energy, agent.max_energy)
                            p.alive = False
                            break 
            
            elif agent.type == 2: # Carnivore
                for h in nearby['herbs']:
                    if h.alive:
                        dx = h.x - agent.x
                        dy = h.y - agent.y
                        if abs(dx) > self.width * 0.5: dx -= self.width if dx > 0 else -self.width
                        if abs(dy) > self.height * 0.5: dy -= self.height if dy > 0 else -self.height
                        
                        dist_sq = dx*dx + dy*dy
                        r_sum = agent.size + h.size
                        if dist_sq < r_sum * r_sum:
                            gain = h.energy * CONFIG["Carn_Eat_Gain"] + 200
                            agent.energy = min(agent.energy + gain, agent.max_energy)
                            h.alive = False
                            self.kill_events.append((agent.x, agent.y, h.x, h.y))
                            break

            if agent.can_reproduce() and random.random() < 0.05:
                self.spawn_child(agent)

        # 4. 清理死者与名人堂维护 (关键修正)
        self.plants = [x for x in self.plants if x.alive]

        # 清理食草动物
        alive_herbs = []
        for h in self.herbs:
            if h.alive: alive_herbs.append(h)
            elif h.age > 500: # 只有长寿者才进名人堂
                self.hof_herbs.append((h.age, h.genome))
        
        if len(self.hof_herbs) > 20:
            self.hof_herbs.sort(key=lambda x: x[0], reverse=True)
            self.hof_herbs = self.hof_herbs[:20]
        self.herbs = alive_herbs

        # 清理食肉动物
        alive_carns = []
        for c in self.carns:
            if c.alive: alive_carns.append(c)
            elif c.age > 800:
                self.hof_carns.append((c.age, c.genome))
        
        if len(self.hof_carns) > 20:
            self.hof_carns.sort(key=lambda x: x[0], reverse=True)
            self.hof_carns = self.hof_carns[:20]
        self.carns = alive_carns

        # 5. 移民/复活机制 (精英克隆)
        if len(self.herbs) < 5:
            new_genome = None
            if self.hof_herbs:
                elite = random.choice(self.hof_herbs)
                new_genome = Genome(parent=elite[1])
            self.herbs.append(Agent(random.uniform(0, self.width), random.uniform(0, self.height), 1, new_genome))
            
        if len(self.carns) < 2:
            new_genome = None
            if self.hof_carns:
                elite = random.choice(self.hof_carns)
                new_genome = Genome(parent=elite[1])
            c = Agent(random.uniform(0, self.width), random.uniform(0, self.height), 2, new_genome)
            c.energy += 500
            self.carns.append(c)

# ==========================================
# 6. 图形界面与渲染管线 (Render)
# ==========================================

class EcoSimApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Optimized EcoSim: Hall of Fame + Object Pooling")
        self.geometry(f"{CONFIG['Width']}x{CONFIG['Height']}")
        self.resizable(False, False)
        
        self.canvas = tk.Canvas(self, width=CONFIG['Width'], height=CONFIG['Height'], bg="#1e1e1e", highlightthickness=0)
        self.canvas.pack()
        
        self.world = World(CONFIG['Width'], CONFIG['Height'])
        self.stats_label = self.canvas.create_text(10, 10, anchor="nw", fill="white", font=("Consolas", 10), text="")
        
        # 渲染缓存：Map[Agent.id] -> CanvasItemID
        self.sprites = {} 
        self.running = True
        self.loop()

    def loop(self):
        if not self.running: return
        
        self.world.update()
        
        # 只清除特效层
        self.canvas.delete("fx")
        
        # 渲染更新
        alive_ids = set()
        all_agents = self.world.plants + self.world.herbs + self.world.carns
        
        for agent in all_agents:
            aid = agent.id
            alive_ids.add(aid)
            x, y, s = agent.x, agent.y, agent.size
            
            if aid in self.sprites:
                cid = self.sprites[aid]
                if agent.type == 2: # Carnivore Triangle Update
                    ang = agent.heading
                    p1 = (x + cos(ang)*s*1.5, y + sin(ang)*s*1.5)
                    p2 = (x + cos(ang + 2.4)*s, y + sin(ang + 2.4)*s)
                    p3 = (x + cos(ang - 2.4)*s, y + sin(ang - 2.4)*s)
                    self.canvas.coords(cid, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1])
                else: # Circle Update
                    self.canvas.coords(cid, x-s, y-s, x+s, y+s)
            else:
                # Create New
                if agent.type == 0: 
                    cid = self.canvas.create_rectangle(x-s, y-s, x+s, y+s, fill=agent.color, outline="", tags="org")
                elif agent.type == 1: 
                    cid = self.canvas.create_oval(x-s, y-s, x+s, y+s, fill=agent.color, outline="", tags="org")
                else: 
                    ang = agent.heading
                    p1 = (x + cos(ang)*s*1.5, y + sin(ang)*s*1.5)
                    p2 = (x + cos(ang + 2.4)*s, y + sin(ang + 2.4)*s)
                    p3 = (x + cos(ang - 2.4)*s, y + sin(ang - 2.4)*s)
                    cid = self.canvas.create_polygon(p1, p2, p3, fill=agent.color, outline="", tags="org")
                self.sprites[aid] = cid

        # 清除无效图形
        dead_ids = [k for k in self.sprites.keys() if k not in alive_ids]
        for did in dead_ids:
            self.canvas.delete(self.sprites[did])
            del self.sprites[did]

        # 绘制击杀特效
        for ax, ay, bx, by in self.world.kill_events:
            if abs(ax-bx) < CONFIG['Width']/2 and abs(ay-by) < CONFIG['Height']/2:
                self.canvas.create_line(ax, ay, bx, by, fill="yellow", width=2, tags="fx")
                self.canvas.create_oval(bx-5, by-5, bx+5, by+5, outline="red", width=2, tags="fx")

        # 统计文本
        txt = (f"Herbs: {len(self.world.herbs)} (HoF: {len(self.world.hof_herbs)}) | "
               f"Carns: {len(self.world.carns)} (HoF: {len(self.world.hof_carns)}) | "
               f"Plants: {len(self.world.plants)}")
        self.canvas.itemconfigure(self.stats_label, text=txt)
        
        self.after(int(1000/CONFIG["FPS"]), self.loop)

if __name__ == "__main__":
    app = EcoSimApp()
    app.mainloop()