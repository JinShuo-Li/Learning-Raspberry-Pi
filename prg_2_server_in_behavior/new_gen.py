#This is an Alife simulator
#The author is Jinshuo Li

import random
import math
import numpy as np
import tkinter as tk

#helper functions
def rand_num(peak=None, left=-1, right=1):
    if peak is None:
        return random.uniform(left, right)
    else:
        m = (peak - left) / (right - left)
        # concentration parameter (larger -> sharper peak)
        k = 6.0
        alpha = 1.0 + m * (k - 2.0)
        beta = 1.0 + (1.0 - m) * (k - 2.0)
        alpha = max(alpha, 1e-6)
        beta = max(beta, 1e-6)
        x = np.random.beta(alpha, beta)
        return left + x * (right - left)
    
