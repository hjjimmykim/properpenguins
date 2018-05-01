#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


# Set up truncated poisson - N is the number of steps this game
lam = 7;
s = np.random.poisson(lam,1000);
s = s[s <= 10];
N = np.random.choice(s,1);



#Initialization
A1_e =  0.0;
A1_m = np.zeros(6);
A1_p = np.zeros(3);

A2_e =  0.6;
A2_m = np.zeros(6);
A2_p = np.zeros(3);


#Game
for i in range(N[0]):
    if i%2 == 0:       
        if A1_e < 0.5:
            A2e,A2_m,A2_p = combined_policy(A1_e,A1_m,A1_p);
        else:
            break
    else:
        if A2_e < 0.5:
            A1_e,A1_m,A2_p = combined_policy(A2_e,A2_m,A2_p);
        else:
            break
        
        
    
