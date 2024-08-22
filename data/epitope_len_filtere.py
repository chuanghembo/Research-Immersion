# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 15:11:06 2024

@author: Chen
"""

infile = "HLA_B_4002_unfiltered.txt"

outfile = 'HLA_B_4002.txt'

with open(infile, 'r') as f:
    for line in f:
        if len(line.split()[0]) == 9:
            with open(outfile, 'a+') as o:
                o.write(line)
            