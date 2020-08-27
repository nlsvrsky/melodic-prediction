#!/usr/bin/env python

## Generate melodic predictions given learned scale degree expectations
## and MIDI test corpus

import glob
import os
import math
import numpy as np
import pandas as pd
from mido import MidiFile

# define parameters and useful data structures

DECAY = 1.6180339887 # decay constant = phi
EXP_FILE = os.path.join('../models', input("Filename of CSV with learned expectations: ")) # directory with training corpus
STEM_DIR = os.path.join('../data', input("Directory containing melodies to predict continuations for: "), '') # directory with test corpus/melodic stems
OUTPUT1 = os.path.join('../models', input("Filename to use for CSV containing predictions with repetition discounting: ")) # output file for predictions with repetition discounting
OUTPUT2 = os.path.join('../models', input("Filename to use for CSV containing predictions without repetition discounting: ")) # output file for predictions without repetition discounting

UPPER2PC = {"A" : 9, "A#": 10, "Bb" : 10, "B" : 11, "C" : 0, "C#" : 1, "Db" : 1, "D" : 2, "D#" : 3, "Eb" : 3, "E" : 4, "F" : 5, "F#" : 6, "Gb" : 6, "G" : 7, "G#" : 8, "Ab" : 8} 
LOWER2PC = {"a" : 9, "b" : 11, "c" : 0, "d" : 2, "e" : 4, "f" : 5, "g" : 7}
        

# read learned expectations from CSV

exp_df = pd.read_csv(EXP_FILE, header=None)
exp = exp_df.values


# make predictions        

predictions1_matrix = np.empty((0, 3))
predictions2_matrix = np.empty((0, 3))

act = np.empty(12) # activations with repetition discounting
act2 = np.empty(12) # activations without repetition discounting

for file_name in glob.iglob(STEM_DIR + "**/*.mid", recursive=True):
    key = None
    act.fill(0) # reset activations for current melody
    act2.fill(0)
    
    # read current melody
    for message in MidiFile(file_name):
        if hasattr(message, 'key'):
            key = UPPER2PC[message.key]
        if message.type == 'note_on' and message.velocity > 0:
            act[(message.note - key) % 12] = 0 # repetition discounting
            act = act / DECAY + exp[(message.note - key) % 12,]
            act2 = act2 / DECAY + exp[(message.note - key) % 12,]
            
    # normalize activations and save predictions in matrix
    act = np.divide(act, sum(act), where=(sum(act)!=0))
    act2 = np.divide(act, sum(act2), where=(sum(act2)!=0))
    
    for pc in range(12): 
        predictions1_matrix = np.append(predictions1_matrix, np.array([file_name.replace(STEM_DIR, ""), str(pc), str(act[pc])], ndmin=2), axis=0)       
        predictions2_matrix = np.append(predictions2_matrix, np.array([file_name.replace(STEM_DIR, ""), str(pc), str(act[pc])], ndmin=2), axis=0)

# write predictions to CSV
predictions1_df = pd.DataFrame(predictions1_matrix, columns=("Melody", "Pitch Class", "Probability"))
predictions2_df = pd.DataFrame(predictions2_matrix, columns=("Melody", "Pitch Class", "Probability"))
predictions1_df.to_csv(OUTPUT1, header=False, index=False)
predictions2_df.to_csv(OUTPUT2, header=False, index=False)