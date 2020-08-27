#!/usr/bin/env python

## Learn scale degree expectations given **kern training corpus

import glob
import os
import re
import numpy as np
import pandas as pd

# define parameters and useful data structures

DECAY = 1.6180339887 # decay constant = phi
CORPUS_DIR = os.path.join('../data', input("Name of training corpus: "), '') # training corpus
OUTPUT = os.path.join('../models', input("Filename to use for trained expectations: ")) # output file for predictions with repetition discounting

UPPER2PC = {"A" : 9, "A#": 10, "Bb" : 10, "B" : 11, "C" : 0, "C#" : 1, "Db" : 1, "D" : 2, "D#" : 3, "Eb" : 3, "E" : 4, "F" : 5, "F#" : 6, "Gb" : 6, "G" : 7, "G#" : 8, "Ab" : 8} 
LOWER2PC = {"a" : 9, "b" : 11, "c" : 0, "d" : 2, "e" : 4, "f" : 5, "g" : 7}



# learn corpus expectations

act = np.zeros(12) # scale degree activations
exp = np.zeros((12, 12)) # expectation network between scale degrees

for file_name in glob.iglob(CORPUS_DIR + "**/*.krn", recursive=True):
    
    # process file
    
    cur_file = open(file_name, "r", encoding="ISO-8859-1")
    seq = np.empty(0, dtype="int8") # pitch-class sequence
    key = float("-inf")
    key_changes = np.empty(0)
    tied = False
    
    for line in cur_file:
        
        # check for key change
        
        key_label = re.search(r'^\*([abcdefgABCDEFG])([#\-]*):', line)
        if key_label != None:
            key = float("-inf") # assume minor key
            if key_label.group(1) in UPPER2PC.keys():
                key = UPPER2PC[key_label.group(1)] # uppercase, major key found
            if key_label.group(2) == '-' or key_label.group(2) == '--' or key_label.group(2) == '---':
                key -= len(key_label.group(2)) # subtract flats
            elif key_label.group(2) == '#' or key_label.group(2) == '##' or key_label.group(2) == '###':
                key += len(key_label.group(2)) # add sharps
            key_changes = np.append(key_changes, len(seq))
            
        # convert note name to pitch class
            
        note_label = re.search(r'^\&{0,1}\{{0,1}\&{0,1}\({0,1}(\[{0,1})([0-9]+)(\.*)([abcdefgABCDEFG])+([#n\-]*)[^\]\t]*(\]{0,1})', line)
        if note_label != None:
            if note_label.group(4) in UPPER2PC.keys():
                note = UPPER2PC[note_label.group(4)]
            else:
                note = LOWER2PC[note_label.group(4)]
            if note_label.group(5) == '-' or note_label.group(5) == '--' or note_label.group(5) == '---':
                note -= len(note_label.group(5)) # subtract flats
            elif note_label.group(5) == '#' or note_label.group(5) == '##' or note_label.group(5) == '###':
                note += len(note_label.group(5)) # add sharps
            if note_label.group(1) == '[':
                tied = True
            elif note_label.group(6) == ']':
                tied = False
            if not tied and key >= 0:
                seq = np.append(seq, (note - key) % 12) # transpose to C major
      
    cur_file.close()
    
    # learn expectations
        
    for i in range(len(seq)):
        
        if i in key_changes:
            act.fill(0) # new key
           
        for pitch_class in range(12):
            if seq[i] >= 0:
                exp[pitch_class, seq[i]] += act[pitch_class] # update expectations
        act /= DECAY # update activations
        act[seq[i]] = 1.0
        
# save learned expectations to CSV

exp_df = pd.DataFrame(exp)
exp_df.to_csv(OUTPUT, header=False, index=False)