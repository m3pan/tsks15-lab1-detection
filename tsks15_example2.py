#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 14:40:32 2018
Example 2: Generate a random melody, listen to it and save it as example.wav
You can listen to the saved file by running 'aplay example.wav' from the 
command line.
@author: marcus
"""

from SignalGenerator import SignalGenerator
# from tsks15audio import play
from tsks15audio import save_wav

import sys
try:
    assert sys.version_info >= (3, 0)
except AssertionError:
    print("This script requires python version 3.4.3 or higher")
    raise
    
sg = SignalGenerator()
melody, idx, mismatch = sg.generate_random_melody(100, 3)  # SNR 100 dB, and 3 tones
# play(melody, sg.sampling_frequency)
save_wav('example.wav', melody, sg.sampling_frequency)
