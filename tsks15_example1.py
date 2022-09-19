#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 14:40:32 2018
Example 1: Spectrum of the first note of a random melody
@author: marcus
"""

from SignalGenerator import SignalGenerator
import numpy as np
import matplotlib.pyplot as plt

import sys
try:
    assert sys.version_info >= (3,0)
except AssertionError:
    print("This script requires python version 3.4.3 or higher")
    raise
    
# Program
sg = SignalGenerator()
# generate a random melody, with SNR 100 dB, and 3 tones
melody, idx, mismatch = sg.generate_random_melody(100, 3)
nr_samples = len(melody)
nr_tones = 12  # all melodies have 12 tones
tone = melody[:int(nr_samples/nr_tones)]  # alla toner lika långa => längden på varje ton fås här
nr_tone_samples = len(tone)
spectrum = np.abs(np.fft.fft(tone))  # skapar ett spektrum där vi ser vilken amplitud den har
fs = sg.sampling_frequency
freqs = np.arange(nr_tone_samples) * fs / nr_tone_samples
plt.figure()
plt.plot(freqs[:int(nr_tone_samples/2)], spectrum[:int(nr_tone_samples/2)])
plt.xlabel('frequency [Hz]')
plt.ylabel('magnitude')
plt.savefig('python-example1.png')
