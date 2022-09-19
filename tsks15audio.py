#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 13:04:04 2018

@author: marcus
"""
import sys
try:
    assert sys.version_info >= (3,0)
except AssertionError:
    print("This script requires python version 3.4.3 or higher")
    raise
    
#import pyaudio
import scipy.io.wavfile as wavfile
import numpy as np

"""def play(signal, fs = 44100):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=fs,
                output=True)    
    stream.write(signal.astype(np.float32).tobytes())
    stream.stop_stream()
    stream.close()
    p.terminate()"""

def save_wav(fname, signal, fs):
    wavfile.write(fname, fs, np.int16(signal/np.max(np.abs(signal)) * 32767))


"""def test_play():
    fs = 8000 # Hz
    fc = 2000 # Hz
    duration = 2 # seconds
    time = np.arange(duration * fs) / fs
    s = np.sin(2*np.pi*fc*time)
    play(s, fs)"""

"""if __name__ == "__main__":
    test_play()"""