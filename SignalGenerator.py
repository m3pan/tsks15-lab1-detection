#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 13:01:04 2018
tested on Python 3.4.3
@author: marcus
"""
import sys

try:
    assert sys.version_info >= (3, 0)
except AssertionError:
    print("This script requires python version 3.4.3 or higher")
    raise

import numpy as np


class SignalGenerator:
    # =============================================================================
    #     Setup
    # =============================================================================
    def __init__(self, f=8820):
        self.setup_constants(f)
        self.setup_note2frequency()
        self.setup_melodies()

    def setup_constants(self, f):
        self.sampling_frequency = f
        self.note_duration = 0.4
        self.pause_duration = 0.01
        self.pitch_mismatches = np.array([0.975, 1.025])

    def setup_note2frequency(self):
        octave = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        notes = [octave[note] + str(oct) for oct in range(1, 8) for note in range(12)]
        n = np.arange(4, 88)
        self.dict_note2frequency = dict(zip(notes, 440.0 * 2.0 ** ((n - 49) / 12)))

    def setup_melodies(self):
        with open('melodies.txt', 'r') as file:
            melodies = file.readlines()
        self.melodies = [melody.strip().split(', ') for melody in melodies]

    # =============================================================================
    #     Getters: Proper Members
    # =============================================================================
    @property
    def sampling_frequency(self):
        return self._sampling_frequency

    @property
    def note_duration(self):
        return self._note_duration

    @property
    def pause_duration(self):
        return self._pause_duration

    @property
    def pitch_mismatches(self):
        return self._pitch_mismatches

    @property
    def dict_note2frequency(self):
        return self._dict_note2frequency

    @property
    def melodies(self):
        return self._melodies

    # =============================================================================
    #     Getters: Implicit Members
    # =============================================================================
    @property
    def pause(self):
        return np.zeros(int(self.pause_duration * self.sampling_frequency))

    @property
    def nr_melodies(self):
        return len(self.melodies)

    @property
    def nr_pitch_mismatches(self):
        return len(self.pitch_mismatches)

    @property
    def nr_hypotheses(self):
        return self.nr_melodies * self.nr_pitch_mismatches

    @property
    def nr_note_samples(self):
        return int(self.note_duration * self.sampling_frequency)

    # =============================================================================
    #     Setters
    # =============================================================================
    @sampling_frequency.setter
    def sampling_frequency(self, value):
        self._sampling_frequency = value

    @note_duration.setter
    def note_duration(self, value):
        self._note_duration = value

    @pause_duration.setter
    def pause_duration(self, value):
        self._pause_duration = value

    @pitch_mismatches.setter
    def pitch_mismatches(self, value):
        self._pitch_mismatches = value

    @dict_note2frequency.setter
    def dict_note2frequency(self, value):
        self._dict_note2frequency = value

    @melodies.setter
    def melodies(self, value):
        self._melodies = value

    # =============================================================================
    # Constants for varying amplitude distribution
    # =============================================================================
    EXPONENTIAL = 1
    UNIFORM = 2
    CRESCENDO = 3

    # =============================================================================
    #     Methods
    # =============================================================================
    def note2frequency(self, note):
        return self.dict_note2frequency[note]

    def get_melody(self, idx_melody):
        return self.melodies[idx_melody]

    def nr_notes(self, melody):
        return len(melody)

    def get_note(self, idx_note, melody):
        return melody[idx_note]

    def generate_melody(self, idx_melody, pitch_mismatch=1, nr_tones=1):
        if nr_tones != 1 and nr_tones != 3:
            raise ValueError('nr_tones should be 1 or 3')
        time_range = np.arange(self.nr_note_samples) / self.sampling_frequency
        melody = np.empty(0)
        for idx_note in range(self.nr_notes(self.get_melody(idx_melody))):
            f = pitch_mismatch * self.note2frequency(self.get_note(idx_note, self.get_melody(idx_melody)))
            note = np.cos(2 * np.pi * f * time_range + 2 * np.pi * np.random.uniform())
            if nr_tones == 3:
                note += 1 * np.cos(2 * np.pi * 3 * f * time_range + 2 * np.pi * np.random.uniform())
                note += 1 * np.cos(2 * np.pi * 5 * f * time_range + 2 * np.pi * np.random.uniform())
                note *= 1 / np.sqrt(3)
            melody = np.append(melody, note * self.get_amplitude(idx_note, self.UNIFORM))
            melody = np.append(melody, self.pause)
        return melody

    def get_amplitude(self, idx_note, opt=None):
        if opt is None:
            return 1
        if opt == self.EXPONENTIAL:
            return np.random.exponential(1)
        if opt == self.CRESCENDO:
            return 0.8 + 0.4 * idx_note / 12
        if opt == self.UNIFORM:
            return np.random.uniform(0.8, 1.2)

    def generate_random_melody(self, snr_db, nr_tones=1):
        idx_melody = np.random.randint(self.nr_melodies)
        pitch_mismatch = self.pitch_mismatches[np.random.randint(self.nr_pitch_mismatches)]
        melody = self.generate_melody(idx_melody, pitch_mismatch, nr_tones)
        snr_lin = 10 ** (snr_db / 10)
        melody += np.sqrt(1 / snr_lin) * np.random.randn(len(melody))
        return melody, idx_melody, pitch_mismatch

    def generate_random_melodies(self, nr_generate, snr_db, nr_tones=1):
        melodies = np.zeros((len(self.generate_random_melody(snr_db)[0]), nr_generate))
        ids = np.empty(nr_generate)
        mismatches = np.empty(nr_generate)
        for idx_generate in range(nr_generate):
            melody, idx_melody, mismatch = self.generate_random_melody(snr_db, nr_tones)
            melodies[:, idx_generate] = melody
            ids[idx_generate] = idx_melody
            mismatches[idx_generate] = mismatch
        return melodies, ids, mismatches


# =============================================================================
# End Class SignalGenerator
# =============================================================================
def print2matlabcell(M, ids, pitch):
    long_string = '{['
    long_string += ';'.join([' '.join(['{:4}'.format(item) for item in row]) for row in M])
    long_string += '], ['
    long_string += ' '.join(['{:4}'.format(item) for item in ids])
    long_string += '], ['
    long_string += ' '.join(['{:4}'.format(item) for item in pitch])
    long_string += ']}'
    print(long_string)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit('Usage: SignalGenerator.py number-of-melodies snr-in-db number-of-tones')
    sg = SignalGenerator()
    melodies, true_ids, true_mismatches = sg.generate_random_melodies(int(sys.argv[1]), float(sys.argv[2]),
                                                                      int(sys.argv[3]))
    print2matlabcell(melodies, true_ids, true_mismatches)
