# Press ⌃R to execute it
# python 3.9

import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import numpy as np
import typing
from typing import Tuple, Union, Any
from tqdm import tqdm

from SignalGenerator import SignalGenerator
from tsks15audio import save_wav

"""
indexes:
- m: 10, the number of melodies
- l: 2, the number of pitch mismatches
- j: 20, the number of different hypotheses (10 melodies * 2 pitch mismatches)
- n: 12, the notes
- K: antal samples för en ton
"""


def setup_note2frequency() -> dict[str, int]:
    octave = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    notes = [octave[note] + str(oct_) for oct_ in range(1, 8) for note in range(12)]
    n = np.arange(4, 88)
    return dict(zip(notes, 440.0 * 2.0 ** ((n - 49) / 12)))


class Classifier:
    def __init__(self):  # melody, nr_tone_samples
        # Constants
        self.note_duration = 0.4
        self.pause_duration = 0.01
        self.pitch_mismatches = np.array([0.975, 1.025])

        # Set up methods
        self.dict_note2frequency = setup_note2frequency()
        self.stored_melodies_notes = self.setup_given_melodies()[0]
        self.stored_melodies_frequencies = self.setup_given_melodies()[1]
        self.melodies_mismatch1 = self.setup_given_melodies()[2]
        self.melodies_mismatch2 = self.setup_given_melodies()[3]

    def setup_given_melodies(self) -> tuple[
        list[list[str]], list[list[int]], list[list[Union[int, Any]]], list[list[Union[int, Any]]]]:
        with open('melodies.txt', 'r') as file:
            melodies = file.readlines()

        # Splits the string into a nested list with the tones (expressed in notes)
        melodies_note = [m.strip().split(', ') for m in melodies]

        # Expresses the melodies as a nested list of frequencies for each tone
        melodies_freq = [[self.dict_note2frequency[n] for n in m] for m in melodies_note]

        m_mismatch_1 = [[n * self.pitch_mismatches[0] for n in m] for m in melodies_freq]
        m_mismatch_2 = [[n * self.pitch_mismatches[1] for n in m] for m in melodies_freq]

        return melodies_note, melodies_freq, m_mismatch_1, m_mismatch_2

    def single_tone_observation_matrix(self):
        """
        Observation matrix
        :return ndarray
        """

        # for k in range(self.nr_tone_samples):
        F = [self.melodies_mismatch1, self.melodies_mismatch2]

        H = np.empty((20, 12, self.nr_tone_samples_div, 2))

        # Plocka frekvensen för varje element i H:
        alpha = 0
        for pitch_mismatch in F:  # 2 mismatches
            n = 0  # rad i H
            if alpha == 1:
                n = n + 10
            for m in pitch_mismatch:  # 10 melodier i m
                j = 0  # kolonn i H
                for f_nj in m:  # 12 toner i en melodi
                    H_nj = np.array([1, 0])
                    # Nu har du frekvensen f_nj
                    for k in range(self.nr_tone_samples_div):
                        new_row = np.array([np.cos(2 * np.pi * f_nj / 8820 * k), np.sin(2 * np.pi * f_nj / 8820 * k)])
                        H_nj = np.vstack([H_nj, new_row])
                    H_nj = np.delete(H_nj, 0, axis=0)
                    H[n, j] = H_nj
                    j += 1
                n += 1
            alpha += 1

        return H

    def three_tone_observation_matrix(self):
        """:return tough stuff"""

        # for k in range(self.nr_tone_samples):
        F = [self.melodies_mismatch1, self.melodies_mismatch2]

        H = np.empty((20, 12, self.nr_tone_samples_div, 6))

        # Plocka frekvensen för varje element i H:
        alpha = 0
        for pitch_mismatch in F:  # 2 mismatches
            n = 0  # rad i H
            if alpha == 1:
                n = n + 10
            for m in pitch_mismatch:  # 10 melodier i m
                j = 0  # kolonn i H
                for f_nj in m:  # 12 toner i en melodi
                    H_nj = np.array([1, 0, 1, 0, 1, 0])
                    # Nu har du frekvensen f_nj
                    for k in range(self.nr_tone_samples_div):
                        new_row = np.array([np.cos(2 * np.pi * f_nj / 8820 * k), np.sin(2 * np.pi * f_nj / 8820 * k),
                                            np.cos(6 * np.pi * f_nj / 8820 * k), np.cos(6 * np.pi * f_nj / 8820 * k),
                                            np.cos(10 * np.pi * f_nj / 8820 * k), np.cos(10 * np.pi * f_nj / 8820 * k)])
                        H_nj = np.vstack([H_nj, new_row])
                    H_nj = np.delete(H_nj, 0, axis=0)
                    H[n, j] = H_nj
                    j += 1
                n += 1
            alpha += 1

        return H

    def setup_y_n(self) -> list[int]:
        y = []

        for n in range(12):
            y.append(self.melody[n * self.nr_tone_samples:(n * self.nr_tone_samples + self.nr_tone_samples_div)])
            n += 1
        return y

    def single_tone_classifier(self, melody) -> tuple[int, float]:
        """
        :return: our hypothesis on melody and pitch mismatch
        """
        self.melody = melody

        nr_samples = len(melody)
        tone = melody[:int(nr_samples / 12)]  # alla toner lika långa => längden på varje ton fås här
        self.nr_tone_samples = int(len(tone))
        self.nr_tone_samples_div = int(len(tone)/10)
        self.y = self.setup_y_n()
        self.H = self.single_tone_observation_matrix()

        # determine the hypothesis j_hat
        j_max = 0
        sum_max = 0
        for j in range(20):  # melodies
            sum = 0
            for n in range(12):  # number of tones
                sum += (np.linalg.norm(np.matmul(np.transpose(self.H[j][n]), self.y[n]))) ** 2
            if sum > sum_max:
                sum_max = sum
                j_max = j

        if j_max < 10:
            cl_id = j_max
            cl_mism = 0.975
        else:
            cl_id = j_max - 10
            cl_mism = 1.025

        return cl_id, cl_mism

    def three_tone_classifier(self, melody):

        self.melody = melody

        # determines the amount of samples
        nr_samples = len(melody)
        tone = melody[:int(nr_samples / 12)]  # alla toner lika långa => längden på varje ton fås här
        self.nr_tone_samples = int(len(tone))
        self.nr_tone_samples_div = int(len(tone)/10)
        self.H = self.three_tone_observation_matrix()
        self.y = self.setup_y_n()

        # determine the hypothesis j_hat
        j_max = 0
        sum_max = 0
        for j in range(20):  # melodies
            sum = 0
            for n in range(12):  # number of tones
                sum += (np.linalg.norm(np.matmul(np.transpose(self.H[j][n]), self.y[n]))) ** 2
            if sum > sum_max:
                sum_max = sum
                j_max = j

        if j_max < 10:
            cl_id = j_max
            cl_mism = 0.975
        else:
            cl_id = j_max - 10
            cl_mism = 1.025

        return cl_id, cl_mism


def monte_carlo_simulation():
    """Determines which classifier that works the best. Plot the misclassification as function of the SNR.
    The interesting SNR range is between -50 dB and -10 dB."""
    sg = SignalGenerator()
    clsfier = Classifier()

    single_or_three_tone = 1  # single-tone melody or three-tone melody

    sim_nr = 5  # the number of simulations
    c_id= -1
    c_mismatch = -1
    nr_of_notes_gen = [1, 3, 1, 3]
    nr_of_notes_cls = [1, 1, 3, 3]

    for run in range(4):
        success_list = []
        SNR_DB_list = []
        classification_sum = 0
        for SNR_DB in tqdm(range(-50, 0, 10)):
            melodies, ids, mismatches = sg.generate_random_melodies(sim_nr, SNR_DB, nr_of_notes_gen[run])
            for melody_index in range(sim_nr):
                if nr_of_notes_cls[run] == 1:
                    c_id, c_mismatch = clsfier.single_tone_classifier(np.transpose(melodies)[melody_index])
                elif nr_of_notes_cls[run] == 3:
                    c_id, c_mismatch = clsfier.three_tone_classifier(np.transpose(melodies)[melody_index])

                if c_id != ids[melody_index] or c_mismatch != mismatches[melody_index]:
                    classification_sum += 1 / sim_nr

            SNR_DB_list.append(SNR_DB)
            success_list.append(classification_sum)
            classification_sum = 0

        plt.figure()
        plt.plot(SNR_DB_list, success_list)
        if run == 0:
            plt.title('Single Tone melody with a single-tone classifier')
            img_name = 'single_tone_single_detector'
        elif run == 1:
            plt.title('Three Tone melody with a single-tone classifier')
            img_name = 'three_tone_single_detector'
        elif run == 2:
            plt.title('Single Tone melody with a three-tone classifier')
            img_name = 'single_tone_three_detector'
        elif run == 3:
            plt.title('Three Tone melody with a three-tone classifier')
            img_name = 'three_tone_three_detector'

        plt.xlabel('SNR')
        plt.ylabel('Misclassification')
        # Show/save figure as desired.
        plt.savefig(img_name)
        plt.show()





if __name__ == '__main__':
    """sg = SignalGenerator()
    # generate a random melody, with SNR 100 dB, and 3 tones
    correct = 0
    for i in range(20):
        melody, idx, mismatch = sg.generate_random_melody(100, 1)
        # nr_samples = len(melody)
        # nr_tones = 12  # all melodies have 12 tones
        # tone = melody[:int(nr_samples / nr_tones)]  # alla toner lika långa => längden på varje ton fås här
        # nr_tone_samples = len(tone)/10

        classifier1 = Classifier()
        classified_id, classified_mismatch = classifier1.single_tone_classifier(melody)
        if classified_id == idx and classified_mismatch == mismatch:
            correct += 1

    print('antal rätt:', correct)"""
    # classified_id, classified_mismatch = classifier1.three_tone_classifier()

    # print("Vi gissar på att det är melodi nr ", classified_id)
    # print("Vi gissar på att det är en pitch mismatch på ", classified_mismatch)
    # print('')
    # print("Rätt melodi: ", classified_id)
    # print("Rätt mismatch: ", classified_mismatch)
    monte_carlo_simulation()
    """
    sg = SignalGenerator()
    # generate a random melody, with SNR 100 dB, and 3 tones
    melodies, ids, mismatches = sg.generate_random_melodies(5, 100)

    nr_samples = len(melodies)
    nr_tones = 12  # all melodies have 12 tones
    tone = melody[:int(nr_samples / nr_tones)]  # alla toner lika långa => längden på varje ton fås här
    nr_tone_samples = len(tone)

    classifier1 = Classifier(melody, nr_tone_samples)
    # classified_id, classified_mismatch = classifier1.single_tone_classifier()
    classified_id, classified_mismatch = classifier1.three_tone_classifier()

    print("Vi gissar på att det är melodi nr ", classified_id)
    print("Vi gissar på att det är en pitch mismatch på ", classified_mismatch)
    print('')
    print("Rätt melodi: ", classified_id)
    print("Rätt mismatch: ", classified_mismatch)"""

