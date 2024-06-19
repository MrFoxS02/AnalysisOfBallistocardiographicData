# Import necessary libraries
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
import scipy

# Define a class for heart rate analysis
class HeartRateAnalysis:
    # Initialize the class with data file, sampling frequency, order of the filter, and cutoff frequencies
    def __init__(self, data_file, Fs = 30000, N = 2, F = np.array([0.1, 1])):
        self.data = loadmat(data_file)  # Load the data file
        self.signal_d = None  # Initialize the signal derivative
        self.filtered_signal = None  # Initialize the filtered signal
        self.peaks = None  # Initialize the peaks
        self.breath = None

    # Formulate the signal
    def signal_formulation(self):
        keys = ['Ch1', 'Ch3', 'Ch4']  # Define the keys for the data
        # Normalize the data and create a DataFrame
        D = {keys[i]: [float(element) for element in self.data[keys[i]]] for i in range(len(keys))}
        self.Data = pd.DataFrame(D)
        self.Data = self.Data.loc[:len(self.Data['Ch1']) - 4, :]

        # Perform some calculations on the data
        p1 = (self.Data['Ch1'] - np.min(self.Data['Ch1'])) / (np.max(self.Data['Ch1']) - np.min(self.Data['Ch1'])) * 2 - 1
        p2 = (self.Data['Ch3'] - np.min(self.Data['Ch3'])) / (np.max(self.Data['Ch3']) - np.min(self.Data['Ch3'])) * 2 - 1
        p3 = (self.Data['Ch4'] - np.min(self.Data['Ch4'])) / (np.max(self.Data['Ch4']) - np.min(self.Data['Ch4'])) * 2 - 1

        P1 = p1 + p3
        P2 = p3 - p1
        P1d = self.__diff(P1)
        P2d = self.__diff(P2)

        Dd = P1[:len(P1) - 1] * P2d - P2[:len(P1) - 1] * P1d
        D = [0]
        D[0] = Dd[0]

        for i in range(1, len(P1) - 1):
            D.append(D[i - 1] + Dd[i])

        self.signal_d = D  # Store the signal derivative
        return self.signal_d  # Return the signal derivative

    # Private method to calculate the difference between adjacent elements in a list
    def __diff(self, A: list) -> list:
        return [A[i + 1] - A[i] for i in range(len(A) - 1)]

    # Filter the signal
    def filtering(self, sfilter = '1', F = np.array([0.1, 1]), N = 2, fs = 30000):
        if sfilter == '1':  # If filter type is '1', use an IIR filter
            b, a = signal.iirfilter(N, np.array(F) / (fs / 2), btype='band', ftype='butter')
            self.filtered_signal = signal.lfilter(b, a, self.signal_d)
        elif sfilter == '2':  # If filter type is '2', use a second-order section (SOS) filter
            sos = signal.butter(N, np.array(F) / (fs / 2), btype='band', output='sos')
            self.filtered_signal = signal.sosfilt(sos, self.signal_d)
        return self.filtered_signal  # Return the filtered signal

    # Visualize the data
    def visualize_data(self, signal, outputSignal, gtype: str = "plot", fs=30000, nperseg=2**17, noverlap=120000, vmin=-40, vmax=30,
                       figsize=(20, 5)):
        if gtype == "plot":  # If plot type is 'plot', plot the signal
            plt.figure(figsize=figsize)
            plt.plot(np.linspace(0, len(signal), len(signal)) / fs, np.array(outputSignal))
            plt.grid()
            plt.xlabel('Time, s')
            plt.show()
        elif gtype == "specgram":  # If plot type is 'specgram', plot a spectrogram of the signal
            plt.specgram(signal, Fs=fs, NFFT=nperseg, noverlap=noverlap, vmin=vmin, vmax=vmax)
            plt.ylim(0, 10)
            plt.colorbar()
            plt.ylabel('Frequency, Hz')
            plt.show()

    # Find peaks in the signal
    def find_peaks_(self, signal, fs=30000, nperseg=2**17, noverlap=120000, prominence = 1, distance = 1, height=0):
        peaks, _ = scipy.signal.find_peaks(signal, prominence=prominence, distance = distance, height=height)
        return peaks

    # Visualize the signal with peaks
    def visualize_with_peaks(self, signal, fs=30000, nperseg=2**17, noverlap=120000, vmin=40, vmax=30,
                                        prominence = 1, distance = 1, height = 0, figsize=(20, 7)):
        self.peaks = self.find_peaks_(signal, prominence=prominence, distance = distance, height=height, 
                                 fs=fs, nperseg=nperseg, noverlap=noverlap)
        plt.figure(figsize=figsize)
        plt.plot(self.peaks, signal[self.peaks], 'rx')  # Plot the peaks
        plt.plot(signal, 'b')  # Plot the signal
        plt.grid()
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time')
        plt.show()
        
    def visualize_b_with_peaks(self, signal, fs=30000, nperseg=2**17, noverlap=120000, vmin=40, vmax=30,
                                        prominence = 1, distance = 1, height = 0, figsize=(20, 7)):
        peaks = self.find_peaks_(signal, prominence=prominence, distance = distance, height=height, 
                                 fs=fs, nperseg=nperseg, noverlap=noverlap)
        peaks_all = self.find_peaks_(signal, prominence=prominence, distance = distance, height=0, 
                                 fs=fs, nperseg=nperseg, noverlap=noverlap)
        self.breath = [peak for peak in peaks_all if peak not in peaks]
        plt.figure(figsize=figsize)
        plt.plot(self.breath, signal[self.breath], 'go')
        plt.plot(signal, 'b')
        plt.grid()
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time')
        plt.show()
        
    # Count the heart rate based on the number of peaks
    def сounting_heart_rate_peaks(self):
        print('heart rate:', len(self.peaks) * (60 / 25))  # Print the heart rate

    def сounting_heart_rate_breath_peaks(self):
        print('breath:', len(self.breath) * (16 / 25))  # Print the breath




