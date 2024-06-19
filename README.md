### AnalysisOfBallistocardiographicData
<p>This program enables normalization and preprocessing of raw data obtained from a ballistocardiographic device as part of a thesis project. 🚀</p>
<p>Within the code block, one can perform signal formation, visualization, filtering where filter frequencies are selected based on the spectrogram, and searching for heart rate peaks with the ability to display an image with highlighted peaks. 📈</p>
<p>An example of using the software solution is presented in the file ExampleOfUse. 📄</p>

```python
from HeartRate import *
data_file = 'andrey_front.mat'
hr_analysis = HeartRateAnalysis(data_file)
signal = hr_analysis.signal_formulation()
filtered_signal = hr_analysis.filtering(sfilter = '2', fs = 30000, N = 3, F = [1.5, 3.5])
hr_analysis.visualize_with_peaks(filtered_signal, fs=30000, prominence = [2.5], distance = 1, height = 2.5, figsize = (20, 7))
hr_analysis.visualize_b_with_peaks(filtered_signal, fs=30000, prominence = [2.5], distance = 1, height = 2.5, figsize = (20, 7))
hr_analysis.сounting_heart_rate_peaks()
hr_analysis.сounting_heart_rate_breath_peaks()
```
