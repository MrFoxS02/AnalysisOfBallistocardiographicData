### AnalysisOfBallistocardiographicData
<p>This program enables normalization and preprocessing of raw data obtained from a ballistocardiographic device as part of a thesis project. 🚀</p>
<p>Within the code block, one can perform signal formation, visualization, filtering where filter frequencies are selected based on the spectrogram, and searching for heart rate peaks with the ability to display an image with highlighted peaks. 📈</p>
<p>An example of using the software solution is presented in the file ExampleOfUse. 📄</p>

```python
from HeartRate import *
data_file = 'andrey_front.mat'
hr_analysis = HeartRateAnalysis(data_file)
....
```
