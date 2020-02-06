# FlowPrint
Original implementation of FlowPrint as in the NDSS '20 paper.
This branch contains the exact scripts used for our experiments in the paper.
**Please note that the `master` branch contains a much better documented and user friendly implementation of FlowPrint.**

## Experiments
The `experiments` directory contains scripts from our evaluation.

### Parameter Selection
Run `experiments/parameters/parameters.py` with different parameter values to compute the optimal parameters.

### App Recognition
FlowPrint experiment: run `experiments/recognition/recognition.py` with desired dataset.
AppScanner experiment: run `experiments/recognition/recognition_appscanner.py` with desired dataset.

**Important: The AppScanner experiment requires AppScanner to be installed**
See [AppScanner](https://github.com/Thijsvanede/AppScanner)
```
pip install appscanner
```

### Unseen App Detection
Run `experiments/anomaly/anomaly.py` with desired dataset.

### Fingerprint insights
Browser isolation: run `experiments/insights/browser/test_detector.py`
Confidence: run `experiments/insights/confidence/confidence.py` with desired dataset.
Cardinality: run `experiments/insights/cardinaltiy/cardinality.py` with desired dataset.

### Mobile Network Traffic challenges
Homogeneous traffic: run `experiments/challenges/homogeneity/homogeneity.py` with ReCon dataset.
Dynamic traffic: N/A

#### Evolving traffic
App updates: run `TODO`
App longitudinal analysis: run `TODO`


 5.1. Mobile Network Traffic Challenges - Homogeneous traffic: `experiments/`
