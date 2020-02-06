# FlowPrint
Original implementation of FlowPrint as in the NDSS '20 paper.
This branch contains the exact scripts used for our experiments in the paper.

**Please note that the `master` branch contains a much better documented and user friendly implementation of FlowPrint.**

# Experiments
The `experiments` directory contains scripts from our evaluation.

## Parameter Selection
 1. Run `experiments/parameters/parameters.py` with different parameter values to compute the optimal parameters.

## App Recognition
 1. FlowPrint experiment: run `experiments/recognition/recognition.py` with desired dataset.
 2. AppScanner experiment: run `experiments/recognition/recognition_appscanner.py` with desired dataset.

**Important: The AppScanner experiment requires AppScanner to be installed**
See [AppScanner](https://github.com/Thijsvanede/AppScanner)
```
pip install appscanner
```

## Unseen App Detection
 1. Run `experiments/anomaly/anomaly.py` with desired dataset.

## Fingerprint insights
 1. Browser isolation: run `experiments/insights/browser/test_detector.py`
 2. Confidence: run `experiments/insights/confidence/confidence.py` with desired dataset.
 3. Cardinality: run `experiments/insights/cardinaltiy/cardinality.py` with desired dataset.

## Mobile Network Traffic challenges
Homogeneous traffic: run `experiments/challenges/homogeneity/homogeneity.py` with ReCon dataset.
Dynamic traffic: N/A

### Evolving traffic
App updates: run `experiments/challenges/evolving/updates/versions.py`
App longitudinal analysis: run`experiments/challenges/evolving/longitudinal/flow_changes.py`
```
python3 flow_changes.py -l <dataset> -k <folds> -f <change> -i <change> (--anomaly|--recognition)

-l PATH          loads dataset
-k INT           number of k-folds
-f FLOAT         changes fraction of certificates
-i FLOAT         changes fraction of IPs
--anomaly        for unseen app detection
--recognition    for app recognition
```

## Training size
This experiments uses the recognition.py from App Recognition and anomaly.py from Unseen App Detection.
The scripts in `experiments/training_size/` automatically run the aforementioned scripts with different number of apps.

## 
