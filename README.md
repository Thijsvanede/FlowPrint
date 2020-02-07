# FlowPrint - In progress
This repository contains the code for FlowPrint by the authors of the NDSS FlowPrint [1] paper.
This `master` branch provides FlowPrint as an out of the box tool.
For the original experiments from the paper, please checkout the `NDSS` branch.

## Installation
TODO

## Usage
```
usage: flowprint.py [-h]
                    (--detection | --fingerprint [FILE] | --recognition)
                    [-b BATCH] [-c CORRELATION], [-s SIMILARITY], [-w WINDOW]
                    [-p PCAPS...] [-rp READ...] [-wp WRITE]

Flowprint: Semi-Supervised Mobile-App
Fingerprinting on Encrypted Network Traffic

Arguments:
  -h, --help                 show this help message and exit

FlowPrint mode (select up to one):
  --fingerprint [FILE]       run in raw fingerprint generation mode (default)
                             outputs to terminal or json FILE
  --detection                run in unseen app detection mode (Unimplemented)
  --recognition              run in app recognition mode      (Unimplemented)

FlowPrint parameters:
  -b, --batch       FLOAT    batch size in seconds       (default=300)
  -c, --correlation FLOAT    cross-correlation threshold (default=0.1)
  -s, --similarity  FLOAT    similarity threshold        (default=0.9)
  -w, --window      FLOAT    window size in seconds      (default=30)

Flow data input/output (either --pcaps or --read required):
  -p, --pcaps PATHS...       path to pcap(ng) files to run through FlowPrint
  -r, --read  PATHS...       read preprocessed data from given files
  -t, --write PATH           write preprocessed data to given file
```

## References
[1] `van Ede, T., Bortolameotti, R., Continella, A., Ren, J., Dubois, D. J., Lindorfer, M., Choffnes, D., van Steen, M. & Peter, A. (2020, February). FlowPrint: Semi-Supervised Mobile-App Fingerprinting on Encrypted Network Traffic. In 2020 NDSS. The Internet Society.`

### Bibtex
```
@inproceedings{vanede2020flowprint,
  title={{FlowPrint: Semi-Supervised Mobile-App Fingerprinting on Encrypted Network Traffic}},
  author={van Ede, Thijs and Bortolameotti, Riccardo and Continella, Andrea and Ren, Jingjing and Dubois, Daniel J. and Lindorfer, Martina and Choffness, David and van Steen, Maarten, and Peter, Andreas}
  booktitle={NDSS},
  year={2020},
  organization={The Internet Society}
}
```
