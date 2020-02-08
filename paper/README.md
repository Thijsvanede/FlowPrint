# Additional resources from paper

## AMI analysis

#### Adjusted Mutual Information of temporal features per TCP/UDP flow
| Feature                                            | AMI       |
|----------------------------------------------------|-----------|
| Inter-flow timing                                  |   0.493   |
| Packet inter-arrival time (incoming) - std         |   0.218   |
| Packet inter-arrival time (outgoing) - std         |   0.211   |
| Packet inter-arrival time (incoming) - mad         |   0.202   |
| Packet inter-arrival time (outgoing) - mad         |   0.197   |
| Packet inter-arrival time (incoming) - max         |   0.196   |
| Packet inter-arrival time (incoming) - 90 pct      |   0.196   |
| Packet inter-arrival time (incoming) - mean        |   0.196   |
| Packet inter-arrival time (incoming) - 80 pct      |   0.196   |
| Packet inter-arrival time (incoming) - 70 pct      |   0.196   |
| Packet inter-arrival time (incoming) - 60 pct      |   0.195   |
| Packet inter-arrival time (incoming) - 40 pct      |   0.193   |
| Packet inter-arrival time (incoming) - 30 pct      |   0.192   |
| Packet inter-arrival time (incoming) - 50 pct      |   0.192   |
| Packet inter-arrival time (incoming) - 10 pct      |   0.190   |
| Packet inter-arrival time (incoming) - 20 pct      |   0.190   |
| Packet inter-arrival time (incoming) - min         |   0.184   |
| Packet inter-arrival time (outgoing) - max         |   0.178   |
| Packet inter-arrival time (outgoing) - mean        |   0.178   |
| Packet inter-arrival time (outgoing) - 90 pct      |   0.178   |
| Packet inter-arrival time (outgoing) - 80 pct      |   0.178   |
| Packet inter-arrival time (outgoing) - 70 pct      |   0.178   |
| Packet inter-arrival time (outgoing) - 60 pct      |   0.177   |
| Packet inter-arrival time (outgoing) - 50 pct      |   0.176   |
| Packet inter-arrival time (outgoing) - min         |   0.175   |
| Packet inter-arrival time (outgoing) - 40 pct      |   0.175   |
| Packet inter-arrival time (outgoing) - 30 pct      |   0.175   |
| Packet inter-arrival time (bidirectional) - std    |   0.173   |
| Packet inter-arrival time (outgoing) - 20 pct      |   0.173   |
| Packet inter-arrival time (outgoing) - 10 pct      |   0.170   |
| Packet inter-arrival time (bidirectional) - max    |   0.067   |
| Packet inter-arrival time (bidirectional) - mad    |   0.039   |
| Packet inter-arrival time (bidirectional) - 90 pct |   0.038   |
| Packet inter-arrival time (bidirectional) - 80 pct |   0.023   |
| Packet inter-arrival time (bidirectional) - mean   |   0.018   |
| Packet inter-arrival time (bidirectional) - 70 pct |   0.015   |
| Packet inter-arrival time (bidirectional) - 60 pct |   0.011   |
| Packet inter-arrival time (bidirectional) - 50 pct |   0.010   |
| Packet inter-arrival time (bidirectional) - 40 pct |   0.008   |
| Packet inter-arrival time (bidirectional) - 30 pct |   0.006   |
| Packet inter-arrival time (bidirectional) - 10 pct |   0.006   |
| Packet inter-arrival time (bidirectional) - 20 pct |   0.005   |
| Packet inter-arrival time (bidirectional) - min    |   0.005   |
| **Average**                                        | **0.144** |

#### Adjusted Mutual Information of size features per TCP/UDP flow

| Feature                              | AMI       |
|--------------------------------------|-----------|
| Packet size (incoming) - std         |   0.235   |
| Packet size (outgoing) - std         |   0.232   |
| Packet size (outgoing) - max         |   0.070   |
| Packet size (incoming) - 90 pct      |   0.067   |
| Packet size (incoming) - max         |   0.065   |
| Packet size (bidirectional) - std    |   0.054   |
| Packet size (incoming) - mad         |   0.053   |
| Packet size (incoming) - 80 pct      |   0.050   |
| Packet size (incoming) - 70 pct      |   0.043   |
| Packet size (outgoing) - 90 pct      |   0.042   |
| Packet size (incoming) - 60 pct      |   0.042   |
| Packet size (incoming) - mean        |   0.040   |
| Packet size (bidirectional) - max    |   0.039   |
| Packet size (outgoing) - mad         |   0.037   |
| Packet size (incoming) - 50 pct      |   0.036   |
| Packet size (bidirectional) - mad    |   0.033   |
| Packet size (bidirectional) - 90 pct |   0.033   |
| Packet size (outgoing) - 80 pct      |   0.033   |
| Packet size (incoming) - 40 pct      |   0.028   |
| Packet size (bidirectional) - 80 pct |   0.026   |
| Packet size (incoming) - 30 pct      |   0.026   |
| Packet size (outgoing) - mean        |   0.024   |
| Packet size (bidirectional) - mean   |   0.022   |
| Packet size (incoming) - 20 pct      |   0.021   |
| Packet size (bidirectional) - 70 pct |   0.020   |
| Packet size (bidirectional) - 60 pct |   0.019   |
| Packet size (incoming) - 10 pct      |   0.016   |
| Packet size (outgoing) - 70 pct      |   0.016   |
| Packet size (bidirectional) - 50 pct |   0.012   |
| Packet size (outgoing) - 50 pct      |   0.009   |
| Packet size (outgoing) - 60 pct      |   0.008   |
| Packet size (bidirectional) - 40 pct |   0.008   |
| Packet size (outgoing) - 30 pct      |   0.007   |
| Packet size (outgoing) - 40 pct      |   0.007   |
| Packet size (outgoing) - 20 pct      |   0.006   |
| Packet size (outgoing) - 10 pct      |   0.006   |
| Packet size (bidirectional) - 30 pct |   0.005   |
| Packet size (outgoing) - min         |   0.005   |
| Packet size (bidirectional) - 20 pct |   0.004   |
| Stream size - packets in stream      |   0.004   |
| Packet size (bidirectional) - 10 pct |   0.004   |
| Packet size (incoming) - min         |   0.003   |
| Packet size (bidirectional) - min    |   0.003   |
| **Average**                          | **0.035** |

#### Adjusted Mutual Information of destination features per TCP/UDP flow
| Feature                                | AMI       |
|----------------------------------------|-----------|
| IP address - destination               |   0.246   |
| TCP port - destination                 |   0.065   |
| **Average**                            | **0.155** |
| TLS server hello - extensions          |   0.107   |
| TLS server hello - session             |   0.089   |
| TLS server hello - ciphersuites        |   0.077   |
| TLS server hello - version             |   0.065   |
| TLS server hello - compression         |   0.023   |
| **Average**                            | **0.072** |
| TLS certificate - validity after       |   0.369   |
| TLS certificate - validity before      |   0.356   |
| TLS certificate - serial               |   0.342   |
| TLS certificate - extensions           |   0.235   |
| TLS certificate - subject name         |   0.170   |
| TLS certificate - subject pk           |   0.111   |
| TLS certificate - issuer name          |   0.086   |
| TLS certificate - subject pk algorithm |   0.031   |
| TLS certificate - signature            |   0.020   |
| TLS certificate - version              |   0.020   |
| **Average**                            | **0.134** |
| **Total Average**                      | **0.142** |

#### Adjusted Mutual Information of source features per TCP/UDP flow
| Feature                         | AMI       |
|---------------------------------|-----------|
| IP address - source             |   0.434   |
| TCP port - source               |   0.173   |
| **Average**                     | **0.304** |
| TLS client hello - extensions   |   0.177   |
| TLS client hello - ciphersuites |   0.176   |
| TLS client hello - session      |   0.147   |
| TLS client hello - version      |   0.065   |
| TLS client hello - compression  |   0.023   |
| **Average**                     | **0.118** |
| **Total Average**               | **0.171** |
