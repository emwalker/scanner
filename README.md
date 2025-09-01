# Scanner

This project is a software-defined radio (SDR) scanner. Eventually it will sweep over a frequency range, look for stations that are transmitting, and attempt to demodulate the signal to determine if there is audio and human voice. The hope is to support various modes like WBFM, NFM, LSB, USB, and AM.

```sh
$ cargo run -- --stations 88.9e6 --duration 60 # 88.9 MHz
$ cargo run -- --band fm
```

At the moment, only very basic FM demodulation is working, and many false positives are incorrectly identified as stations.

## Test bed for coding agents

This project is being used a test bed for experimenting with agentic coding, in order to better understand its strengths and limitations. The coding style may be inconsistent or unnecessarily complex and should not be taken as a reflection of the author's style.
