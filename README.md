# Scanner

This project is a software-defined radio (SDR) scanner. Eventually it will sweep over a frequency range, look for stations that are transmitting, and attempt to demodulate the signal to determine if there is audio and human voice. The hope is to support various modes like WBFM, NFM, LSB, USB, and AM.

```sh
$ cargo run -- --duration 10 --tune-freq 88.9e6 # 88.9 MHz
```

## Test bed for coding agents

This project is being used a test bed for experimenting with agentic coding, in order to better understand its strengths and limitations. The coding style may be inconsistent or unnecessarily complex and should not be taken as a reflection of the author's style.
