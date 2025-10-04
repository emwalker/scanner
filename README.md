# Scanner

This project is a software-defined radio (SDR) scanner. Eventually it will sweep over a frequency range, look for stations that are transmitting, and attempt to demodulate the signal to determine if there is audio and human voice. The hope is to support various modes like WBFM, NFM, LSB, USB, and AM.

```sh
$ cargo run -- scan --band fm
$ cargo run -- scan --duration 60 --stations 88.9e6 # 88.9 MHz
$ cargo run -- scan --band fm --headless --verbose --json
```

At the moment, only basic FM demodulation is working.

<img width="2576" height="1456" alt="Screenshot From 2025-10-03 22-57-13" src="https://github.com/user-attachments/assets/03e7719c-eedc-4107-9e23-99060f606867" />

## Test bed for coding agents

This project is being used a test bed for experimenting with agentic coding, in order to better understand its strengths and limitations. The coding style may be inconsistent or unnecessarily complex and should not be taken as a reflection of the author's style.
