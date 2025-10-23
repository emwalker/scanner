#!/usr/bin/env python3
"""
SDR Filter Designer

Design FIR filters for SDR decimation chains, with focus on FM demodulation
and multi-stage decimation. Generates filter coefficients, frequency response
plots, and Rust code for integration.

Usage:
    python3 filter_designer.py --mode decimation --sample-rate 2048000 --decimation 4 --passband 100000
    python3 filter_designer.py --mode hilbert --num-taps 65
    python3 filter_designer.py --mode deemphasis --sample-rate 48000 --time-constant 75
    python3 filter_designer.py --mode chain --input-rate 2048000 --output-rate 48000
"""

import argparse
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


def estimate_filter_taps(sample_rate: float, transition_width: float,
                         stopband_atten_db: float = 60) -> int:
    """
    Estimate number of taps required for FIR filter using Kaiser formula.

    Args:
        sample_rate: Sample rate in Hz
        transition_width: Transition bandwidth in Hz
        stopband_atten_db: Required stopband attenuation in dB

    Returns:
        Number of filter taps (odd number)
    """
    delta_f = transition_width / sample_rate
    num_taps = int((stopband_atten_db - 8) / (2.285 * 2 * np.pi * delta_f)) + 1

    # Round up to next odd number
    if num_taps % 2 == 0:
        num_taps += 1

    return num_taps


def design_decimation_filter(sample_rate: float, decimation: int,
                             passband_edge: float, stopband_atten_db: float = 60,
                             max_taps: Optional[int] = None) -> np.ndarray:
    """
    Design FIR lowpass filter for decimation.

    Args:
        sample_rate: Input sample rate (Hz)
        decimation: Decimation factor
        passband_edge: Passband edge frequency (Hz)
        stopband_atten_db: Stopband attenuation (dB)
        max_taps: Maximum number of taps (None for auto)

    Returns:
        FIR filter coefficients
    """
    nyquist = sample_rate / 2
    stopband_edge = sample_rate / decimation / 2  # Prevent aliasing
    transition_width = stopband_edge - passband_edge

    # Estimate required taps
    num_taps = estimate_filter_taps(sample_rate, transition_width, stopband_atten_db)

    if max_taps and num_taps > max_taps:
        print(f"Warning: Estimated {num_taps} taps, limiting to {max_taps}")
        print(f"  This may result in insufficient stopband attenuation")
        num_taps = max_taps

    # Design filter using Kaiser window
    beta = signal.kaiser_beta(stopband_atten_db)
    taps = signal.firwin(
        num_taps,
        cutoff=passband_edge,
        window=('kaiser', beta),
        fs=sample_rate
    )

    return taps


def design_hilbert_filter(num_taps: int = 65) -> np.ndarray:
    """
    Design FIR Hilbert transformer for SSB demodulation.

    Args:
        num_taps: Number of filter taps (must be odd)

    Returns:
        FIR Hilbert filter coefficients
    """
    if num_taps % 2 == 0:
        raise ValueError("num_taps must be odd for Hilbert filter")

    # Parks-McClellan optimal FIR design
    # Passband: 0.1 to 0.9 of Nyquist (avoid DC and Nyquist freq)
    bands = np.array([0.1, 0.9])
    desired = np.array([1, 1])
    taps = signal.remez(num_taps, bands, desired, type='hilbert')

    return taps


def design_deemphasis_filter(sample_rate: float, time_constant_us: float = 75) -> Tuple[np.ndarray, np.ndarray]:
    """
    Design de-emphasis filter for FM broadcast.

    Args:
        sample_rate: Audio sample rate (Hz)
        time_constant_us: Time constant in microseconds (75 for US, 50 for EU)

    Returns:
        (sos, zi): Second-order sections and initial state
    """
    tau = time_constant_us * 1e-6
    cutoff_freq = 1.0 / (2 * np.pi * tau)  # ~2122 Hz for 75us

    # 1-pole lowpass filter
    sos = signal.butter(1, cutoff_freq, btype='low', fs=sample_rate, output='sos')
    zi = signal.sosfilt_zi(sos)

    return sos, zi


def design_decimation_chain(input_rate: float, output_rate: float,
                            passband_edge: Optional[float] = None) -> List[dict]:
    """
    Design multi-stage decimation chain.

    Args:
        input_rate: Input sample rate (Hz)
        output_rate: Desired output sample rate (Hz)
        passband_edge: Passband edge (Hz), defaults to output_rate / 2.5

    Returns:
        List of decimation stages with filter coefficients
    """
    if passband_edge is None:
        passband_edge = output_rate / 2.5  # Leave some margin

    total_decimation = input_rate / output_rate

    # Decompose into stages (prefer decimation by 2, 4, or 8)
    stages = []
    current_rate = input_rate

    while current_rate > output_rate * 1.1:  # 10% tolerance
        # Choose decimation factor
        if current_rate / output_rate >= 8:
            decim = 8
        elif current_rate / output_rate >= 4:
            decim = 4
        elif current_rate / output_rate >= 2:
            decim = 2
        else:
            break

        # Design filter for this stage
        taps = design_decimation_filter(current_rate, decim, passband_edge)

        stages.append({
            'input_rate': current_rate,
            'output_rate': current_rate / decim,
            'decimation': decim,
            'taps': taps,
            'num_taps': len(taps),
        })

        current_rate /= decim

    # Add final resampling stage if needed
    if abs(current_rate - output_rate) > 1:
        ratio = output_rate / current_rate
        stages.append({
            'input_rate': current_rate,
            'output_rate': output_rate,
            'decimation': None,  # Resampling
            'ratio': ratio,
            'taps': None,
            'num_taps': 0,
        })

    return stages


def plot_frequency_response(taps: np.ndarray, sample_rate: float, title: str):
    """Plot frequency response of FIR filter."""
    w, h = signal.freqz(taps, fs=sample_rate, worN=2048)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Magnitude response
    ax1.plot(w, 20 * np.log10(abs(h)))
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Magnitude (dB)')
    ax1.set_title(f'{title} - Magnitude Response')
    ax1.grid(True)
    ax1.axhline(-6, color='r', linestyle='--', label='-6 dB')
    ax1.axhline(-60, color='orange', linestyle='--', label='-60 dB')
    ax1.legend()

    # Phase response
    ax2.plot(w, np.angle(h))
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Phase (radians)')
    ax2.set_title(f'{title} - Phase Response')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def generate_rust_code(taps: np.ndarray, name: str) -> str:
    """Generate Rust code for filter coefficients."""
    code = f"// Auto-generated filter coefficients: {name}\n"
    code += f"// Number of taps: {len(taps)}\n\n"
    code += f"pub const {name.upper()}_TAPS: [f32; {len(taps)}] = [\n"

    for i, tap in enumerate(taps):
        code += f"    {tap:.10f}f32,\n"

    code += "];\n"
    return code


def main():
    parser = argparse.ArgumentParser(description='SDR Filter Designer')
    parser.add_argument('--mode', choices=['decimation', 'hilbert', 'deemphasis', 'chain'],
                       required=True, help='Filter design mode')

    # Decimation filter args
    parser.add_argument('--sample-rate', type=float, help='Sample rate (Hz)')
    parser.add_argument('--decimation', type=int, help='Decimation factor')
    parser.add_argument('--passband', type=float, help='Passband edge (Hz)')
    parser.add_argument('--stopband-atten', type=float, default=60,
                       help='Stopband attenuation (dB)')
    parser.add_argument('--max-taps', type=int, help='Maximum number of taps')

    # Hilbert filter args
    parser.add_argument('--num-taps', type=int, default=65, help='Number of taps')

    # De-emphasis filter args
    parser.add_argument('--time-constant', type=float, default=75,
                       help='Time constant (us)')

    # Decimation chain args
    parser.add_argument('--input-rate', type=float, help='Input sample rate (Hz)')
    parser.add_argument('--output-rate', type=float, help='Output sample rate (Hz)')

    # Output options
    parser.add_argument('--plot', action='store_true', help='Show frequency response plot')
    parser.add_argument('--rust', action='store_true', help='Generate Rust code')
    parser.add_argument('--name', type=str, default='filter', help='Filter name for Rust code')

    args = parser.parse_args()

    if args.mode == 'decimation':
        if not all([args.sample_rate, args.decimation, args.passband]):
            parser.error('decimation mode requires --sample-rate, --decimation, --passband')

        print(f"Designing decimation filter:")
        print(f"  Sample rate: {args.sample_rate:,.0f} Hz")
        print(f"  Decimation: {args.decimation}")
        print(f"  Passband edge: {args.passband:,.0f} Hz")
        print(f"  Stopband attenuation: {args.stopband_atten} dB")

        taps = design_decimation_filter(
            args.sample_rate, args.decimation, args.passband,
            args.stopband_atten, args.max_taps
        )

        print(f"\nFilter designed:")
        print(f"  Number of taps: {len(taps)}")
        print(f"  Output rate: {args.sample_rate / args.decimation:,.0f} Hz")

        # Estimate computational cost
        cost = len(taps) * args.sample_rate / 1e6
        print(f"  Computational cost: {cost:.1f} million multiplies/sec")

        if args.plot:
            plot_frequency_response(taps, args.sample_rate, 'Decimation Filter')

        if args.rust:
            print("\nRust code:\n")
            print(generate_rust_code(taps, args.name))

    elif args.mode == 'hilbert':
        print(f"Designing Hilbert transformer:")
        print(f"  Number of taps: {args.num_taps}")

        taps = design_hilbert_filter(args.num_taps)

        print(f"\nHilbert filter designed")

        if args.plot:
            # For Hilbert filter, sample rate doesn't matter (it's normalized)
            plot_frequency_response(taps, 1.0, 'Hilbert Transformer')

        if args.rust:
            print("\nRust code:\n")
            print(generate_rust_code(taps, args.name or 'hilbert'))

    elif args.mode == 'deemphasis':
        if not args.sample_rate:
            parser.error('deemphasis mode requires --sample-rate')

        print(f"Designing de-emphasis filter:")
        print(f"  Sample rate: {args.sample_rate:,.0f} Hz")
        print(f"  Time constant: {args.time_constant} us")

        sos, _ = design_deemphasis_filter(args.sample_rate, args.time_constant)

        tau = args.time_constant * 1e-6
        cutoff = 1.0 / (2 * np.pi * tau)
        print(f"\nDe-emphasis filter designed:")
        print(f"  Cutoff frequency: {cutoff:.1f} Hz")

        if args.plot:
            # Create impulse response for plotting
            impulse = np.zeros(1000)
            impulse[0] = 1.0
            response = signal.sosfilt(sos, impulse)

            plt.figure(figsize=(10, 6))
            plt.plot(response)
            plt.title('De-emphasis Filter Impulse Response')
            plt.xlabel('Sample')
            plt.ylabel('Amplitude')
            plt.grid(True)
            plt.show()

    elif args.mode == 'chain':
        if not all([args.input_rate, args.output_rate]):
            parser.error('chain mode requires --input-rate and --output-rate')

        print(f"Designing decimation chain:")
        print(f"  Input rate: {args.input_rate:,.0f} Hz")
        print(f"  Output rate: {args.output_rate:,.0f} Hz")
        print(f"  Total decimation: {args.input_rate / args.output_rate:.2f}")

        stages = design_decimation_chain(args.input_rate, args.output_rate, args.passband)

        print(f"\nDecimation chain ({len(stages)} stages):")
        total_cost = 0

        for i, stage in enumerate(stages, 1):
            print(f"\nStage {i}:")
            print(f"  {stage['input_rate']:,.0f} Hz → {stage['output_rate']:,.0f} Hz")

            if stage['decimation']:
                print(f"  Decimation: {stage['decimation']}")
                print(f"  Taps: {stage['num_taps']}")
                cost = stage['num_taps'] * stage['input_rate'] / 1e6
                total_cost += cost
                print(f"  Cost: {cost:.1f} million multiplies/sec")
            else:
                print(f"  Resampling: {stage['ratio']:.3f}")

        print(f"\nTotal computational cost: {total_cost:.1f} million multiplies/sec")

        if args.rust:
            print("\nRust code:\n")
            for i, stage in enumerate(stages, 1):
                if stage['taps'] is not None:
                    name = f"{args.name}_stage{i}"
                    print(generate_rust_code(stage['taps'], name))
                    print()


if __name__ == '__main__':
    main()
