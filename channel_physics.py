"""Pure numerical helpers shared by the physical-channel plugins.

This module intentionally has no Streamlit, Sionna, Torch, or Mitsuba imports.
It keeps the conventions from ``docs/sionna_rt_revised_spec.md`` testable on
machines without a ray-tracing GPU backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

BOLTZMANN = 1.380649e-23


@dataclass(frozen=True)
class DuplexTiming:
    output_time_steps: int
    internal_time_steps: int
    internal_sampling_frequency: float | None
    requested_time_steps_ignored: bool


def resolve_duplex_timing(
    model: str,
    requested_time_steps: int,
    sampling_frequency: float | None,
    duplex_interval: float,
) -> DuplexTiming:
    """Resolve non-overlapping CIR and duplex time controls."""
    if requested_time_steps < 1:
        raise ValueError("num_time_steps must be at least one")
    if model in {"Disabled", "Legacy Duplicate"}:
        if requested_time_steps > 1 and (
            sampling_frequency is None or sampling_frequency <= 0
        ):
            raise ValueError(
                "sampling_frequency must be positive when num_time_steps > 1"
            )
        return DuplexTiming(
            requested_time_steps,
            requested_time_steps,
            sampling_frequency if requested_time_steps > 1 else None,
            False,
        )
    if model == "Reciprocal at Same Time":
        return DuplexTiming(1, 1, None, requested_time_steps != 1)
    if model == "Time-Separated Duplex":
        if duplex_interval <= 0:
            raise ValueError("Time-Separated Duplex requires duplex_interval > 0")
        return DuplexTiming(1, 2, 1.0 / duplex_interval, requested_time_steps != 1)
    raise ValueError(f"Unknown duplex model: {model}")


def automatic_device_batch_sizes(
    *, num_tx: int, num_rx: int, max_num_paths_per_src: int,
    tx_antennas: int = 1, synthetic_array: bool = True,
    target_capacity_per_rx: int = 10_000,
    max_buffer_entries: int = 10_000_000, max_tx: int = 128,
    max_rx: int = 128, safety_factor: float = 1.25,
) -> tuple[int, int]:
    """Apply the candidate-capacity heuristic from section 1 of the spec."""
    if min(num_tx, num_rx, max_num_paths_per_src, target_capacity_per_rx) < 1:
        raise ValueError("device counts and capacities must be positive")
    if safety_factor < 1:
        raise ValueError("safety_factor must be at least one")
    rx_size = max_num_paths_per_src // int(np.ceil(safety_factor * target_capacity_per_rx))
    sources_per_tx = 1 if synthetic_array else max(1, int(tx_antennas))
    tx_size = max_buffer_entries // (sources_per_tx * max_num_paths_per_src)
    return min(num_tx, max_tx, max(1, tx_size)), min(num_rx, max_rx, max(1, rx_size))


def random_velocity_vectors(
    count: int, speed_min: float, speed_max: float, *, seed: int = 0,
    heading_mode: str = "Random", heading_min_deg: float = 0.0,
    heading_max_deg: float = 360.0, fixed_heading_deg: float = 0.0,
    vertical_velocity: float = 0.0,
) -> np.ndarray:
    """Generate reproducible polar (not Cartesian-uniform) RX velocities."""
    if count < 0 or speed_min < 0 or speed_max < speed_min:
        raise ValueError("invalid count or speed interval")
    rng = np.random.default_rng(seed)
    speed = rng.uniform(speed_min, speed_max, count)
    if heading_mode.lower() == "random":
        heading = rng.uniform(heading_min_deg, heading_max_deg, count)
    else:
        heading = np.full(count, fixed_heading_deg)
    angle = np.deg2rad(heading)
    return np.column_stack((speed * np.cos(angle), speed * np.sin(angle), np.full(count, vertical_velocity)))


@dataclass(frozen=True)
class TrajectoryState:
    position: np.ndarray
    velocity: np.ndarray
    direction: np.ndarray
    extrapolated: bool


def interpolate_trajectory(
    positions: Sequence[Sequence[float]], timestamps: Sequence[float], time_s: float,
    *, allow_extrapolation: bool = False,
) -> TrajectoryState:
    """Piecewise-linear position and its matching derivative for one segment."""
    p = np.asarray(positions, dtype=float)
    t = np.asarray(timestamps, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) != len(t) or len(t) < 2:
        raise ValueError("trajectory needs at least two 3-D positions and timestamps")
    if np.any(np.diff(t) <= 0):
        raise ValueError("trajectory timestamps must be strictly increasing")
    outside = time_s < t[0] or time_s > t[-1]
    if outside and not allow_extrapolation:
        raise ValueError("requested time is outside the trajectory segment")
    i = int(np.clip(np.searchsorted(t, time_s, side="right") - 1, 0, len(t) - 2))
    velocity = (p[i + 1] - p[i]) / (t[i + 1] - t[i])
    # At an exact internal knot use the duration-weighted two-sided tangent.
    knot = np.flatnonzero(np.isclose(t, time_s, rtol=0, atol=1e-12))
    if knot.size and 0 < knot[0] < len(t) - 1:
        k = int(knot[0])
        dl, dr = t[k] - t[k - 1], t[k + 1] - t[k]
        vl, vr = (p[k] - p[k - 1]) / dl, (p[k + 1] - p[k]) / dr
        velocity = (vl * dl + vr * dr) / (dl + dr)
    position = p[i] + (time_s - t[i]) * (p[i + 1] - p[i]) / (t[i + 1] - t[i])
    norm = np.linalg.norm(velocity)
    direction = velocity / norm if norm else np.zeros(3)
    return TrajectoryState(position, velocity, direction, outside)


def scale_waveform_to_power(signal: np.ndarray, power_dbm: float) -> np.ndarray:
    """Interpret dBm as total device RMS waveform power."""
    x = np.asarray(signal)
    rms = float(np.sqrt(np.mean(np.abs(x) ** 2))) if x.size else 0.0
    if rms == 0:
        return np.zeros_like(x)
    power_watt = 10.0 ** ((float(power_dbm) - 30.0) / 10.0)
    return x / rms * np.sqrt(power_watt)


def thermal_noise_variance(
    temperature_k: float, bandwidth_hz: float, noise_figure_db: float,
    rx_gain_db: float = 0.0,
) -> float:
    """Output complex-baseband E[|n|²] = G k T B F."""
    if temperature_k < 0 or bandwidth_hz < 0:
        raise ValueError("temperature and bandwidth cannot be negative")
    return BOLTZMANN * temperature_k * bandwidth_hz * 10 ** ((noise_figure_db + rx_gain_db) / 10)


def add_complex_noise(signal: np.ndarray, variance: float, seed: int) -> np.ndarray:
    """Add circular complex Gaussian noise with total complex variance."""
    if variance < 0:
        raise ValueError("variance cannot be negative")
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(variance / 2)
    noise = rng.normal(0, sigma, np.shape(signal)) + 1j * rng.normal(0, sigma, np.shape(signal))
    return np.asarray(signal) + noise


def occupied_power_bandwidth(signal: np.ndarray, sample_rate: float, fraction: float = 0.99) -> dict:
    """Return the narrowest contiguous FFT interval containing a power fraction."""
    x = np.asarray(signal).reshape(-1)
    if x.size < 2 or sample_rate <= 0 or not 0 < fraction <= 1:
        raise ValueError("invalid signal, sample rate, or occupied fraction")
    power = np.abs(np.fft.fftshift(np.fft.fft(x))) ** 2
    freq = np.fft.fftshift(np.fft.fftfreq(x.size, 1 / sample_rate))
    total = power.sum()
    if total == 0:
        return {"lower_hz": 0.0, "upper_hz": 0.0, "bandwidth_hz": 0.0, "fraction": fraction}
    csum = np.concatenate(([0.0], np.cumsum(power)))
    best = (0, len(power) - 1)
    for lo in range(len(power)):
        hi = int(np.searchsorted(csum, csum[lo] + fraction * total, side="left") - 1)
        if hi < len(power) and hi - lo < best[1] - best[0]:
            best = lo, max(lo, hi)
    lower, upper = float(freq[best[0]]), float(freq[best[1]])
    return {"lower_hz": lower, "upper_hz": upper, "bandwidth_hz": upper - lower, "fraction": fraction}


def duplex_grid_time(interval_s: float, sampling_frequency: float) -> tuple[int, float, float]:
    """Map Δt onto the CIR grid and return (index, actual time, error)."""
    if interval_s < 0 or sampling_frequency <= 0:
        raise ValueError("invalid duplex interval or sampling frequency")
    index = int(round(interval_s * sampling_frequency))
    actual = index / sampling_frequency
    return index, actual, actual - interval_s


def split_tdl_duplex_windows(
    a, tau, samples_per_side: int, second_window_start: int | None = None
):
    """Pack consecutive TDL time windows into first/second batch halves."""
    coeff = np.asarray(a)
    delay = np.asarray(tau)
    second_start = (
        samples_per_side
        if second_window_start is None
        else int(second_window_start)
    )
    if (
        samples_per_side < 1
        or second_start < 0
        or coeff.shape[-1] < second_start + samples_per_side
    ):
        raise ValueError("TDL CIR does not contain two complete side windows")
    first = coeff[..., :samples_per_side]
    second = coeff[..., second_start : second_start + samples_per_side]
    return (
        np.concatenate([first, second], axis=0),
        np.concatenate([delay, delay], axis=0),
    )
