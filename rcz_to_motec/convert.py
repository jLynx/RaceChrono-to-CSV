#!/usr/bin/env python3
"""
RCZ to MoTeC Direct Converter

Converts RaceChrono .rcz session files directly to MoTeC .ld + .ldx files,
bypassing the intermediate CSV format for a binary-to-binary conversion.

Usage:
    python -m rcz_to_motec.convert session.rcz
    python -m rcz_to_motec.convert session.rcz --output output.ld --driver "Name"
"""

import sys
import os
import math
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

# Add parent directory to path so we can import decoder and channel_ids
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from decoder import RaceChronoDecoder
from rcz_to_motec.motec_writer import MotecWriter
from rcz_to_motec.motec_ldx import write_ldx


def detect_frequency(timestamps):
    """Detect the native sample frequency from a timestamp array.

    Returns the frequency in Hz (integer, minimum 1).
    """
    if len(timestamps) < 2:
        return 1
    dt = (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1)
    if dt <= 0:
        return 1
    return max(1, int(round(1.0 / dt)))


def resample_to_uniform(timestamps, values, freq, method='linear'):
    """Resample irregular timestamps to a uniform grid at the given frequency.

    Args:
        timestamps: numpy array of timestamps in seconds
        values: numpy array of channel values
        freq: target frequency in Hz
        method: 'linear' for linear interpolation, 'zoh' for zero-order hold

    Returns:
        numpy array of resampled values on a uniform grid
    """
    if len(timestamps) < 2:
        return values.copy()

    duration = timestamps[-1] - timestamps[0]
    n_samples = max(1, int(duration * freq) + 1)
    uniform_t = np.linspace(timestamps[0], timestamps[0] + (n_samples - 1) / freq, n_samples)

    # Filter out NaN values for interpolation
    valid = ~np.isnan(values)
    if np.sum(valid) < 2:
        return np.full(n_samples, 0.0)

    if method == 'zoh':
        # Zero-order hold: use most recent value
        result = np.zeros(n_samples)
        valid_ts = timestamps[valid]
        valid_vals = values[valid]
        for i, t in enumerate(uniform_t):
            idx = np.searchsorted(valid_ts, t, side='right') - 1
            if 0 <= idx < len(valid_vals):
                result[i] = valid_vals[idx]
        return result
    else:
        # Linear interpolation
        return np.interp(uniform_t, timestamps[valid], values[valid])


def compute_calculated_fields(decoder):
    """Compute derived fields from raw GPS data.

    Returns dict of {name: (data_array, unit, freq)} for calculated channels.
    """
    fields = {}

    if 'speed' not in decoder.channels or decoder.gps_timestamps is None:
        return fields

    raw_speed = decoder.channels['speed'].values
    raw_ts = decoder.channels['speed'].timestamps
    n_gps = len(raw_ts)
    freq = detect_frequency(raw_ts)

    N = 3  # Central difference width

    # Longitudinal acceleration (from GPS speed derivative)
    raw_long_acc = np.full(n_gps, np.nan)
    for i in range(N, n_gps - N):
        dt = raw_ts[i + N] - raw_ts[i - N]
        if dt > 0 and not np.isnan(raw_speed[i + N]) and not np.isnan(raw_speed[i - N]):
            raw_long_acc[i] = (raw_speed[i + N] - raw_speed[i - N]) / dt / 9.81

    # Fill edges
    for i in range(1, min(N, n_gps - 1)):
        dt = raw_ts[i + 1] - raw_ts[i - 1]
        if dt > 0 and not np.isnan(raw_speed[i + 1]) and not np.isnan(raw_speed[i - 1]):
            raw_long_acc[i] = (raw_speed[i + 1] - raw_speed[i - 1]) / dt / 9.81
    for i in range(max(n_gps - N, 1), n_gps - 1):
        hi = min(i + 1, n_gps - 1)
        lo = max(i - 1, 0)
        dt = raw_ts[hi] - raw_ts[lo]
        if dt > 0:
            raw_long_acc[i] = (raw_speed[hi] - raw_speed[lo]) / dt / 9.81

    fields['Longitudinal Acc'] = (raw_long_acc, 'G', raw_ts, freq)

    # Lateral acceleration (from GPS speed and bearing)
    raw_lat_acc = np.full(n_gps, np.nan)
    if 'bearing' in decoder.channels:
        raw_bearing = decoder.channels['bearing'].values
        for i in range(N, n_gps - N):
            dt = raw_ts[i + N] - raw_ts[i - N]
            if dt > 0 and not np.isnan(raw_bearing[i + N]) and not np.isnan(raw_bearing[i - N]):
                db = raw_bearing[i + N] - raw_bearing[i - N]
                if db > 180:
                    db -= 360
                elif db < -180:
                    db += 360
                yaw_rate = math.radians(db) / dt
                spd = raw_speed[i] if not np.isnan(raw_speed[i]) else 0
                raw_lat_acc[i] = -(spd * yaw_rate / 9.81)

    fields['Lateral Acc'] = (raw_lat_acc, 'G', raw_ts, freq)

    # Combined acceleration
    valid = ~np.isnan(raw_long_acc) & ~np.isnan(raw_lat_acc)
    combined_acc = np.full(n_gps, np.nan)
    combined_acc[valid] = np.sqrt(raw_long_acc[valid] ** 2 + raw_lat_acc[valid] ** 2)
    fields['Combined Acc'] = (combined_acc, 'G', raw_ts, freq)

    # Lean angle
    lean_angle = np.full(n_gps, np.nan)
    lean_angle[valid] = np.degrees(np.arctan(-raw_lat_acc[valid]))
    fields['Lean Angle'] = (lean_angle, 'deg', raw_ts, freq)

    return fields


# Channel name mapping for MoTeC display: raw_name -> (motec_name, unit)
GPS_CHANNEL_NAMES = {
    'latitude': ('GPS Latitude', 'deg'),
    'longitude': ('GPS Longitude', 'deg'),
    'speed': ('GPS Speed', 'm/s'),
    'altitude': ('GPS Altitude', 'm'),
    'bearing': ('GPS Bearing', 'deg'),
    'distance_traveled': ('GPS Distance', 'm'),
    'accuracy': ('GPS Accuracy', 'm'),
    'satellites': ('GPS Satellites', ''),
    'fix_type': ('GPS Fix Type', ''),
    'device_battery_level': ('GPS Battery', '%'),
}

SENSOR_CHANNEL_NAMES = {
    'x_acc': ('X Acc', 'G'),
    'y_acc': ('Y Acc', 'G'),
    'z_acc': ('Z Acc', 'G'),
    'x_rate_of_rotation': ('X Gyro', 'deg/s'),
    'y_rate_of_rotation': ('Y Gyro', 'deg/s'),
    'z_rate_of_rotation': ('Z Gyro', 'deg/s'),
    'x_magnetic_field': ('X Mag', 'uT'),
    'y_magnetic_field': ('Y Mag', 'uT'),
    'z_magnetic_field': ('Z Mag', 'uT'),
}


def convert_rcz_to_motec(rcz_path, output_path=None, driver="", vehicle_id="",
                          venue_name="", event_name="", event_session="",
                          short_comment="", long_comment=""):
    """Convert an RCZ file directly to MoTeC .ld + .ldx files.

    Args:
        rcz_path: Path to .rcz file
        output_path: Output .ld file path (default: same name as input with .ld extension)
        driver: Driver name for MoTeC metadata
        vehicle_id: Vehicle ID for MoTeC metadata
        venue_name: Venue/track name for MoTeC metadata
        event_name: Event name for MoTeC metadata
        event_session: Session name for MoTeC metadata
        short_comment: Short comment for MoTeC metadata
        long_comment: Long comment for MoTeC metadata
    """
    rcz_path = Path(rcz_path)

    if output_path:
        ld_path = Path(output_path)
    else:
        ld_path = rcz_path.with_suffix('.ld')
    ldx_path = ld_path.with_suffix('.ldx')

    print(f"Converting: {rcz_path.name} -> {ld_path.name}")
    print("=" * 70)

    # Step 1: Parse RCZ binary data using existing decoder
    with RaceChronoDecoder(rcz_path) as decoder:
        if not decoder.load_metadata():
            print("ERROR: Failed to load metadata")
            return False

        decoder.decode_gps_channels()
        decoder.decode_processed_channels()
        decoder.decode_sensor_channels()
        # We intentionally skip resample_to_common_timeline() and export_to_csv()

        # Get session metadata
        session_info = decoder.session_info or {}
        if not venue_name:
            venue_name = session_info.get('trackName', '') or session_info.get('firstPositionReverseGeocoding', '')
        if not event_name:
            event_name = session_info.get('title', '') or venue_name
        if not event_session:
            event_session = session_info.get('title', '')

        # Determine session datetime
        created_ts = session_info.get('timeCreated', session_info.get('firstTimestamp', 0)) / 1000
        session_dt = datetime.fromtimestamp(created_ts, tz=timezone.utc)
        # Convert to naive datetime for MoTeC (it doesn't handle timezone)
        session_dt = session_dt.replace(tzinfo=None)

        # Build comment from session stats if not provided
        if not short_comment:
            lap_count = session_info.get('lapCount', 0)
            best_lap_ms = session_info.get('bestLaptime', 0)
            if lap_count > 0 and best_lap_ms > 0 and best_lap_ms < 3600000:
                best_min = int(best_lap_ms / 60000)
                best_sec = (best_lap_ms % 60000) / 1000
                short_comment = f"{lap_count} laps, best {best_min}:{best_sec:06.3f}"
            elif lap_count > 0:
                short_comment = f"{lap_count} laps"

        if not long_comment:
            notes = session_info.get('description', '')
            duration_sec = session_info.get('lengthTime', 0) / 1000
            distance_m = session_info.get('lengthDistance', 0) / 1000
            parts = []
            if notes:
                parts.append(notes)
            if duration_sec > 0:
                mins = int(duration_sec // 60)
                secs = int(duration_sec % 60)
                parts.append(f"Duration: {mins}m {secs}s")
            if distance_m > 0:
                parts.append(f"Distance: {distance_m/1000:.1f} km")
            long_comment = '. '.join(parts)

        # Step 2: Create MoTeC writer
        print("\n" + "=" * 70)
        print("BUILDING MOTEC FILE")
        print("=" * 70)
        print(f"  Venue: {venue_name}")
        print(f"  Event: {event_name}")
        print(f"  Session: {event_session}")
        print(f"  Date: {session_dt.strftime('%d/%m/%Y %H:%M')}")
        if short_comment:
            print(f"  Info: {short_comment}")

        writer = MotecWriter(
            dt=session_dt,
            driver=driver,
            vehicle_id=vehicle_id,
            venue_name=venue_name,
            event_name=event_name,
            event_session=event_session,
            short_comment=short_comment,
            long_comment=long_comment,
        )

        # Step 3: Add GPS channels at their native rate
        if decoder.gps_timestamps is not None and len(decoder.gps_timestamps) > 1:
            gps_freq = detect_frequency(decoder.gps_timestamps)
            gps_ts = decoder.gps_timestamps
            print(f"GPS channels at {gps_freq} Hz")

            for raw_name, (motec_name, unit) in GPS_CHANNEL_NAMES.items():
                if raw_name not in decoder.channels:
                    continue
                ch = decoder.channels[raw_name]
                method = 'zoh' if raw_name == 'bearing' else 'linear'
                data = resample_to_uniform(ch.timestamps, ch.values, gps_freq, method=method)
                writer.add_channel(motec_name, unit, data, gps_freq)
                print(f"  [+] {motec_name}: {len(data)} samples")

        # Step 4: Add CAN bus channels at their native rates
        if decoder.canbus_channel_order:
            print(f"CAN bus channels: {len(decoder.canbus_channel_order)}")
            for can_name in decoder.canbus_channel_order:
                key = f'canbus_{can_name}'
                if key not in decoder.channels:
                    continue
                ch = decoder.channels[key]
                freq = detect_frequency(ch.timestamps)
                data = resample_to_uniform(ch.timestamps, ch.values, freq)

                # Format name for MoTeC display
                motec_name = can_name.replace('_', ' ').title()
                unit = decoder.canbus_channel_units.get(can_name, '')
                writer.add_channel(motec_name, unit, data, freq)
                print(f"  [+] {motec_name}: {len(data)} samples at {freq} Hz")

        # Step 5: Add sensor channels at their native rates
        # Count devices per sensor type to detect duplicates
        type_counts = {}
        for sensor in decoder.sensor_devices:
            dev_type = sensor['dev_type']
            type_counts[dev_type] = type_counts.get(dev_type, 0) + 1

        for sensor in decoder.sensor_devices:
            dev_id = sensor['dev_id']
            dev_type = sensor['dev_type']
            source = sensor['source']
            sensor_ts = sensor['timestamps']
            freq = detect_frequency(sensor_ts)
            # Add device ID suffix when multiple devices share the same type
            needs_suffix = type_counts.get(dev_type, 1) > 1
            suffix = f" {dev_id}" if needs_suffix else ""
            print(f"Sensor {source} at {freq} Hz")

            for col_name in sensor['columns']:
                data_key = f'sensor_{dev_id}_{col_name}'
                if data_key not in decoder.channels:
                    continue
                ch = decoder.channels[data_key]

                if col_name in SENSOR_CHANNEL_NAMES:
                    motec_name, unit = SENSOR_CHANNEL_NAMES[col_name]
                else:
                    motec_name = col_name.replace('_', ' ').title()
                    unit = ch.unit

                motec_name = motec_name + suffix
                data = resample_to_uniform(ch.timestamps, ch.values, freq)
                writer.add_channel(motec_name, unit, data, freq)
                print(f"  [+] {motec_name}: {len(data)} samples")

        # Step 6: Add calculated fields at GPS rate
        calc_fields = compute_calculated_fields(decoder)
        if calc_fields:
            print("Calculated fields:")
            for name, (values, unit, timestamps, freq) in calc_fields.items():
                data = resample_to_uniform(timestamps, values, freq)
                writer.add_channel(name, unit, data, freq)
                print(f"  [+] {name}: {len(data)} samples at {freq} Hz")

        # Step 7: Write .ld file
        print()
        output_dir = ld_path.parent
        if output_dir and not output_dir.exists():
            os.makedirs(output_dir)

        writer.write(str(ld_path))

        # Step 8: Write .ldx file with lap markers
        laps = session_info.get('laps', [])
        if laps and decoder.gps_timestamps is not None:
            data_start_ms = decoder.gps_timestamps[0] * 1000
            write_ldx(str(ldx_path), laps, data_start_ms)
        else:
            print("No lap data found, skipping .ldx file")

    print("\nDone!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert RaceChrono .rcz files directly to MoTeC .ld format"
    )
    parser.add_argument("rcz_file", type=str, help="Path to .rcz file")
    parser.add_argument("--output", type=str, help="Output .ld file path")
    parser.add_argument("--driver", type=str, default="", help="Driver name")
    parser.add_argument("--vehicle_id", type=str, default="", help="Vehicle ID")
    parser.add_argument("--venue_name", type=str, default="", help="Venue/track name")
    parser.add_argument("--event_name", type=str, default="", help="Event name")
    parser.add_argument("--event_session", type=str, default="", help="Session name")
    parser.add_argument("--short_comment", type=str, default="", help="Short comment")
    parser.add_argument("--long_comment", type=str, default="", help="Long comment")

    args = parser.parse_args()

    rcz_path = Path(args.rcz_file).resolve()
    if not rcz_path.exists():
        print(f"ERROR: File not found: {rcz_path}")
        sys.exit(1)

    if not rcz_path.suffix.lower() == '.rcz':
        print(f"WARNING: File does not have .rcz extension: {rcz_path.name}")

    success = convert_rcz_to_motec(
        rcz_path,
        output_path=args.output,
        driver=args.driver,
        vehicle_id=args.vehicle_id,
        venue_name=args.venue_name,
        event_name=args.event_name,
        event_session=args.event_session,
        short_comment=args.short_comment,
        long_comment=args.long_comment,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
