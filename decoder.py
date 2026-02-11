"""
RaceChrono Session Decoder

Handles decoding of RaceChrono session files and exports to CSV format
matching the official RaceChrono app export.
"""

import math
import struct
import json
import csv
import zipfile
import tempfile
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

import re
import channel_ids


class ChannelData:
    """Container for channel data with timestamps"""
    def __init__(self, name: str, timestamps: List[float], values: List[Any], unit: str = "", source: str = ""):
        self.name = name
        self.timestamps = np.array(timestamps)  # in seconds
        self.values = np.array(values)
        self.unit = unit
        self.source = source


class RaceChronoDecoder:
    """Decoder for RaceChrono session files"""

    # Channel scaling factors and configurations
    CHANNEL_CONFIG = {
        1: {'name': 'timestamp', 'scale': 1, 'unit': 's'},
        2: {'name': 'distance_traveled', 'scale': 1000, 'unit': 'm', 'source': '100: gps'},
        3: {'name': 'position', 'scale': 6000000, 'unit': 'deg', 'source': '100: gps'},  # lat/lon
        4: {'name': 'speed', 'scale': 1000, 'unit': 'm/s', 'source': '100: gps'},  # Fixed: was 10000
        5: {'name': 'altitude', 'scale': 1000, 'unit': 'm', 'source': '100: gps'},
        6: {'name': 'bearing', 'scale': 1000, 'unit': 'deg', 'source': '100: gps'},
        46: {'name': 'device_battery_level', 'scale': 1000, 'unit': '%'},
        30002: {'name': 'satellites', 'scale': 1, 'unit': 'sats', 'source': '100: gps'},  # No scaling
        30003: {'name': 'fix_type', 'scale': 1, 'unit': '', 'source': '100: gps'},  # No scaling
        30007: {'name': 'accuracy', 'scale': 1000, 'unit': 'm', 'source': '100: gps'},
        10024: {'name': 'rpm', 'scale': 1, 'unit': 'rpm', 'source': '200: canbus'},
    }

    # CAN bus channel ID → CSV column name and unit (keyed by channel_ids constants)
    CAN_CHANNEL_MAP = {
        channel_ids.Speed: {'name': 'speed', 'unit': 'm/s'},
        channel_ids.SteeringAngle: {'name': 'steering_angle', 'unit': 'deg'},
        channel_ids.BrakePedalPosition: {'name': 'brake_pos', 'unit': '%'},
        channel_ids.ClutchPedalPosition: {'name': 'clutch_pos', 'unit': '%'},
        channel_ids.Gear: {'name': 'gear', 'unit': ''},
        channel_ids.GearboxTemperature: {'name': 'gearbox_temp', 'unit': '.C'},
        channel_ids.EngineOilPressure: {'name': 'oil_pressure', 'unit': 'kPa'},
        channel_ids.ThrottleAngle: {'name': 'throttle_angle', 'unit': 'deg'},
        channel_ids.TyreTemperature: {'name': 'tyre_temp', 'unit': '.C'},
        channel_ids.TyrePressure: {'name': 'tyre_pressure', 'unit': 'kPa'},
        channel_ids.WheelSpeed: {'name': 'wheel_speed', 'unit': 'm/s'},
        channel_ids.SuspensionTravel: {'name': 'suspension_travel', 'unit': 'mm'},
        channel_ids.BrakePressure: {'name': 'brake_pressure', 'unit': 'kPa'},
        channel_ids.EnginePower: {'name': 'engine_power', 'unit': 'kW'},
        channel_ids.EngineTorque: {'name': 'engine_torque', 'unit': 'Nm'},
        channel_ids.EngineRpm: {'name': 'rpm', 'unit': 'rpm'},
        channel_ids.ThrottlePosition: {'name': 'throttle_pos', 'unit': '%'},
        channel_ids.EngineCoolantTemperature: {'name': 'coolant_temp', 'unit': '.C'},
        channel_ids.IntakeManifoldPressure: {'name': 'intake_pressure', 'unit': 'kPa'},
        channel_ids.TimingAdvance: {'name': 'timing_advance', 'unit': 'deg'},
        channel_ids.IntakeAirTemperature: {'name': 'intake_air_temp', 'unit': '.C'},
        channel_ids.MafAirflowRate: {'name': 'maf', 'unit': 'g/s'},
        channel_ids.AmbientAirTemperature: {'name': 'air_temp', 'unit': '.C'},
        channel_ids.EngineLoad: {'name': 'engine_load', 'unit': '%'},
        channel_ids.FuelPressure: {'name': 'fuel_pressure', 'unit': 'kPa'},
        channel_ids.FuelLevel: {'name': 'fuel_level', 'unit': '%'},
        channel_ids.BarometricPressure: {'name': 'baro_pressure', 'unit': 'kPa'},
        channel_ids.EngineOilTemperature: {'name': 'engine_oil_temp', 'unit': '.C'},
        channel_ids.EngineFuelRate: {'name': 'fuel_rate', 'unit': 'L/h'},
        channel_ids.AcceleratorPedalPosition: {'name': 'accelerator_pos', 'unit': '%'},
        channel_ids.Lambda: {'name': 'lambda', 'unit': ''},
    }

    # Sensor position suffix mapping (wheel positions)
    POSITION_SUFFIX = {
        0: '',
        3: '_rr',   # rear right
        4: '_rl',   # rear left
        5: '_fr',   # front right
        6: '_fl',   # front left
    }

    # Sensor device type → source label
    SENSOR_TYPE_LABEL = {
        2: 'acc',     # accelerometer
        3: 'gyro',    # gyroscope
        8: 'magn',    # magnetometer
    }

    # Sensor channels by device type (channel_id → name, unit)
    SENSOR_CHANNELS = {
        2: {  # accelerometer
            9:  {'name': 'x_acc', 'unit': 'G'},
            10: {'name': 'y_acc', 'unit': 'G'},
            11: {'name': 'z_acc', 'unit': 'G'},
        },
        3: {  # gyroscope
            12: {'name': 'x_rate_of_rotation', 'unit': 'deg/s'},
            13: {'name': 'y_rate_of_rotation', 'unit': 'deg/s'},
            14: {'name': 'z_rate_of_rotation', 'unit': 'deg/s'},
        },
        8: {  # magnetometer
            28: {'name': 'x_magnetic_field', 'unit': 'uT'},
            29: {'name': 'y_magnetic_field', 'unit': 'uT'},
            30: {'name': 'z_magnetic_field', 'unit': 'uT'},
        },
    }

    # Value that indicates "no data" in channel files
    NO_DATA_INT32 = 0x7FFFFFFF

    @classmethod
    def _get_canbus_column_info(cls, channel_id):
        """Get CSV column name and unit for a CAN bus channel ID.
        Handles sensor position (fl/fr/rl/rr) and sensor index suffixes."""
        base_id = channel_ids.get_base_channel_id(channel_id)
        position = channel_ids.get_sensor_position(channel_id)
        index = channel_ids.get_sensor_index(channel_id)

        # Look up base name from CAN_CHANNEL_MAP
        if base_id in cls.CAN_CHANNEL_MAP:
            cfg = cls.CAN_CHANNEL_MAP[base_id]
            name = cfg['name']
            unit = cfg['unit']
        else:
            # Fallback: CamelCase to snake_case
            camel_name = channel_ids.get_channel_name(base_id)
            snake_name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', camel_name)
            snake_name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', snake_name).lower()
            name = snake_name
            unit = ''

        # Append position suffix (fl/fr/rl/rr)
        pos_suffix = cls.POSITION_SUFFIX.get(position, f'_pos{position}')
        name += pos_suffix

        # Append sensor index suffix
        if index > 0:
            name += f'_{index}'

        return name, unit

    def __init__(self, session_path: Path):
        """
        Initialize decoder with either a .rcz file or an extracted session directory

        Args:
            session_path: Path to .rcz file or extracted session directory
        """
        self.session_path = Path(session_path)
        self.session_dir = None
        self.temp_dir = None
        self.session_info = None
        self.fragment_info = None
        self.channels = {}  # Dict of ChannelData objects
        self.resampled_data = {}  # Resampled data on common timeline
        self.gps_timestamps = None  # Raw GPS timestamps (seconds)
        self.gps_native_rate_hz = 1  # 1Hz for phone GPS, 25Hz for RaceBox
        self.primary_gps_device_id = 100  # From sessionfragment.json
        self.devices = []  # Device list from sessionfragment.json
        self.all_canbus_timestamps = []  # Timestamp arrays from all CAN channels
        self.canbus_channel_order = []  # CAN channel names in alphabetical order
        self.canbus_channel_units = {}  # CAN channel name → unit
        self.sensor_devices = []  # Discovered sensor device configs for CSV ordering
        self.all_sensor_timestamps = []  # Timestamp arrays from all sensor devices

        # Determine if we need to extract a .rcz file
        if self.session_path.is_file() and self.session_path.suffix.lower() == '.rcz':
            self._extract_rcz()
        elif self.session_path.is_dir():
            self.session_dir = self.session_path
        else:
            raise ValueError(f"Invalid path: {session_path}. Must be a .rcz file or directory.")

    def _extract_rcz(self):
        """Extract .rcz (ZIP) file to a temporary directory"""
        print(f"Extracting RCZ file: {self.session_path.name}")

        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix='racechrono_')
        self.session_dir = Path(self.temp_dir)

        # Extract ZIP file
        try:
            with zipfile.ZipFile(self.session_path, 'r') as zip_ref:
                zip_ref.extractall(self.session_dir)
            print(f"Extracted to temporary location: {self.session_dir}")
        except zipfile.BadZipFile:
            self._cleanup()
            raise ValueError(f"Invalid RCZ file: {self.session_path}")

    def _cleanup(self):
        """Clean up temporary directory if it was created"""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
            print(f"Cleaned up temporary files")

    def __del__(self):
        """Destructor to ensure cleanup"""
        self._cleanup()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self._cleanup()
        return False

    def _gps_source(self):
        """Get source label for GPS channels"""
        return f'{self.primary_gps_device_id}: gps'

    def load_metadata(self) -> bool:
        """Load session.json and sessionfragment.json metadata"""
        session_file = self.session_dir / "session.json"

        if not session_file.exists():
            print(f"Error: session.json not found in {self.session_dir}")
            return False

        with open(session_file, 'r') as f:
            self.session_info = json.load(f)

        # Load fragment metadata to discover device configuration
        fragment_file = self.session_dir / "sessionfragment.json"
        if fragment_file.exists():
            with open(fragment_file, 'r') as f:
                self.fragment_info = json.load(f)
            self.primary_gps_device_id = self.fragment_info.get('primaryGpsDeviceIndex', 100)
            devices_data = self.fragment_info.get('devices', {})
            if isinstance(devices_data, dict):
                self.devices = devices_data.get('items', [])
            elif isinstance(devices_data, list):
                self.devices = devices_data
            print(f"Primary GPS device: {self.primary_gps_device_id}")
            print(f"Devices found: {len(self.devices)}")

        return True

    def print_session_info(self):
        """Print session metadata"""
        if not self.session_info:
            return

        print("=" * 70)
        print("SESSION INFORMATION")
        print("=" * 70)
        print(f"Location: {self.session_info.get('firstPositionReverseGeocoding', 'Unknown')}")
        print(f"Start Time: {datetime.fromtimestamp(self.session_info['firstTimestamp']/1000)}")
        print(f"End Time: {datetime.fromtimestamp(self.session_info['latestTimestamp']/1000)}")
        print(f"Duration: {self.session_info['lengthTime']/1000:.1f} seconds")
        print(f"Distance: {self.session_info['lengthDistance']} meters")

        if 'firstPositionLatitude' in self.session_info:
            lat = self.session_info['firstPositionLatitude'] / 6_000_000
            lon = self.session_info['firstPositionLongitude'] / 6_000_000
            print(f"First Position: {lat:.6f}°, {lon:.6f}°")
        print()

    def _read_timestamps(self, filepath: Path) -> Optional[np.ndarray]:
        """Read timestamp file (64-bit unsigned integers in milliseconds)"""
        if not filepath.exists():
            return None

        with open(filepath, 'rb') as f:
            data = f.read()
            count = len(data) // 8
            timestamps_ms = struct.unpack(f'<{count}Q', data)
            # Convert to seconds
            return np.array([ts / 1000.0 for ts in timestamps_ms])

    def _read_int32_channel(self, filepath: Path, scale: float = 1.0) -> Optional[np.ndarray]:
        """Read channel file with 32-bit signed integers"""
        if not filepath.exists():
            return None

        with open(filepath, 'rb') as f:
            data = f.read()
            count = len(data) // 4
            values = struct.unpack(f'<{count}i', data)
            # Apply scaling and filter out NO_DATA values
            scaled_values = []
            for v in values:
                if v == self.NO_DATA_INT32:
                    scaled_values.append(np.nan)
                else:
                    scaled_values.append(v / scale)
            return np.array(scaled_values)

    def _read_double_channel(self, filepath: Path) -> Optional[np.ndarray]:
        """Read channel file with 64-bit doubles"""
        if not filepath.exists():
            return None

        with open(filepath, 'rb') as f:
            data = f.read()
            count = len(data) // 8
            values = struct.unpack(f'<{count}d', data)
            return np.array(values)

    def decode_gps_channels(self):
        """Decode all GPS-related channels using detected GPS device type"""
        print("=" * 70)
        print("DECODING GPS CHANNELS")
        print("=" * 70)

        dev_id = self.primary_gps_device_id
        gps_source = self._gps_source()

        # Read GPS timestamps
        gps_timestamps_file = self.session_dir / f"channel_1_{dev_id}_0_1_1"
        gps_timestamps = self._read_timestamps(gps_timestamps_file)

        if gps_timestamps is None:
            print(f"No GPS timestamps found (looked for channel_1_{dev_id}_0_1_1)")
            return

        self.gps_timestamps = gps_timestamps

        # Detect native GPS rate from actual timestamps
        if len(gps_timestamps) > 1:
            avg_dt = (gps_timestamps[-1] - gps_timestamps[0]) / (len(gps_timestamps) - 1)
            self.gps_native_rate_hz = round(1.0 / avg_dt) if avg_dt > 0 else 1
        print(f"[OK] GPS timestamps: {len(gps_timestamps)} points (~{self.gps_native_rate_hz}Hz)")

        # Decode GPS position (lat/lon) - special handling (interleaved int32 pairs)
        gps_pos_file = self.session_dir / f"channel_1_{dev_id}_0_3_1"
        if gps_pos_file.exists():
            with open(gps_pos_file, 'rb') as f:
                data = f.read()
                count = len(data) // 8
                lat_values = []
                lon_values = []
                for i in range(count):
                    lat_int, lon_int = struct.unpack('<ii', data[i*8:(i+1)*8])
                    lat_values.append(lat_int / 6_000_000)
                    lon_values.append(lon_int / 6_000_000)

                self.channels['latitude'] = ChannelData('latitude', gps_timestamps, lat_values, 'deg', gps_source)
                self.channels['longitude'] = ChannelData('longitude', gps_timestamps, lon_values, 'deg', gps_source)
                print(f"[OK] GPS position: {len(lat_values)} points")

        # Decode other GPS channels
        gps_channels = {
            2: ('distance_traveled', 2),  # Every other value (interleaved)
            4: ('speed', 1),
            5: ('altitude', 1),
            6: ('bearing', 1),
            46: ('device_battery_level', 1),
            30002: ('satellites', 1),
            30003: ('fix_type', 1),
            30007: ('accuracy', 1),
        }

        for channel_id, (name, stride) in gps_channels.items():
            config = self.CHANNEL_CONFIG.get(channel_id, {})
            scale = config.get('scale', 1.0)
            unit = config.get('unit', '')

            filepath = self.session_dir / f"channel_1_{dev_id}_0_{channel_id}_{'1' if channel_id == 2 else '0'}"
            values = self._read_int32_channel(filepath, scale)

            if values is not None:
                # Handle interleaved data (distance_traveled)
                if stride == 2:
                    values = values[::2]  # Take every other value

                # Ensure same length as timestamps
                if len(values) > len(gps_timestamps):
                    values = values[:len(gps_timestamps)]

                self.channels[name] = ChannelData(name, gps_timestamps, values, unit, gps_source)
                print(f"[OK] {name}: {len(values)} points")

    def decode_processed_channels(self):
        """Decode processed sensor channels (channel2_12_200_X_Y_3 files).
        Each CAN channel has its own timestamp file (channel_12_200_{id}_1_1)."""
        print("\n" + "=" * 70)
        print("DECODING PROCESSED CHANNELS")
        print("=" * 70)

        # Find all channel2 files
        channel2_files = list(self.session_dir.glob("channel2_12_200_*_*_3"))

        if not channel2_files:
            print("No processed channels found")
            return

        canbus_names = []

        for filepath in channel2_files:
            parts = filepath.stem.split('_')
            if len(parts) < 4:
                continue
            try:
                channel_id = int(parts[3])
            except (ValueError, IndexError):
                continue

            # Get channel name and unit from channel_ids mapping
            name, unit = self._get_canbus_column_info(channel_id)

            # Prefix CAN channel names to avoid collision with GPS channels
            # (e.g., both GPS and CAN have "speed")
            canbus_name = f'canbus_{name}'

            # Find per-channel timestamp file
            ts_file = self.session_dir / f"channel_12_200_{channel_id}_1_1"
            if not ts_file.exists():
                # Fallback: try shared timestamp file
                ts_file = self.session_dir / "channel_12_200_1_1_1"

            timestamps = self._read_timestamps(ts_file)
            if timestamps is None:
                print(f"[SKIP] No timestamps for channel {channel_id} ({name})")
                continue

            # Read values (doubles)
            values = self._read_double_channel(filepath)
            if values is None:
                continue

            # Match lengths (use minimum)
            min_len = min(len(timestamps), len(values))
            timestamps = timestamps[:min_len]
            values = values[:min_len]

            self.channels[canbus_name] = ChannelData(canbus_name, timestamps, values, unit, '200: canbus')
            self.all_canbus_timestamps.append(timestamps)
            canbus_names.append(name)
            self.canbus_channel_units[name] = unit
            print(f"[OK] {name} (id={channel_id}): {len(values)} points, range: {np.nanmin(values):.2f} - {np.nanmax(values):.2f}")

        # Store sorted CAN channel names for column ordering
        self.canbus_channel_order = sorted(set(canbus_names))
        print(f"CAN bus channels found: {', '.join(self.canbus_channel_order)}")

    def decode_sensor_channels(self):
        """Decode IMU sensor channels (accelerometer, gyroscope, magnetometer)."""
        if not self.devices:
            return

        print("\n" + "=" * 70)
        print("DECODING SENSOR CHANNELS")
        print("=" * 70)

        # Group devices by sensor type, sort by device ID within each type
        sensor_groups = {}  # type -> [(dev_id, dev_type)]
        for dev in self.devices:
            dev_id = dev.get('id')
            dev_type = dev.get('type')
            if dev_type in self.SENSOR_TYPE_LABEL:
                sensor_groups.setdefault(dev_type, []).append(dev_id)

        # Process in type order: 2 (acc), 3 (gyro), 8 (magn)
        for dev_type in sorted(sensor_groups.keys()):
            device_ids = sorted(sensor_groups[dev_type])
            type_label = self.SENSOR_TYPE_LABEL[dev_type]
            channel_defs = self.SENSOR_CHANNELS.get(dev_type, {})

            for dev_id in device_ids:
                source = f'{dev_id}: {type_label}'

                # Read timestamps
                ts_file = self.session_dir / f"channel_{dev_type}_{dev_id}_0_1_1"
                timestamps = self._read_timestamps(ts_file)
                if timestamps is None:
                    continue

                self.all_sensor_timestamps.append(timestamps)
                col_names = []

                for ch_id in sorted(channel_defs.keys()):
                    ch_info = channel_defs[ch_id]
                    data_file = self.session_dir / f"channel_{dev_type}_{dev_id}_0_{ch_id}_0"
                    values = self._read_int32_channel(data_file, 1000.0)
                    if values is None:
                        continue

                    # Convert accelerometer from m/s² to G
                    if dev_type == 2:
                        values = values / 9.80665

                    # Trim to match timestamps
                    min_len = min(len(timestamps), len(values))
                    ts_trimmed = timestamps[:min_len]
                    values = values[:min_len]

                    key = f'sensor_{dev_id}_{ch_info["name"]}'
                    self.channels[key] = ChannelData(key, ts_trimmed, values, ch_info['unit'], source)
                    col_names.append(ch_info['name'])

                if col_names:
                    self.sensor_devices.append({
                        'dev_id': dev_id,
                        'dev_type': dev_type,
                        'source': source,
                        'columns': col_names,
                        'timestamps': timestamps,
                    })
                    print(f"[OK] {source}: {len(timestamps)} points, channels: {', '.join(col_names)}")

    def _compute_device_update_rates(self, timestamps):
        """Compute instantaneous update rate (Hz) from a timestamp array.
        Returns array of same length with rate = 1/dt for each sample."""
        if len(timestamps) < 2:
            return np.full(len(timestamps), np.nan)
        rates = np.full(len(timestamps), np.nan)
        dt = np.diff(timestamps)
        rates[1:] = 1.0 / np.where(dt > 0, dt, np.nan)
        # Set first sample rate = second sample rate
        rates[0] = rates[1]
        return rates

    def _interp_nan_aware(self, x_new, x_orig, y_orig):
        """Linear interpolation that properly handles NaN values in source data.
        Only interpolates between valid (non-NaN) points."""
        mask = ~np.isnan(y_orig)
        if np.sum(mask) < 2:
            # Not enough valid points for interpolation
            result = np.full_like(x_new, np.nan)
            # For single valid points, do zero-order hold within range
            if np.sum(mask) == 1:
                valid_idx = np.where(mask)[0][0]
                valid_time = x_orig[valid_idx]
                valid_val = y_orig[valid_idx]
                # Find the GPS epoch boundaries (1Hz = 1 second intervals)
                epoch_start = valid_time
                if valid_idx + 1 < len(x_orig):
                    epoch_end = x_orig[valid_idx + 1]
                else:
                    epoch_end = valid_time + 1.0
                # Hold value for the epoch
                hold_mask = (x_new >= epoch_start) & (x_new < epoch_end)
                result[hold_mask] = valid_val
            return result
        return np.interp(x_new, x_orig[mask], y_orig[mask], left=np.nan, right=np.nan)

    def _zero_order_hold(self, x_new, x_orig, y_orig):
        """Zero-order hold interpolation: for each new point, use the value
        from the most recent original sample at or before that time.
        NaN values propagate (if source sample is NaN, output is NaN)."""
        result = np.full_like(x_new, np.nan, dtype=float)
        for i, t in enumerate(x_new):
            # Find the index of the most recent original sample <= t
            idx = np.searchsorted(x_orig, t, side='right') - 1
            if 0 <= idx < len(y_orig):
                result[i] = y_orig[idx]
        return result

    def resample_to_common_timeline(self, sample_rate_hz: float = 20.0):
        """Build combined timeline from all data sources."""
        print("\n" + "=" * 70)
        print("BUILDING COMBINED TIMELINE")
        print("=" * 70)

        if not self.channels or self.gps_timestamps is None:
            print("No channels to resample")
            return

        gps_start = self.gps_timestamps[0]
        gps_end = self.gps_timestamps[-1]
        duration = gps_end - gps_start

        print(f"GPS range: {gps_start:.3f}s to {gps_end:.3f}s ({duration:.1f}s, {len(self.gps_timestamps)} points at ~{self.gps_native_rate_hz}Hz)")

        if self.gps_native_rate_hz <= 1:
            # OLD FORMAT: Build 20Hz grid from 1Hz GPS, merge with CAN timestamps
            dt = 1.0 / sample_rate_hz
            gps_grid = np.arange(gps_start + dt, gps_end + dt/2, dt)
            gps_grid = np.round(gps_grid, 3)
            print(f"GPS 20Hz grid: {len(gps_grid)} points")

            if self.all_canbus_timestamps:
                all_can_ts = np.concatenate(self.all_canbus_timestamps)
                canbus_in_range = all_can_ts[
                    (all_can_ts >= gps_start) & (all_can_ts <= gps_end)
                ]
                canbus_in_range = np.round(canbus_in_range, 3)
                combined = np.union1d(gps_grid, canbus_in_range)
                print(f"CAN bus timestamps in range: {len(np.unique(canbus_in_range))} unique points")
            else:
                combined = gps_grid
        else:
            # NEW FORMAT: Union of ALL device timestamps (GPS + CAN + sensors)
            # plus a 20Hz interpolation grid for calculated fields
            all_ts = [np.round(self.gps_timestamps, 3)]
            if self.all_canbus_timestamps:
                for ts in self.all_canbus_timestamps:
                    all_ts.append(np.round(ts, 3))
            if self.all_sensor_timestamps:
                for ts in self.all_sensor_timestamps:
                    all_ts.append(np.round(ts, 3))

            # Generate 20Hz interpolation grid (matches official app behavior)
            # Grid origin = firstTimestamp/1000 + 0.04 (the elapsed_reference)
            # Use integer millisecond arithmetic to avoid floating-point drift
            if self.session_info and 'firstTimestamp' in self.session_info:
                grid_origin_ms = self.session_info['firstTimestamp'] + 40  # +40ms
            else:
                grid_origin_ms = round(gps_start * 1000)
            dt_ms = round(1000 / sample_rate_hz)  # 50ms for 20Hz
            gps_start_ms = round(gps_start * 1000)
            gps_end_ms = round(gps_end * 1000)
            # First grid step at or after GPS start
            first_step = math.ceil((gps_start_ms - grid_origin_ms) / dt_ms)
            # Last grid step at or before GPS end
            last_step = math.floor((gps_end_ms - grid_origin_ms) / dt_ms)
            calc_grid_ms = np.arange(first_step, last_step + 1) * dt_ms + grid_origin_ms
            calc_grid = calc_grid_ms / 1000.0
            all_ts.append(calc_grid)
            print(f"20Hz calc grid: {len(calc_grid)} points")

            # Inject lap boundary timestamps so lap crossings land on exact rows
            if self.session_info and 'laps' in self.session_info:
                lap_ts = []
                for lap in self.session_info['laps']:
                    start_ms = lap.get('startTimestamp', 0)
                    if start_ms:
                        lap_ts.append(start_ms / 1000.0)
                    finish_ms = lap.get('finishTimestamp', None)
                    if finish_ms:
                        lap_ts.append(finish_ms / 1000.0)
                if lap_ts:
                    all_ts.append(np.array(lap_ts))
                    print(f"Lap boundary timestamps: {len(lap_ts)} points")

            combined = np.unique(np.concatenate(all_ts))
            # Clip to GPS range (exclude sensor timestamps before first GPS fix)
            combined = combined[(combined >= gps_start) & (combined <= combined[-1])]
            print(f"Union of all timestamps: {len(combined)} unique points")

        combined = np.sort(combined)
        print(f"Combined timeline: {len(combined)} points")

        # Compute elapsed time reference from session firstTimestamp
        if self.session_info and 'firstTimestamp' in self.session_info:
            first_ts_sec = self.session_info['firstTimestamp'] / 1000.0
            if self.gps_native_rate_hz <= 1:
                elapsed_reference = math.ceil(first_ts_sec)
            else:
                elapsed_reference = first_ts_sec + 0.04
        else:
            elapsed_reference = gps_start

        # Interpolate all channels onto combined timeline
        self.resampled_data['timestamp'] = combined
        self.resampled_data['elapsed_time'] = combined - elapsed_reference

        # Channels that use zero-order hold
        zoh_channels = {'bearing'}

        for name, channel in self.channels.items():
            if name in zoh_channels:
                resampled = self._zero_order_hold(combined, channel.timestamps, channel.values)
            else:
                resampled = self._interp_nan_aware(combined, channel.timestamps, channel.values)

            self.resampled_data[name] = resampled
            valid_count = np.sum(~np.isnan(resampled))
            print(f"[OK] Resampled {name}: {valid_count} valid points")

        # Compute device_update_rate for GPS
        gps_rates = self._compute_device_update_rates(self.gps_timestamps)
        gps_rate_interp = self._interp_nan_aware(combined, self.gps_timestamps, gps_rates)
        self.resampled_data['gps_device_update_rate'] = gps_rate_interp

        # Compute device_update_rate for each sensor device
        for sensor in self.sensor_devices:
            rates = self._compute_device_update_rates(sensor['timestamps'])
            rate_interp = self._interp_nan_aware(combined, sensor['timestamps'], rates)
            self.resampled_data[f'sensor_{sensor["dev_id"]}_update_rate'] = rate_interp

    def decode_all_channels(self):
        """Decode all available channels"""
        self.decode_gps_channels()
        self.decode_processed_channels()
        self.decode_sensor_channels()
        self.resample_to_common_timeline()

    def _compute_calculated_fields(self):
        """Compute derived fields: longitudinal_acc, calc speed."""
        if 'speed' not in self.resampled_data:
            return

        timestamps = self.resampled_data['timestamp']
        speed = self.resampled_data['speed']
        num_rows = len(timestamps)

        # Compute longitudinal acceleration (G)
        # longitudinal_acc = delta_speed / delta_time / 9.81
        # Official app has ~0.4s warmup before producing longitudinal_acc values
        elapsed = self.resampled_data['elapsed_time']
        long_acc = np.full(num_rows, np.nan)
        for i in range(1, num_rows):
            if elapsed[i] < 0.45 - 1e-6:
                continue  # Warmup period - no output
            if not np.isnan(speed[i]) and not np.isnan(speed[i-1]):
                dt = timestamps[i] - timestamps[i-1]
                if dt > 0:
                    long_acc[i] = (speed[i] - speed[i-1]) / dt / 9.81

        self.resampled_data['longitudinal_acc'] = long_acc

        # Calc speed = copy of GPS speed
        self.resampled_data['calc_speed'] = speed.copy()

        print(f"[OK] Computed longitudinal_acc and calc speed")

    @staticmethod
    def _fmt(value, decimals=5):
        """Format a float: N decimal places, strip trailing zeros, keep min 1 dp."""
        if np.isnan(value):
            return ''
        s = f'{value:.{decimals}f}'
        s = s.rstrip('0')
        if s.endswith('.'):
            s += '0'
        return s

    def export_to_csv(self, output_file: str = None):
        """Export all decoded data to CSV in RaceChrono format"""
        if not self.resampled_data:
            print("No data to export!")
            return

        # Compute calculated fields before export
        self._compute_calculated_fields()

        # Determine output filename
        if output_file is None:
            output_filename = self.session_path.stem + ".csv"
        else:
            output_filename = output_file

        # Save next to the .rcz file or in the session directory
        if self.temp_dir:
            output_path = self.session_path.parent / output_filename
        else:
            output_path = self.session_dir / output_filename

        print("\n" + "=" * 70)
        print("EXPORTING TO RACECHRONO CSV FORMAT")
        print("=" * 70)

        # Get session metadata for header
        session_title = self.session_info.get('title', '') or self.session_info.get('firstPositionReverseGeocoding', 'Unknown')
        track_name = self.session_info.get('trackName', '')
        note = self.session_info.get('description', '')
        has_track = bool(track_name)
        session_type = 'Lap timing' if has_track else 'Data logging'

        created_timestamp = self.session_info.get('timeCreated', self.session_info.get('firstTimestamp', 0)) / 1000
        created_date = datetime.fromtimestamp(created_timestamp, tz=timezone.utc)

        # Build lap lookup from session.json laps data
        laps = self.session_info.get('laps', [])
        lap_boundaries = []
        for lap in laps:
            start_ms = lap.get('startTimestamp', 0)
            end_ms = lap.get('finishTimestamp', None)
            lap_boundaries.append((start_ms / 1000.0, end_ms / 1000.0 if end_ms else float('inf'), lap.get('number', 0)))

        gps_source = self._gps_source()
        has_battery = 'device_battery_level' in self.resampled_data
        canbus_columns = self.canbus_channel_order

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Write RaceChrono header
            writer.writerow(['This file is created using RaceChrono Pro v9.1.3 ( http://racechrono.com/ ).'])
            writer.writerow(['Format', '3'])
            f.write(f'Session title,"{session_title}"\n')
            writer.writerow(['Session type', session_type])
            if has_track:
                f.write(f'Track name,"{track_name}"\n')
            else:
                writer.writerow([])
            writer.writerow(['Driver name', ''])
            writer.writerow(['Created', created_date.strftime('%d/%m/%Y'), created_date.strftime('%H:%M')])
            if note:
                f.write(f'Note,"{note}"\n')
            else:
                writer.writerow(['Note', ''])
            writer.writerow([])

            # --- Build column names ---
            base_gps_columns = [
                'timestamp', 'fragment_id', 'lap_number', 'elapsed_time', 'distance_traveled',
                'accuracy', 'altitude', 'bearing',
            ]
            if has_battery:
                base_gps_columns.append('device_battery_level')
            base_gps_columns.extend([
                'device_update_rate', 'fix_type',
                'latitude', 'longitude', 'satellites', 'speed',
            ])
            calc_columns = [
                'combined_acc', 'device_update_rate', 'lateral_acc', 'lean_angle', 'longitudinal_acc',
                'speed',
            ]

            # Sensor columns: each device group has device_update_rate + data channels
            sensor_col_names = []
            sensor_col_units = []
            sensor_col_sources = []
            for sensor in self.sensor_devices:
                sensor_col_names.append('device_update_rate')
                sensor_col_units.append('Hz')
                sensor_col_sources.append(sensor['source'])
                for col in sensor['columns']:
                    sensor_col_names.append(col)
                    # Look up unit from SENSOR_CHANNELS
                    unit = ''
                    for ch_defs in self.SENSOR_CHANNELS.values():
                        for ch_info in ch_defs.values():
                            if ch_info['name'] == col:
                                unit = ch_info['unit']
                                break
                    sensor_col_units.append(unit)
                    sensor_col_sources.append(sensor['source'])

            columns = base_gps_columns + calc_columns + canbus_columns + sensor_col_names
            writer.writerow(columns)

            # --- Build units row ---
            base_gps_units = ['unix time', '', '', 's', 'm', 'm', 'm', 'deg']
            if has_battery:
                base_gps_units.append('%')
            base_gps_units.extend(['Hz', '', 'deg', 'deg', 'sats', 'm/s'])
            calc_units = ['G', 'Hz', 'G', 'deg', 'G', 'm/s']
            canbus_units = [self.canbus_channel_units.get(name, '') for name in canbus_columns]
            writer.writerow(base_gps_units + calc_units + canbus_units + sensor_col_units)

            # --- Build source row ---
            base_gps_sources = ['', '', '', '', '']
            gps_col_count = len(base_gps_columns) - 5  # everything after the first 5 non-sourced columns
            base_gps_sources.extend([gps_source] * gps_col_count)
            calc_sources = ['calc'] * len(calc_columns)
            canbus_sources = ['200: canbus'] * len(canbus_columns)
            writer.writerow(base_gps_sources + calc_sources + canbus_sources + sensor_col_sources)

            # --- Write data rows ---
            timestamps = self.resampled_data['timestamp']
            elapsed_arr = self.resampled_data['elapsed_time']
            num_rows = len(timestamps)
            fmt = self._fmt

            def fmt_coord(v):
                if np.isnan(v):
                    return ''
                s = f'{v:.7f}'.rstrip('0')
                if s.endswith('.'):
                    s += '0'
                return s

            def get_lap_number(ts):
                for start, end, num in lap_boundaries:
                    if start <= ts < end:
                        return str(num)
                return ''

            gps_start_time = self.gps_timestamps[0] if self.gps_timestamps is not None else 0
            use_dynamic_gps_rate = self.gps_native_rate_hz > 1

            for i in range(num_rows):
                ts = timestamps[i]
                elapsed = elapsed_arr[i]
                lap_num = get_lap_number(ts) if lap_boundaries else ''

                row = [
                    fmt(ts, 3),
                    '0',
                    lap_num,
                    fmt(elapsed, 3),
                ]

                # GPS data columns
                for ch_name in ['distance_traveled', 'accuracy', 'altitude', 'bearing']:
                    if ch_name in self.resampled_data:
                        row.append(fmt(self.resampled_data[ch_name][i], 5))
                    else:
                        row.append('')

                # device_battery_level (only for BLE GPS)
                if has_battery:
                    row.append(fmt(self.resampled_data['device_battery_level'][i], 5))

                # device_update_rate (GPS)
                if use_dynamic_gps_rate:
                    rate_key = 'gps_device_update_rate'
                    if rate_key in self.resampled_data:
                        row.append(fmt(self.resampled_data[rate_key][i], 5))
                    else:
                        row.append('')
                else:
                    time_since_start = ts - gps_start_time
                    row.append('1.0' if time_since_start >= 1.0 - 1e-6 else '')

                # fix_type
                if 'fix_type' in self.resampled_data:
                    row.append(fmt(self.resampled_data['fix_type'][i], 5))
                else:
                    row.append('')

                # lat, lon
                row.append(fmt_coord(self.resampled_data['latitude'][i]) if 'latitude' in self.resampled_data else '')
                row.append(fmt_coord(self.resampled_data['longitude'][i]) if 'longitude' in self.resampled_data else '')

                # satellites, speed
                for ch_name in ['satellites', 'speed']:
                    if ch_name in self.resampled_data:
                        row.append(fmt(self.resampled_data[ch_name][i], 5))
                    else:
                        row.append('')

                # Calc columns
                row.append('')  # combined_acc
                row.append('20.0')  # device_update_rate (calc)
                row.append('')  # lateral_acc
                row.append('')  # lean_angle
                if 'longitudinal_acc' in self.resampled_data:
                    row.append(fmt(self.resampled_data['longitudinal_acc'][i], 5))
                else:
                    row.append('')
                if 'calc_speed' in self.resampled_data:
                    row.append(fmt(self.resampled_data['calc_speed'][i], 5))
                else:
                    row.append('')

                # CAN bus columns
                for can_name in canbus_columns:
                    key = f'canbus_{can_name}'
                    if key in self.resampled_data:
                        row.append(fmt(self.resampled_data[key][i], 5))
                    else:
                        row.append('')

                # Sensor columns
                for sensor in self.sensor_devices:
                    # device_update_rate for this sensor
                    rate_key = f'sensor_{sensor["dev_id"]}_update_rate'
                    if rate_key in self.resampled_data:
                        row.append(fmt(self.resampled_data[rate_key][i], 5))
                    else:
                        row.append('')
                    # Data channels
                    for col in sensor['columns']:
                        data_key = f'sensor_{sensor["dev_id"]}_{col}'
                        if data_key in self.resampled_data:
                            row.append(fmt(self.resampled_data[data_key][i], 5))
                        else:
                            row.append('')

                writer.writerow(row)

        print(f"Data exported to: {output_path}")
        print(f"Total rows: {num_rows}")
        print(f"Total columns: {len(columns)}")
        print()
