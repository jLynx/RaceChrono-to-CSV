"""
RaceChrono Session Decoder

Handles decoding of RaceChrono session files and exports to CSV format.
"""

import struct
import json
import csv
import zipfile
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from channel_ids import get_channel_name


class RaceChronoDecoder:
    """Decoder for RaceChrono session files"""

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
        self.channels_data = {}

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

    def load_metadata(self) -> bool:
        """Load session.json metadata"""
        session_file = self.session_dir / "session.json"

        if not session_file.exists():
            print(f"Error: session.json not found in {self.session_dir}")
            return False

        with open(session_file, 'r') as f:
            self.session_info = json.load(f)

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

    def decode_gps_channel(self) -> Optional[List[Tuple[float, float]]]:
        """Decode GPS coordinates (channel 3)"""
        filepath = self.session_dir / "channel_1_100_0_3_1"
        if not filepath.exists():
            return None

        data = []
        with open(filepath, 'rb') as f:
            while True:
                chunk = f.read(8)
                if not chunk or len(chunk) < 8:
                    break

                lat_int, lon_int = struct.unpack('<ii', chunk)
                lat = lat_int / 6_000_000
                lon = lon_int / 6_000_000
                data.append((lat, lon))

        return data

    def decode_processed_channel(self, channel_id: int) -> Optional[List[float]]:
        """
        Decode processed sensor channel (channel2 files with 64-bit doubles)

        Args:
            channel_id: The channel ID to decode
        """
        # Look for channel2 file with this ID
        pattern = f"channel2_12_200_{channel_id}_{channel_id}_3"
        filepath = self.session_dir / pattern

        if not filepath.exists():
            return None

        data = []
        with open(filepath, 'rb') as f:
            while True:
                chunk = f.read(8)
                if not chunk or len(chunk) < 8:
                    break

                value = struct.unpack('<d', chunk)[0]
                data.append(value)

        return data

    def decode_timestamps(self) -> Optional[List[Tuple[int, datetime]]]:
        """Decode timestamp channel"""
        filepath = self.session_dir / "channel_12_200_1_1_1"
        if not filepath.exists():
            return None

        data = []
        with open(filepath, 'rb') as f:
            while True:
                chunk = f.read(8)
                if not chunk or len(chunk) < 8:
                    break

                timestamp_ms = struct.unpack('<Q', chunk)[0]
                if timestamp_ms > 0:
                    timestamp_sec = timestamp_ms / 1000
                    dt = datetime.fromtimestamp(timestamp_sec)
                    data.append((timestamp_ms, dt))

        return data

    def discover_channels(self) -> List[int]:
        """Discover all available channel2 files and extract channel IDs"""
        channel_ids = []

        for filepath in self.session_dir.glob("channel2_12_200_*_*_3"):
            # Extract channel ID from filename: channel2_12_200_XXXXX_XXXXX_3
            parts = filepath.stem.split('_')
            if len(parts) >= 4:
                try:
                    channel_id = int(parts[3])
                    channel_ids.append(channel_id)
                except ValueError:
                    pass

        return sorted(set(channel_ids))

    def decode_all_channels(self):
        """Decode all available channels"""
        print("=" * 70)
        print("DISCOVERING CHANNELS")
        print("=" * 70)

        # Decode GPS
        gps_data = self.decode_gps_channel()
        if gps_data:
            self.channels_data['GPS_Latitude'] = [lat for lat, lon in gps_data]
            self.channels_data['GPS_Longitude'] = [lon for lat, lon in gps_data]
            print(f"[OK] GPS Position: {len(gps_data)} points")

        # Decode timestamps
        timestamps = self.decode_timestamps()
        if timestamps:
            self.channels_data['Timestamp_ms'] = [ts for ts, dt in timestamps]
            self.channels_data['DateTime'] = [dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] for ts, dt in timestamps]
            print(f"[OK] Timestamps: {len(timestamps)} points")

        # Discover and decode all sensor channels
        channel_ids = self.discover_channels()
        print(f"\nFound {len(channel_ids)} sensor channels:")

        for channel_id in channel_ids:
            data = self.decode_processed_channel(channel_id)
            if data:
                channel_name = get_channel_name(channel_id)
                self.channels_data[channel_name] = data

                # Print summary
                min_val = min(data)
                max_val = max(data)
                avg_val = sum(data) / len(data)
                print(f"  [OK] {channel_name} (ID: {channel_id}): {len(data)} values, "
                      f"range: {min_val:.2f} - {max_val:.2f}, avg: {avg_val:.2f}")

        print()

    def export_to_csv(self, output_file: str = None):
        """Export all decoded data to CSV"""
        if not self.channels_data:
            print("No data to export!")
            return

        # Determine output filename and location
        # If no output file specified, use the input filename with .csv extension
        if output_file is None:
            output_filename = self.session_path.stem + ".csv"
        else:
            output_filename = output_file

        # If we extracted from .rcz, save next to the .rcz file
        # If we're working with a folder, save in that folder
        if self.temp_dir:
            # Working with .rcz file - save next to it
            output_path = self.session_path.parent / output_filename
        else:
            # Working with extracted folder - save in it
            output_path = self.session_dir / output_filename

        # Find maximum length
        max_len = max(len(values) for values in self.channels_data.values())

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Write header
            headers = sorted(self.channels_data.keys())
            writer.writerow(headers)

            # Write data rows
            for i in range(max_len):
                row = []
                for header in headers:
                    values = self.channels_data[header]
                    if i < len(values):
                        value = values[i]
                        if isinstance(value, float):
                            row.append(f'{value:.6f}')
                        else:
                            row.append(str(value))
                    else:
                        row.append('')
                writer.writerow(row)

        print("=" * 70)
        print("EXPORT COMPLETE")
        print("=" * 70)
        print(f"Data exported to: {output_path}")
        print(f"Total rows: {max_len}")
        print(f"Total columns: {len(headers)}")
        print()
        print("Column summary:")
        for header in headers:
            print(f"  - {header}")
