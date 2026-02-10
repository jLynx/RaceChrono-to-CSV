# RaceChrono to CSV Decoder

Python tool to decode RaceChrono session files (.rcz) and export telemetry data to CSV format.

## Features

- ✅ Decode .rcz files (ZIP archives) automatically
- ✅ Support extracted session folders
- ✅ Export all channels to CSV format
- ✅ Complete channel ID mapping from RaceChrono source
- ✅ Automatic temporary file cleanup
- ✅ Support for all RaceChrono channel types (GPS, OBD, sensors, etc.)

## Project Structure

```
RaceChrono-to-CSV/
├── channel_ids.py         # Channel ID constants and helper functions (from ChannelId.java)
├── decoder.py             # RaceChronoDecoder class - handles extraction and decoding
├── decode_racechrono.py   # Main entry point
└── README.md             # This file
```

## Requirements

- Python 3.6+
- Standard library only (no external dependencies)

## Usage

### Decode a .rcz file:
```bash
python decode_racechrono.py session.rcz
```

The tool will:
1. Extract the .rcz file to a temporary directory
2. Decode all channels
3. Export to `session.csv` (same name as input, in the same directory as the .rcz file)
4. Clean up temporary files

### Decode an extracted folder:
```bash
python decode_racechrono.py test_session
```

The CSV file will be saved inside the session folder as `test_session.csv` (matching the folder name).

## Output Format

The tool exports a CSV file with:
- **Timestamps** - Millisecond precision timestamps and human-readable DateTime
- **GPS Data** - Latitude and Longitude
- **Sensor Channels** - All available sensor data (RPM, speed, temperatures, etc.)

Example columns:
- `DateTime` - Human-readable timestamp
- `Timestamp_ms` - Unix timestamp in milliseconds
- `GPS_Latitude`, `GPS_Longitude` - GPS coordinates
- `Speed` - Vehicle speed
- `EngineRpm` - Engine RPM
- `EngineCoolantTemperature` - Coolant temperature
- `TyreTemperature_X_posY` - Tire temperatures (multiple sensors)
- And many more...

## Channel ID Structure

Channels use a 32-bit ID with the following structure:

```
  bit
  3 3 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0
  1 0 9 8 7 6 5 4 3 2 1 0 9 8 7 6 5 4 3 2 1 0 9 8 7 6 5 4 3 2 1 0
 -----------------------------------------------------------------
 | | |Unused | Sensor    | Sensor  |       Base channel ID       |
 | | |       | Index     | position|                             |
 | |^ Paused channel
  ^ Delta channel
 -----------------------------------------------------------------
```

- **Sensor position**: 0 = no postfix, 1..31 = postfix
- **Sensor index**: 0 = no postfix, 1..63 = postfix number

## Supported Channel Types

### General Channels
- Timestamp, GPS Position, Speed, Altitude, Bearing
- Acceleration (Lateral, Longitudinal, X, Y, Z)
- Rotation rates and vectors
- Battery and device information

### Common OBD/Data Logger Channels (1000-2000)
- Steering Angle, Brake/Clutch Pedal Position
- Gear, Temperatures (Gearbox, Clutch, Brakes, Tires)
- Tire Pressure, Wheel Speed, Suspension Travel
- Fuel Level, Battery Status

### Engine/OBD Channels (10000+)
- Engine RPM, Throttle Position, Coolant Temperature
- Intake Manifold Pressure, Timing Advance
- MAF Airflow, Engine Load, Fuel Pressure
- Oil Temperature, Fuel Rate, Lambda

### GPS Channels (30000+)
- Satellites, Fix Type, Accuracy
- HDOP, VDOP, PDOP
- Satellite details (elevation, azimuth, SNR)

### Lap/Timer Channels (32000+)
- Current/Previous/Best lap times
- Lap numbers, Race distance/time
- Sector times

## Example Session Data

The included `test_session.rcz` contains:
- **Duration**: 75 seconds
- **Distance**: 8,083 meters
- **Location**: Auckland, New Zealand
- **Channels**: 50+ sensor channels
- **Data points**: 10,110 timestamp entries

## Copyright

Based on RaceChrono source code:
```
Copyright (C) RaceChrono Oy - All Rights Reserved
Unauthorized copying, distribution and creating derived works of this file,
via any medium is strictly prohibited
Proprietary and confidential
Written by Antti Lammi in aol@racechrono.com in 2015
```

Python port for decoding session files.

## License

This is a tool for decoding RaceChrono session files. Use in accordance with RaceChrono's terms of service.
