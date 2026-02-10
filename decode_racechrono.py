#!/usr/bin/env python3
"""
RaceChrono Session Decoder - Main Entry Point

Decodes .rcz session files based on the channel ID mappings from RaceChrono source code.
Usage: python decode_racechrono.py <session_folder>

Copyright (C) RaceChrono Oy - All Rights Reserved
Unauthorized copying, distribution and creating derived works of this file, via any medium is strictly prohibited
Proprietary and confidential
Written by Antti Lammi in aol@racechrono.com in 2015

Python port for decoding session files.
"""

import sys
from pathlib import Path

from decoder import RaceChronoDecoder


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("RaceChrono Session Decoder")
        print("=" * 70)
        print("Usage: python decode_racechrono.py <path>")
        print()
        print("Path can be either:")
        print("  - A .rcz file (will be automatically extracted)")
        print("  - An extracted session folder")
        print()
        print("Examples:")
        print("  python decode_racechrono.py session.rcz")
        print("  python decode_racechrono.py test_session")
        print("  python decode_racechrono.py \"5016 903 1163\"")
        sys.exit(1)

    session_path = Path(sys.argv[1])

    if not session_path.exists():
        print(f"Error: Path not found: {session_path}")
        sys.exit(1)

    # Use context manager to ensure cleanup
    try:
        with RaceChronoDecoder(session_path) as decoder:
            # Load metadata
            if not decoder.load_metadata():
                sys.exit(1)

            # Print session info
            decoder.print_session_info()

            # Decode all channels
            decoder.decode_all_channels()

            # Export to CSV
            decoder.export_to_csv()

            csv_filename = session_path.stem + ".csv"
            print(f"Done! You can now open {csv_filename} in Excel or any spreadsheet program.")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
