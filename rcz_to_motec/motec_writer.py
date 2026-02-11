"""
MoTeC .ld file writer

Writes MoTeC i2 Pro-enabled .ld files directly from numpy arrays.
Based on the ldparser reverse-engineered format.
"""

import datetime
import struct
import numpy as np

from .ldparser import ldVehicle, ldVenue, ldEvent, ldHead, ldChan, ldData


class MotecWriter:
    """Writes MoTeC .ld files from numpy channel data.

    Unlike the CSV-based MotecLog, this accepts numpy arrays directly,
    skipping the intermediate Message/Channel objects.
    """

    # Fixed file section pointers (consistent across MoTeC .ld files)
    VEHICLE_PTR = 1762
    VENUE_PTR = 5078
    EVENT_PTR = 8180
    HEADER_PTR = 11336

    CHANNEL_HEADER_SIZE = struct.calcsize(ldChan.fmt)

    def __init__(self, dt=None, driver="", vehicle_id="", venue_name="",
                 event_name="", event_session="", short_comment="", long_comment="",
                 vehicle_weight=0, vehicle_type="", vehicle_comment=""):
        self.driver = driver
        self.vehicle_id = vehicle_id
        self.vehicle_weight = vehicle_weight
        self.vehicle_type = vehicle_type
        self.vehicle_comment = vehicle_comment
        self.venue_name = venue_name
        self.event_name = event_name
        self.event_session = event_session
        self.short_comment = short_comment
        self.long_comment = long_comment
        self.dt = dt or datetime.datetime.now()

        self.ld_header = None
        self.ld_channels = []

        self._initialize()

    def _initialize(self):
        """Set up the MoTeC metadata hierarchy."""
        ld_vehicle = ldVehicle(
            self.vehicle_id, self.vehicle_weight,
            self.vehicle_type, self.vehicle_comment
        )
        ld_venue = ldVenue(self.venue_name, self.VEHICLE_PTR, ld_vehicle)
        ld_event = ldEvent(
            self.event_name, self.event_session,
            self.long_comment, self.VENUE_PTR, ld_venue
        )
        self.ld_header = ldHead(
            self.HEADER_PTR, self.HEADER_PTR, self.EVENT_PTR, ld_event,
            self.driver, self.vehicle_id, self.venue_name, self.dt,
            self.short_comment
        )

    def add_channel(self, name, unit, data, freq, decimals=0, short_name=""):
        """Add a channel directly from a numpy array.

        Args:
            name: Channel name (max 32 chars, will be truncated)
            unit: Unit string (max 12 chars, will be truncated)
            data: numpy array of float values at uniform frequency
            freq: Sample frequency in Hz (integer)
            decimals: Decimal places for MoTeC display (0 = auto)
            short_name: Short name (max 8 chars, optional)
        """
        name = name[:32]
        unit = unit[:12]
        short_name = short_name[:8]
        freq = max(1, int(round(freq)))

        # Advance header data pointer for this new channel
        self.ld_header.data_ptr += self.CHANNEL_HEADER_SIZE

        # Advance data pointers of all previous channels
        for ch in self.ld_channels:
            ch.data_ptr += self.CHANNEL_HEADER_SIZE

        # Compute file pointers
        if self.ld_channels:
            meta_ptr = self.ld_channels[-1].next_meta_ptr
            prev_meta_ptr = self.ld_channels[-1].meta_ptr
            data_ptr = self.ld_channels[-1].data_ptr + self.ld_channels[-1]._data.nbytes
        else:
            meta_ptr = self.HEADER_PTR
            prev_meta_ptr = 0
            data_ptr = self.ld_header.data_ptr
        next_meta_ptr = meta_ptr + self.CHANNEL_HEADER_SIZE

        # Store as float32
        data_arr = np.asarray(data, dtype=np.float32)
        # Replace NaN with 0 (MoTeC doesn't handle NaN)
        data_arr = np.nan_to_num(data_arr, nan=0.0)

        ld_chan = ldChan(
            None, meta_ptr, prev_meta_ptr, next_meta_ptr,
            data_ptr, len(data_arr),
            np.float32, freq,
            0, 1, 1, decimals,  # shift=0, mul=1, scale=1
            name, short_name, unit
        )
        ld_chan._data = data_arr

        self.ld_channels.append(ld_chan)

    def write(self, filename):
        """Write the MoTeC .ld file to disk."""
        if not self.ld_channels:
            print("No channels to write")
            return

        # Zero out the final channel's next pointer (end of linked list)
        self.ld_channels[-1].next_meta_ptr = 0

        ld_data = ldData(self.ld_header, self.ld_channels)
        ld_data.write(filename)
        print(f"Wrote MoTeC .ld file: {filename} ({len(self.ld_channels)} channels)")
