"""
MoTeC .ldx file writer

Generates XML-based .ldx files containing lap beacon markers
for use with MoTeC i2 analysis software.
"""

from xml.dom import minidom


def write_ldx(filename, laps, data_start_ms):
    """Write a MoTeC .ldx file with lap beacon markers.

    Args:
        filename: Output .ldx file path
        laps: List of dicts from session.json 'laps' array.
              Each lap has 'startTimestamp' and 'finishTimestamp' (ms).
        data_start_ms: Start timestamp of the data in the .ld file (ms),
                       i.e. the first GPS timestamp. Beacons are placed
                       relative to this time.
    """
    if not laps or len(laps) < 1:
        return

    # Compute beacon times as elapsed from data start
    # Each beacon marks a lap finish (= start/finish line crossing)
    # Skip laps that finish before data starts (fragment files)
    beacon_times_sec = []
    lap_durations = []
    for lap in laps:
        finish_ms = lap.get('finishTimestamp', None)
        start_ms = lap.get('startTimestamp', 0)
        if finish_ms is not None:
            elapsed_sec = (finish_ms - data_start_ms) / 1000.0
            if elapsed_sec < 0:
                continue  # lap finished before this data starts
            lap_duration = (finish_ms - start_ms) / 1000.0
            beacon_times_sec.append(elapsed_sec)
            lap_durations.append(lap_duration)

    if not beacon_times_sec:
        return

    # Build XML
    root = minidom.Document()

    ldx = root.createElement("LDXFile")
    root.appendChild(ldx)
    ldx.setAttribute("locale", "English_United Kingdom.1252")
    ldx.setAttribute("DefaultLocale", "C")
    ldx.setAttribute("Version", "1.6")

    layers = root.createElement("Layers")
    ldx.appendChild(layers)

    layer = root.createElement("Layer")
    layers.appendChild(layer)

    markerblock = root.createElement("MarkerBlock")
    layer.appendChild(markerblock)

    markergroup = root.createElement("MarkerGroup")
    markerblock.appendChild(markergroup)
    markergroup.setAttribute("Name", "Beacons")
    markergroup.setAttribute("Index", str(len(beacon_times_sec) - 1))

    # Create beacon markers at elapsed times from data start
    for idx, elapsed_sec in enumerate(beacon_times_sec):
        elapsed_us = elapsed_sec * 1_000_000  # microseconds
        marker = root.createElement("Marker")
        marker.setAttribute("Version", "100")
        marker.setAttribute("ClassName", "BCN")
        marker.setAttribute("Name", f"Manual.{idx + 1}")
        marker.setAttribute("Flags", "77")
        marker.setAttribute("Time", f"{elapsed_us:0.2f}")
        markergroup.appendChild(marker)

    # Details section
    details = root.createElement("Details")
    layers.appendChild(details)

    total_laps = root.createElement("String")
    details.appendChild(total_laps)
    total_laps.setAttribute("Id", "Total Laps")
    total_laps.setAttribute("Value", str(len(beacon_times_sec) + 1))  # include in-lap

    # Find fastest lap (skip outlap = first lap)
    if len(lap_durations) > 1:
        search_laps = lap_durations[1:]  # skip outlap
        fastest_idx = min(range(len(search_laps)), key=lambda i: search_laps[i])
        fastest_time = search_laps[fastest_idx]
        fastest_lap_num = fastest_idx + 2  # +1 for outlap, +1 for 1-based
    else:
        fastest_time = lap_durations[0]
        fastest_lap_num = 1

    minutes = int(fastest_time % 3600 // 60)
    seconds = fastest_time % 3600 % 60

    ft = root.createElement("String")
    details.appendChild(ft)
    ft.setAttribute("Id", "Fastest Time")
    ft.setAttribute("Value", f"{minutes:02d}:{seconds:06.3f}")

    fl = root.createElement("String")
    details.appendChild(fl)
    fl.setAttribute("Id", "Fastest Lap")
    fl.setAttribute("Value", str(fastest_lap_num))

    # Write file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(root.toprettyxml(indent="  "))

    print(f"Wrote MoTeC .ldx file: {filename} ({len(beacon_times_sec)} laps)")
