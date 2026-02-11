"""
MoTeC .ldx file writer

Generates XML-based .ldx files containing lap beacon markers
for use with MoTeC i2 analysis software.
"""

from xml.dom import minidom


def write_ldx(filename, laps):
    """Write a MoTeC .ldx file with lap beacon markers.

    Args:
        filename: Output .ldx file path
        laps: List of dicts from session.json 'laps' array.
              Each lap has 'startTimestamp' and 'finishTimestamp' (ms).
    """
    if not laps or len(laps) < 1:
        return

    # Compute lap times from start/finish timestamps
    lap_times = []
    first_start_ms = laps[0].get('startTimestamp', 0)

    for lap in laps:
        start_ms = lap.get('startTimestamp', 0)
        finish_ms = lap.get('finishTimestamp', None)
        if finish_ms is not None:
            lap_time_sec = (finish_ms - start_ms) / 1000.0
            lap_times.append(lap_time_sec)

    if not lap_times:
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
    markergroup.setAttribute("Index", str(len(lap_times) - 1))

    # Create beacon markers at cumulative elapsed times
    elapsed_us = 0.0
    for idx, lap_time in enumerate(lap_times):
        elapsed_us += lap_time * 1_000_000  # microseconds
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
    total_laps.setAttribute("Value", str(len(lap_times) + 1))  # include in-lap

    # Find fastest lap (skip outlap = first lap)
    if len(lap_times) > 1:
        search_laps = lap_times[1:]  # skip outlap
        fastest_idx = min(range(len(search_laps)), key=lambda i: search_laps[i])
        fastest_time = search_laps[fastest_idx]
        fastest_lap_num = fastest_idx + 2  # +1 for outlap, +1 for 1-based
    else:
        fastest_time = lap_times[0]
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

    print(f"Wrote MoTeC .ldx file: {filename} ({len(lap_times)} laps)")
