#!/usr/bin/env python3
import sys
from collections import Counter

def read_csv(path):
    with open(path, "r") as f:
        lines = f.readlines()
    header_line = lines[9].strip()
    headers = header_line.split(",")
    data = []
    for line in lines[10:]:
        stripped = line.strip()
        if stripped:
            vals = stripped.split(",")
            data.append(dict(zip(headers, vals)))
    return headers, data

OUR = r"a:/Users/jLynx/Documents/Code/Python/RaceChrono-to-CSV/Test/session_20260103_135924_race_1.csv"
OFF = r"a:/Users/jLynx/Documents/Code/Python/RaceChrono-to-CSV/Test/session_20260103_135924_race_1_1_v3.csv"

_, our_data = read_csv(OUR)
_, off_data = read_csv(OFF)

print(f"Our CSV: {len(our_data)} rows")
print(f"Official CSV: {len(off_data)} rows")
print(f"Row difference: {len(off_data) - len(our_data)}")
print()

off_by_ts = {}
for row in off_data:
    ts = row.get("timestamp", "")
    if ts:
        off_by_ts[ts] = row

def find_boundaries(data):
    bounds = []
    for i in range(1, len(data)):
        pl = data[i-1].get("lap_number", "")
        cl = data[i].get("lap_number", "")
        if pl != cl and pl and cl:
            bounds.append((i, pl, cl, data[i-1], data[i]))
    return bounds

ob = find_boundaries(our_data)
rb = find_boundaries(off_data)

print(f"Our lap boundaries: {len(ob)}")
print(f"Official lap boundaries: {len(rb)}")
print()

cols = ["timestamp", "elapsed_time", "distance_traveled", "latitude", "longitude", "speed", "lap_number"]

def pr(row):
    return ", ".join(f"{c}={row.get(c,"")!s}" for c in cols)

print("=" * 130)
print("LAP BOUNDARY COMPARISON")
print("=" * 130)

for idx_i, (ix, fl, tl, lr, fr) in enumerate(ob):
    print()
    print(f"--- Boundary {idx_i+1}: Lap {fl} -> {tl} ---")
    print(f"  OURS last row of lap {fl}:")
    print(f"    {pr(lr)}")
    print(f"  OURS first row of lap {tl}:")
    print(f"    {pr(fr)}")

    mo = None
    for r in rb:
        if r[1] == fl and r[2] == tl:
            mo = r
            break

    if mo:
        _, _, _, mlr, mfr = mo
        print(f"  OFFICIAL last row of lap {fl}:")
        print(f"    {pr(mlr)}")
        print(f"  OFFICIAL first row of lap {tl}:")
        print(f"    {pr(mfr)}")

        olt = lr.get("timestamp", "")
        rlt = mlr.get("timestamp", "")
        oft = fr.get("timestamp", "")
        rft = mfr.get("timestamp", "")
        print(f"  TS MATCH last={olt==rlt} first={oft==rft}")

        try:
            d1 = float(lr.get("elapsed_time","0")) - float(mlr.get("elapsed_time","0"))
            d2 = float(fr.get("elapsed_time","0")) - float(mfr.get("elapsed_time","0"))
            print(f"  ELAPSED DIFF last={d1:.6f} first={d2:.6f}")
        except Exception as e:
            print(f"  ELAPSED error: {e}")

        try:
            d1 = float(lr.get("distance_traveled","") or "0") - float(mlr.get("distance_traveled","") or "0")
            d2 = float(fr.get("distance_traveled","") or "0") - float(mfr.get("distance_traveled","") or "0")
            print(f"  DIST DIFF last={d1:.6f}m first={d2:.6f}m")
        except Exception as e:
            print(f"  DIST error: {e}")
    else:
        print("  WARNING: No matching official boundary!")

print()
print("=" * 130)
print("NEARBY TIMESTAMP COMPARISON AT LAP BOUNDARIES")
print("=" * 130)

hf = "  {:<20} {:<8} {:<8} {:<14} {:<14} {:<14} {:<14} {:<12} {:<12} {:<12}"

for idx_i, (ix, fl, tl, _, _) in enumerate(ob):
    mo = None
    for r in rb:
        if r[1] == fl and r[2] == tl:
            mo = r
            break
    if not mo:
        continue

    print()
    print(f"--- Around Boundary {idx_i+1}: Lap {fl} -> {tl} ---")
    print(hf.format("Timestamp","Lap(O)","Lap(R)","Elapsed(O)","Elapsed(R)","Dist(O)","Dist(R)","DistDiff","Speed(O)","Speed(R)"))
    print("  " + "-"*128)

    s = max(0, ix - 3)
    e = min(len(our_data), ix + 3)
    for j in range(s, e):
        r = our_data[j]
        ts = r.get("timestamp","")
        ol = r.get("lap_number","")
        oe = r.get("elapsed_time","")
        od = r.get("distance_traveled","")
        ospd = r.get("speed","")
        rr = off_by_ts.get(ts)
        if rr:
            rl = rr.get("lap_number","")
            re = rr.get("elapsed_time","")
            rd = rr.get("distance_traveled","")
            rspd = rr.get("speed","")
            try:
                dd = f"{float(od)-float(rd):.6f}" if od and rd else "N/A"
            except:
                dd = "N/A"
            mk = " ***MISMATCH" if ol != rl else ""
            print(hf.format(ts,ol,rl,oe,re,od,rd,dd,ospd,rspd)+mk)
        else:
            print(hf.format(ts,ol,"N/A",oe,"N/A",od,"N/A","N/A",ospd,"N/A"))

print()
olc = Counter(r.get("lap_number","") for r in our_data)
rlc = Counter(r.get("lap_number","") for r in off_data)
print("Rows per lap:")
for lap in sorted(set(list(olc.keys()) + list(rlc.keys())))[:20]:
    oc = olc.get(lap, 0)
    rc = rlc.get(lap, 0)
    print(f"  Lap {lap}: ours={oc}, official={rc}, diff={oc-rc}")
