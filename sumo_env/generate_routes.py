#!/usr/bin/env python3
"""
generate_routes.py — creates a traffic route file with randomised vehicle arrivals.
Supports configurable traffic density (low / medium / high / mixed).
Output: sumo_env/traffic.rou.xml
"""
import random, pathlib, argparse

BASE = pathlib.Path(__file__).parent

VEHICLE_TYPES = {
    "car":        {"accel": "2.9", "decel": "7.5", "length": "5.0",  "maxSpeed": "13.89", "color": "0.8,0.8,0.8,1"},
    "truck":      {"accel": "1.5", "decel": "5.0", "length": "12.0", "maxSpeed": "10.00", "color": "0.6,0.4,0.2,1"},
    "bus":        {"accel": "1.2", "decel": "4.5", "length": "14.0", "maxSpeed": "9.00",  "color": "0.2,0.2,0.9,1"},
    "motorcycle": {"accel": "4.0", "decel": "8.0", "length": "2.5",  "maxSpeed": "16.67", "color": "1.0,0.5,0.0,1"},
}

# Weighted mix: mostly cars, some trucks/buses/motorcycles
VEHICLE_MIX = {
    "car": 0.65, "motorcycle": 0.15, "truck": 0.12, "bus": 0.08
}

ROUTES = [
    ("north_south", "north_in south_out"),
    ("south_north", "south_in north_out"),
    ("east_west",   "east_in  west_out"),
    ("west_east",   "west_in  east_out"),
    ("north_east",  "north_in east_out"),
    ("east_north",  "east_in  north_out"),
    ("south_west",  "south_in west_out"),
    ("west_south",  "west_in  south_out"),
]

def weighted_choice(choices: dict, rng: random.Random) -> str:
    keys = list(choices.keys())
    weights = list(choices.values())
    return rng.choices(keys, weights=weights, k=1)[0]

def generate(seed: int = 42, episode_length: int = 3600,
             vehicles_per_hour: int = 600, output: pathlib.Path | None = None):
    """
    vehicles_per_hour: controls traffic density
        low ~= 200, medium ~= 600, high ~= 1200
    """
    rng = random.Random(seed)
    output = output or (BASE / "traffic.rou.xml")

    # Poisson arrivals: mean inter-arrival = 3600 / vph seconds
    mean_gap = 3600.0 / vehicles_per_hour
    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"')
    lines.append('        xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">')
    lines.append('')

    # Vehicle type definitions
    for vt, attrs in VEHICLE_TYPES.items():
        attr_str = " ".join(f'{k}="{v}"' for k, v in attrs.items())
        lines.append(f'    <vType id="{vt}" {attr_str}/>')
    lines.append('')

    # Route definitions
    for route_id, edges in ROUTES:
        lines.append(f'    <route id="{route_id}" edges="{edges}"/>')
    lines.append('')

    # Vehicle departures (Poisson process)
    t = 0.0
    vid = 0
    while t < episode_length:
        # inter-arrival drawn from Exponential distribution (Poisson process)
        gap = rng.expovariate(1.0 / mean_gap)
        t += gap
        if t >= episode_length:
            break
        vtype = weighted_choice(VEHICLE_MIX, rng)
        route  = rng.choice(ROUTES)[0]
        depart = f"{t:.2f}"
        lines.append(f'    <vehicle id="v{vid}" type="{vtype}" route="{route}" depart="{depart}"/>')
        vid += 1

    lines.append('</routes>')
    output.write_text("\n".join(lines))
    print(f"✅  Generated {vid} vehicles → {output}  (density={vehicles_per_hour} veh/hr)")
    return str(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SUMO route file")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--length",  type=int, default=3600, help="Episode length in seconds")
    parser.add_argument("--density", type=int, default=600,  help="Vehicles per hour")
    parser.add_argument("--output",  type=str, default=None)
    args = parser.parse_args()
    out = pathlib.Path(args.output) if args.output else None
    generate(seed=args.seed, episode_length=args.length,
             vehicles_per_hour=args.density, output=out)
