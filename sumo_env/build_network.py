#!/usr/bin/env python3
"""
build_network.py — generates intersection.net.xml from raw XML files using netconvert.
Run once before training: python sumo_env/build_network.py
"""
import subprocess, sys, pathlib

BASE = pathlib.Path(__file__).parent

def build():
    cmd = [
        "netconvert",
        "--node-files",       str(BASE / "intersection.nod.xml"),
        "--edge-files",       str(BASE / "intersection.edg.xml"),
        "--connection-files", str(BASE / "intersection.con.xml"),
        "--tllogic-files",    str(BASE / "intersection.tll.xml"),
        "--output-file",      str(BASE / "intersection.net.xml"),
        "--no-turnarounds",   "true",
        "--tls.discard-simple", "false",
    ]
    print("Running netconvert ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("netconvert FAILED:\n", result.stderr)
        sys.exit(1)
    print("✅  intersection.net.xml created successfully.")

if __name__ == "__main__":
    build()
