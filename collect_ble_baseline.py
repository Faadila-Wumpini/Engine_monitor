# collect_ble_baseline.py
# Connects to the ESP32 over Bluetooth and logs RAW sensor readings
# (accX, accY, accZ, temp) to a CSV while the engine runs NORMALLY.
#
# WHY THIS EXISTS:
#   model.pkl / scaler.pkl were trained on the AI4I 2020 dataset, which has
#   completely different columns (Air temperature, Torque, Rotational speed,
#   etc.) to the live ESP32 sensor readings (accX, accY, accZ, temp). Scoring
#   live BLE data against the AI4I-trained model produces meaningless results
#   — see train_bluetooth.py for the model that actually understands your
#   sensor data, which this script generates the training data for.
#
# HOW TO USE:
#   1. Mount the ESP32 + sensors on the engine as normal.
#   2. Start the engine and let it idle / run under NORMAL conditions
#      (no faults). Do NOT introduce faults during collection — this data
#      becomes the model's definition of "normal".
#   3. Run:  python collect_ble_baseline.py --minutes 10
#   4. Let it run for the full duration, then stop the engine.
#
# OUTPUT:
#   ble_baseline_normal.csv — one row per BLE notification received
#
# TIPS FOR A GOOD BASELINE:
#   - Collect at least 5 minutes (ideally 10-15). At 50Hz that's 15,000-45,000
#     rows, giving several hundred training windows.
#   - Capture the engine at a few different normal states if possible (idle,
#     light throttle) so the model learns realistic variation, not just one
#     fixed RPM.
#   - Keep the laptop within Bluetooth range the whole time — a dropped
#     connection just ends collection early with whatever was captured so far.

import argparse
import asyncio
import csv
import json
import os
import time
from datetime import datetime

BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH       = os.path.join(BASE_DIR, 'ble_baseline_normal.csv')
# Every unit advertises as "EngineIQ_Sensor_<DEVICE_ID>" (see the firmware's
# DEVICE_ID constant) — this is the shared prefix, not a full name, so
# scanning finds any of them; --device targets one specifically.
BLE_DEVICE_PREFIX = "EngineIQ_Sensor"
CHAR_UUID         = "abcd1234-ab12-ab12-ab12-abcdef123456"
BLE_SCAN_TIMEOUT  = 8.0


async def collect(duration_seconds, device_id=None):
    try:
        from bleak import BleakClient, BleakScanner
    except ImportError:
        print("❌ bleak not installed. Run:  pip install bleak")
        return

    def matches(device, advertisement_data):
        name = device.name or ''
        if not name.startswith(BLE_DEVICE_PREFIX):
            return False
        if device_id is None:
            return True
        return name == f"{BLE_DEVICE_PREFIX}_{device_id}"

    label = f"'{BLE_DEVICE_PREFIX}_{device_id}'" if device_id else f"any '{BLE_DEVICE_PREFIX}_*' sensor"
    print("=" * 55)
    print("  BLE BASELINE COLLECTION")
    print(f"  Scanning for {label}...")
    print("=" * 55)

    device = await BleakScanner.find_device_by_filter(matches, timeout=BLE_SCAN_TIMEOUT)

    if device is None:
        print(f"\n⚠ Could not find a matching EngineIQ sensor nearby.")
        print("  Check the ESP32 is powered on and Bluetooth is enabled.")
        if device_id:
            print(f"  Check its DEVICE_ID matches '{device_id}', or drop --device to accept any unit.")
        return

    print(f"\n  ✓ Found: {device.name} [{device.address}]")
    print(f"  Connecting...\n")

    rows = []
    start_time = time.monotonic()
    connected_device_id = device.name.replace(f'{BLE_DEVICE_PREFIX}_', '', 1) if device.name else 'unknown'

    file_exists = os.path.exists(OUTPUT_PATH)
    csv_file = open(OUTPUT_PATH, 'a', newline='')
    writer = csv.writer(csv_file)
    if not file_exists:
        writer.writerow(['timestamp', 'accX', 'accY', 'accZ', 'temp', 'device_id'])

    async with BleakClient(device) as client:
        print(f"  ✓ Connected via Bluetooth!")
        print(f"  Logging normal-running data for {duration_seconds:.0f} seconds...")
        print(f"  → Writing to: {OUTPUT_PATH}")
        print(f"  Press Ctrl+C to stop early.\n")

        def handle_notification(sender, data):
            try:
                payload = json.loads(data.decode('utf-8'))
                row = [
                    datetime.now().isoformat(),
                    float(payload.get('accX', 0)),
                    float(payload.get('accY', 0)),
                    float(payload.get('accZ', 0)),
                    float(payload.get('temp', 0)),
                    payload.get('device_id', connected_device_id),
                ]
                writer.writerow(row)
                rows.append(row)
                if len(rows) % 250 == 0:
                    elapsed = time.monotonic() - start_time
                    print(f"  [{elapsed:6.1f}s] {len(rows)} readings logged...")
            except Exception as e:
                print(f"  ⚠ BLE data error: {e}")

        await client.start_notify(CHAR_UUID, handle_notification)

        try:
            while (time.monotonic() - start_time) < duration_seconds:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\n  Stopped early by user.")
        finally:
            await client.stop_notify(CHAR_UUID)
            csv_file.close()

    print(f"\n{'='*55}")
    print(f"  COLLECTION COMPLETE")
    print(f"{'='*55}")
    print(f"  Readings logged this session : {len(rows)}")
    print(f"  Total file size              : {OUTPUT_PATH}")
    print(f"\n  Next step:")
    print(f"    python train_bluetooth.py")
    print(f"{'='*55}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Collect a normal-running BLE sensor baseline for training'
    )
    parser.add_argument('--minutes', type=float, default=10.0,
                         help='How many minutes to collect (default: 10)')
    parser.add_argument('--device', type=str, default=None,
                         help="Target a specific ESP32 by its DEVICE_ID (e.g. --device ESP32-02) "
                              "when more than one EngineIQ sensor might be nearby. "
                              "Omit to connect to whichever matching unit is found first.")
    args = parser.parse_args()

    print("\n╔═══════════════════════════════════════════╗")
    print("║  EngineIQ — BLE Baseline Collector        ║")
    print("║  Group 11, KNUST                          ║")
    print("╚═══════════════════════════════════════════╝\n")

    asyncio.run(collect(args.minutes * 60, device_id=args.device))