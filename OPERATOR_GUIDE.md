# Interactive Operator Configuration

## Overview

The system now prompts the operator at startup to configure inspection parameters interactively.

## What the Operator is Asked

When you run the application, you will be prompted for:

### 1. Model Selection
```
SELECT MODEL MODE:
  1. OFFLINE - Local YOLO model (no internet required)
  2. ONLINE  - Roboflow Rapid API (requires internet)

Enter choice [1/2] (default: 1):
```

**Options:**
- **1. OFFLINE**: Uses your trained YOLO model locally (recommended for Raspberry Pi)
- **2. ONLINE**: Uses Roboflow's cloud API (requires internet connection)

### 2. Pipeline Length
```
PIPELINE LENGTH:
  Enter pipeline length in meters (default: 100.0):
```

**Input:** Enter the total length of the pipeline being inspected in meters.

**Examples:**
- `50` for 50 meter pipe
- `100.5` for 100.5 meter pipe
- `250` for 250 meter pipe

### 3. Robot Velocity
```
ROBOT VELOCITY:
  Available units: m/s or km/h
  Enter unit [m/s/km/h] (default: m/s):
  Enter velocity in m/s (default: 0.5):
```

**First**, choose the unit:
- `m/s` for meters per second
- `km/h` for kilometers per hour

**Then**, enter the robot's inspection speed in that unit.

**Common Values:**
- `0.3` m/s - Slow, detailed inspection
- `0.5` m/s - Standard inspection speed
- `1.0` m/s - Fast inspection
- `1.8` km/h - Standard (equivalent to 0.5 m/s)
- `3.6` km/h - Fast (equivalent to 1.0 m/s)

## Example Session

```
============================================================
PIPELINE CRACK DETECTION SYSTEM
Inspection Configuration
============================================================

SELECT MODEL MODE:
  1. OFFLINE - Local YOLO model (no internet required)
  2. ONLINE  - Roboflow Rapid API (requires internet)

Enter choice [1/2] (default: 1): 1

PIPELINE LENGTH:
  Enter pipeline length in meters (default: 100.0): 150

ROBOT VELOCITY:
  Available units: m/s or km/h
  Enter unit [m/s/km/h] (default: m/s): km/h
  Enter velocity in km/h (default: 0.5): 2.5

------------------------------------------------------------
CONFIGURATION SUMMARY:
------------------------------------------------------------
  Model Mode:       OFFLINE
  Pipeline Length:  150.00 meters
  Robot Velocity:   2.50 km/h
  Estimated Time:   216.0 seconds (3.6 minutes)
------------------------------------------------------------

Proceed with this configuration? [Y/n]: y

[INIT] Starting system with operator configuration...
```

## Using Default Values

You can quickly accept defaults by pressing Enter without typing anything:

```
Enter choice [1/2] (default: 1): [ENTER]        → Uses default (1)
Enter pipeline length (default: 100.0): [ENTER] → Uses default (100.0)
Enter unit (default: m/s): [ENTER]              → Uses default (m/s)
Enter velocity (default: 0.5): [ENTER]          → Uses default (0.5)
```

This makes it quick for repeated testing with the same parameters.

## Validation

The system validates all inputs:

**Pipeline Length:**
- Must be a positive number
- Cannot be zero or negative

**Velocity:**
- Must be a positive number
- Cannot be zero or negative

**Unit:**
- Must be exactly `m/s` or `km/h`
- Case-insensitive (M/S, KM/H also work)

**Model Selection:**
- Must be `1` or `2`

If invalid input is detected, you'll be prompted again:
```
Error: Length must be positive.
Enter pipeline length in meters (default: 100.0):
```

## Configuration Summary

Before starting, the system shows:
- Model mode selected
- Pipeline length
- Robot velocity with units
- **Estimated inspection time** (calculated automatically)

**Formula**: Time = Distance / Velocity

## Confirmation

After reviewing the summary:
```
Proceed with this configuration? [Y/n]:
```

- Press `y` or Enter to continue
- Press `n` to cancel and exit

## Starting the Application

Simply run:
```bash
python realtime_pi5_dual_web.py
```

The interactive prompt will start automatically.

## For Automated/Headless Operation

If you need to run without interaction (e.g., on boot), you can:

**Option 1**: Modify the code to skip prompts
**Option 2**: Use environment variables as before (will require code modification)
**Option 3**: Create a wrapper script that feeds inputs

### Wrapper Script Example:
```bash
#!/bin/bash
# auto_start.sh

# Feed inputs: model=1 (offline), length=100, unit=m/s, velocity=0.5, confirm=y
echo -e "1\n100\nm/s\n0.5\ny" | python realtime_pi5_dual_web.py
```

## Benefits

1. **No manual .env editing** - Configure on-the-fly
2. **Operator-friendly** - Clear prompts and defaults
3. **Validated input** - Prevents configuration errors
4. **Visual confirmation** - See settings before starting
5. **Flexible** - Easy to change between inspections
6. **Professional** - Guided setup process

## Tips for Operators

1. **Keep a log** - Write down your settings for each inspection
2. **Test first** - Run a short test to verify velocity is correct
3. **Use consistent units** - Stick to one unit system (m/s or km/h)
4. **Check the summary** - Always verify calculated inspection time
5. **For repeated runs** - Use default values (just press Enter)

## Troubleshooting

**Q: I entered wrong values, how do I restart?**
A: Press `n` when asked to confirm, or press Ctrl+C and restart

**Q: Can I change settings after starting?**
A: No, you need to stop and restart the application

**Q: What if I don't know the exact velocity?**
A: Start with the default (0.5 m/s) and measure actual time vs. expected time, then adjust

**Q: The estimated time seems wrong?**
A: Check that velocity units match your input (m/s vs km/h)

**Q: Can I skip the prompts for testing?**
A: Press Enter repeatedly to use all defaults

## Integration with Existing .env

The interactive system uses `.env` values as **defaults**, so:
- Your `.env` settings still matter
- They appear as default values in prompts
- Operator can accept or override them
- Changes are **not saved** to `.env` (session only)

This design allows:
- Quick setup for production (use defaults)
- Easy customization when needed (change values)
- No accidental .env modifications
