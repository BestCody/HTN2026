# Arm Setup — Wiring, SD Card, Install

This is the end-to-end runbook: hardware wiring first, then getting the code onto
the Raspberry Pi's SD card, then bringing the arm to life. Follow the sections in
order the first time.

The scope of this document is **one 4-DOF arm** driven by four MG996R servos
through a PCA9685 PWM driver, controlled by a Raspberry Pi, with a browser
dashboard that renders the arm in 3D. The second arm plugs into channels 4-7 of
the same PCA9685 later.

> **Read this before touching a wire.** MG996R servos can each draw ~2.5 A at
> stall. Four of them together will brown out a Raspberry Pi if you power them
> from its 5 V rail — use the dedicated **Shnitpwr SNT-0312-120** described
> below and jumper its ground to the Pi's ground. Never let the Pi's 5 V rail
> see the servos' V+ rail. **Also: set the Shnitpwr's output to 6 V and verify
> it on the LCD before connecting anything.** It ships adjustable up to 12 V,
> which will destroy MG996Rs (max input ~7.2 V).

---

## 1. Bill of materials

| Qty | Part                                                        | Notes                                                    |
|-----|-------------------------------------------------------------|----------------------------------------------------------|
| 1   | Raspberry Pi (any model with a 40-pin header, Zero 2 W or 3/4/5) | Runs the controller. Wi-Fi is convenient but not required. |
| 1   | microSD card, 16 GB+                                        | Class 10 / A1 recommended.                               |
| 1   | PCA9685 16-channel 12-bit PWM/servo driver breakout          | I²C, 5 V logic-tolerant, screw terminal for V+.          |
| 4   | MG996R servos (per arm — 8 total for the dual system)       | All identical.                                           |
| 1   | **Shnitpwr SNT-0312-120** adjustable AC→DC supply           | 3.5–12 V adjustable, 10 A / 120 W max, LCD readout, ships with 14 barrel-jack tips and a **DC screw-terminal adapter tip** that you'll use to bring the leads onto the breadboard. Set it to **6 V** for the MG996Rs (Step 1 in §2). Adequate for one arm; when you add the second arm consider stepping up to a 5 V/20 A supply so peak stall current doesn't clip. |
| *(optional)* | 1000 µF **radial (through-hole) aluminum electrolytic** capacitor, **rated ≥16 V** | Bulk cap across the PCA9685 V+ / GND rail. Skip for now — the Shnitpwr's own filtering is usually enough. Only order one if you later see servo jitter, PCA9685 resets during motion, or LCD voltage sag under load. Amazon search: *"1000uF 25V radial electrolytic capacitor"* (~$5 for a bag of 10). |
| 1   | Half-size breadboard                                        | Common ground / V+ rail + capacitor.                     |
| —   | Male–male and male–female jumper wires                       | ~10 pieces.                                              |
| 1   | Pi power supply (5 V/3 A USB-C for Pi 4/5, or 5 V/2.5 A microUSB for older) | Independent of the servo PSU.                         |
| 1   | Ethernet cable *or* known-good Wi-Fi network                 | For SSH onto the Pi.                                     |

Optional but nice: an inline switch on the servo PSU so you can cut motor power
without unplugging the Pi.

---

## 2. Wiring — step by step

Follow steps 1–10 in order. Every step assumes the Shnitpwr is **unplugged
from AC** and the Pi is **unplugged from USB power** unless the step says
otherwise. There are only three power domains in this project — Pi, servo PSU,
and breadboard rails — and they meet at exactly one place: a common ground
wire between the breadboard and Pi pin 6.

### Cheat sheet (keep this visible while wiring)

**Raspberry Pi 40-pin header — you only use pins 1, 3, 5, 6:**

```
  Pin 1  = 3.3 V   → PCA9685 VCC
  Pin 3  = SDA     → PCA9685 SDA
  Pin 5  = SCL     → PCA9685 SCL
  Pin 6  = GND     → breadboard blue (–) rail
```

**PCA9685 servo channel → joint:**

| Channel | Joint          |
|---------|----------------|
| 0       | Base rotation  |
| 1       | Shoulder       |
| 2       | Elbow          |
| 3       | Gripper        |
| 4–7     | *(second arm — leave empty for now)* |

**MG996R plug orientation:** brown = GND (outer edge of the PCA9685),
red = V+ (middle), orange = PWM (inner edge).

---

### Step 1 — Set the Shnitpwr to 6 V, alone

Do this before you connect the Shnitpwr to anything.

1. Do **not** attach any tip to the barrel plug yet. Nothing connected.
2. Plug the Shnitpwr AC lead into the wall. The LCD lights up.
3. Turn the rotary voltage knob until the LCD reads exactly **6.0 V**.
4. Unplug the Shnitpwr from AC.

The Shnitpwr can output up to 12 V, which destroys an MG996R. Doing this
first, in isolation, means nothing can be fried by the wrong default voltage.

### Step 2 — Attach the DC screw-terminal tip to the Shnitpwr

The Shnitpwr's cable ends in a small round **barrel plug**. On its own it has
nothing you can screw a wire into. The Shnitpwr ships with a bag of ~14
interchangeable tips (all the different barrel-jack sizes for laptops etc.);
one of them is the adapter you want here.

1. Open the plastic bag/box of tips that came inside the Shnitpwr's retail
   box. Spread them out.
2. Pick out the **DC screw-terminal tip**. It's the odd one out — instead of
   another barrel connector, its front face is a small plastic block with
   two tiny **screw heads** and the markings **`+`** and **`–`**. The back
   end is a barrel socket that fits over the Shnitpwr's barrel plug. It's
   sometimes labelled "DC terminal connector" on the tip itself or in the
   Shnitpwr manual.
3. Push that tip onto the Shnitpwr's barrel plug until it clicks / seats
   firmly.
4. Strip ~1 cm off two short pieces of solid-core hookup wire — one red,
   one black.
5. Loosen the two screws on the tip a couple of turns. Insert the **red**
   wire into the **`+`** terminal and tighten. Insert the **black** wire
   into the **`–`** terminal and tighten.
6. Tug gently on both wires — if either pulls out, tighten harder.

*Can't find the screw-terminal tip?* Any one of these three fallbacks works
and lands you in the same place (two bare wires — red for +, black for –):

- A **5.5 × 2.1 mm female barrel jack to screw terminal adapter** (~$3 on
  Amazon). Plug the Shnitpwr's stock 5.5 × 2.1 mm tip into it.
- A **barrel-plug-to-pigtail** cable — a matching barrel jack on one end,
  bare wires on the other.
- Snip the barrel plug off the Shnitpwr's cable entirely and strip its two
  internal wires. Destructive; only do this if you're sure the Shnitpwr is
  a permanent part of this build.

### Step 3 — Build the breadboard power rails

Half-size breadboard, long side up. Both long-edge rails run the length of
the board — one red (+), one blue (–).

1. Push the red wire from Step 2 into any hole on the **red rail** (+).
2. Push the black wire from Step 2 into any hole on the **blue rail** (–).

At this point the rails have voltage available at their far ends but
everything is still unplugged from AC — the rails are cold.

### Step 4 — *(Optional)* Install a 1000 µF capacitor across the rails

**Skip this step for now.** The Shnitpwr has its own internal filtering and
the PCA9685 has a small onboard cap; together that's usually enough to get
the arm moving. Only come back and add a bulk cap if you later see:

- Servos jittering when idle or twitching mid-motion.
- The PCA9685 dropping off the I²C bus during aggressive moves.
- The Shnitpwr's LCD dipping noticeably below 6 V under load.

If any of those show up, buy a **1000 µF, ≥16 V, radial through-hole
aluminum electrolytic** capacitor (Amazon: *"1000uF 25V radial
electrolytic capacitor"*, ~$5 / bag of 10, next-day shipping) and install
it here:

1. Identify the legs: **long leg = +**, **short leg = – (the can also has a
   printed stripe running down the – side)**.
2. Push the **long leg into the red (+) rail**.
3. Push the **short leg into the blue (–) rail**.
4. Place it as close to the PCA9685 V+ / GND screw terminals as you can —
   shorter loop = better damping of servo current spikes.

Polarity matters — installed backwards, it pops when powered.

**Otherwise, move on to Step 5.**

### Step 5 — Wire the breadboard rails to the PCA9685 screw terminals

The PCA9685 has two large screw terminals in one corner. They are labeled
**V+** and **GND**.

1. Jumper: breadboard **red rail** → PCA9685 **V+** screw terminal.
2. Jumper: breadboard **blue rail** → PCA9685 **GND** screw terminal.
3. Tighten both screws.

### Step 6 — Wire the Pi to the PCA9685 header

The PCA9685 has a row of six small header pins (VCC, GND, SCL, SDA, OE, and
one more) on the opposite side from the screw terminals. Use female-to-female
jumpers to reach from the Pi's 40-pin header.

**Pi is still unplugged.** Wire these four:

1. Pi **pin 1 (3.3 V)** → PCA9685 **VCC**.
2. Pi **pin 3 (SDA)** → PCA9685 **SDA**.
3. Pi **pin 5 (SCL)** → PCA9685 **SCL**.
4. Pi **pin 6 (GND)** → breadboard **blue rail** (any empty hole).

Leave the PCA9685's **OE** and header-side **GND** pins alone — you don't need
them.

That last wire (Pi pin 6 → blue rail) is the common ground. Without it the
PCA9685's logic and the servo PSU float relative to each other and the servos
will twitch or refuse to move. Non-optional.

### Step 7 — Visual sanity check (no power yet)

Before you plug anything in, look at your build:

1. **No wire from Pi pins 2 or 4 goes anywhere.** Those are the Pi's 5 V rail
   and must not touch the servo rail.
2. Cap's **short leg** is on the blue rail, long leg on the red rail.
3. Red rail meets **V+** on the PCA9685; blue rail meets **GND**.
4. Only wires to the Pi: pin 1, pin 3, pin 5, pin 6.
5. No servos plugged in yet.

### Step 8 — First power-on, still no servos

### Step 8 — First power-on, still no servos

**Optional pre-check (recommended if you own a multimeter):** with the
Shnitpwr *unplugged from AC*, set your meter to continuity mode (the
beeper) and touch one probe to the red (+) breadboard rail and the other
to the blue (–) rail. You want **no beep**. A beep means the two rails
are shorted somewhere — find it before applying power.

**Part A — verify the I²C link (Pi only, Shnitpwr stays unplugged).**

You don't need the Shnitpwr on for this. The PCA9685's logic side is
powered by the Pi's 3.3 V (Step 6), which is all `i2cdetect` talks to.

1. Plug the Pi into its USB power supply. Wait ~30 seconds for it to boot.
2. SSH into the Pi (see §3 if you haven't set it up yet) and run:
   ```bash
   i2cdetect -y 1
   ```
   You should see `40` in the grid. If not, recheck Step 6 wiring.

**Part B — verify the servo rail (Shnitpwr on, still no servos plugged in).**

1. Plug the Shnitpwr AC into the wall. The LCD should light up and read
   **6.0 V**. (This model's LCD only shows voltage, not current, so the
   voltage reading is your only live indicator.)
2. Warning signs that mean **unplug the Shnitpwr immediately**:
   - LCD reads noticeably below 6 V, blinks, or drops out on plug-in.
   - The Shnitpwr, PCA9685, breadboard rails, or DC-terminal tip get warm
     within the first ~10 seconds.
   - You hear a high-pitched whine or clicking, or smell hot plastic.
   Any of those = a short. Unplug, recheck Steps 3, 5, and 6, then retry.
3. If everything looks quiet, unplug the Shnitpwr AC and move to Step 9.

### Step 9 — Plug in the four servos

Servos plug **directly into the 3-pin headers** along the edge of the
PCA9685 — never through the breadboard.

Each 3-pin header has the same layout: **outer pin = GND**, middle pin = V+,
inner pin = PWM signal. On the MG996R cable this maps to brown = GND,
red = V+, orange = PWM. So on every channel, the brown wire faces the outer
edge of the board.

1. **Channel 0** ← base rotation servo.
2. **Channel 1** ← shoulder servo.
3. **Channel 2** ← elbow servo.
4. **Channel 3** ← gripper servo.

If a servo is silent when you later command it, or the driver chip gets
noticeably warm, that servo is on backwards — unplug the PSU and reseat.

### Step 10 — Second power-on, live

1. Pi is still on from Step 8.
2. Plug the Shnitpwr AC back in. LCD should still read 6.0 V. Idle current
   should be under ~0.2 A (servos hold their current position but no PWM has
   been sent yet).
3. Open the dashboard, connect to the Pi's WebSocket, click **Enable** or
   **Home**. The servos energize and move to their home positions.

You're wired.

---

### Every-time power-on / power-off order

Once wiring is done, from cold:

1. Pi power on (wait ~30 s).
2. Shnitpwr AC in — confirm LCD still reads **6.0 V**.
3. Dashboard → Connect → **Enable** (or **Home**).

To power down: Dashboard → **Disable** → unplug Shnitpwr AC → on the Pi run
`sudo shutdown now`, then unplug Pi USB.

### Capacity note (why this matters for arm 2)

At 6 V the Shnitpwr's ceiling is 10 A. Four MG996Rs in normal motion pull
1–3 A together; a four-servo simultaneous stall would clip. If the LCD dims
or the voltage sags during motion, lower `max_deg_per_sec` in
`firmware/config.yaml`.

When you bring the second arm online (channels 4–7), eight MG996Rs share the
same rail — 10 A is not enough headroom. Plan on a bigger supply
(e.g. 5 V / 20 A) before wiring arm two.

---

## 3. Get the code onto the Pi's SD card

Two paths. Pick one.

### 3.1 Option A — Flash headless, then `git clone` (recommended)

1. On your laptop, install **Raspberry Pi Imager** (<https://www.raspberrypi.com/software/>).
2. Insert the microSD card into your laptop.
3. In Imager:
   - **Device**: your Pi model.
   - **Operating system**: *Raspberry Pi OS Lite (64-bit)* — no desktop needed.
   - **Storage**: the microSD.
   - Click the ⚙ gear (or *Edit Settings*) and fill in:
     - Hostname: `arm`
     - Enable SSH → *Use password authentication*, set a username (`pi`) and
       password. (Or paste your SSH public key.)
     - Configure Wi-Fi → SSID + password + your country code.
     - Locale / timezone.
   - Write. Wait ~5 minutes.
4. Eject, put the microSD into the Pi, power the Pi on. Give it ~60 s to
   boot and connect to Wi-Fi.
5. From your laptop:
   ```bash
   ssh pi@arm.local
   ```
   If `arm.local` doesn't resolve (Windows / some Linux distros), find the Pi's
   IP from your router and use that instead.
6. On the Pi, enable I²C and install prerequisites:
   ```bash
   sudo raspi-config nonint do_i2c 0
   sudo apt update
   sudo apt install -y git python3-venv python3-pip i2c-tools
   ```
7. Verify the driver is visible on the bus:
   ```bash
   i2cdetect -y 1
   ```
   You should see `40` in the grid. If not, re-check SDA/SCL/VCC wiring.
8. Clone the repo and install the firmware:
   ```bash
   git clone https://github.com/<you>/HTN2026.git ~/HTN2026
   cd ~/HTN2026/firmware
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[pi]"
   ```
9. Smoke-test it in the foreground:
   ```bash
   arm-controller --host 0.0.0.0 --port 8000
   ```
   You should see `PCA9685 attached at 0x40 @ 50 Hz`. Ctrl-C to stop.
10. Install as a systemd service so it starts on boot:
    ```bash
    sudo cp systemd/arm-controller.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable --now arm-controller
    sudo systemctl status arm-controller
    ```

### 3.2 Option B — No network, copy from a USB stick

If the Pi can't reach the internet:

1. On your laptop, `git clone` the repo and copy the whole `HTN2026` directory
   to a USB drive.
2. Flash Raspberry Pi OS Lite via Imager (steps 1–3 above).
3. Boot the Pi with a keyboard + monitor attached. Log in.
4. Plug in the USB stick and copy the repo:
   ```bash
   sudo mount /dev/sda1 /mnt
   cp -r /mnt/HTN2026 ~/
   sudo umount /mnt
   ```
5. Continue from step 6 of Option A. You will still need internet once (to
   install `python3-venv` and pip packages) — either hotspot from your phone
   or download the wheels ahead of time.

### 3.3 Option C — Edit directly on the SD card (development only)

The `/boot` partition of a Raspberry Pi OS SD card is FAT32 and mounts on any
OS. You can drop a `firstrun.sh` there to run once at boot. This is fine for
setting up SSH keys or Wi-Fi if the Imager UI didn't stick, but is not a great
way to ship the actual firmware — use A or B for that.

---

## 4. Dashboard

The dashboard is a Vite + React app. It can run on your laptop against the Pi,
or you can build it once and let the Pi serve it.

### 4.1 Run on your laptop (fastest iteration)

```bash
cd dashboard
npm install
npm run dev
```

Open the printed URL (usually `http://localhost:5173`). In the WS URL box at the
top, set `ws://arm.local:8000/ws` (or `ws://<pi-ip>:8000/ws`) and hit Connect.

### 4.2 Serve the dashboard from the Pi

```bash
cd dashboard
npm install
npm run build
# copy the built assets to the Pi
rsync -av dist/ pi@arm.local:~/HTN2026/dashboard/dist/
```

Then on the Pi, set the environment variable that tells the firmware where the
built files are, and restart the service:

```bash
sudo systemctl edit arm-controller
# add:
# [Service]
# Environment=ARM_DASHBOARD_DIST=/home/pi/HTN2026/dashboard/dist
sudo systemctl restart arm-controller
```

Now visit `http://arm.local:8000/` in a browser.

---

## 5. Calibrate each servo before you mount the arm

Do this **once per servo, with the servo unmounted and free to spin**. It finds
the pulse width limits that correspond to the mechanical endpoints of each
joint, and saves you from cooking a servo against a hard stop.

```bash
cd ~/HTN2026/firmware
source .venv/bin/activate
python -m arm_controller.calibrate --channel 0
```

At the `us>` prompt, type integer microseconds. The MG996R responds to roughly
500–2500 µs at 50 Hz. Start at 1500 (centre), then sweep down 100 µs at a time
until the servo just reaches its lower mechanical stop *without buzzing*. Note
the value; sweep up the same way to find the upper stop.

Repeat for channels 1, 2, 3. Update the `min_us` / `max_us` values under each
joint in `firmware/config.yaml`, and restart the service:

```bash
sudo systemctl restart arm-controller
```

You can also set `invert: true` for a joint if the CAD orientation and servo
mounting direction disagree.

---

## 6. Optional: Fusion 360 → 3D viewer meshes

The viewer works out of the box with placeholder cylinders and boxes. To make
it look like your actual arm, drop STL exports of the CAD components in
`dashboard/public/models/`:

1. Open `arm_model.f3d` in Fusion 360.
2. In the browser panel, right-click a component (e.g. *Base*) → **Save as Mesh**.
3. Format: STL Binary. Structure: One File. Units: meters. Click OK, save to
   `dashboard/public/models/base.stl`.
4. Repeat for the shoulder, elbow, and gripper components, using the filenames
   `shoulder.stl`, `elbow.stl`, `gripper.stl`.
5. Rebuild the dashboard (`npm run build`) or refresh in dev mode.

You don't need GLB — Fusion 360's free tier doesn't export it, but STL is
supported and the dashboard loads it natively. If a file is missing, that link
falls back to its placeholder shape, so partial exports still work.

---

## 7. Troubleshooting

| Symptom                                   | Likely cause                                           | Fix                                                                                     |
|-------------------------------------------|--------------------------------------------------------|-----------------------------------------------------------------------------------------|
| `i2cdetect` shows no `40`                 | I²C not enabled, or SDA/SCL swapped                    | `sudo raspi-config` → Interface Options → I²C; recheck wiring.                          |
| Log says "Hardware driver unavailable"    | Ran on laptop, or the `pi` optional deps aren't installed | On the Pi: `pip install -e ".[pi]"`. The mock driver is fine for laptop dev.            |
| Servos jitter / Pi reboots when servo moves | PSU sagging under load, or ground not shared with Pi | Lower `max_deg_per_sec` in `config.yaml`; confirm the Shnitpwr LCD holds 6 V under motion; confirm PSU – rail jumpers to Pi GND. |
| Shnitpwr LCD voltage sags under load     | 10 A ceiling being clipped by peak stall current       | Reduce speed limits, avoid commanding all joints at once, or upgrade to a larger supply.  |
| Servo hums but doesn't move               | Pulse width outside its mechanical range               | Recalibrate that channel; widen `min_us`/`max_us` cautiously.                           |
| Dashboard connects but joints don't move  | Arm is disabled                                        | Click **Enable** — the firmware refuses to drive PWM until enabled.                     |
| Browser can't reach `arm.local`           | mDNS not resolving on your OS                          | Use the Pi's IP (`hostname -I` on the Pi) instead.                                      |
| Cap installed backwards → pop / smoke     | Reversed polarity                                      | Replace the cap. Long leg = + rail. Do not power on again until confirmed.              |

---

## 8. What each command actually did (mental model)

- `raspi-config nonint do_i2c 0` — turns on the kernel's I²C driver so
  `/dev/i2c-1` exists.
- `pip install -e ".[pi]"` — installs the firmware in editable mode plus the
  Adafruit Blinka / PCA9685 libraries that only make sense on real hardware.
- `systemctl enable --now arm-controller` — starts the service now and every
  boot after.
- The FastAPI process opens two loops: a **control loop** at
  `control_rate_hz` that moves current joint angles toward targets and writes
  PCA9685 duty cycles, and a **broadcast loop** at `broadcast_rate_hz` that
  pushes state to any connected browser over the WebSocket at `/ws`.

Good luck. If something looks wrong, unplug the servo PSU first, then debug.
