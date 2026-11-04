#!/bin/bash
# Run on the excavator's Pi by deploy.sh, from ~/umi-dashboard (compose.yaml, imus.json, src/ = the
# image's build context). Safe to rerun, including after a fresh OS flash: host setup (hostname, I2C
# buses, wired links, Wi-Fi, Docker, access token, .env), then builds the image unless --no-build,
# (re)starts it with docker compose and prints the private dashboard link.
# Environment (deploy.sh passes these on):
#   UMI_PI_WLAN_MAC    refuse unless wlan0 has this MAC (guards against setting up the wrong Pi)
#   UMI_PI_ETH0_ADDR   eth0's static link-local address (default 169.254.10.2/16)
#   UMI_PI_USB0_ADDR   the USB-C gadget's address (default 169.254.10.3/16)
set -euo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cd "$(dirname "$0")"
task_hostname=bucket
task_build=1
task_up_args=(-d --remove-orphans)
[ "${1:-}" = --no-build ] && task_build=0
if [ "$(uname -s)" != Linux ] || ! command -v raspi-config >/dev/null; then
  echo 'Run this on the Raspberry Pi (Raspberry Pi OS).' >&2
  exit 1
fi
if [ "$(id -un)" = root ]; then
  echo 'Run as your normal Pi user. The script requests sudo when needed.' >&2
  exit 1
fi
if [ -n "${UMI_PI_WLAN_MAC:-}" ] && [ "$(cat /sys/class/net/wlan0/address)" != "$UMI_PI_WLAN_MAC" ]; then
  echo "This Pi's wlan0 MAC is not UMI_PI_WLAN_MAC; refusing to set it up." >&2
  exit 1
fi
task_eth0_addr=${UMI_PI_ETH0_ADDR:-169.254.10.2/16}
task_usb0_addr=${UMI_PI_USB0_ADDR:-169.254.10.3/16}
echo 'Setting up the Pi. Enter its password if requested.'
sudo -v

# Hostname: the dashboard is <hostname>.local, and time-calibration.json is bound to it.
task_old_hostname=$(hostname)
if [ "$task_old_hostname" != "$task_hostname" ]; then
  echo "Renaming the Pi from $task_old_hostname to $task_hostname..."
  sudo hostnamectl set-hostname "$task_hostname"
  if grep -q '^127\.0\.1\.1[[:space:]]' /etc/hosts; then
    sudo sed -i "s/^127\.0\.1\.1[[:space:]].*/127.0.1.1\t$task_hostname/" /etc/hosts
  else
    printf '127.0.1.1\t%s\n' "$task_hostname" | sudo tee -a /etc/hosts >/dev/null
  fi
  task_up_args+=(--force-recreate)   # the recorder reads the hostname and time correction at start
  if systemctl -q is-active avahi-daemon; then sudo systemctl restart avahi-daemon; fi
  # Same Pi under a new name: keep its time correction loadable.
  if [ -s time-calibration.json ]; then
    python3 - "$task_old_hostname" "$task_hostname" <<'PY'
import json, pathlib, sys
path = pathlib.Path('time-calibration.json')
data = json.loads(path.read_text())
if data.get('deployment', {}).get('hostname') == sys.argv[1]:
    data['deployment']['hostname'] = sys.argv[2]
    path.write_text(json.dumps(data, indent=2) + '\n')
PY
  fi
fi

sudo raspi-config nonint do_i2c 0
sudo udevadm settle --timeout=10
if [ ! -e /dev/i2c-1 ]; then
  echo 'I2C bus 1 is missing. Enable I2C (sudo raspi-config, Interface Options), reboot, then rerun.' >&2
  exit 1
fi
# Arm IMUs (inference-time joint state): stick on I2C bus 3 (GPIO4 SDA pin 7, GPIO14 SCL pin 8),
# boom on I2C bus 5 (GPIO12 SDA pin 32, GPIO13 SCL pin 33). The bucket IMU stays on bus 1.
# GPIO14 has no hardware I2C function, so bus 3 is the kernel's bit-banged i2c-gpio driver
# (delay 5 us, about 100 kHz), and the serial console that would write to GPIO14 is turned off. All
# buses stay at 100 kHz: at 400 kHz the bucket IMU (bus 1) hit I/O errors when the boom hit its stop.
task_boot_config=/boot/firmware/config.txt
[ -f "$task_boot_config" ] || task_boot_config=/boot/config.txt
task_reboot=0
sudo raspi-config nonint do_serial_cons 1
# The earlier hardware bus 3 on GPIO4/5 would claim GPIO4 too; the 400 kHz bus lines are undone.
for task_old in 'dtoverlay=i2c3,pins_4_5,baudrate=100000' 'dtparam=i2c_arm_baudrate=400000' 'dtoverlay=i2c-gpio,bus=3,i2c_gpio_sda=4,i2c_gpio_scl=14,i2c_gpio_delay_us=2' 'dtoverlay=i2c5,pins_12_13,baudrate=400000'; do
  if grep -qxF "$task_old" "$task_boot_config"; then
    sudo sed -i "\\|^${task_old}\$|d" "$task_boot_config"
    task_reboot=1
  fi
done
# Wired links to the Mac (work without Wi-Fi). eth0 = $task_eth0_addr, the wired address the Mac
# tools try first (list it first in hosts.pi.urls of configs/hub/local.json); without it NetworkManager
# waits for DHCP that a direct cable never answers and drops eth0 for 5 minutes at a time. The USB-C
# port becomes a USB Ethernet device, usb0 = $task_usb0_addr. Fixed MACs keep the Mac from creating a
# new network interface on every boot.
for task_overlay in 'dtoverlay=i2c-gpio,bus=3,i2c_gpio_sda=4,i2c_gpio_scl=14,i2c_gpio_delay_us=5' 'dtoverlay=i2c5,pins_12_13,baudrate=100000' 'dtoverlay=dwc2,dr_mode=peripheral'; do
  if ! grep -qxF "$task_overlay" "$task_boot_config"; then
    echo "$task_overlay" | sudo tee -a "$task_boot_config" >/dev/null
    task_reboot=1
  fi
done
task_cmdline=/boot/firmware/cmdline.txt
[ -f "$task_cmdline" ] || task_cmdline=/boot/cmdline.txt
if ! grep -q 'modules-load=dwc2,g_ether' "$task_cmdline"; then
  sudo sed -i 's/ rootwait / rootwait modules-load=dwc2,g_ether /' "$task_cmdline"
  grep -q 'modules-load=dwc2,g_ether' "$task_cmdline"
  task_reboot=1
fi
task_gadget_macs='options g_ether host_addr=02:62:6b:32:00:01 dev_addr=02:62:6b:32:00:02'
if [ "$(cat /etc/modprobe.d/g_ether.conf 2>/dev/null)" != "$task_gadget_macs" ]; then
  echo "$task_gadget_macs" | sudo tee /etc/modprobe.d/g_ether.conf >/dev/null
  task_reboot=1
fi
# NetworkManager's 85-nm-unmanaged.rules leaves every USB gadget interface unmanaged.
task_gadget_rule='SUBSYSTEM=="net", ENV{DEVTYPE}=="gadget", ENV{NM_UNMANAGED}="0"'
if [ "$(cat /etc/udev/rules.d/86-usb-gadget-managed.rules 2>/dev/null)" != "$task_gadget_rule" ]; then
  echo "$task_gadget_rule" | sudo tee /etc/udev/rules.d/86-usb-gadget-managed.rules >/dev/null
  sudo udevadm control --reload
  if [ -e /sys/class/net/usb0 ]; then
    sudo udevadm trigger --action=change /sys/class/net/usb0
    sudo udevadm settle --timeout=10
    sudo nmcli device set usb0 managed yes
  fi
fi
if ! nmcli -t -f NAME connection show | grep -qxF usb-gadget; then
  sudo nmcli connection add type ethernet ifname usb0 con-name usb-gadget >/dev/null
fi
sudo nmcli connection modify usb-gadget connection.autoconnect yes ipv4.method manual \
  ipv4.addresses "$task_usb0_addr" ipv4.never-default yes ipv6.method link-local
if ! nmcli -t -f NAME connection show | grep -qxF wired-link; then
  sudo nmcli connection add type ethernet ifname eth0 con-name wired-link >/dev/null
fi
sudo nmcli connection modify wired-link connection.autoconnect yes connection.autoconnect-priority 10 \
  ipv4.method manual ipv4.addresses "$task_eth0_addr" ipv4.never-default yes ipv6.method link-local
task_wifi_uuid=$(nmcli -g GENERAL.CON-UUID device show wlan0)
if [ -z "$task_wifi_uuid" ] || [ "$task_wifi_uuid" = -- ]; then
  echo 'No active Wi-Fi profile; leaving Wi-Fi alone (the USB-C link is enough).'
else
  sudo nmcli connection modify uuid "$task_wifi_uuid" connection.autoconnect yes \
    802-11-wireless.powersave 2 802-11-wireless.cloned-mac-address permanent \
    ipv4.dhcp-hostname "$task_hostname"
  if command -v iw >/dev/null; then
    sudo iw dev wlan0 set power_save off
  fi
fi

if ! command -v docker >/dev/null || ! sudo docker compose version >/dev/null 2>&1; then
  echo 'Installing Docker (needs internet on the Pi)...'
  curl -fsSL https://get.docker.com | sudo sh
fi
sudo systemctl enable --now docker >/dev/null
if ! id -nG | tr ' ' '\n' | grep -qxF docker; then
  # Lets the Pi user run docker without sudo after the next login; this script uses sudo.
  sudo usermod -aG docker "$(id -un)"
fi
if [ ! -s access-token ]; then
  (umask 077; python3 -c 'import secrets; print(secrets.token_urlsafe(32))' >access-token)
fi
chmod 600 access-token
mkdir -p -m 700 recordings
task_gid() { getent group "$1" | cut -d: -f3; }
task_env=$(printf 'UMI_UID=%s\nUMI_GID=%s\nVIDEO_GID=%s\nI2C_GID=%s\n' \
  "$(id -u)" "$(id -g)" "$(task_gid video)" "$(task_gid i2c)")
if grep -q '=$' <<<"$task_env"; then
  echo 'The video or i2c group is missing on this Pi:' >&2
  echo "$task_env" >&2
  exit 1
fi
if [ "$(cat .env 2>/dev/null)" != "$task_env" ]; then
  printf '%s\n' "$task_env" >.env
fi

if [ "$task_build" = 1 ]; then
  echo 'Building the dashboard image (the first build downloads packages and takes a few minutes)...'
  sudo docker build -f src/umi-data-collection/deploy/Dockerfile -t umi-dashboard:latest src
fi
# Restarting the container interrupts a take, so refuse while one is recording or saving.
python3 - <<'PY'
import json, pathlib, urllib.request
request = urllib.request.Request('http://127.0.0.1:8000/api/recordings/status',
    headers={'Cookie': 'umi_access=' + pathlib.Path('access-token').read_text().strip()})
try:
    with urllib.request.urlopen(request, timeout=3) as response:
        state = json.load(response)
except (OSError, ValueError):
    state = {}
if state.get('status') in ('recording', 'saving'):
    raise SystemExit('A take is recording or saving. Stop it in the dashboard, wait for saving to finish, then rerun.')
PY
show_failure() {
  echo 'The dashboard could not start. Details from the Pi:'
  sudo docker compose ps -a || true
  sudo docker compose logs --tail 30 || true
}
if ! sudo docker compose up "${task_up_args[@]}"; then
  show_failure
  exit 1
fi
sudo docker image prune -f >/dev/null
if ! python3 - <<'PY'
import json, pathlib, socket, time, urllib.request
token = pathlib.Path('access-token').read_text().strip()
request = urllib.request.Request('http://127.0.0.1:8000/api/status', headers={'Cookie': 'umi_access=' + token})
for attempt in range(30):
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            data = json.load(response)
        break
    except OSError:
        time.sleep(1)
else:
    raise SystemExit('Dashboard did not start.')
alignment = data.get('time_alignment', {})
if pathlib.Path('time-calibration.json').exists():
    if not alignment.get('configured'):
        raise SystemExit(alignment.get('error') or 'The installed time correction did not load.')
    print(f"Loaded time correction: {alignment['offset_ms']:+.3f} ms for future recordings.")
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
    try:
        sock.connect(('192.0.2.1', 53))   # any routable address: picks the default route's source, sends nothing
        address = sock.getsockname()[0]
    except OSError:
        address = socket.gethostname() + '.local'
print('\nDashboard is running. Open this private link on your Mac:\n')
print('http://' + address + ':8000/?token=' + token)
print('\nIt starts automatically when the Pi boots.')
print('To stop it: cd ~/umi-dashboard && sudo docker compose down')
PY
then
  show_failure
  exit 1
fi
if [ "$task_reboot" = 1 ]; then
  echo 'Boot settings changed (arm IMU I2C buses and/or the USB-C link). Reboot the Pi once (sudo reboot) to start them.'
fi
