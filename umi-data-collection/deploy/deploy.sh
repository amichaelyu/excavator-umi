#!/bin/bash
# Run on the Mac: deploy the UMI recording dashboard to the excavator's Pi. Packs the image's build
# context, compose.yaml, deploy-on-pi.sh and devices/imus-bucket.json (as imus.json) into
# ~/umi-dashboard on the Pi, runs deploy-on-pi.sh there (host setup, image build, docker compose up),
# then saves the Pi's access token to umi-data-collection/.devices/bucket/access-token.
# Normal SSH authentication (keys / ~/.ssh/config, or the password when prompted); nothing is saved.
# Usage: deploy.sh [--no-build | --diagnose] USER@PI   (or set UMI_PI_TARGET)
#   --no-build   set up and restart with the image already on the Pi
#   --diagnose   read-only checks (Wi-Fi, power, container); report saved to umi-data-collection/pi-diagnostics.txt
# Passed on to deploy-on-pi.sh: UMI_PI_WLAN_MAC, UMI_PI_ETH0_ADDR, UMI_PI_USB0_ADDR (see there).
set -euo pipefail
umask 077
task_dir=$(cd "$(dirname "$0")" && pwd)
task_home=$(dirname "$task_dir")   # umi-data-collection/
task_root=$(dirname "$task_home")
task_mode=deploy
case ${1:-} in
  --no-build) task_mode=no-build; shift ;;
  --diagnose) task_mode=diagnose; shift ;;
  -h|--help) sed -n '2,10s/^# \{0,1\}//p' "$0"; exit 0 ;;
esac
task_target=${1:-${UMI_PI_TARGET:-}}
if ! [[ "$task_target" =~ ^[a-z_][a-z0-9_-]*@[A-Za-z0-9.-]+$ ]]; then
  echo 'Usage: bash umi-data-collection/deploy/deploy.sh [--no-build | --diagnose] USER@PI   (or set UMI_PI_TARGET)' >&2
  exit 2
fi

if [ "$task_mode" = diagnose ]; then
  echo 'Checking Pi Wi-Fi, power, and dashboard status. Enter the Pi password if prompted.'
  ssh -tt -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=3 "$task_target" '
    export PATH=$PATH:/usr/sbin:/sbin PAGER=cat SYSTEMD_PAGER=cat SYSTEMD_COLORS=0
    echo "Host and uptime"; hostname; uptime
    echo "Wi-Fi connection and power saving"
    if command -v iw >/dev/null; then
      iw dev wlan0 link; iw dev wlan0 get power_save
    else
      echo "iw is not installed; showing kernel wireless statistics. Power-saving state is unknown."
      cat /proc/net/wireless
    fi
    echo "Network state"
    nmcli -f GENERAL.STATE,GENERAL.CONNECTION,IP4.ADDRESS device show wlan0
    echo "Power and temperature"
    vcgencmd get_throttled; vcgencmd measure_temp
    echo "Dashboard container"
    task_docker=docker
    docker info >/dev/null 2>&1 || task_docker="sudo docker"
    cd "$HOME/umi-dashboard" && $task_docker compose ps -a && $task_docker compose logs --tail 30 --no-color
  ' 2>&1 | tee "$task_home/pi-diagnostics.txt"
  echo "Saved report to $task_home/pi-diagnostics.txt"
  exit 0
fi

task_temp=$(mktemp -d "${TMPDIR:-/tmp}/umi-dashboard-deploy.XXXXXX")
task_socket="$task_temp/ssh"
cleanup() {
  ssh -S "$task_socket" -O exit "$task_target" >/dev/null 2>&1 || true
  rm -rf "$task_temp"
}
trap cleanup EXIT

# The tarball is laid out like ~/umi-dashboard; src/ holds the same files the Dockerfile copies
# (theme.css is a symlink into hub/web, which is packed too).
task_stage="$task_temp/stage"
task_tar=(--no-xattrs --no-acls --no-fflags)
mkdir -p "$task_stage/src"
(cd "$task_root" && COPYFILE_DISABLE=1 tar "${task_tar[@]}" -cf - \
  umi-data-collection/deploy/Dockerfile umi-data-collection/deploy/Dockerfile.dockerignore \
  umi-data-collection/recorder/*.py umi-data-collection/recorder/capture.c \
  hub/web/data-collection hub/web/theme.css) | tar -xf - -C "$task_stage/src"
cp "$task_dir/compose.yaml" "$task_dir/deploy-on-pi.sh" "$task_stage/"
cp "$task_home/devices/imus-bucket.json" "$task_stage/imus.json"
# The image must be readable by the unprivileged container user whatever the Mac's umask was.
chmod -R u+rwX,go+rX "$task_stage/src"
# Entries by name, not ".", so extracting does not change ~/umi-dashboard's own mode.
(cd "$task_stage" && COPYFILE_DISABLE=1 tar "${task_tar[@]}" -czf "$task_temp/dashboard.tar.gz" -- *)

task_env=$(printf 'UMI_PI_WLAN_MAC=%q UMI_PI_ETH0_ADDR=%q UMI_PI_USB0_ADDR=%q' \
  "${UMI_PI_WLAN_MAC:-}" "${UMI_PI_ETH0_ADDR:-169.254.10.2/16}" "${UMI_PI_USB0_ADDR:-169.254.10.3/16}")
task_args=
[ "$task_mode" = no-build ] && task_args=--no-build
echo 'Connecting to the Pi. Enter its login password if prompted.'
ssh -M -S "$task_socket" -o ControlPersist=60 -o ConnectTimeout=10 -fN "$task_target"
scp -o "ControlPath=$task_socket" "$task_temp/dashboard.tar.gz" "$task_target:umi-dashboard-install.tar.gz"
ssh -S "$task_socket" -tt "$task_target" 'mkdir -p "$HOME/umi-dashboard" && rm -rf "$HOME/umi-dashboard/src" &&
  tar -xzf "$HOME/umi-dashboard-install.tar.gz" -C "$HOME/umi-dashboard" && rm "$HOME/umi-dashboard-install.tar.gz" &&
  cd "$HOME/umi-dashboard" && '"$task_env"' bash deploy-on-pi.sh '"$task_args"
mkdir -p "$task_home/.devices/bucket"
chmod 700 "$task_home/.devices" "$task_home/.devices/bucket"
scp -q -o "ControlPath=$task_socket" "$task_target:umi-dashboard/access-token" "$task_home/.devices/bucket/access-token"
chmod 600 "$task_home/.devices/bucket/access-token"
echo "Saved the Pi's access token to $task_home/.devices/bucket/access-token"
