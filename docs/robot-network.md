# Robot Ethernet / radio

The deployment app defaults to **Robot Ethernet / radio**, team **509**, device
**12**, and TCP **5808**: `10.5.9.12:5808`. Enter your own team number and choose an
unused device address from `.6` through `.19`. Connect the Jetson Ethernet to the
robot radio or switch, and connect the laptop to that same robot network.

## First-time setup, entirely through the app

1. Select **USB recovery / first-time setup**. The app connects to
   `192.168.55.1:22`; enter the Jetson account and SSH password.
2. Use **Initialize Jetson** if needed. Initialization installs NetworkManager and
   Avahi alongside the vision prerequisites. It supports a matching offline bundle.
3. Set the robot team, unused device number and unused robot SSH port. Connect the
   laptop to the robot network while retaining its USB connection to the Jetson.
4. Expand **Configure robot Ethernet**, click **Inspect Ethernet**, and select the
   Ethernet interface plugged into the radio/switch. USB recovery interfaces and
   unmanaged devices are excluded.
5. Enter the administrator password and click **Configure Ethernet**. The app
   applies the address, reconnects to it, verifies the same SSH host key, and saves
   the confirmed robot connection as the active profile. No SSH terminal is needed.

Setup creates a separate NetworkManager profile with a `/24` mask and no default
route or DNS override. Other interfaces retain their configuration. NetworkManager
checkpoints the selected interface and restores it if the app cannot confirm the
new address. The checkpoint expires after 90 seconds; the worker normally rolls
back earlier if confirmation does not arrive. Pending profiles do not autoconnect,
so a reboot before confirmation retains the previous profile. Confirmation enables
autoconnect at priority 999. Address conflict detection runs during activation.

A separate `vortex-radio-ssh.service` listens on the selected team TCP port. It uses
the device's standard SSH host keys, allows only the configured non-root account,
supports password/key authentication and SFTP, and disables SSH forwarding. The
existing SSH service and USB port 22 stay available. A rolled-back Ethernet change
leaves this additional listener installed for the next attempt.

## Finding and checking the device

**Find Device** checks the configured robot IP and optional `.local` hostname on
the configured SSH port and initial-setup port 22. It checks SSH greetings without
sending credentials or scanning the subnet. **Check Connection** authenticates and
shows the actual endpoint and device hostname, then remembers its SSH identity.
Authentication failures and identity mismatches do not silently try other devices.

Robot connections automatically fall back to port 22 in the pits if the radio SSH
listener has not been configured. USB and custom modes use their explicit endpoint.
Existing custom addresses and pending deployment-job endpoints are preserved when
upgrading the app. Changing the target team/device or custom address clears the
previous device identity; a subsequent check pins the newly selected device.

The robot connection team and the runtime's NetworkTables/UDP settings are separate.
Configure those output settings for the robot or receiver you intend to use.

## Traffic and field access

**Limit radio traffic** is enabled by default. Uploads are streamed at 128 KiB/s,
and diagnostics request grayscale previews at up to 320×240, JPEG quality 50 and
1 FPS. Detector input resolution and production pose processing are unchanged.
Activate a release built with this version for the smaller previews. USB/custom
connections retain the usual 640×480 / 5 FPS preview; the limit can also be turned
off for a wired robot connection. Prefer USB for large offline bundles.

Use pit/tethered connections for deployment and field access for diagnostics. TCP
22 is absent from the 2026 FMS open-port list; an unused TCP port in 5800–5810 is
the radio listener's supported range. Coordinate ports with other services on the
same device. A roboRIO USB tether can optionally forward a team TCP port to the
Jetson, using WPILib's raw TCP port forwarder and the app's Custom address mode.

The [VM rehearsal](../deployment/rehearsal/README.md) exercises real NetworkManager
confirmation and timeout rollback on a second isolated Ethernet NIC, concurrent
change rejection, USB-interface protection and both SSH listeners. It cannot
establish actual radio throughput, field connectivity or Jetson NIC compatibility.

References:

- [Limelight Ethernet and addressing](https://docs.limelightvision.io/docs/docs-limelight/getting-started/limelight-3)
- [WPILib IP configurations](https://docs.wpilib.org/en/stable/docs/networking/networking-introduction/ip-configurations.html)
- [2026 FRC manual, R704 and open FMS ports](https://firstfrc.blob.core.windows.net/frc2026/Manual/HTML/2026GameManual.htm)
- [WPILib TCP port forwarding](https://docs.wpilib.org/en/stable/docs/networking/networking-utilities/portforwarding.html)
- [NetworkManager checkpoints](https://networkmanager.pages.freedesktop.org/NetworkManager/NetworkManager/gdbus-org.freedesktop.NetworkManager.html)
