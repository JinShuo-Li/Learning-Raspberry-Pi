# Raspberry Pi Bluetooth/WiFi Local Chat Server

This is a simple local network chat room application designed to run on a Linux device (like a Raspberry Pi). By configuring the device as a WiFi hotspot, users can connect to the network and access the chat interface via a web browser to share text messages, images, and files.

## Directory Structure

```
bt_server/
├── venv/                 # Python virtual environment
├── templates/            # HTML templates folder
│   └── index.html        # Frontend chat interface
├── uploads/              # (Auto-generated) Stores uploaded files
├── app.py                # Backend Flask server code
```

## 1. Setting Up The Environment

First, create the project directory and set up the Python virtual environment:

```bash
mkdir ~/bt_server
cd ~/bt_server
python3 -m venv venv
source venv/bin/activate
pip install flask
```

## 2. Project Code

You will need to create the following two files:

- Create the file `app.py` inside the `~/bt_server/` directory.

- Create the directory `templates/` and then create `index.html` inside it.

You can see the full code for both files in the repository.

## 3. Network Configuration (WiFi Hotspot Setup)

To facilitate connection to the chat server, you will need to turn your Linux/Raspberry Pi device into a WiFi Access Point.

```bash
# Create a hotspot named "Pi_Chat_Room" with password "12345678"
sudo nmcli device wifi hotspot ssid "Pi_Chat_Room" password "12345678" ifname wlan0
```

Check the IP address assigned to your wireless interface (this will be the chat room address):

```bash
ip addr show wlan0
```
*Note: The default IP address for hotspots is often `10.42.0.1`, but verify via the command above.*

## 4. Running & Usage

### Start the Server
Run the following inside your project directory:

```bash
cd ~/bt_server
python app.py
```

### Access User Interface
1.  **Connect to WiFi**: On your phone or laptop, connect to the WiFi network `Pi_Chat_Room` using password `12345678`.
2.  **Open Browser**: Go to `http://10.42.0.1:5000` (or the IP address you found in step 3).

