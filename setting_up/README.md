## **How to Boot Raspberry Pi 5 from USB Drive (Headless Setup)**

Traditionally, Raspberry Pi devices have booted from microSD cards, but with the release of the Raspberry Pi 5, USB booting is now natively supported. This means you can run your Raspberry Pi 5 directly from a USB drive, which can offer better performance and reliability compared to microSD cards.

Setting up a Raspberry Pi 5 to boot from a USB drive with a "headless" configuration (no monitor or keyboard attached) is straightforward thanks to the new Raspberry Pi 5 hardware and the official Imager software.

Because the Raspberry Pi 5 supports USB boot natively, you do not need a microSD card for this process.

### **Prerequisites**

* **Raspberry Pi 5 (4GB)**
* **USB Drive:** A fast USB 3.0 flash drive or, ideally, an external USB SSD (Solid State Drive) for much better performance.
* **Power Supply:** The official 27W USB-C PD Power Supply is highly recommended.
* *Note: Using a standard phone charger may not provide enough power to boot a USB drive reliable.*


* **A separate computer** (Windows, Mac, or Linux) to prepare the drive.

---

### **Step 1: Prepare the Software**

The **Raspberry Pi Imager** is the critical tool here because it allows you to pre-configure Wi-Fi and SSH before you ever turn the Pi on.

1. Download and install the **Raspberry Pi Imager** from the [official website](https://www.raspberrypi.com/software/).
2. Insert your USB drive into your computer.
3. Open the Raspberry Pi Imager.

### **Step 2: Select OS and Storage**

1. **Choose Device:** Click "Choose Device" and select **Raspberry Pi 5**.
2. **Choose OS:** Click "Choose OS".
* Select **Raspberry Pi OS (64-bit)**. This is the standard, recommended version.


3. **Choose Storage:** Click "Choose Storage" and select your **USB Drive**.
* *Warning: This will erase all data on the USB drive.*



### **Step 3: Configure Wireless & SSH (The "Headless" Part)**

**This is the most important step.** Do not click "Next" yet.

1. Click "Next". You will see a pop-up asking: *"Would you like to apply OS customisation settings?"*
2. Select **EDIT SETTINGS**.
3. **General Tab:**
* **Set specific hostname:** Give your Pi a name (e.g., `raspberrypi`).
* **Set username and password:** Create a username (e.g., `pi`) and a strong password. You will need these to log in later.
* **Configure Wireless LAN:**
* Check this box.
* Enter your Wi-Fi **SSID** (Network Name) and **Password**.
* **Country:** Select your 2-letter country code (e.g., US, UK, CA).




4. **Services Tab:**
* Check **Enable SSH**.
* Select **Use password authentication**.


5. Click **SAVE** and then **YES** to apply the settings.
6. Click **YES** to confirm erasing the drive and begin writing.

### **Step 4: Boot the Raspberry Pi 5**

1. Once the Imager says "Write Successful," remove the USB drive from your computer.
2. Plug the USB drive into one of the **Blue USB 3.0 ports** on the Raspberry Pi 5 (these are faster).
3. **Ensure no microSD card is inserted** in the Pi. (The Pi 5 will automatically look for a USB boot device if the SD slot is empty).
4. Connect your power supply to the USB-C port on the Pi.
5. Wait about **2–3 minutes**. The first boot takes longer than usual as the Pi resizes the file system and generates SSH keys.

### **Step 5: Connect via SSH**

Now you need to find your Pi on the network and connect to it.

1. Open a terminal (Command Prompt or PowerShell on Windows, Terminal on Mac/Linux) on your computer.
2. Type the following command:
```bash
ssh username@hostname.local
```


* Replace `username` with the user you created in Step 3 (e.g., `pi`).
* Replace `hostname` with the name you set in Step 3 (e.g., `raspberrypi`).
* *Example:* `ssh pi@raspberrypi.local`


3. If asked "Are you sure you want to continue connecting?", type `yes`.
4. Enter the password you created in Step 3.

You should now be logged into your Raspberry Pi 5 terminal!

### **Troubleshooting**

* **"Host not found":** If `ssh pi@raspberrypi.local` doesn't work, your router might not support "mDNS" names. You will need to find the Pi's IP address:
* Log into your home router's admin page and look for "Connected Devices". Find the Raspberry Pi and note its IP (e.g., `192.168.1.50`).
* Then connect using the IP: `ssh pi@192.168.1.50`.


* **Low Power Warning:** If the Pi LED blinks red or the drive doesn't spin up, your power supply might be too weak to power both the Pi 5 and the USB drive. Ensure you are using a high-quality 27W (5V/5A) supply.

### Remark

Sometimes, the first boot can take up to 5 minutes, especially if you are using a large USB drive or an SSD. Be patient and do not unplug the power during this time. If you are anxious to see the Pi boot, you can modify the `config.txt` to change the default booting behavior, but this is not necessary for most users.

**Add the following line to the end of the `config.txt` file on the USB drive:**
```bash
program_usb_boot_mode=1
usb_max_current_enable=1
```
This will ensure that the Pi prioritizes USB booting and provides enough power to the drive.