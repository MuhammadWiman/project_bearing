# arduino_controller.py
import serial
import serial.tools.list_ports
import time

class ArduinoController:
    def __init__(self, port=None, baud=9600):
        self.port = port
        self.baud = baud
        self.serial = None
        self.connected = False
    
    def find_arduino_port(self):
        """Mencari port Arduino yang terhubung"""
        print("🔍 Scanning for Arduino ports...")
        ports = serial.tools.list_ports.comports()
        
        for port in ports:
            port_desc = port.description.lower()
            port_name = port.device
            print(f"  Found: {port_name} - {port.description}")
            
            if 'arduino' in port_desc or 'usb serial' in port_desc or 'ch340' in port_desc:
                print(f"  ✅ Arduino detected on {port_name}")
                return port_name
            
            if port.vid and port.pid:
                if port.vid == 0x2341 or port.vid == 0x1A86:
                    print(f"  ✅ Arduino detected on {port_name}")
                    return port_name
        
        print("  ❌ No Arduino found")
        return None
    
    def connect(self):
        """Koneksi ke Arduino"""
        try:
            if self.serial:
                self.serial.close()
            
            if not self.port:
                self.port = self.find_arduino_port()
            
            if not self.port:
                print("❌ No Arduino port found")
                self.connected = False
                return False
            
            print(f"🔌 Connecting to {self.port}...")
            self.serial = serial.Serial(self.port, self.baud, timeout=1)
            time.sleep(2)
            self.connected = True
            print(f"✅ Arduino connected on {self.port}")
            return True
            
        except Exception as e:
            print(f"❌ Arduino connection failed: {e}")
            self.connected = False
            return False
    
    def send(self, command):
        """Kirim command ke Arduino"""
        if not self.connected or not self.serial:
            print(f"⚠️ Arduino not connected")
            return None
        
        try:
            self.serial.write(f"{command}\n".encode())
            print(f"📤 Sent: {command}")
            time.sleep(0.2)
            
            responses = []
            while self.serial.in_waiting:
                resp = self.serial.readline().decode().strip()
                if resp:
                    responses.append(resp)
                    print(f"📥 Arduino: {resp}")
            
            return responses[-1] if responses else None
            
        except Exception as e:
            print(f"❌ Send error: {e}")
            return None
    
    def close(self):
        if self.serial:
            self.serial.close()
        self.connected = False