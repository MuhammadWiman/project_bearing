# conveyor_controller.py
from datetime import datetime

class ConveyorController:
    def __init__(self, arduino):
        self.arduino = arduino
        self.status = {
            'running': False,
            'speed': 70,
            'auto_stop': True,
            'defect_count': 0,
            'total_count': 0,
            'last_stop_reason': None,
            'last_stop_time': None,
            'direction': 'RIGHT_TO_LEFT'
        }
    
    def get_status(self):
        return self.status
    
    def start(self):
        self.status['running'] = True
        self.status['last_stop_reason'] = None
        self.arduino.send("START")
        return {'success': True, 'status': self.status}
    
    def stop(self, reason='Manual stop'):
        self.status['running'] = False
        self.status['last_stop_reason'] = reason
        self.status['last_stop_time'] = datetime.now().isoformat()
        self.arduino.send("STOP")
        return {'success': True, 'status': self.status}
    
    def set_speed(self, speed):
        speed = max(10, min(100, speed))
        self.status['speed'] = speed
        self.arduino.send(f"SPEED:{speed}")
        return {'success': True, 'status': self.status}
    
    def set_auto_stop(self, enabled):
        self.status['auto_stop'] = enabled
        return {'success': True, 'status': self.status}
    
    def reset_counter(self):
        self.status['defect_count'] = 0
        self.status['total_count'] = 0
        return {'success': True, 'status': self.status}
    
    def add_defect(self, count=1):
        self.status['defect_count'] += count
        self.arduino.send("DEFECT")
        return {'success': True}
    
    def add_production(self, count):
        self.status['total_count'] += count
    
    def reverse(self):
        self.status['direction'] = 'LEFT_TO_RIGHT' if self.status['direction'] == 'RIGHT_TO_LEFT' else 'RIGHT_TO_LEFT'
        self.arduino.send("REVERSE")
        return {'success': True, 'status': self.status}
    
    def auto_stop_if_needed(self):
        if self.status['auto_stop'] and self.status['running']:
            self.status['running'] = False
            self.status['last_stop_reason'] = f'Defect at {datetime.now().strftime("%H:%M:%S")}'
            self.status['last_stop_time'] = datetime.now().isoformat()
            self.arduino.send("STOP")
            return True
        return False