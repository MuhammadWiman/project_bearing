# inspection_manager.py
import json
import os
from datetime import datetime
from config import DAILY_TARGET, SHIFT_HOURS

class InspectionManager:
    def __init__(self, data_file='inspection_log.json'):
        self.data_file = data_file
        self.logs = []
        self.load()
    
    def load(self):
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    self.logs = json.load(f)
            except:
                self.logs = []
    
    def save(self):
        with open(self.data_file, 'w') as f:
            json.dump(self.logs, f, indent=2)
    
    def add(self, class_name, confidence, image_name='upload', measurement=None):
        entry = {
            'timestamp': datetime.now().isoformat(),
            'class': class_name,
            'confidence': confidence,
            'shift': self.get_current_shift(),
            'image_name': image_name,
            # Measurement is optional so older classification-only calls still work.
            # Example value:
            # {
            #   "class": "608Z",
            #   "outer_diameter_mm": 22.1,
            #   "inner_diameter_mm": 8.0,
            #   "status": "PASS"
            # }
            'measurement': measurement,
            'status': measurement.get('status') if measurement else None
        }
        self.logs.append(entry)
        self.save()
        return entry
    
    def get_current_shift(self):
        hour = datetime.now().hour
        for shift, (start, end) in SHIFT_HOURS.items():
            if start <= end:
                if start <= hour < end:
                    return shift
            else:
                if hour >= start or hour < end:
                    return shift
        return 'Morning'
    
    def get_statistics(self):
        today = datetime.now().date()
        today_logs = [l for l in self.logs if datetime.fromisoformat(l['timestamp']).date() == today]
        
        total = len(today_logs)
        small = sum(1 for l in today_logs if l['class'] == 'small')
        medium = sum(1 for l in today_logs if l['class'] == 'medium')
        large = sum(1 for l in today_logs if l['class'] == 'large')
        no_bearing = sum(1 for l in today_logs if l['class'] == 'no_bearing')
        
        return {
            'total': total,
            'small': small,
            'medium': medium,
            'large': large,
            'no_bearing': no_bearing,
            'ok': small,
            'warning': medium,
            'reject': large,
            'defect_rate': (large / total * 100) if total > 0 else 0,
            'target_achievement': (total / DAILY_TARGET * 100) if DAILY_TARGET > 0 else 0,
            'shift': self.get_current_shift(),
            'recent_logs': self.logs[-20:]
        }
    
    def get_logs(self, class_filter=None, date_from=None, date_to=None, limit=100):
        logs = self.logs[::-1]
        if class_filter:
            logs = [l for l in logs if l['class'] == class_filter]
        if date_from:
            logs = [l for l in logs if l['timestamp'] >= date_from]
        if date_to:
            logs = [l for l in logs if l['timestamp'] <= date_to]
        return logs[:limit]
