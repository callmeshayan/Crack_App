import time

class Marker:
    def fire(self, duration_ms: int = 100):
        print(f"[MARK] {duration_ms}ms")
        time.sleep(duration_ms / 1000.0)