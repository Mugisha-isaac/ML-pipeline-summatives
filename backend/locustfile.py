"""Load testing script using Locust"""
import io
import random
from locust import HttpUser, task, between, events
from locust.contrib.fasthttp import FastHttpUser

class AudioTalentLoadTest(FastHttpUser):
    """Load test scenarios for Audio Talent Classification API"""
    
    wait_time = between(0.5, 2)
    
    def on_start(self):
        """Executed when a test starts"""
        # Health check
        self.client.get("/api/v1/health")
    
    @task(1)
    def health_check(self):
        """Task 1: Health check"""
        self.client.get("/api/v1/health")
    
    @task(2)
    def get_model_info(self):
        """Task 2: Get model information"""
        self.client.get("/api/v1/model-info")
    
    @task(3)
    def single_prediction(self):
        """Task 3: Make single prediction"""
        # Create dummy audio file (1 second of silence - 22050 samples)
        import wave
        import numpy as np
        
        audio_data = np.zeros(22050, dtype=np.int16)
        audio_bytes = io.BytesIO()
        
        with wave.open(audio_bytes, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(22050)
            wav.writeframes(audio_data.tobytes())
        
        audio_bytes.seek(0)
        
        self.client.post(
            "/api/v1/predictions/single",
            files={"file": ("test.wav", audio_bytes, "audio/wav")},
            name="/api/v1/predictions/single"
        )
    
    @task(1)
    def get_training_status(self):
        """Task 4: Check training status"""
        self.client.get("/api/v1/train-status")
    
    @task(1)
    def get_model_metrics(self):
        """Task 5: Get model metrics"""
        self.client.get("/api/v1/model-metrics")

class AudioTalentStressTest(FastHttpUser):
    """Stress test - aggressive prediction requests"""
    
    wait_time = between(0, 1)
    
    @task(10)
    def stress_prediction(self):
        """Aggressive prediction requests"""
        import wave
        import numpy as np
        
        audio_data = np.random.randint(-32768, 32767, 22050, dtype=np.int16)
        audio_bytes = io.BytesIO()
        
        with wave.open(audio_bytes, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(22050)
            wav.writeframes(audio_data.tobytes())
        
        audio_bytes.seek(0)
        
        self.client.post(
            "/api/v1/predictions/single",
            files={"file": ("test.wav", audio_bytes, "audio/wav")},
            name="/api/v1/predictions/single [stress]"
        )

# Event handler for detailed logging
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print("\n" + "="*60)
    print("LOAD TEST STARTED")
    print("="*60)

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    print("\n" + "="*60)
    print("LOAD TEST COMPLETED")
    print("="*60)
    print(f"Total requests: {environment.stats.total.num_requests}")
    print(f"Total failures: {environment.stats.total.num_failures}")
    print(f"Average response time: {environment.stats.total.avg_response_time:.2f}ms")
    print("="*60)
