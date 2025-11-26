"""Comprehensive Locust load testing and monitoring for Audio Talent Classification API"""
import io
import random
import wave
import numpy as np
from locust import HttpUser, task, between, events
from locust.contrib.fasthttp import FastHttpUser


def create_test_audio(duration_ms=1000, sample_rate=22050):
    """Generate synthetic audio data"""
    samples = int(sample_rate * duration_ms / 1000)
    audio_data = np.random.randint(-32768, 32767, samples, dtype=np.int16)
    
    audio_bytes = io.BytesIO()
    with wave.open(audio_bytes, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio_data.tobytes())
    
    audio_bytes.seek(0)
    return audio_bytes


class HealthCheckUser(FastHttpUser):
    """User focused on health check and monitoring endpoints"""
    
    wait_time = between(1, 3)
    
    @task(5)
    def health_check(self):
        """Check API health"""
        with self.client.get(
            "/api/v1/health",
            catch_response=True,
            name="/api/v1/health [monitoring]"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")
    
    @task(2)
    def get_model_info(self):
        """Get model information"""
        with self.client.get(
            "/api/v1/model-info",
            catch_response=True,
            name="/api/v1/model-info [monitoring]"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")
    
    @task(1)
    def get_model_metrics(self):
        """Get model metrics and performance"""
        with self.client.get(
            "/api/v1/model-metrics",
            catch_response=True,
            name="/api/v1/model-metrics [monitoring]"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")
    
    @task(1)
    def check_training_status(self):
        """Check if training is in progress"""
        with self.client.get(
            "/api/v1/train-status",
            catch_response=True,
            name="/api/v1/train-status [monitoring]"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")


class SinglePredictionUser(FastHttpUser):
    """User focused on single audio predictions"""
    
    wait_time = between(1, 4)
    
    @task(10)
    def single_prediction(self):
        """Make single prediction"""
        audio_file = create_test_audio()
        
        with self.client.post(
            "/api/v1/predictions/single",
            files={"file": ("test.wav", audio_file, "audio/wav")},
            catch_response=True,
            name="/api/v1/predictions/single [prediction]"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if 'prediction' in data:
                        response.success()
                    else:
                        response.failure("Missing prediction in response")
                except:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Status: {response.status_code}")


class BatchPredictionUser(FastHttpUser):
    """User focused on batch audio predictions"""
    
    wait_time = between(3, 8)
    
    @task(5)
    def batch_prediction(self):
        """Make batch predictions (3-5 files)"""
        num_files = random.randint(3, 5)
        files = []
        
        for i in range(num_files):
            audio_file = create_test_audio()
            files.append(("files", (f"test_{i}.wav", audio_file, "audio/wav")))
        
        with self.client.post(
            "/api/v1/predictions/batch",
            files=files,
            catch_response=True,
            name=f"/api/v1/predictions/batch [batch-{num_files}]"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if 'predictions' in data:
                        response.success()
                    else:
                        response.failure("Missing predictions in response")
                except:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Status: {response.status_code}")


class MixedWorkloadUser(FastHttpUser):
    """User with mixed workload - realistic user behavior"""
    
    wait_time = between(2, 5)
    
    def on_start(self):
        """Initialize with health check"""
        self.client.get("/api/v1/health")
    
    @task(2)
    def health_check(self):
        """Regular health checks"""
        self.client.get("/api/v1/health")
    
    @task(3)
    def get_model_info(self):
        """Get model info"""
        self.client.get("/api/v1/model-info")
    
    @task(8)
    def predict_single(self):
        """Single predictions - most common"""
        audio_file = create_test_audio()
        self.client.post(
            "/api/v1/predictions/single",
            files={"file": ("test.wav", audio_file, "audio/wav")},
            catch_response=True
        )
    
    @task(2)
    def predict_batch(self):
        """Batch predictions - less frequent"""
        num_files = random.randint(2, 4)
        files = []
        for i in range(num_files):
            audio_file = create_test_audio()
            files.append(("files", (f"test_{i}.wav", audio_file, "audio/wav")))
        
        self.client.post(
            "/api/v1/predictions/batch",
            files=files,
            catch_response=True
        )
    
    @task(1)
    def check_metrics(self):
        """Occasionally check metrics"""
        self.client.get("/api/v1/model-metrics")


class StressTestUser(FastHttpUser):
    """High-load stress testing"""
    
    wait_time = between(0.1, 0.5)
    
    @task(15)
    def stress_single_prediction(self):
        """Aggressive prediction requests"""
        audio_file = create_test_audio()
        self.client.post(
            "/api/v1/predictions/single",
            files={"file": ("test.wav", audio_file, "audio/wav")},
            catch_response=True,
            name="/api/v1/predictions/single [stress]"
        )


class SpikeTestUser(FastHttpUser):
    """Sudden traffic spike simulation"""
    
    wait_time = between(0, 0.3)
    
    @task(20)
    def spike_requests(self):
        """Burst traffic requests"""
        # Mix of health checks and predictions
        if random.random() < 0.3:
            self.client.get("/api/v1/health")
        else:
            audio_file = create_test_audio()
            self.client.post(
                "/api/v1/predictions/single",
                files={"file": ("test.wav", audio_file, "audio/wav")},
                catch_response=True,
                name="/api/v1/predictions/single [spike]"
            )


# Event handlers for monitoring
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when load test starts"""
    print("\n" + "="*80)
    print(" " * 20 + "LOAD TEST STARTED")
    print("="*80)
    print(f"Target: {environment.host}")
    print("="*80 + "\n")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when load test stops"""
    print("\n" + "="*80)
    print(" " * 20 + "LOAD TEST COMPLETED")
    print("="*80)
    
    # Print summary statistics
    print(f"\nTotal Requests: {environment.stats.total.num_requests}")
    print(f"Total Failures: {environment.stats.total.num_failures}")
    print(f"Success Rate: {(environment.stats.total.num_requests - environment.stats.total.num_failures) / environment.stats.total.num_requests * 100:.2f}%")
    print(f"Average Response Time: {environment.stats.total.avg_response_time:.2f}ms")
    print(f"Min Response Time: {environment.stats.total.min_response_time:.2f}ms")
    print(f"Max Response Time: {environment.stats.total.max_response_time:.2f}ms")
    print(f"Requests/sec: {environment.stats.total.total_rps:.2f}")
    
    print("\n" + "="*80)


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, environment, **kwargs):
    """Monitor individual requests"""
    if exception:
        print(f"  ✗ {name}: {exception}")
    else:
        print(f"  ✓ {name}: {response_time:.0f}ms")
    # Only print stats every 10 requests to reduce noise
    if environment.stats.total.num_requests % 10 == 0:
        print(f"Total failures: {environment.stats.total.num_failures}")
        print(f"Average response time: {environment.stats.total.avg_response_time:.2f}ms")
        print("="*60)
