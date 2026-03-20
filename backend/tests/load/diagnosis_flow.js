import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const diagnosisLatency = new Trend('diagnosis_latency');

export const options = {
  stages: [
    { duration: '30s', target: 10 },   // Ramp up
    { duration: '1m', target: 50 },     // Steady load
    { duration: '30s', target: 0 },     // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<2000'],  // 95% under 2s
    errors: ['rate<0.01'],               // <1% error rate
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

export default function () {
  // 1. Health check
  const healthRes = http.get(`${BASE_URL}/health/live`);
  check(healthRes, {
    'health status 200': (r) => r.status === 200,
  });

  // 2. DTC search
  const dtcRes = http.get(`${BASE_URL}/api/v1/dtc/search?q=P0300`);
  check(dtcRes, {
    'dtc search status 200': (r) => r.status === 200,
  });
  errorRate.add(dtcRes.status !== 200);

  // 3. Diagnosis request (authenticated)
  const diagnosisPayload = JSON.stringify({
    vehicle_make: 'Volkswagen',
    vehicle_model: 'Golf',
    vehicle_year: 2018,
    dtc_codes: ['P0300'],
    symptoms: 'A motor rázkódik és egyenetlenül jár, különösen hidegindításnál.',
  });

  const diagnosisRes = http.post(
    `${BASE_URL}/api/v1/diagnosis/analyze`,
    diagnosisPayload,
    {
      headers: { 'Content-Type': 'application/json' },
      tags: { name: 'diagnosis' },
    }
  );

  diagnosisLatency.add(diagnosisRes.timings.duration);
  check(diagnosisRes, {
    'diagnosis status 2xx': (r) => r.status >= 200 && r.status < 300,
  });
  errorRate.add(diagnosisRes.status >= 400);

  sleep(1);
}
