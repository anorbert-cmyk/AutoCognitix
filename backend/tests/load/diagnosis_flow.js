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
const TEST_EMAIL = __ENV.TEST_EMAIL || 'loadtest@autocognitix.hu';
const TEST_PASSWORD = __ENV.TEST_PASSWORD || 'LoadTest123!';

// One-time setup: login and obtain auth token
export function setup() {
  const loginRes = http.post(
    `${BASE_URL}/api/v1/auth/login`,
    `username=${encodeURIComponent(TEST_EMAIL)}&password=${encodeURIComponent(TEST_PASSWORD)}`,
    { headers: { 'Content-Type': 'application/x-www-form-urlencoded' } }
  );

  const success = check(loginRes, {
    'login status 200': (r) => r.status === 200,
  });

  if (!success) {
    console.error(`Login failed (status ${loginRes.status}). Set TEST_EMAIL and TEST_PASSWORD env vars.`);
    return { token: null };
  }

  const body = JSON.parse(loginRes.body);
  return { token: body.access_token };
}

export default function (data) {
  const authHeaders = data.token
    ? { 'Content-Type': 'application/json', 'Authorization': `Bearer ${data.token}` }
    : { 'Content-Type': 'application/json' };

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
      headers: authHeaders,
      tags: { name: 'diagnosis' },
    }
  );

  diagnosisLatency.add(diagnosisRes.timings.duration);
  check(diagnosisRes, {
    'diagnosis status 200': (r) => r.status === 200,
  });
  errorRate.add(diagnosisRes.status >= 400);

  sleep(1);
}
