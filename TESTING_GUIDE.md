# 🧪 Complete Testing Guide for Fraud Detection System

## 🚀 **Quick Start Testing**

### **1. Start the System**
```bash
poetry run devrun
```

This will:
- Load your 1.7M transaction dataset
- Train ML models (if not already trained)
- Start the API server
- Launch the web dashboard

### **2. Access the Dashboard**
Open your browser and go to: **http://localhost:8000**

You should see:
- Real metrics from your dataset
- Modern fraud detection dashboard
- Transaction analysis form

---

## 🔍 **Testing Methods**

### **Method 1: Web Dashboard Testing**

#### **Step 1: Check Dashboard Metrics**
- **Total Transactions**: Should show ~1,754,155
- **Fraud Detected**: Should show ~14,680
- **Fraud Rate**: Should show ~0.84%
- **Status**: Should show "Online" with green indicator

#### **Step 2: Test Transaction Analysis**
1. **Fill in the form** with real transaction data:
   ```
   Transaction ID: TX_00000001
   Customer ID: CUST_000001
   Terminal ID: TERM_000001
   Amount: 150.00
   ```

2. **Click "Analyze Transaction"**
3. **Check results**:
   - Fraud probability (0-1)
   - Fraud prediction (Yes/No)
   - Confidence level (Low/Medium/High)
   - Risk factors explanation
   - Processing time

#### **Step 3: Test Different Scenarios**

**Test Scenario 1 - Amount > 220 (Should be flagged):**
```
Transaction ID: TX_TEST_001
Customer ID: CUST_000001
Terminal ID: TERM_000001
Amount: 250.00
```
**Expected**: High fraud probability, risk factor "High transaction amount (>220)"

**Test Scenario 2 - Normal Transaction:**
```
Transaction ID: TX_TEST_002
Customer ID: CUST_000001
Terminal ID: TERM_000001
Amount: 50.00
```
**Expected**: Low fraud probability, few risk factors

**Test Scenario 3 - Unusual Time:**
```
Transaction ID: TX_TEST_003
Customer ID: CUST_000001
Terminal ID: TERM_000001
Amount: 100.00
```
**Expected**: Medium fraud probability, risk factor "Unusual transaction time"

---

### **Method 2: API Testing**

#### **Step 1: Test Metrics API**
```bash
curl http://localhost:8000/metrics
```

**Expected Response:**
```json
{
  "total_transactions": 1754155,
  "fraud_detected": 14680,
  "fraud_rate": 0.84,
  "accuracy": 99.2,
  "live_transactions": 100,
  "avg_processing_time": 45,
  "model_confidence": 94.5,
  "status": "online",
  "last_updated": "2024-01-01T12:00:00"
}
```

#### **Step 2: Test Prediction API**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "TX_TEST_001",
    "customer_id": "CUST_000001",
    "terminal_id": "TERM_000001",
    "tx_amount": 250.00,
    "tx_datetime": "2024-01-01T12:00:00"
  }'
```

**Expected Response:**
```json
{
  "transaction_id": "TX_TEST_001",
  "fraud_probability": 0.85,
  "is_fraud": true,
  "confidence": "High",
  "explanation": {
    "ensemble_probability": 0.85,
    "individual_predictions": {
      "xgboost": 0.82,
      "lightgbm": 0.87,
      "catboost": 0.86,
      "neural_network": 0.84
    },
    "top_features": [
      {"feature": "TX_AMOUNT", "value": 250.0, "importance": 0.45},
      {"feature": "amount_vs_avg_ratio", "value": 5.0, "importance": 0.35}
    ],
    "risk_factors": ["High transaction amount (>220)"]
  },
  "processing_time_ms": 45
}
```

#### **Step 3: Test Health Check**
```bash
curl http://localhost:8000/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00",
  "models_loaded": true,
  "version": "1.0.0"
}
```

---

### **Method 3: Python Script Testing**

Create a test script:

```python
import requests
import json

# Test API endpoints
base_url = "http://localhost:8000"

def test_metrics():
    """Test metrics endpoint"""
    response = requests.get(f"{base_url}/metrics")
    print("📊 Metrics Test:")
    print(f"Status: {response.status_code}")
    print(f"Data: {response.json()}")
    print()

def test_prediction():
    """Test prediction endpoint"""
    test_transaction = {
        "transaction_id": "TX_TEST_001",
        "customer_id": "CUST_000001", 
        "terminal_id": "TERM_000001",
        "tx_amount": 250.00,
        "tx_datetime": "2024-01-01T12:00:00"
    }
    
    response = requests.post(
        f"{base_url}/predict",
        headers={"Content-Type": "application/json"},
        data=json.dumps(test_transaction)
    )
    
    print("🔍 Prediction Test:")
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Fraud Probability: {result['fraud_probability']:.3f}")
    print(f"Is Fraud: {result['is_fraud']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Risk Factors: {result['explanation']['risk_factors']}")
    print()

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{base_url}/health")
    print("❤️ Health Test:")
    print(f"Status: {response.status_code}")
    print(f"Health: {response.json()}")
    print()

if __name__ == "__main__":
    print("🧪 Testing Fraud Detection System")
    print("=" * 50)
    
    test_health()
    test_metrics()
    test_prediction()
    
    print("✅ All tests completed!")
```

---

## 📊 **What to Look For**

### **✅ Success Indicators**

#### **Dashboard Metrics:**
- Real numbers from your dataset (not random)
- Fraud rate around 0.84%
- Status shows "Online"
- Charts update with real data

#### **API Responses:**
- Fast response times (<100ms)
- Consistent fraud probability scores
- Meaningful risk factor explanations
- Proper error handling

#### **Model Performance:**
- High fraud probability for amounts > 220
- Low fraud probability for normal amounts
- Reasonable confidence levels
- Explainable risk factors

### **❌ Warning Signs**

#### **Dashboard Issues:**
- Random/fluctuating numbers
- "Offline" status
- Empty charts
- Error messages

#### **API Issues:**
- Slow response times (>500ms)
- HTTP 500 errors
- Inconsistent predictions
- Missing risk factors

---

## 🎯 **Test Scenarios**

### **Scenario 1: Baseline Fraud Detection**
```
Amount: 250.00 (above 220 threshold)
Expected: High fraud probability, "High transaction amount" risk factor
```

### **Scenario 2: Normal Transaction**
```
Amount: 50.00
Expected: Low fraud probability, few risk factors
```

### **Scenario 3: Edge Case**
```
Amount: 220.00 (exactly at threshold)
Expected: Medium fraud probability
```

### **Scenario 4: Unusual Time**
```
Amount: 100.00, Time: 3:00 AM
Expected: Medium fraud probability, "Unusual transaction time" risk factor
```

### **Scenario 5: Weekend Transaction**
```
Amount: 150.00, Day: Saturday
Expected: Low-medium fraud probability, "Weekend transaction" risk factor
```

---

## 🔧 **Troubleshooting**

### **Common Issues:**

#### **1. System Won't Start**
```bash
# Check if Poetry is installed
poetry --version

# Reinstall dependencies
poetry install

# Check Python version
poetry run python --version
```

#### **2. API Not Responding**
```bash
# Check if server is running
curl http://localhost:8000/health

# Check logs
tail -f logs/fraud_detection.log
```

#### **3. Dashboard Shows Errors**
- Check browser console for JavaScript errors
- Verify API endpoints are working
- Check network connectivity

#### **4. Models Not Loading**
```bash
# Check if models exist
ls data/models/

# Retrain models
poetry run python main.py --mode train
```

---

## 📈 **Performance Testing**

### **Load Testing:**
```bash
# Test multiple requests
for i in {1..10}; do
  curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"transaction_id":"TX_'$i'","customer_id":"CUST_000001","terminal_id":"TERM_000001","tx_amount":100.00,"tx_datetime":"2024-01-01T12:00:00"}' &
done
wait
```

### **Response Time Testing:**
```bash
# Measure response time
time curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"transaction_id":"TX_TEST","customer_id":"CUST_000001","terminal_id":"TERM_000001","tx_amount":150.00,"tx_datetime":"2024-01-01T12:00:00"}'
```

---

## 🎉 **Success Criteria**

### **✅ System is Working If:**
- Dashboard loads with real metrics
- API responds in <100ms
- Fraud detection works for amount > 220
- Risk factors are meaningful
- Models show high confidence
- No error messages in logs

### **🚀 Ready for Production If:**
- All test scenarios pass
- Response times are consistent
- Error handling works properly
- Dashboard shows real data
- GPU acceleration is working
- Monitoring is active

---

## 🎯 **Quick Test Commands**

```bash
# Start system
poetry run devrun

# Test in another terminal
curl http://localhost:8000/health
curl http://localhost:8000/metrics
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"transaction_id":"TX_TEST","customer_id":"CUST_000001","terminal_id":"TERM_000001","tx_amount":250.00,"tx_datetime":"2024-01-01T12:00:00"}'
```

**Your fraud detection system is ready to test!** 🚀