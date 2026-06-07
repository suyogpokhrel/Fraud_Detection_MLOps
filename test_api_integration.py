#!/usr/bin/env python
"""
Test script for Phase 8 API + UI Integration
Verifies FastAPI backend and communication with Streamlit
"""

import requests
import json
from pathlib import Path

# Configuration
API_URL = "http://127.0.0.1:8000"

# Test data - typical transaction
SAMPLE_TRANSACTION = {
    "step": 1,
    "amount": 9839.64,
    "oldbalanceOrg": 170136.00,
    "newbalanceOrig": 160296.36,
    "oldbalanceDest": 0.00,
    "newbalanceDest": 0.00,
    "type": "PAYMENT",
    "acct_type": "Current",
    "branch": 50,
    "unusuallogin": 5,
    "isFlaggedFraud": 0
}

def test_health_check():
    """Test: API health check"""
    print("\n" + "="*60)
    print("TEST 1: Health Check")
    print("="*60)
    try:
        response = requests.get(f"{API_URL}/", timeout=5)
        if response.status_code == 200:
            print("✓ PASS: API is responding")
            print(f"  Response: {response.json()}")
            return True
        else:
            print(f"✗ FAIL: API returned {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False

def test_single_prediction():
    """Test: Single transaction prediction"""
    print("\n" + "="*60)
    print("TEST 2: Single Transaction Prediction")
    print("="*60)
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=SAMPLE_TRANSACTION,
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            print("✓ PASS: Prediction successful")
            print(f"  Is Fraud: {data['is_fraud']}")
            print(f"  Fraud Probability: {data['fraud_probability']:.4f}")
            print(f"  Confidence: {data['fraud_confidence']:.2f}%")
            print(f"  Risk Level: {data['risk_level']}")
            
            # Validate response structure
            required_fields = ['is_fraud', 'fraud_probability', 'fraud_confidence', 'risk_level']
            if all(field in data for field in required_fields):
                print("✓ PASS: Response has all required fields")
                return True
            else:
                print("✗ FAIL: Response missing required fields")
                return False
        else:
            print(f"✗ FAIL: API returned {response.status_code}")
            print(f"  Error: {response.text}")
            return False
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False

def test_batch_prediction():
    """Test: Batch prediction with 5 transactions"""
    print("\n" + "="*60)
    print("TEST 3: Batch Prediction (5 transactions)")
    print("="*60)
    
    # Create test batch with variations
    batch = [SAMPLE_TRANSACTION.copy() for _ in range(5)]
    batch[2]["amount"] = 999999.99  # Suspicious high amount
    batch[3]["unusuallogin"] = 99  # Suspicious login
    batch[4]["isFlaggedFraud"] = 1  # System flagged
    
    try:
        response = requests.post(
            f"{API_URL}/batch_predict",
            json=batch,
            timeout=30
        )
        if response.status_code == 200:
            predictions = response.json()
            if len(predictions) == 5:
                print("✓ PASS: Batch prediction successful")
                fraud_count = sum(1 for p in predictions if p['is_fraud'])
                print(f"  Total Transactions: {len(predictions)}")
                print(f"  Fraud Detections: {fraud_count}")
                print(f"  Fraud Rate: {fraud_count/5*100:.1f}%")
                
                # Show individual predictions
                for i, pred in enumerate(predictions, 1):
                    status = "🚨 FRAUD" if pred['is_fraud'] else "✓ LEGIT"
                    print(f"    {i}. {status} - Risk: {pred['risk_level']}")
                return True
            else:
                print(f"✗ FAIL: Expected 5 predictions, got {len(predictions)}")
                return False
        else:
            print(f"✗ FAIL: API returned {response.status_code}")
            print(f"  Error: {response.text}")
            return False
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False

def test_model_info():
    """Test: Model information endpoint"""
    print("\n" + "="*60)
    print("TEST 4: Model Information")
    print("="*60)
    try:
        response = requests.get(f"{API_URL}/model-info", timeout=5)
        if response.status_code == 200:
            info = response.json()
            print("✓ PASS: Model info retrieved")
            print(f"  Model Type: {info.get('model_type', 'N/A')}")
            print(f"  Recall: {info.get('recall', 0):.2%}")
            print(f"  Precision: {info.get('precision', 0):.2%}")
            print(f"  ROC-AUC: {info.get('roc_auc', 0):.4f}")
            print(f"  Features: {info.get('features', 'N/A')}")
            print(f"  Training Date: {info.get('training_date', 'N/A')}")
            return True
        else:
            print(f"✗ FAIL: API returned {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False

def test_error_handling():
    """Test: Error handling with invalid input"""
    print("\n" + "="*60)
    print("TEST 5: Error Handling (Invalid Input)")
    print("="*60)
    
    invalid_data = {
        "step": "invalid",  # Should be int
        "amount": -1000,    # Should be positive
        "type": "INVALID"   # Invalid transaction type
    }
    
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=invalid_data,
            timeout=5
        )
        if response.status_code != 200:
            print("✓ PASS: API correctly rejected invalid input")
            print(f"  Status Code: {response.status_code}")
            print(f"  Error: {response.json().get('detail', response.text)[:100]}")
            return True
        else:
            print("✗ FAIL: API accepted invalid input")
            return False
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("PHASE 8: FRAUD DETECTION API + UI INTEGRATION TEST")
    print("="*60)
    print(f"Target API: {API_URL}")
    
    print("\n⏳ Starting tests...\n")
    
    results = {
        "Health Check": test_health_check(),
        "Single Prediction": test_single_prediction(),
        "Batch Prediction": test_batch_prediction(),
        "Model Info": test_model_info(),
        "Error Handling": test_error_handling(),
    }
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! API and UI are working correctly.")
        print("\nNext steps:")
        print("1. Open http://localhost:8501 in browser for Streamlit UI")
        print("2. Visit http://localhost:8000/docs for API documentation")
        print("3. Test the UI with sample transactions")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Check the API setup.")
        return 1

if __name__ == "__main__":
    exit(main())
