#!/usr/bin/env python3
"""
Test script for the Climate Science Translator API
"""

import requests
import json

# API endpoint
BASE_URL = "http://localhost:8080"

def test_experts_endpoint():
    """Test the experts endpoint"""
    print("Testing /api/experts endpoint...")
    response = requests.get(f"{BASE_URL}/api/experts")
    if response.status_code == 200:
        experts = response.json()
        print(f"✓ Successfully loaded {len(experts)} experts")
        print(f"  Sample expert: {experts[0]['name']} - {experts[0]['expertise_tags']}")
    else:
        print(f"✗ Failed to load experts: {response.status_code}")
    print()

def test_assistant_endpoint(question, expected_keywords=None):
    """Test the assistant endpoint with a question"""
    print(f"Testing question: {question}")
    
    payload = {
        "onboarding": {
            "affiliation": "NGO",
            "thematic_area": "project planning",
            "contact": {
                "name": "María Pérez",
                "affiliation_text": "Climate Aid NGO",
                "email": "maria@ngo.org"
            }
        },
        "question": question
    }
    
    response = requests.post(f"{BASE_URL}/api/assistant", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Answer: {data['answer'][:100]}...")
        print(f"✓ Confidence: {data['confidence']}%")
        print(f"✓ Experts found: {len(data['recommended_experts'])}")
        
        if data['recommended_experts']:
            for i, expert in enumerate(data['recommended_experts'][:2], 1):
                print(f"  {i}. {expert['name']} ({expert['match_score']}%) - {expert['reason']}")
        
        if data['follow_up']:
            print(f"✓ Follow-up: {data['follow_up']}")
        
        if expected_keywords:
            matched_tags = data['debug']['matched_tags']
            print(f"✓ Matched tags: {matched_tags}")
        
    else:
        print(f"✗ Failed: {response.status_code} - {response.text}")
    
    print()

def main():
    """Run all tests"""
    print("🌍 Climate Science Translator API Test")
    print("=" * 50)
    
    # Test experts endpoint
    test_experts_endpoint()
    
    # Test various questions
    test_questions = [
        {
            "question": "How can we implement carbon sequestration in agricultural systems?",
            "expected": ["carbon", "sequestration", "agricultural"]
        },
        {
            "question": "What are effective climate adaptation strategies for coastal communities?",
            "expected": ["adaptation", "coastal", "climate"]
        },
        {
            "question": "What monitoring protocols should we use for climate mitigation projects?",
            "expected": ["mitigation", "monitoring", "climate"]
        },
        {
            "question": "How do we assess climate risk for renewable energy projects?",
            "expected": ["risk", "assessment", "renewable", "energy"]
        }
    ]
    
    for test in test_questions:
        test_assistant_endpoint(test["question"], test["expected"])
    
    print("✅ All tests completed!")
    print(f"\n🌐 Web interface available at: {BASE_URL}")

if __name__ == "__main__":
    main()
