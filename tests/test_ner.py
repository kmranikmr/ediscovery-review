#!/usr/bin/env python3
"""Test NER functionality with different inputs"""

import requests
import json

def test_ner_endpoint():
    """Test NER endpoint with various inputs"""
    
    test_cases = [
        {
            "name": "Simple Case",
            "data": {
                "text": "John Smith works at ABC Corp",
                "include_pii": True
            }
        },
        {
            "name": "Complex Case",
            "data": {
                "text": "Dr. Sarah Johnson from TechCorp Inc. sent an email to mike.brown@example.com on March 15, 2024",
                "entity_types": ["PERSON", "ORGANIZATION", "EMAIL", "DATE"]
            }
        },
        {
            "name": "Email and Phone",
            "data": {
                "text": "Contact Jane Doe at jane@company.com or call 555-123-4567",
                "entity_types": ["PERSON", "EMAIL", "PHONE"]
            }
        }
    ]
    
    print("🔍 Testing NER Endpoint")
    print("=" * 50)
    
    for test_case in test_cases:
        print(f"\n🧪 {test_case['name']}")
        print(f"   Input: {test_case['data']['text']}")
        
        try:
            response = requests.post(
                "http://localhost:8001/ner/extract",
                json=test_case['data'],
                timeout=20
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get("success"):
                    entities = result.get("result", {}).get("entities", {})
                    stats = result.get("result", {}).get("statistics", {})
                    
                    print(f"   ✅ Status: Success")
                    print(f"   📊 Total Entities: {stats.get('total_entities', 0)}")
                    print(f"   🎯 Entity Types Found: {stats.get('entity_types', 0)}")
                    print(f"   📈 Avg Confidence: {stats.get('avg_confidence', 0):.2f}")
                    
                    if entities:
                        print(f"   📝 Entities Found:")
                        for entity_type, entity_list in entities.items():
                            for entity in entity_list:
                                print(f"      {entity_type}: '{entity['text']}' (confidence: {entity['confidence']})")
                    else:
                        print(f"   ⚠️  No entities found")
                        
                else:
                    print(f"   ❌ API Error: {result.get('error')}")
                    
            else:
                print(f"   ❌ HTTP Error {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Request failed: {e}")
    
    print(f"\n" + "=" * 50)
    print("NER Testing Complete")

if __name__ == "__main__":
    test_ner_endpoint()
