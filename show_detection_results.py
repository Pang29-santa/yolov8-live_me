import json

def display_detection_results(json_data):
    data = json.loads(json_data)
    
    if data["OK"] and data["objects"]:
        for obj in data["objects"]:
            print(f"Label: {obj['label']}")
            print(f"Score: {float(obj['score']):.2%}")

# Example usage
sample_data = '''{
    "OK": true,
    "filename": "./uploads/2025/02/28/20250228104510.png",
    "objects": [
        {
            "label": "(เดาว่า) ต้มยำกุ้งน้ำข้น",
            "rank": "0",
            "result": "ต้มยำกุ้งน้ำข้น",
            "score": "0.07687537"
        }
    ]
}'''

display_detection_results(sample_data)
