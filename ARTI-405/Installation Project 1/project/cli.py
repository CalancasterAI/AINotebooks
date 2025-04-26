import os
from inference import predict

def main():
    print("Interactive Rice Classifier. Type 'exit' to quit.")
    while True:
        image_path = input("Enter path to rice grain image (or 'exit' to quit): ").strip()
        if image_path.lower() in ('exit', 'quit'):
            print("Goodbye.")
            break
        if not os.path.isfile(image_path):
            print(f"File not found: {image_path}")
            continue
        result = predict(image_path)
        print(f"\nPrediction: {result}")

if __name__ == "__main__":
    main()