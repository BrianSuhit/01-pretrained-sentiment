from transformers import pipeline

print(">>> Downloading and uploading the smart model...")
sorter = pipeline("sentiment-analysis")

phrases = [
    "I am very happy learning AI Engineering!", # Simple positivo
    "The movie was not that bad, I liked it.",   # Complejo
    "I hate errors and bugs in my code.",        # Simple negativo
    "I'm feeling neutral about this situation."  # Neutro
]

print("\n" + "="*40)
print("   INFERENCE RESULTS")
print("="*40)

results = sorter(phrases)

for i, phrase in enumerate(phrases):
    feeling = results[i]['label']
    trust = results[i]['score']
    
    emoji = "😊" if feeling == "POSITIVE" else "😡"
    
    print(f"PHRASE: {phrase}")
    print(f"VERDICT: {feeling} {emoji} | Trust: {trust:.4f}")
    print("-" * 20)