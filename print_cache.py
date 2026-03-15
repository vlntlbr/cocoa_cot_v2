import pickle
import glob

# Find the latest cache file
cache_files = sorted(glob.glob('results/cache/generations_*.pkl'))
if cache_files:
    latest = cache_files[-1]
    print(f"Loading: {latest}")
    
    with open(latest, 'rb') as f:
        data = pickle.load(f)
    
    print(f"\nTotal cached prompts: {len(data)}")
    print(f"\nFirst item structure:")
    item = data[0]
    print(f"  Keys: {item.keys()}")
    print(f"  Prompt: {item['prompt'][:100]}...")
    print(f"  Greedy text: {item['greedy'].text[:200]}...")
    print(f"  Number of samples: {len(item['samples'])}")
    print(f"  Sample 1 text: {item['samples'][0].text[:200]}...")
else:
    print("No cache files found")
