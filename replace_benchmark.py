import os

target_dirs = ['scripts', 'docs', 'benchmarks', 'configs', 'validation']
target_root_files = ['README.md', 'test.sh', 'run.sh', 'setup.sh', 'report.py']
target_files = []

for d in target_dirs:
    for root, _, files in os.walk(d):
        for f in files:
            if f.endswith(('.py', '.md', '.sh', '.json')):
                target_files.append(os.path.join(root, f))

target_files.extend(target_root_files)

for filepath in target_files:
    if not os.path.exists(filepath): continue
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Blindly replace substrings
        new_content = content.replace('Campaign', 'Benchmark')
        new_content = new_content.replace('campaign', 'benchmark')
        new_content = new_content.replace('CAMPAIGN', 'BENCHMARK')
        
        if new_content != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f'Updated {filepath}')
    except Exception as e:
        print(f"Error on {filepath}: {e}")
