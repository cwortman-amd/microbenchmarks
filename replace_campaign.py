import os, re
exclude_dirs = {'.git', 'results', '.microbenchmarks-rocm-venv', '__pycache__', 'venv', 'aiter'}
for root, dirs, files in os.walk('.'):
    dirs[:] = [d for d in dirs if d not in exclude_dirs]
    for f in files:
        if f.endswith(('.py', '.md', '.sh', '.json')):
            filepath = os.path.join(root, f)
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                new_content = re.sub(r'\bCampaign\b', 'Benchmark', content)
                new_content = re.sub(r'\bcampaign\b', 'benchmark', new_content)
                new_content = re.sub(r'\bCAMPAIGN\b', 'BENCHMARK', new_content)
                
                if new_content != content:
                    with open(filepath, 'w', encoding='utf-8') as file:
                        file.write(new_content)
                    print(f'Updated {filepath}')
            except Exception:
                pass
