#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
python -m pip wheel --no-deps --wheel-dir "$tmp/dist" "$root"
python -m venv "$tmp/env"
"$tmp/env/bin/python" -m pip install --no-deps "$tmp"/dist/*.whl
cp -R "$root/assets" "$tmp/assets"
cd "$tmp"
"$tmp/env/bin/python" -I -m unittest discover -s "$root/tests" -v
"$tmp/env/bin/python" -I -m flowlab demo > demo.json
for kind in demo score conditional windows; do
    "$tmp/env/bin/python" -I "$root/examples/render_$kind.py" demo.json --check
done
"$tmp/env/bin/python" -I - "$root" <<'PY'
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
root = Path(sys.argv[1]).resolve()
modules = ['flowlab', 'flowlab.core', 'flowlab.cli', 'flowlab.demo',
           'flowlab.conditional', 'flowlab.__main__']
origins = {m: importlib.util.find_spec(m).origin for m in modules}
for path in origins.values():
    assert 'site-packages' in path and not Path(path).resolve().is_relative_to(root)
exe = str(Path(sys.executable).parent/'flowlab')
points = str(root/'examples/points.json')
commands = [
    ['train', points, 'model.json', '--updates', '1000'],
    ['score', 'model.json', points],
    ['gradient', 'model.json', points],
    ['conditional', 'model.json', '--axis', '0', '--value', '1.5',
     '--low', '-4', '--high', '5', '--grid-size', '257'],
    ['windows', 'model.json', '--axis', '0', '--value', '1.5', '--low', '-4',
     '--high', '4', '--grid-size', '513', '--inner-start', '192', '--inner-end', '320'],
    ['sample', 'model.json', '--count', '128', '--seed', '19'],
]
for i, command in enumerate(commands):
    result = subprocess.run([exe, *command], check=True, capture_output=True, text=True)
    report = json.loads(result.stdout)
    Path('command-'+str(i)+'.json').write_text(result.stdout)
    assert isinstance(report, dict)
print(json.dumps({'installed_origins': origins, 'non_demo_console_commands': len(commands)}))
PY
