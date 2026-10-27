"""Exercise the single launcher without installing packages or starting hardware."""
from pathlib import Path
import fcntl
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory(prefix='app-launcher-') as directory:
        base = Path(directory)
        root = base / 'application with spaces'
        for name in ('scripts', 'data', '.venv/bin', '.offline-python/python/bin', 'frontend/dist', 'fakebin'):
            (root / name).mkdir(parents=True)
        for name in ('app.sh', 'scripts/setup_state.py'):
            shutil.copyfile(ROOT / name, root / name)
        for name in ('.venv/pyvenv.cfg', 'frontend/dist/index.html', 'frontend/index.html', 'frontend/package.json', 'frontend/package-lock.json', 'requirements.txt'):
            (root / name).write_text('test')
        assets = ['models/detection/bead_finder.onnx', 'models/packet/barcodescanner.onnx', 'models/segmentation/testbed.onnx', 'models/segmentation/essaetestbed.onnx', 'models/segmentation/anomolies.onnx']
        assets += ['Classification/siglip2-base-patch16-384/' + name for name in ('vision_model.onnx', 'vision_model.onnx_data', 'text_model.onnx', 'text_model.onnx_data', 'config.json', 'preprocessor_config.json', 'tokenizer.json', 'tokenizer_config.json')]
        for name in assets:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('test asset')
        log = root / 'calls.log'

        def executable(path, content):
            path.write_text('#!/usr/bin/env bash\nset -eu\n' + content)
            path.chmod(0o755)

        wrapper = ('if [[ "${1:-}" == scripts/setup_state.py || "${1:-}" == -c || "${1:-}" == - ]]; then\n'
                   f'  exec {shlex.quote(sys.executable)} "$@"\nfi\n'
                   'echo "$*" >> "$TEST_STARTUP_LOG"\n'
                   'if [[ "${FAIL_SETUP:-}" == 1 && "${1:-}" == scripts/verify_offline_install.py ]]; then exit 9; fi\n')
        executable(root / '.venv/bin/python', wrapper)
        executable(root / '.offline-python/python/bin/python3', wrapper)
        for name in ('node', 'npm', 'v4l2-ctl'):
            executable(root / 'fakebin' / name, f'echo "{name} $*" >> "$TEST_STARTUP_LOG"\n')
        executable(root / 'fakebin/id', 'echo "user dialout"\n')
        env = {**os.environ, 'PATH': str(root / 'fakebin') + os.pathsep + os.environ['PATH'], 'PYTHON_BIN': sys.executable, 'TEST_STARTUP_LOG': str(log)}

        def run(*args, code=0, failure=False):
            log.write_text('')
            result = subprocess.run(['bash', str(root / 'app.sh'), *args], cwd=base, env={**env, 'FAIL_SETUP': '1' if failure else '0'}, capture_output=True, text=True)
            assert result.returncode == code, result.stdout + result.stderr
            return log.read_text(), result.stdout + result.stderr

        calls, _ = run('install')
        assert 'pip install' in calls and 'npm run build' in calls
        assert 'start_linux_backend.py' not in calls
        for args in ((), ('start',)):
            calls, _ = run(*args)
            assert calls == 'scripts/start_linux_backend.py\n', calls
        packages = base / 'offline/linux-cp310'
        packages.mkdir(parents=True)
        (packages / 'requirements-lock.txt').write_text('test==1')
        calls, _ = run('install')
        assert 'pip --isolated install' in calls and 'verify_offline_install.py' in calls
        assert 'scripts/offline_runtime.py' not in calls
        for args in ((), ('start',)):
            calls, _ = run(*args)
            assert calls == 'scripts/offline_runtime.py\n', calls
        run('install', code=9, failure=True)
        assert json.loads((root / 'data/setup-offline.json').read_text())['status'] == 'incomplete'
        calls, _ = run('start')
        assert calls == 'scripts/offline_runtime.py\n', calls
        asset = root / assets[0]
        asset.unlink()
        calls, output = run('start', code=1)
        assert not calls and 'Missing or empty asset' in output
        asset.write_text('version https://git-lfs.github.com/spec/v1\n')
        calls, output = run('start', code=1)
        assert not calls and 'Git LFS pointer' in output
        asset.write_text('test asset')
        with (root / 'data/backend.lock').open('w') as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            calls, output = run('start', code=1)
            assert not calls and 'already running' in output
        (root / '.venv/bin/python').unlink()
        calls, output = run(code=1)
        assert not calls and 'bash app.sh install' in output
        run('--help')
        run('invalid', code=2)
        run('start', 'extra', code=2)
    assert [p.name for p in ROOT.glob('*.sh')] == ['app.sh']
    print('PASS: install-only, daily start, offline routing, setup failures, model checks, duplicate lock, missing environment and usage')


if __name__ == '__main__':
    main()
