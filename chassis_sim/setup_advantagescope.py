"""Add a camera tab to a closed AdvantageScope 2026 profile, keeping a backup."""
import copy
import json
import os
from pathlib import Path
import time
from uuid import uuid4


def configure(profile):
    files = sorted(profile.glob('state-26_*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError('Open and close AdvantageScope once before setting up its camera tab.')
    path = files[0]
    state = json.loads(path.read_text(encoding='utf-8'))
    for hub in state.get('hubs', []):
        tabs = hub['state']['tabs']
        field = next((tab for tab in tabs['tabs'] if tab['type'] == 3), None)
        if field is None:
            continue
        override = dict(type='cameraOverride', logKey='NT:/VortexSim/Camera', logType='Pose3d', visible=True, options={})
        sources = field['controller']['sources']
        if not any(source.get('logKey') == override['logKey'] for source in sources):
            sources.append(override)
        camera = next((tab for tab in tabs['tabs'] if tab.get('title') == 'Vortex Camera'), None)
        if camera is None:
            camera = copy.deepcopy(field)
            camera['title'] = 'Vortex Camera'
            camera['controllerUUID'] = uuid4().hex
            tabs['tabs'].append(camera)
        camera['controller']['sources'] = [
            dict(type='robot', logKey='NT:/VortexSim/Actual', logType='Pose3d', visible=True,
                 options=dict(model='2026 KitBot')), override]
        camera['controller']['game'] = 'FRC:2026 Field'
        camera['renderer'] = dict(cameraIndex=0, orbitFov=50, cameraPosition=[0, 0, 0], cameraTarget=[0, 0, 0])
        tabs['selected'] = tabs['tabs'].index(camera)
    backup = path.with_name(path.name + f'.backup-{time.time_ns()}')
    backup.write_bytes(path.read_bytes())
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state), encoding='utf-8')
    temporary.replace(path)
    return path, backup


def main():
    profile = Path(os.environ.get('APPDATA', Path.home()/'.config'))/'AdvantageScope'
    path, backup = configure(profile)
    print(f'Configured {path}\nBackup: {backup}\nLaunch AdvantageScope and connect to the simulator.')


if __name__ == '__main__':
    main()
