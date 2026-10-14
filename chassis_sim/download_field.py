"""Download the official AdvantageScope 2026 field asset into the simulator cache."""
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile


def main():
    root = Path(__file__).resolve().parent/'generated/assets/Field3d_2026FRCFieldV2'
    url = ('https://github.com/Mechanical-Advantage/AdvantageScopeAssets/releases/download/'
           'default-assets-v2/Field3d_2026FRCFieldV2.zip')
    root.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=60) as response:
        data = response.read()
    with ZipFile(BytesIO(data)) as archive:
        for entry in archive.infolist():
            target = (root/entry.filename).resolve()
            if not target.is_relative_to(root.resolve()):
                raise ValueError('Unexpected archive path')
        archive.extractall(root)
    print(f'Field asset: {root}')


if __name__ == '__main__':
    main()
