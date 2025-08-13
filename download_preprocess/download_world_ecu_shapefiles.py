"""
This module downloads shape files for Ecuador and the world.
"""

import os
import zipfile
from urllib.request import urlretrieve
from tqdm import tqdm
from pathlib import Path

from config import DATA_PATH

def download_with_progress(url, filename):
    """
    Download a file from URL to filename with a progress bar.
    """
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=filename.name) as t:
        urlretrieve(url, filename, reporthook=t.update_to)

def download_and_extract_shapes(data_path, download_url, folder_name):
    """
    Downloads and extracts a ZIP file containing geographic shapefiles, with progress bars.
    """
    # Define paths
    geography_path = os.path.join(data_path, "geography", folder_name)
    zip_path = Path(data_path) / "geography" / f"{folder_name}.zip"

    # Create directory if it doesn't exist
    os.makedirs(geography_path, exist_ok=True)

    # Download the zip file
    print(f"Downloading to: {zip_path}")
    download_with_progress(download_url, zip_path)

    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.infolist()
        for member in tqdm(members, desc="Extracting", unit="file"):
            zip_ref.extract(member, geography_path)

    print(f"Extracted to: {geography_path}")
    return geography_path

# Download and extract
for name, url in [('world_countries',
                   "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"),
                  ('ecu_adm_2024',
                   "https://data.humdata.org/dataset/ab3c7592-3b0c-41cd-999a-2919a6b243f2/resource/d00145f6-141c-4bf2-a881-c32341ddec75/download/ecu_adm_2024.zip")
                   ]:
    download_and_extract_shapes(DATA_PATH, download_url=url, folder_name=name)
