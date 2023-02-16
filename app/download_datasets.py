file_url  = 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938134-1629953256/toothbrush.tar.xz'


import os
import argparse

try:
    from urllib.request import urlretrieve
except ImportError:  # Python 2 compat
    from urllib import urlretrieve
    
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def parse_user_arguments():
    """Parse user arguments for the evaluation of a method on the MVTec AD
    dataset.

    Returns:
        Parsed user arguments.
    """
    parser = argparse.ArgumentParser(description="""Parse user arguments.""")

    parser.add_argument('--url',
                        required=True,
                        help="""Url to the dataset to download. Must be a .tar.xz file.""")

    args = parser.parse_args()

    assert args.url.endswith('.tar.xz') 
    
    return args


def download_and_extract(url, download_dir):
    import tarfile
    
    FILENAME = url.rsplit('/', 1)[1]
    FILENAME_NO_EXT = FILENAME.rsplit('.', 2)[0]

    
    directory = os.path.join(download_dir, FILENAME)

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
        
    if not os.path.exists(directory):
            print('Downloading %s to %s...' % (url, download_dir))
        
            urlretrieve(url, directory)
            
            print('Download finished')
            
    if not os.path.exists(os.path.join(download_dir, FILENAME_NO_EXT)):
        print('Extracting %s to %s...' % (FILENAME, download_dir))
        
        tar = tarfile.open(directory)
        tar.extractall(download_dir)
        tar.close()
                    
    
    return directory

if __name__ == "__main__":
    args = parse_user_arguments()
    
    dataset_dir = download_and_extract(args.url, 'datasets')
    