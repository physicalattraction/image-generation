import io
import requests
from PIL import Image
from datetime import datetime
from typing import Collection, Iterator, Union


def download_and_save(url: str, output_file: str):
    """
    Download the contents from the given URL into a local file

    :param url: URL to download from
    :param output_file: Full path to the output file to write
    """

    print(f'Downloading from {url} and saving it to {output_file}')
    response = requests.get(url)
    response.raise_for_status()
    with open(output_file, 'wb') as f:
        f.write(response.content)


def download_image(url: str) -> Image:
    """
    Download the image from the given URL, and return it as a PIL Image
    """

    response = requests.get(url)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content))


def generate_filename(prefix: str = '', suffix: str = '',
                      use_date: bool = True, use_time: bool = True, extension: str = '.jpg') -> str:
    """
    Generate a unique filename
    """

    if extension and not extension.startswith('.'):
        extension = '.' + extension

    base_elements = []
    if prefix:
        base_elements.append(prefix)
    if use_date:
        timestamp = f'{datetime.now():%Y%m%d}'
        base_elements.append(timestamp)
    if use_time:
        timestamp = f'{datetime.now():%H%M%S}'
        base_elements.append(timestamp)
    if suffix:
        base_elements.append(suffix)
    basename = '_'.join(base_elements)
    return basename + extension


def split_to_batches(collection: Collection, batch_size: int) -> Iterator:
    """
    Split the given collection into batches of (max) the given size

    >>> list(split_to_batches([1, 2, 3, 4], 2))
    [[1, 2], [3, 4]]
    >>> list(split_to_batches([1, 2, 3, 4, 5], 2))
    [[1, 2], [3, 4], [5]]
    """

    for i in range(0, len(collection), batch_size):
        yield collection[i:i + batch_size]


def str2bool(v: Union[str, bool]) -> bool:
    """
    Convert the input string to a boolean

    >>> all([str2bool(elem) for elem in [True, 'yes', 'true', 't', 'y', '1']])
    True
    >>> any([str2bool(elem) for elem in [False, 'no', 'false', 'f', 'n', '0']])
    False
    """

    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise AssertionError('Boolean value expected.')
