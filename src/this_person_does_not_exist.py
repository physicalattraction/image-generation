import os.path
from abc import ABC
from time import sleep

from utils import download_image, generate_filename


class Downloader(ABC):
    _base_dir = os.path.dirname(os.path.dirname(__file__))
    _img_dir: str = None

    def __init__(self, classifier: str):
        self.classifier = classifier

    @property
    def url(self) -> str:
        if self.classifier == 'person':
            return f'https://this{self.classifier}doesnotexist.com/image'
        else:
            return f'https://this{self.classifier}doesnotexist.com'

    @property
    def img_dir(self) -> str:
        if not self._img_dir:
            self._img_dir = os.path.join(self._base_dir, 'img', f'this_{self.classifier}_does_not_exist')
            if not os.path.exists(self._img_dir):
                os.mkdir(self._img_dir)
                with open(os.path.join(self._img_dir, '.gitignore'), 'w') as f:
                    f.write('*\n!.gitignore')
        return self._img_dir

    def download(self, suffix: str):
        img = download_image(self.url)
        filename = generate_filename(use_time=False, suffix=suffix)
        print(f'Downloading {os.path.join(self.img_dir, filename)}')
        img.save(os.path.join(self.img_dir, filename))
        # I noticed that when you make the same request too fast, you will get the same image.
        # I found duplicates when sleeping for 0.5 seconds, or sleeping until the filename changed.
        # That's why I go for the safe option of sleeping 1 second.
        sleep(0.5)


if __name__ == '__main__':
    person_downloader = Downloader('person')
    cat_downloader = Downloader('cat')
    for index in range(820, 10000):
        person_downloader.download(str(index).zfill(4))
        cat_downloader.download(str(index).zfill(4))
