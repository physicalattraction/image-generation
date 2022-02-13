import os
from typing import Optional

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from dall_e import load_model, map_pixels, unmap_pixels

from utils import download_and_save, download_image

target_image_size = 256


class DallEUsage:
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_dir = os.path.join(base_dir, 'model')
    img_dir = os.path.join(base_dir, 'img', 'dall-e')

    def __init__(self):
        self._dec_file = None
        self._dec_file = None

        self.dev = torch.device('cpu')
        self.enc = load_model(self.enc_file, self.dev)
        self.dec = load_model(self.dec_file, self.dev)
        self._preprocessed_image: Optional[torch.Tensor] = None

    @property
    def enc_file(self) -> str:
        """
        Return the location to the enc file

        Download the file if you don't have it locally yet
        """

        self._dec_file = os.path.join(self.model_dir, 'encoder.pkl')
        if not os.path.exists(self._dec_file):
            download_and_save('https://cdn.openai.com/dall-e/encoder.pkl', self._dec_file)
        return self._dec_file

    @property
    def dec_file(self) -> str:
        """
        Return the location to the dec file

        Download the file if you don't have it locally yet
        """

        self._dec_file = os.path.join(self.model_dir, 'decoder.pkl')
        if not os.path.exists(self._dec_file):
            download_and_save('https://cdn.openai.com/dall-e/decoder.pkl', self._dec_file)
        return self._dec_file

    def load_image(self, url: str):
        """
        Download the given image, and store as a Torch inside this object
        """

        self._preprocessed_image = self._preprocess(download_image(url))

    def display(self):
        """
        Display the loaded image
        """

        assert self._preprocessed_image is not None
        img = T.ToPILImage(mode='RGB')(self._preprocessed_image[0])
        img.show()
        img.save(os.path.join(self.img_dir, 'original.jpeg'))

    def transform(self):
        """
        Transform the loaded image
        """

        assert self._preprocessed_image is not None
        z_logits = self.enc(self._preprocessed_image)
        z = torch.argmax(z_logits, axis=1)
        z = F.one_hot(z, num_classes=self.enc.vocab_size).permute(0, 3, 1, 2).float()

        x_stats = self.dec(z).float()
        x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))
        x_rec = T.ToPILImage(mode='RGB')(x_rec[0])

        x_rec.show()
        x_rec.save(os.path.join(self.img_dir, 'transformed.jpeg'))

    @staticmethod
    def _preprocess(img: Image) -> torch.Tensor:
        s = min(img.size)

        if s < target_image_size:
            raise ValueError(f'min dim for image {s} < {target_image_size}')

        r = target_image_size / s
        s = [round(r * img.size[1]), round(r * img.size[0])]
        img = TF.resize(img, s, interpolation=Image.LANCZOS)
        img = TF.center_crop(img, output_size=2 * [target_image_size])
        img = torch.unsqueeze(T.ToTensor()(img), 0)
        return map_pixels(img)


if __name__ == '__main__':
    usage = DallEUsage()
    usage.load_image(url='https://assets.bwbx.io/images/users/iqjWHBFdfxIU/iKIWgaiJUtss/v2/1000x-1.jpg')
    usage.display()
    usage.transform()
