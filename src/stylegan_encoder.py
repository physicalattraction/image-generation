"""
Youtube explanation: https://www.youtube.com/watch?v=dCKbRCUyop8
Notebooks: https://drive.google.com/drive/folders/1LBWcmnUPoHDeaYlRiHokGyjywIdyhAQb
"""

import bz2
import os
import os.path
import pickle

import PIL.Image
import numpy as np
from PIL import Image, ImageFilter
from keras.models import load_model
from keras.utils.data_utils import get_file
from tqdm import tqdm

import dnnlib
import dnnlib.tflib as tflib
from encoder.generator_model import Generator
from encoder.perceptual_model import PerceptualModel, load_images
from ffhq_dataset.face_alignment import image_align
from ffhq_dataset.landmarks_detector import LandmarksDetector
from utils import download_and_save, split_to_batches
from keras.applications.resnet import preprocess_input


# noinspection PyBroadException
class StyleGanEncoder:
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_dir = os.path.join(base_dir, 'model')
    img_dir = os.path.join(base_dir, 'img', 'stylegan_encoder')
    raw_img_dir = os.path.join(img_dir, 'raw_images')
    aligned_img_dir = os.path.join(img_dir, 'aligned_images')
    generated_img_dir = os.path.join(img_dir, 'generated_images')
    latent_dir = os.path.join(img_dir, 'latent_repesentations')
    mask_dir = os.path.join(img_dir, 'masks')
    video_dir = os.path.join(img_dir, 'videos')

    def __init__(self):
        self._resnet_file = None
        self._stylegan_model_file = None
        self._landmarks_model_file = None

    @property
    def resnet_file(self) -> str:
        if not self._resnet_file:
            self._resnet_file = os.path.join(self.model_dir, 'finetuned_resnet.h5')
            if not os.path.exists(self._resnet_file):
                download_and_save('https://drive.google.com/uc?id=1aT59NFy9-bNyXjDuZOTMl0qX0jmZc6Zb',
                                  self._resnet_file)
        return self._resnet_file

    @property
    def stylegan_model_file(self) -> str:
        if not self._stylegan_model_file:
            self._stylegan_model_file = os.path.join(self.model_dir, 'karras2019stylegan-ffhq-1024x1024.pkl')
            if not os.path.exists(self._stylegan_model_file):
                msg = 'You have to manually download this file from ' \
                      'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ ' \
                      'and place it in the models folder. We cannot automate this, ' \
                      'since the result for the webpage is a Google Virus warning that the file cannot be scanned.'
                raise AssertionError(msg)
        return self._stylegan_model_file

    def display_raw_images(self, res=256):
        self._display_images(self.raw_img_dir, res)

    def display_aligned_images(self, res=256):
        self._display_images(self.aligned_img_dir, res)

    @property
    def landmarks_model_file(self):
        """
        Return the location to the landsmark model file

        Download the file if you don't have it locally yet
        """

        if not self._landmarks_model_file:
            def unpack_bz2(src_path):
                dst_path = src_path[:-4]
                if not os.path.isfile(dst_path):
                    data = bz2.BZ2File(src_path).read()
                    with open(dst_path, 'wb') as fp:
                        fp.write(data)
                return dst_path

            landmarks_model_url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
            self._landmarks_model_file = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                                             landmarks_model_url, cache_subdir=self.model_dir))
        return self._landmarks_model_file

    def align_images(self, output_size: int = 1024, x_scale: float = 1, y_scale: float = 1,
                     em_scale: float = 0.1, use_alpha: bool = False):
        """
        Extracts and aligns all faces from raw images,
        using DLib and a function from original FFHQ dataset preparation step

        :param output_size: The dimension of images for input to the model
        :param x_scale: Scaling factor for x dimension
        :param y_scale: Scaling factor for y dimension
        :param em_scale: Scaling factor for eye-mouth distance
        :param use_alpha: Add an alpha channel for masking
        """

        landmarks_detector = LandmarksDetector(self.landmarks_model_file)

        for raw_img_name in self._images_in_folder(self.raw_img_dir):
            print(f'Aligning {raw_img_name}')
            try:
                raw_img_path = os.path.join(self.raw_img_dir, raw_img_name)
                base_name, ext = os.path.splitext(raw_img_name)
                # The align_image function saves images as PNG
                aligned_img_name = f'{base_name}_01.png'
                aligned_img_path = os.path.join(self.aligned_img_dir, aligned_img_name)
                if os.path.isfile(aligned_img_path):
                    print(f'{aligned_img_path} already exists')
                    continue
                print('Getting landmarks...')
                for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(raw_img_path), start=1):
                    try:
                        print('Starting face alignment...')
                        aligned_img_name = f'{base_name}_{i:02d}.png'
                        aligned_img_path = os.path.join(self.aligned_img_dir, aligned_img_name)
                        image_align(raw_img_path, aligned_img_path, face_landmarks, output_size=output_size,
                                    x_scale=x_scale, y_scale=y_scale, em_scale=em_scale, alpha=use_alpha)
                        print('Wrote result %s' % aligned_img_path)
                    except:
                        print('Exception in face alignment!')
            except:
                print("Exception in landmark detection!")

    def encode_images(
            self,
            dlatent_avg: str = '',
            model_res: int = 1024,
            batch_size: int = 1,
            optimizer: str = 'ggt',
            # Perceptual model params
            image_size: int = 256,
            resnet_image_size: int = 256,
            lr: float = 0.25,
            decay_rate: float = 4,
            iterations: int = 100,
            decay_steps: int = 4,
            early_stopping: bool = True,
            early_stopping_threshold: float = 0.5,
            early_stopping_patience: int = 10,
            load_resnet: str = 'data/finetuned_resnet.h5',
            use_preprocess_input: bool = True,
            use_best_loss: bool = True,
            average_best_loss: float = 0.25,
            sharpen_input: bool = True,
            # Loss function options
            use_vgg_loss: float = 0.4,
            use_vgg_layer: int = 9,
            use_pixel_loss: float = 1.5,
            use_mssim_loss: float = 200,
            use_lpips_loss: float = 100,
            use_l1_penalty: float = 0.5,
            use_discriminator_loss: float = 0.5,
            use_adaptive_loss: bool = False,
            # Generator params
            randomize_noise: bool = False,
            tile_dlatents: bool = False,
            clipping_threshold: float = 2,
            # Masking params
            load_mask: bool = False,
            face_mask: bool = True,
            use_grabcut: bool = True,
            scale_mask: float = 1.4,
            composite_mask: bool = True,
            composite_blur: int = 8,
            # Video params
            output_video: bool = False,
            video_codec: str = 'MJPG',
            video_frame_rate: int = 24,
            video_size: int = 512,
            video_skip: int = 1
    ):
        """
        Find latent representation of reference images using perceptual losses

        General params
        :param dlatent_avg: Use dlatent from file specified here for truncation instead of dlatent_avg from Gs
        :param model_res: The dimension of images in the StyleGAN model
        :param batch_size: Batch size for generator and perceptual model
        :param optimizer: Optimization algorithm used for optimizing dlatents

        Perceptual model params
        :param image_size: Size of images for perceptual model
        :param resnet_image_size: Size of images for the Resnet model
        :param lr: Learning rate for perceptual model
        :param decay_rate: Decay rate for learning rate
        :param iterations: Number of optimization steps for each batch
        :param decay_steps: Decay steps for learning rate decay (as a percent of iterations)
        :param early_stopping: Stop early once training stabilizes
        :param early_stopping_threshold: Stop after this threshold has been reached
        :param early_stopping_patience: Number of iterations to wait below threshold
        :param load_resnet: Model to load for ResNet approximation of dlatents
        :param use_preprocess_input: Call process_input() first before using feed forward net
        :param use_best_loss: Output the lowest loss value found as the solution
        :param average_best_loss: Do a running weighted average with the previous best dlatents found
        :param sharpen_input: Sharpen the input images

        Loss function options
        :param use_vgg_loss: Use VGG perceptual loss; 0 to disable, > 0 to scale
        :param use_vgg_layer: Pick which VGG layer to use
        :param use_pixel_loss: Use logcosh image pixel loss; 0 to disable, > 0 to scale
        :param use_mssim_loss: Use MS-SIM perceptual loss; 0 to disable, > 0 to scale.
        :param use_lpips_loss: Use LPIPS perceptual loss; 0 to disable, > 0 to scale
        :param use_l1_penalty: Use L1 penalty on latents; 0 to disable, > 0 to scale
        :param use_discriminator_loss: Use trained discriminator to evaluate realism
        :param use_adaptive_loss: Use the adaptive robust loss function from Google Research for pixel and VGG feature loss

        Generator params
        :param randomize_noise: Add noise to dlatents during optimization
        :param tile_dlatents: Tile dlatents to use a single vector at each scale
        :param clipping_threshold: Stochastic clipping of gradient values outside of this threshold

        Masking params
        :param load_mask: Load segmentation masks
        :param face_mask: Generate a mask for predicting only the face area
        :param use_grabcut: Use grabcut algorithm on the face mask to better segment the foreground
        :param scale_mask: Look over a wider section of foreground for grabcut
        :param composite_mask: Merge the unmasked area back into the generated image
        :param composite_blur: Size of blur filter to smoothly composite the images

        # Video params
        :param output_video: Generate videos of the optimization process
        :param video_codec: FOURCC-supported video codec name
        :param video_frame_rate: Video frames per second
        :param video_size: Video size in pixels
        :param video_skip: Only write every n frames (1 = write every frame)
        """

        decay_steps *= 0.01 * iterations  # Calculate steps as a percent of total iterations

        if output_video:
            import cv2
            synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=False),
                                    minibatch_size=batch_size)

        ref_images = self._images_in_folder(self.aligned_img_dir)
        if len(ref_images) == 0:
            raise Exception(f'{self.aligned_img_dir} is empty. Run align_images() first')

        # Initialize generator and perceptual model
        tflib.init_tf()
        with dnnlib.util.open_url(self.stylegan_model_file, cache_dir='cache') as f:
            generator_network, discriminator_network, Gs_network = pickle.load(f)

        generator = Generator(Gs_network, batch_size, clipping_threshold=clipping_threshold,
                              tiled_dlatent=tile_dlatents, model_res=model_res,
                              randomize_noise=randomize_noise)
        if dlatent_avg != '':
            generator.set_dlatent_avg(np.load(dlatent_avg))

        perc_model = None
        if use_lpips_loss > 0.00000001:
            with dnnlib.util.open_url('https://drive.google.com/uc?id=1N2-m9qszOeVC9Tq77WxsLnuWwOedQiD2',
                                      cache_dir='cache') as f:
                perc_model = pickle.load(f)
        perceptual_model = PerceptualModel(perc_model=perc_model, batch_size=batch_size)
        perceptual_model.build_perceptual_model(generator, discriminator_network)

        ff_model = None

        # Optimize (only) dlatents by minimizing perceptual loss between reference and generated images in feature space
        for images_batch in tqdm(split_to_batches(ref_images, batch_size),
                                 total=len(ref_images) // batch_size):
            names = [os.path.splitext(os.path.basename(x))[0] for x in images_batch]
            if output_video:
                video_out = {}
                for name in names:
                    video_out[name] = cv2.VideoWriter(os.path.join(self.video_dir, f'{name}.avi'),
                                                      cv2.VideoWriter_fourcc(*video_codec),
                                                      video_frame_rate, (video_size, video_size))

            perceptual_model.set_reference_images(images_batch)
            dlatents = None
            if ff_model is None:
                if os.path.exists(load_resnet):
                    print("Loading ResNet Model:")
                    ff_model = load_model(load_resnet)
            if ff_model is not None:  # predict initial dlatents with ResNet model
                images = load_images(images_batch, image_size=resnet_image_size)
                if use_preprocess_input:
                    images = preprocess_input(images)
                dlatents = ff_model.predict(images)
            if dlatents is not None:
                generator.set_dlatents(dlatents)

            op = perceptual_model.optimize(generator.dlatent_variable, iterations=iterations,
                                           use_optimizer=optimizer)
            pbar = tqdm(op, leave=False, total=iterations)
            vid_count = 0
            best_loss = None
            best_dlatent = None
            avg_loss_count = 0
            avg_loss = prev_loss = None
            for loss_dict in pbar:
                if early_stopping:  # early stopping feature
                    if prev_loss is not None:
                        if avg_loss is not None:
                            avg_loss = 0.5 * avg_loss + (prev_loss - loss_dict["loss"])
                            if avg_loss < early_stopping_threshold:  # count while under threshold; else reset
                                avg_loss_count += 1
                            else:
                                avg_loss_count = 0
                            if avg_loss_count > early_stopping_patience:  # stop once threshold is reached
                                print("")
                                break
                        else:
                            avg_loss = prev_loss - loss_dict["loss"]
                pbar.set_description(
                    " ".join(names) + ": " + "; ".join(["{} {:.4f}".format(k, v) for k, v in loss_dict.items()]))
                if best_loss is None or loss_dict["loss"] < best_loss:
                    if best_dlatent is None or average_best_loss <= 0.00000001:
                        best_dlatent = generator.get_dlatents()
                    else:
                        best_dlatent = 0.25 * best_dlatent + 0.75 * generator.get_dlatents()
                    if use_best_loss:
                        generator.set_dlatents(best_dlatent)
                    best_loss = loss_dict["loss"]
                if output_video and (vid_count % video_skip == 0):
                    batch_frames = generator.generate_images()
                    for i, name in enumerate(names):
                        video_frame = PIL.Image.fromarray(batch_frames[i], 'RGB').resize(
                            (video_size, video_size), PIL.Image.LANCZOS)
                        video_out[name].write(
                            cv2.cvtColor(np.array(video_frame).astype('uint8'), cv2.COLOR_RGB2BGR))
                generator.stochastic_clip_dlatents()
                prev_loss = loss_dict["loss"]

            if not use_best_loss:
                best_loss = prev_loss
            print(" ".join(names), f" Loss {best_loss:.4f}")

            if output_video:
                for name in names:
                    video_out[name].release()

            # Generate images from found dlatents and save them
            if use_best_loss:
                generator.set_dlatents(best_dlatent)
            generated_images = generator.generate_images()
            generated_dlatents = generator.get_dlatents()
            for img_array, dlatent, img_path, img_name in zip(generated_images, generated_dlatents, images_batch,
                                                              names):
                mask_img = None
                if composite_mask and (load_mask or face_mask):
                    _, im_name = os.path.split(img_path)
                    mask_img = os.path.join(self.mask_dir, f'{im_name}')
                if composite_mask and mask_img is not None and os.path.isfile(mask_img):
                    orig_img = PIL.Image.open(img_path).convert('RGB')
                    width, height = orig_img.size
                    imask = PIL.Image.open(mask_img).convert('L').resize((width, height))
                    imask = imask.filter(ImageFilter.GaussianBlur(composite_blur))
                    mask = np.array(imask) / 255
                    mask = np.expand_dims(mask, axis=-1)
                    img_array = mask * np.array(img_array) + (1.0 - mask) * np.array(orig_img)
                    img_array = img_array.astype(np.uint8)
                    # img_array = np.where(mask, np.array(img_array), orig_img)
                img = PIL.Image.fromarray(img_array, 'RGB')
                img.save(os.path.join(self.generated_img_dir, f'{img_name}.png'), 'PNG')
                np.save(os.path.join(self.latent_dir, f'{img_name}.npy'), dlatent)

            generator.reset_dlatents()

    def _display_images(self, folder: str, res: int):
        filenames = self._images_in_folder(folder)
        print(f'Found {len(filenames)} images in {folder}')
        for filename in filenames:
            img = Image.open(os.path.join(folder, filename)).resize((res, res))
            # img.show()

    def _images_in_folder(self, folder: str):
        return [filename for filename in sorted(os.listdir(folder))
                if filename.endswith('.jpg') or filename.endswith('.png')]


if __name__ == '__main__':
    sge = StyleGanEncoder()
    # sge.display_raw_images(res=256)
    # sge.align_images()
    sge.display_aligned_images(res=256)
    sge.encode_images()
