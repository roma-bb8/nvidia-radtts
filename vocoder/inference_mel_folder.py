import os
import json
import numpy as np

import torch
from scipy.io.wavfile import write

from .models import Generator
from .denoiser import Denoiser


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_vocoder(vocoder_path, config_path, to_cuda=False):
    with open(config_path) as f:
        data_vocoder = f.read()
    config_vocoder = json.loads(data_vocoder)
    h = AttrDict(config_vocoder)
    if 'blur' in vocoder_path:
        config_vocoder['gaussian_blur']['p_blurring'] = 0.5
    else:
        if 'gaussian_blur' in config_vocoder:
            config_vocoder['gaussian_blur']['p_blurring'] = 0.0
        else:
            config_vocoder['gaussian_blur'] = {'p_blurring': 0.0}
            h['gaussian_blur'] = {'p_blurring': 0.0}

    state_dict_g = torch.load(vocoder_path, map_location='cpu')['generator']

    # load hifigan
    vocoder = Generator(h)
    vocoder.load_state_dict(state_dict_g)
    denoiser = Denoiser(vocoder)
    if to_cuda:
        vocoder.cuda()
        denoiser.cuda()
    vocoder.eval()
    denoiser.eval()

    return vocoder, denoiser


def float2pcm(sig, dtype='int16'):
    """Convert floating point signal with a range from -1 to 1 to PCM.
    Any signal values outside the interval [-1.0, 1.0) are clipped.
    No dithering is used.
    Note that there are different possibilities for scaling floating
    point numbers to PCM numbers, this function implements just one of
    them.  For an overview of alternatives see
    http://blog.bjornroche.com/2009/12/int-float-int-its-jungle-out-there.html
    Parameters
    ----------
    sig : array_like
        Input array, must have floating point type.
    dtype : data type, optional
        Desired (integer) data type.
    Returns
    -------
    numpy.ndarray
        Integer data, scaled and clipped to the range of the given
        *dtype*.
    See Also
    --------
    pcm2float, dtype
    """
    sig = np.asarray(sig)
    if sig.dtype.kind != 'f':
        raise TypeError("'sig' must be a float array")
    dtype = np.dtype(dtype)
    if dtype.kind not in 'iu':
        raise TypeError("'dtype' must be an integer type")

    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)


def process_folder(input_mel_folder, vocoder_path, vocoder_config_path, denoising_strength, file_name):
    vocoder, denoiser = load_vocoder(vocoder_path, vocoder_config_path)

    with torch.no_grad():
        input_mel_file = input_mel_folder + f'{file_name}.mel'
        mel_file = torch.load(input_mel_file)
        audio = vocoder(mel_file).float()[0]

        audio_denoised = denoiser(audio, strength=denoising_strength)[0].float()
        audio_denoised = audio_denoised[0].cpu().numpy()
        audio_denoised = audio_denoised / np.max(np.abs(audio_denoised))

        sig = float2pcm(audio_denoised, dtype='int16')

        output_file = input_mel_file.replace('.mel', '.wav')
        write(output_file, 22050, sig)  # 22050
        print('<<--', output_file)

        os.remove(input_mel_file)
