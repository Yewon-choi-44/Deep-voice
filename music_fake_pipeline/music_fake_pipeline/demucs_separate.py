"""
HTDemucs 로 원본 오디오를 vocals / accompaniment(음악) 성분으로 분리하는 모듈.

baseline_submit/script.py 의 load_htdemucs_model(), separate_voice_and_music()
로직을 그대로 재사용합니다. 학습용 REAL/FAKE 음악 데이터도 실제 추론과 똑같이
이 함수를 통과시켜야, "학습 시엔 원본 음악을 그대로 쓰고 추론 시엔 HTDemucs를
거친 결과를 쓰는" 분포 불일치(distribution shift)를 피할 수 있습니다.

사용하려면 baseline_submit.zip 안의 model/htdemucs/ 디렉토리 경로가 필요합니다
(config.HTDEMUCS_MODEL_DIR 또는 --htdemucs-dir 인자로 지정).
"""

from pathlib import Path

import numpy as np
import torch
import torchaudio
from demucs.apply import apply_model
from demucs.pretrained import get_model
from demucs.separate import load_track

import config


def load_htdemucs_model(htdemucs_dir: Path):
    """baseline script.py의 load_htdemucs_model과 동일."""
    original_torch_load = torch.load

    def load_trusted_checkpoint(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = load_trusted_checkpoint
    try:
        model = get_model("htdemucs", repo=htdemucs_dir)
    finally:
        torch.load = original_torch_load
    return model.cpu().eval()


def separate_voice_and_music(audio_path: Path, model, device: torch.device):
    """baseline script.py의 separate_voice_and_music과 동일.

    Returns
    -------
    voice_audio, music_audio : np.ndarray (16kHz, mono, float32)
    """
    waveform = load_track(audio_path, model.audio_channels, model.samplerate).float()
    mono_waveform = waveform.mean(0)
    mean = mono_waveform.mean()
    std = mono_waveform.std()

    if float(std) < 1e-8:
        length = round(waveform.shape[-1] * config.AUDIO_SAMPLE_RATE / model.samplerate)
        silence = np.zeros(max(1, length), dtype=np.float32)
        return silence, silence.copy()

    normalized_waveform = (waveform - mean) / std
    with torch.inference_mode():
        sources = apply_model(
            model,
            normalized_waveform[None],
            device=device,
            shifts=0,
            split=True,
            overlap=0.25,
            progress=False,
        )[0]
    sources = sources * std + mean

    vocal_index = model.sources.index("vocals")
    voice_audio = sources[vocal_index].mean(0, keepdim=True)

    music_sources = []
    for index, source_name in enumerate(model.sources):
        if source_name != "vocals":
            music_sources.append(sources[index])
    music_audio = torch.stack(music_sources).sum(0).mean(0, keepdim=True)

    voice_audio = torchaudio.functional.resample(
        voice_audio, model.samplerate, config.AUDIO_SAMPLE_RATE
    )[0]
    music_audio = torchaudio.functional.resample(
        music_audio, model.samplerate, config.AUDIO_SAMPLE_RATE
    )[0]
    return (
        voice_audio.cpu().numpy().astype(np.float32),
        music_audio.cpu().numpy().astype(np.float32),
    )
