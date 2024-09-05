import torch
import torch.nn.functional as F
import soundfile as sf

from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForPreTraining,
    Wav2Vec2Model,
    WavLMModel,
    WavLMPreTrainedModel,
    WavLMForCTC,
    WhisperForConditionalGeneration,
    WhisperForAudioClassification,
    get_constant_schedule
)
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices

model = WhisperForConditionalGeneration.from_pretrained('../models/whisper-base')
# model = Wav2Vec2ForPreTraining.from_pretrained('models/wavlm-large')
# model = WavLMModel.from_pretrained('models/wavlm-base')

print(model)

total_params = 0
for name, parameters in model.named_parameters():
    if 'decoder' not in name:
        print(name)
        total_params += parameters.numel()

print(
    f"total parameters:{total_params}"
)