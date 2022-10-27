# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/whisper/expert.py ]
#   Synopsis     [ the whisper wrapper ]
#   Author       [ OpenAI ]
"""*********************************************************************************************"""

###############
# IMPORTATION #
###############

import torch

from ..interfaces import UpstreamBase

############
# CONSTANT #
############
SAMPLE_RATE = 16000
EXAMPLE_SEC = 5


###################
# UPSTREAM EXPERT #
###################
class UpstreamExpert(UpstreamBase):
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)

        import importlib
        spam_spec = importlib.util.find_spec("whisper")
        if not spam_spec:
            assert ("Please install the whisper package first: pip install git+https://github.com/openai/whisper.git")

        import whisper
        self.model = whisper.load_model(name)

        if len(self.hooks) == 0:
            module_name = "self.model.encoder.blocks"
            for module_id in range(len(eval(module_name))):
                self.add_hook(
                    f"{module_name}[{module_id}]",
                    lambda input, output: input[0].transpose(0, 1),
                )
            self.add_hook("self.model.encoder", lambda input, output: output[0])

    def get_downsample_rates(self, key: str) -> int:
        return 320

    def forward(self, wavs):

        device = wavs[0].device

        batch = []
        for w in wavs:
            audio = whisper.pad_or_trim(w)
            mel = whisper.log_mel_spectrogram(audio).to(device)
            batch.append(self.model.encoder(mel.unsqueeze(0)).squeeze(0))

        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=-100)
        ## compute mask
        mask = (batch != -100).to(device)
        return batch, mask
