# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/whisper/expert.py ]
#   Synopsis     [ the whisper wrapper ]
#   Author       [ OpenAI ]
from collections import defaultdict

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
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.layer_num = len(self.model.encoder.blocks)
        self.model.to(self.device)

    def get_downsample_rates(self, key: str) -> int:
        return 1

    def chunks(self, l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def collate_fn_pad(self, batch, device):
        '''
        Padds batch of variable length
        note: it converts things ToTensor manually here since the ToTensor transform
        assume it takes in images rather than arbitrary tensors.
        '''
        ## get sequence lengths
        lengths = torch.tensor([t.shape[0] for t in batch]).to(device)
        ## padd
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)
        ## compute mask
        mask = (batch != 0).to(device)
        return batch, lengths, mask

    def forward(self, wavs):
        device = wavs[0].device
        import whisper
        with torch.no_grad():
            wav_features = []
            wav_features_map = []
            for w_id, w in enumerate(wavs):
                mel = whisper.log_mel_spectrogram(w).to(device)
                for m_30s in torch.split(mel, 3000, -1):
                    audio = whisper.pad_or_trim(m_30s, 3000)
                    wav_features.append(audio)
                    wav_features_map.append(w_id)

            code_result = defaultdict(list)
            for bd, bm in zip(self.chunks(wav_features, len(wavs)), self.chunks(wav_features_map, len(wavs))):
                batch, lengths, masks = self.collate_fn_pad(bd, self.device)
                masks_ratio = lengths / torch.max(lengths)
                hidden = self.model.encoder(batch)
                mask_len = (hidden.shape[1] * masks_ratio).int()
                for a, h, ml in zip(bm, hidden, mask_len):
                    code_result[a].append(h[:ml, :])
            for k, v in code_result.items():
                code_result[k] = torch.cat(v, 0)

            states = {
                "hidden_states": [torch.stack(list(code_result.values()), 0)]
            }

            return states
