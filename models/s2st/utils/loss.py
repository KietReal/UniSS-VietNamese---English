# pyrefly: ignore [missing-import]
import torch.nn as nn
import torch

class compute_parallel_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, text_prediction: torch.tensor, text_gt: torch.tensor,
        audio_prediction: torch.tensor, audio_gt: torch.tensor):
        """
        Args:
            text_prediction: argmax text header. shape: (batch, seq)
            text_gt: text grouth truth. shape: (batch, seq)

            audio_prediction: argmax audio header. shape: (batch, seq)
            audio_gt: audio grouth truth. shape: (batch, seq)
        """
        len_text_gt = text_gt.shape[1]

        text_prediction = text_prediction[:len_text_gt]
        audio_prediction= audio_prediction[len_text_gt:]

        loss_text = self.loss(text_prediction, text_gt)
        loss_audio= self.loss(audio_prediction, audio_gt)

        total_loss= loss_text+loss_audio
        return total_loss
