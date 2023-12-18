from typing import Tuple, Union

from transformers import BatchFeature, Seq2SeqTrainer

from models.ctc_encoder_plus_autoregressive_decoder import (
    JointCTCAttentionEncoderDecoder,
)


class AdditionalLossTrackerTrainer(Seq2SeqTrainer):
    """Custom trainer to log both losses"""

    def compute_loss(
        self, model: JointCTCAttentionEncoderDecoder, inputs: BatchFeature, return_outputs=False
    ) -> Union[float, Tuple[float, BatchFeature]]:
        """
        MAX: Subclassed to compute training accuracy.

        How the loss is computed by Trainer. By default, all models return the loss in
        the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)

        if hasattr(self.state, "additional_logs"):
            self.state.additional_logs.append([outputs.enc_loss.mean(), outputs.dec_loss.mean()])

        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of
            # ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
