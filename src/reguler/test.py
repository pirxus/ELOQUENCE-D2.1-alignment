from reguler.configuration_reguler import JointCTCAttentionEncoderDecoderConfig
from reguler.modeling_reguler import JointCTCAttentionEncoderDecoder

JointCTCAttentionEncoderDecoderConfig.register_for_auto_class()
JointCTCAttentionEncoderDecoder.register_for_auto_class("AutoModel")
JointCTCAttentionEncoderDecoder.register_for_auto_class("AutoModelForSpeechSeq2Seq")
model = JointCTCAttentionEncoderDecoder.from_pretrained(
    "/Users/alexanderpolok/PycharmProjects/huggingface_asr/checkpoint-378950"
)
model.push_to_hub("Lakoc/reguler_medium")
