from models.decoders.multi_head_gpt2 import GPT2LMMultiHeadModel, GPT2MultiHeadConfig
from models.encoders.e_branchformer import (
    BestRQEBranchformerConfig,
    BestRQEBranchformerForPreTraining,
    Wav2Vec2EBranchformerConfig,
    Wav2Vec2EBranchformerModel,
)

# Wav2vec2 base like model
configuration = Wav2Vec2EBranchformerConfig()
configuration.num_hidden_layers = 6
configuration.hidden_size = 128
configuration.output_hidden_size = 128
configuration.num_attention_heads = 8
configuration.num_feat_extract_layers = 2
configuration.intermediate_size = 1024
configuration.max_source_positions = 1024
configuration.ebranchformer_conv_dropout = 0.1
configuration.csgu_activation = "identity"
configuration.csgu_kernel_size = 31
configuration.csgu_use_linear_after_conv = False
configuration.merge_conv_kernel = 31
configuration.use_macaron_ff = True
configuration.use_fbanks = True
configuration.ctc_zero_infinity = True
configuration.apply_spec_augment = True
configuration.conv_dim = [128, 128, 128, 128, 128, 128, 128]

# pylint: disable=not-callable
configuration.push_to_hub("Lakoc/ebranchformer_6_128h")

model_enc = Wav2Vec2EBranchformerModel(configuration)
print(model_enc.num_parameters())

# Wav2vec2 base like model for 2d input
configuration = Wav2Vec2EBranchformerConfig()
configuration.num_hidden_layers = 6
configuration.hidden_size = 128
configuration.output_hidden_size = 128
configuration.num_attention_heads = 8
configuration.num_feat_extract_layers = 2
configuration.intermediate_size = 1024
configuration.max_source_positions = 1024
configuration.ebranchformer_conv_dropout = 0.1
configuration.csgu_activation = "identity"
configuration.csgu_kernel_size = 31
configuration.csgu_use_linear_after_conv = False
configuration.merge_conv_kernel = 31
configuration.use_macaron_ff = True
configuration.use_fbanks = True
configuration.ctc_zero_infinity = True
configuration.apply_spec_augment = True
configuration.conv_dim = [128, 128]
configuration.conv_stride = [2, 2]
configuration.conv_kernel = [3, 3]

# pylint: disable=not-callable
configuration.push_to_hub("Lakoc/ebranchformer_6_128h_2d")

model_enc = Wav2Vec2EBranchformerModel(configuration)
print(model_enc.num_parameters())


configuration = BestRQEBranchformerConfig()
configuration.num_hidden_layers = 6
configuration.hidden_size = 128
configuration.output_hidden_size = 128
configuration.num_attention_heads = 8
configuration.num_feat_extract_layers = 2
configuration.intermediate_size = 1024
configuration.max_source_positions = 1024
configuration.ebranchformer_conv_dropout = 0.1
configuration.csgu_activation = "identity"
configuration.csgu_kernel_size = 31
configuration.csgu_use_linear_after_conv = False
configuration.merge_conv_kernel = 31
configuration.use_macaron_ff = True
configuration.use_fbanks = True
configuration.ctc_zero_infinity = True
configuration.apply_spec_augment = True
configuration.conv_dim = [128, 128]
configuration.conv_stride = [2, 2]
configuration.conv_kernel = [3, 3]
configuration.best_rq_codebook_dim = 16
configuration.best_rq_codebook_size = 8192
configuration.best_rq_num_books = 8

# pylint: disable=not-callable
configuration.push_to_hub("Lakoc/ebranchformer_6_128h_2d_bestrq")

model_enc = BestRQEBranchformerForPreTraining(configuration)
print(model_enc.num_parameters())

config = GPT2MultiHeadConfig(
    n_head=8,
    n_layer=12,
    vocab_size=50000,
    bos_token_id=0,
    eos_token_id=1,
    n_positions=2048,
    head_locations=[10],
    head_weights=[0.6, 0.4],
    n_embd=704,
    n_inner=2816,
    average_logits=True,
    tie_word_embeddings=False,
)
# pylint: disable=not-callable
config.push_to_hub("Lakoc/gpt2_704h_12l_add_head10_04")
model_dec = GPT2LMMultiHeadModel(config)
print(model_dec.num_parameters())
