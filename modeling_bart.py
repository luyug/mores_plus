from typing import Optional

from jax import numpy as jnp
from transformers import BartConfig, FlaxBartPreTrainedModel
from transformers.modeling_flax_outputs import FlaxSeq2SeqModelOutput, FlaxSeq2SeqSequenceClassifierOutput
from transformers.models.bart.modeling_flax_bart import FlaxBartModule, FlaxBartForSequenceClassificationModule, \
    FlaxBartClassificationHead


class FlaxBartMoresModule(FlaxBartModule):
    config: BartConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def __call__(
            self,
            input_ids,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            position_ids,
            decoder_position_ids,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
            deterministic: bool = True,
    ):
        # Here we assume the inputs are already chunked
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        hiddens = encoder_outputs[0]
        encoder_mask = attention_mask

        # We reshape inputs for joint attention
        dec_bsz = decoder_input_ids.shape[0]
        hiddens = hiddens.reshape((dec_bsz, -1, hiddens.shape[-1]))
        encoder_mask = encoder_mask.reshape((dec_bsz, -1))

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            position_ids=decoder_position_ids,
            encoder_hidden_states=hiddens,
            encoder_attention_mask=encoder_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return FlaxSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class FlaxBartMoresRankerModule(FlaxBartForSequenceClassificationModule):
    config: BartConfig
    dtype: jnp.dtype = jnp.float32
    num_labels: Optional[int] = None

    def setup(self):
        self.model = FlaxBartMoresModule(config=self.config, dtype=self.dtype)
        self.classification_head = FlaxBartClassificationHead(
            config=self.config,
            inner_dim=self.config.d_model,
            num_classes=1,
            dtype=jnp.float32,
            pooler_dropout=self.config.classifier_dropout,
        )

    def __call__(
            self,
            input_ids,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            position_ids,
            decoder_position_ids,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
            deterministic: bool = True,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            position_ids=position_ids,
            decoder_position_ids=decoder_position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        hidden_states = outputs[0]  # last hidden state

        eos_mask = jnp.where(decoder_input_ids == self.config.eos_token_id, 1, 0).astype(hidden_states.dtype)
        sentence_representation = (eos_mask.reshape(eos_mask.shape + (1,)) * hidden_states).sum(1)
        logits = self.classification_head(sentence_representation, deterministic=deterministic)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return output

        return FlaxSeq2SeqSequenceClassifierOutput(
            logits=logits,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


class FlaxBartMoresRanker(FlaxBartPreTrainedModel):
    module_class = FlaxBartMoresRankerModule
    dtype = jnp.float32