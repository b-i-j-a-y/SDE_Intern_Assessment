from torch.nn import functional as F
import torch


# ============================================================
# Skip Attention
# ============================================================

class SkipAttnProcessor(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()


    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        return hidden_states



# ============================================================
# Memory Efficient Attention Processor
# ============================================================

class AttnProcessor2_0(torch.nn.Module):

    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
        **kwargs
    ):

        super().__init__()

        if not hasattr(
            F,
            "scaled_dot_product_attention"
        ):

            raise ImportError(
                "PyTorch 2.0 required"
            )



    # --------------------------------------------------------
    # Chunked attention
    # --------------------------------------------------------

    def memory_efficient_attention(
        self,
        query,
        key,
        value,
        attention_mask=None
    ):


        q_len = query.shape[-2]
        k_len = key.shape[-2]


        # Normal SDPA for small tensors

        if q_len * k_len < 1000000:

            return F.scaled_dot_product_attention(

                query,

                key,

                value,

                attn_mask=attention_mask,

                dropout_p=0.0,

                is_causal=False
            )



        # Chunk mode for MPS

        chunk_size = 256


        outputs = []


        for start in range(
            0,
            q_len,
            chunk_size
        ):


            q_chunk = query[
                :,
                :,
                start:start + chunk_size,
                :
            ]


            scores = torch.matmul(

                q_chunk,

                key.transpose(
                    -2,
                    -1
                )

            )


            scale = (
                query.shape[-1]
            ) ** -0.5


            scores = scores * scale



            if attention_mask is not None:

                scores = scores + attention_mask



            probs = torch.softmax(

                scores,

                dim=-1

            )



            out = torch.matmul(

                probs,

                value

            )


            outputs.append(out)



            # release memory

            del scores
            del probs



        return torch.cat(

            outputs,

            dim=-2

        )



    # --------------------------------------------------------
    # Main attention call
    # --------------------------------------------------------

    def __call__(

        self,

        attn,

        hidden_states,

        encoder_hidden_states=None,

        attention_mask=None,

        temb=None,

        *args,

        **kwargs

    ):


        residual = hidden_states



        if attn.spatial_norm is not None:

            hidden_states = attn.spatial_norm(
                hidden_states,
                temb
            )



        input_ndim = hidden_states.ndim



        if input_ndim == 4:

            batch_size, channel, height, width = hidden_states.shape


            hidden_states = hidden_states.reshape(

                batch_size,

                channel,

                height * width

            ).transpose(

                1,

                2

            )



        batch_size = hidden_states.shape[0]



        sequence_length = hidden_states.shape[1]



        if attention_mask is not None:


            attention_mask = attn.prepare_attention_mask(

                attention_mask,

                sequence_length,

                batch_size

            )


            attention_mask = attention_mask.view(

                batch_size,

                attn.heads,

                -1,

                attention_mask.shape[-1]

            )



        if attn.group_norm is not None:

            hidden_states = attn.group_norm(

                hidden_states.transpose(
                    1,
                    2

                )

            ).transpose(

                1,

                2

            )



        query = attn.to_q(
            hidden_states
        )



        if encoder_hidden_states is None:

            encoder_hidden_states = hidden_states


        elif attn.norm_cross:

            encoder_hidden_states = attn.norm_encoder_hidden_states(

                encoder_hidden_states

            )



        key = attn.to_k(

            encoder_hidden_states

        )


        value = attn.to_v(

            encoder_hidden_states

        )



        inner_dim = key.shape[-1]


        head_dim = inner_dim // attn.heads



        query = query.view(

            batch_size,

            -1,

            attn.heads,

            head_dim

        ).transpose(

            1,

            2

        )



        key = key.view(

            batch_size,

            -1,

            attn.heads,

            head_dim

        ).transpose(

            1,

            2

        )



        value = value.view(

            batch_size,

            -1,

            attn.heads,

            head_dim

        ).transpose(

            1,

            2

        )



        # ====================================================
        # Memory safe attention
        # ====================================================

        hidden_states = self.memory_efficient_attention(

            query,

            key,

            value,

            attention_mask

        )



        hidden_states = hidden_states.transpose(

            1,

            2

        ).reshape(

            batch_size,

            -1,

            attn.heads * head_dim

        )



        hidden_states = hidden_states.to(
            query.dtype
        )



        # output projection

        hidden_states = attn.to_out[0](

            hidden_states

        )


        hidden_states = attn.to_out[1](

            hidden_states

        )



        if input_ndim == 4:

            hidden_states = hidden_states.transpose(

                -1,

                -2

            ).reshape(

                batch_size,

                channel,

                height,

                width

            )



        if attn.residual_connection:

            hidden_states = hidden_states + residual



        hidden_states = hidden_states / attn.rescale_output_factor



        return hidden_states