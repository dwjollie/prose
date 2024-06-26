import numpy as np
import torch
import torch.nn as nn


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def eval_batch_data(data, t_eval, output_dim=1, dims=None):
    """
    prepare and batch data together
    """
    length = t_eval.size(0)
    lengths = torch.LongTensor([len(eq) for eq in data])

    data_input = torch.zeros(length, len(data), output_dim + 1, dtype=t_eval.dtype)

    for i, eq in enumerate(data):
        dim = eq.size(-1) if dims is None else dims[i]
        data_input[:, i, 0].copy_(t_eval)
        data_input[:, i, 1 : (dim + 1)].copy_(eq[..., :dim])

    return data_input, lengths


def eval_batch_data_pde(data, t_input, x_grid_size, output_dim=1, dims=None):
    """
    Batch list of data into a Tensor ready for operator data decoder
    Inputs:
        data           (bs, t_num, x_grid_size, dim)
    Outputs:
        data_input     (input_len, bs, 1+output_dim*x_grid_size)
        lengths        (bs, )
    """
    length = t_input.size(0)
    input_len = length
    lengths = torch.LongTensor([input_len for _ in data])

    if dims is None:
        data_input1 = torch.zeros(input_len, len(data), 1 + output_dim * x_grid_size, dtype=t_input.dtype)

        for i, eq in enumerate(data):
            dim = eq.size(-1) if dims is None else dims[i]
            data_input1[:, i, 0].copy_(t_input)
            data_input1[:, i, 1 : (dim * x_grid_size + 1)].copy_(eq[..., :dim].reshape(input_len, dim * x_grid_size))

        return data_input1, lengths
    else:
        bs = len(lengths)
        reshape_dim = x_grid_size * output_dim
        data_input = torch.stack(data).transpose(0, 1).reshape(input_len, bs, reshape_dim)
        t_input = t_input[:, None, None].expand(input_len, bs, 1)  # (input_len, bs, 1)
        data_input = torch.cat((t_input, data_input), dim=-1)

        # assert torch.allclose(data_input1, data_input)

        return data_input, lengths


class ModelWrapper(nn.Module):
    """"""

    def __init__(
        self,
        env=None,
        embedder=None,
        text_encoder=None,
        text_decoder=None,
        data_encoder=None,
        data_decoder=None,
        fusion=None,
        normalizer = None,
        beam_type="search",
        beam_length_penalty=1,
        beam_size=1,
        beam_early_stopping=True,
        max_generated_output_len=200,
        beam_temperature=1.0,
        text_only=False,
        data_only=False,
        no_text=False,
        output_dim=1,
        use_skeleton=False,
        input_len=-1,
        input_step=1,
        output_step =1,
        amp=-1,
        space_dim=0,
        # input_tx_grid=None,
        # output_tx_grid=None,
        x_grid_size=1,
    ):
        super().__init__()

        self.env = env
        self.embedder = embedder
        self.text_encoder = text_encoder
        self.text_decoder = text_decoder
        self.data_encoder = data_encoder
        self.data_decoder = data_decoder
        self.normalizer = normalizer
        self.fusion = fusion
        self.beam_type = beam_type
        self.beam_early_stopping = beam_early_stopping
        self.max_generated_output_len = max_generated_output_len
        self.beam_size = beam_size
        self.beam_length_penalty = beam_length_penalty
        self.beam_temperature = beam_temperature
        try:
            self.device = next(self.text_encoder.parameters()).device
        except:
            self.device = next(self.data_encoder.parameters()).device
        self.text_only = text_only
        self.data_only = data_only
        self.no_text = no_text
        self.query_emb = None
        self.output_dim = output_dim
        self.use_skeleton = use_skeleton
        self.input_len = input_len
        self.input_step = input_step
        self.output_step = output_step
        self.amp = amp
        self.space_dim = space_dim
        # self.input_tx_grid = input_tx_grid
        # self.output_tx_grid = output_tx_grid
        self.x_grid_size = x_grid_size

    @torch.no_grad()
    def forward(
        self,
        data_input,
        text_input,
        dims=None,
        eval_output_start = -1,
        eval_output_end = 1,
        logger=None,
        # norms = None,
    ):
        """
        data_input: bags of data sequences (B, data_T, output_dim) or (B, data_T, x_grid_size, output_dim) data_T = input_len
        text_input: bags of text sequences (B, text_T)
        """

        env = self.env
        embedder, data_encoder, data_decoder, normalizer = (
            self.embedder,
            self.data_encoder,
            self.data_decoder,
            self.normalizer
        )
        fusion, text_encoder, text_decoder = (
            self.fusion,
            self.text_encoder,
            self.text_decoder,
        )
        t_eval = torch.from_numpy(env.generator.t_eval.astype(np.single))  # full time sequence

        B, data_T = len(data_input), max([len(xi) for xi in data_input])
        B2, text_T = len(text_input), max([len(xi) for xi in text_input])
        assert B == B2

        t_input = t_eval[0 : self.input_len : self.input_step]  # input time locations

        text_outputs = []
        input_len = self.input_len

        output_len = int(np.ceil((eval_output_end - eval_output_start) / self.output_step))

        if self.space_dim == 0:
            data_outputs = torch.zeros(B, output_len, self.output_dim, dtype=t_input.dtype, device=self.device)
        else:
            data_outputs = torch.zeros(
                B, output_len, self.x_grid_size, self.output_dim, dtype=t_input.dtype, device=self.device
            )

        if self.query_emb is None and not self.text_only:
            # if self.space_dim == 0:
            t_eval = t_eval.to(self.device)
            query_emb = data_decoder("query_emb", query_times=t_eval[eval_output_start: eval_output_end:self.output_step])  # (data_len, dim)
            self.query_emb = query_emb
            # else:
            # query_emb = data_decoder("query_emb", query_times=self.output_tx_grid)  # (data_len*x_grid_size, dim)
            # self.query_emb = query_emb
        else:
            # query embedding will be the same, reuse it
            query_emb = self.query_emb

        for chunk in chunks(
            np.arange(B),
            B
            # min(
            #     int(50000 / data_T),
            #     int(50000 / text_T),
            #     int(100000 / self.beam_size / self.max_generated_output_len),
            # ),
        ):
            # prepare data input
            chunk_min, chunk_max = min(chunk), max(chunk)
            # if norms == None:
            #     cur_data_input = [data_input[idx] for idx in chunk]
            # else:
            #     cur_data_input = [data_input[idx]/norms[idx] for idx in chunk]

            cur_data_input = [data_input[idx] for idx in chunk]

            if dims is None:
                cur_dims = None
            else:
                cur_dims = [dims[idx] for idx in chunk]
            if self.space_dim == 0:
                cur_data_input, data_len = eval_batch_data(cur_data_input, t_input, self.output_dim)
            else:
                cur_data_input, data_len = eval_batch_data_pde(
                    cur_data_input, t_input, self.x_grid_size, self.output_dim, dims=cur_dims
                )
            cur_data_input, data_len = cur_data_input.to(self.device), data_len.to(self.device)

            with (torch.cuda.amp.autocast(enabled=(self.amp >= 0), dtype=torch.bfloat16)):
                if normalizer != None:
                    cur_data_input,mu,var = normalizer(cur_data_input)
                cur_data_input = embedder(cur_data_input)
                data_encoded = data_encoder("fwd", x=cur_data_input, lengths=data_len, causal=False)

                if self.no_text:
                    fused_features = data_encoded.transpose(0, 1)
                    text_len_encoder = None
                else:
                    cur_text_input = [text_input[idx] for idx in chunk]
                    if self.use_skeleton:
                        (
                            cur_text_input_encoder,
                            text_len_encoder,
                        ) = env.batch_equations_placeholder(env.word_to_idx(cur_text_input, float_input=False))
                    else:
                        cur_text_input, text_len = env.batch_equations(
                            env.word_to_idx(cur_text_input, float_input=False)
                        )
                        cur_text_input_encoder = cur_text_input[1:-1, :]
                        text_len_encoder = text_len - 2

                    cur_text_input_encoder, text_len_encoder = cur_text_input_encoder.to(
                        self.device
                    ), text_len_encoder.to(self.device)

                    # data_len, text_len shape: (bs, )

                    text_encoded = text_encoder("fwd", x=cur_text_input_encoder, lengths=text_len_encoder, causal=False)

                    fused_features = fusion(
                        "fwd",
                        x_data=data_encoded,
                        x_text=text_encoded,
                        lengths_data=data_len,
                        lengths_text=text_len_encoder,
                        causal=False,
                    ).transpose(
                        0, 1
                    )  # (bs, slen, dim)

                ### generate data output

                if not self.text_only:
                    data_pred_generated = data_decoder.generate(
                        src_enc=fused_features,  # (bs, slen, dim)
                        src_len=(data_len, text_len_encoder),
                        query_emb=query_emb,  # (slen, dim)
                        # norms = norms,
                    )
                    if normalizer != None:
                        data_pred_generated = normalizer.reverse(data_pred_generated,mu,var)
                    data_pred_generated = data_pred_generated.float().transpose(
                        0, 1
                    )  # (bs, slen, output_dim)

                    if self.space_dim == 0:
                        data_outputs[chunk_min : chunk_max + 1, :, :] = data_pred_generated
                    else:
                        data_outputs[chunk_min : chunk_max + 1, :, :, :] = data_pred_generated.reshape(
                            data_pred_generated.size(0), output_len, self.x_grid_size, self.output_dim
                        )

                ### greedy text solution

                if not self.data_only and not self.no_text:
                    text_generations, _ = text_decoder.generate(
                        src_enc=fused_features,
                        src_len=(data_len, text_len_encoder),
                        max_len=self.max_generated_output_len,
                    )  # (text_len, bs)

                    bs = text_generations.size(1)

                    text_generations = text_generations.unsqueeze(-1).view(text_generations.shape[0], bs, 1)
                    text_generations = (
                        text_generations.transpose(0, 1).transpose(1, 2).cpu().tolist()
                    )  # (bs, 1, text_len)
                    text_generations = [
                        list(
                            filter(
                                lambda x: x is not None,
                                [
                                    env.idx_to_infix(hyp[1:-1], is_float=False, str_array=False)
                                    for hyp in text_generations[i]
                                ],
                            )
                        )
                        for i in range(bs)
                    ]  # nested list of shape (bs, 1), some inner lists are possibly empty

                    """
                    if self.beam_type == "search":
                        ## search method needs update

                        raise NotImplementedError("Search method not implemented")
                        _, _, search_generations = text_decoder.generate_beam(
                            encoded,
                            x_len,
                            beam_size=self.beam_size,
                            length_penalty=self.beam_length_penalty,
                            max_len=self.max_generated_output_len,
                            early_stopping=self.beam_early_stopping,
                        )
                        search_generations = [
                            sorted(
                                [hyp for hyp in search_generations[i].hyp],
                                key=lambda s: s[0],
                                reverse=True,
                            )
                            for i in range(bs)
                        ]
                        search_generations = [
                            list(
                                filter(
                                    lambda x: x is not None,
                                    [
                                        env.idx_to_infix(
                                            hyp.cpu().tolist()[1:],
                                            is_float=False,
                                            str_array=False,
                                        )
                                        for (_, hyp) in search_generations[i]
                                    ],
                                )
                            )
                            for i in range(bs)
                        ]
                        for i in range(bs):
                            text_generations[i].extend(search_generations[i])

                    elif self.beam_type == "sampling":
                        num_samples = self.beam_size
                        fused_features = (
                            fused_features.unsqueeze(1)
                            .expand((bs, num_samples) + fused_features.shape[1:])
                            .contiguous()
                            .view((bs * num_samples,) + fused_features.shape[1:])
                        )  # (bs*beam_size, slen, dim)
                        data_len = data_len.unsqueeze(1).expand(bs, num_samples).contiguous().view(-1)
                        text_len_encoder = text_len_encoder.unsqueeze(1).expand(bs, num_samples).contiguous().view(-1)

                        sampling_generations, _ = text_decoder.generate(
                            src_enc=fused_features,
                            src_len=(data_len, text_len_encoder),
                            sample_temperature=self.beam_temperature,
                            max_len=self.max_generated_output_len,
                        )  # (text_len, bs)
                        sampling_generations = sampling_generations.unsqueeze(-1).view(
                            sampling_generations.shape[0], bs, num_samples
                        )
                        sampling_generations = (
                            sampling_generations.transpose(0, 1).transpose(1, 2).cpu().tolist()  # (bs, beam_size, text_len)
                        )
                        sampling_generations = [
                            list(
                                filter(
                                    lambda x: x is not None,
                                    [
                                        env.idx_to_infix(hyp[1:-1], is_float=False, str_array=False)
                                        for hyp in sampling_generations[i]
                                    ],
                                )
                            )
                            for i in range(bs)
                        ]  # nested list of shape (bs, beam_size), some inner lists are possibly empty, elements are trees
                        for i in range(bs):
                            text_generations[i].extend(sampling_generations[i])
                    else:
                        raise NotImplementedError
                    """
                    text_outputs.extend(text_generations)
        return data_outputs, text_outputs
