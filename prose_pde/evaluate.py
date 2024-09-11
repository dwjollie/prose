from logging import getLogger
import os
import torch
import torch.distributed as dist
import numpy as np
from copy import deepcopy
from plot import plot,plot_sample_output,plot_one,plot_sample_output_noerror

import symbolicregression
from symbolicregression.model.model_wrapper import ModelWrapper
import pandas as pd
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from tabulate import tabulate
import sympy as sy
import scipy
from jax import numpy as jnp

from tqdm import tqdm
import h5py
from sklearn.metrics import r2_score

from symbolicregression.envs.data_gen_NLE import  diff_react_1D_f, burgers_f
from symbolicregression.envs.fplanck import fokker_planck, boundary, gaussian_pdf, delta_function, uniform_pdf

# np.seterr(all="raise")
np.seterr(divide="raise", under="ignore", over="raise", invalid="raise")

logger = getLogger()


def ODE_solver(tree, IC, output_grid, logger=None, type=None):
    # solve ODE where RHS is given by tree

    y0 = IC.numpy(force=True)  # (dim,)
    t_span = [0.0, 6.0]

    def f(t, y):
        shape = y.shape  # (dim,)
        return tree.val(y.reshape(1, -1)).reshape(shape)

    try:
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method="BDF",
            t_eval=output_grid,
            rtol=1e-4,
            atol=1e-6,
        )

        if (sol.status) == 0:
            return torch.from_numpy(sol.y.transpose().astype(np.single))  # (t_num, dim)
        else:
            logger.info(f"{type} solver error: sol status {sol.status}")
    except Exception as e:
        logger.info(f"{type} error is {e}")

    return None


def compute_losses(output, target, output_len, eps, dx):
    """
    output: (output_len, dim) or (output_len, x_grid_size, dim)
    target: (output_len, dim) or (output_len, x_grid_size, dim)

    RETURN: MSE, relative SE, first half relative SE, second half relative SE, mean_relative SE
    """
    half_len = output_len // 2
    target_first_half_sum = torch.sum(target[:half_len] ** 2)
    target_second_half_sum = torch.sum(target[half_len:] ** 2)
    first_half_diff = torch.sum((output[:half_len] - target[:half_len]) ** 2)
    second_half_diff = torch.sum((output[half_len:] - target[half_len:]) ** 2)
    # total_sum = torch.sum(target ** 2)

    # abs_loss = torch.sum((output - target) ** 2)
    # rel_loss = torch.sqrt(abs_loss / (eps + total_sum))
    abs_loss = first_half_diff + second_half_diff
    rel_loss = torch.sqrt(abs_loss / ( target_first_half_sum + target_second_half_sum))
    mean_rel_loss = torch.sqrt(abs_loss / ( torch.sum((target - target.mean()) ** 2)))

    l1_loss = torch.sum(torch.abs(output - target)) / ( torch.sum(torch.abs(target)))

    rel_loss_first_half = torch.sqrt(first_half_diff / (target_first_half_sum))
    rel_loss_second_half = torch.sqrt(second_half_diff / (target_second_half_sum))

    x_grid_size = 1 if output.dim() == 2 else output.size(-2)

    return (
        abs_loss / output_len / output.size(-1) / x_grid_size,
        rel_loss,
        rel_loss_first_half,
        rel_loss_second_half,
        mean_rel_loss,
        l1_loss,
    )


class Evaluator(object):
    ENV = None

    def __init__(self, trainer):
        """
        Initialize evaluator.
        """
        self.trainer = trainer
        self.modules = trainer.modules
        self.params = trainer.params
        self.env = trainer.env
        Evaluator.ENV = trainer.env
        self.eval_num_input_points = 50
        self.dataloader = None
        self.output_dim = self.params.max_output_dimension
        self.space_dim = self.params.max_input_dimension

        self.t_eval = torch.from_numpy(self.env.generator.t_eval.astype(np.single))
        self.x_grid = torch.from_numpy(self.env.generator.x_grid.astype(np.single))
        if self.space_dim > 0:
            # self.x_grid = torch.from_numpy(self.env.generator.x_grid.astype(np.single))
            self.x_grid_size = self.env.generator.x_grid_size
            # self.input_tx_grid = self.env.get_tx_grid(
            #     self.t_eval[0 : self.params.input_len], self.x_grid, self.space_dim, self.x_grid_size
            # )  # (t_num*x_num, 1+space_dim)
            # self.output_tx_grid = self.env.get_tx_grid(
            #     self.t_eval[self.params.input_len :], self.x_grid, self.space_dim, self.x_grid_size
            # ).cuda()
        else:
            # self.x_grid = None
            self.x_grid_size = 1
            # self.input_tx_grid = None
            # self.output_tx_grid = None

        # if self.params.ode_gen:
        #     self.types = self.env.generator.ode_generator.types
        # else:
        #     self.types = self.env.generator.pde_generator.types
        # if str(self.params.eval_types) != "":
        #     if str(self.params.eval_types).startswith("pde"):
        #         self.types = self.env.generator.pde_generator.eval_types
        #     else:
        #         self.types = self.env.generator.ode_generator.eval_types
        # else:
        self.all_types = [
            "heat",
            "porous_medium",
            "advection",
            "kdv",
            "fplanck",
            "diff_logisreact_1D",
            "diff_linearreact_1D",
            "diff_bistablereact_1D",
            "diff_squarelogisticreact_1D",
            "burgers",
            "conservation_linearflux",
            "conservation_sinflux",
            "conservation_cosflux",
            "conservation_cubicflux",
            "inviscid_burgers",
            "inviscid_conservation_sinflux",
            "inviscid_conservation_cosflux",
            "inviscid_conservation_cubicflux",
            "cahnhilliard_1D",
            "wave",
            "Klein_Gordon",
            "Sine_Gordon",
        ]
        self.alltypes_to_idx = {s: i for i, s in enumerate(self.all_types)}
        if str(self.params.types).startswith("pde"):
            self.types = self.env.generator.pde_generator.types
        else:
            self.types = self.env.generator.ode_generator.types
        self.types_to_idx = {s: i for i, s in enumerate(self.types)}

        if self.space_dim == 0:
            self.input_points = np.random.uniform(-5, 5, size=(self.eval_num_input_points, self.output_dim, 1))
        else:
            t_grid = np.linspace(0.0, self.params.t_range, self.params.t_num)
            x_grid = np.linspace(0.0, self.params.x_range, self.params.x_num)
            coeff = np.random.uniform(-5, 5, size=(8, self.params.max_input_dimension))
            # Create mesh grids
            T, X = np.meshgrid(t_grid, x_grid, indexing="ij")

            # Calculate the terms using vectorized operations
            input_points = np.zeros((self.params.max_input_dimension, self.params.t_num, self.params.x_num, 8))
            for i in range(self.params.max_input_dimension):
                input_points[i, :, :, 0] = (coeff[0, i] + coeff[1, i] * T + coeff[2, i] * T**2) * (
                    coeff[3, i] + coeff[4, i] * X + coeff[5, i] * X**2 + coeff[6, i] * X**3 + coeff[7, i] * X**4
                )
                input_points[i, :, :, 1] = (coeff[1, i] + 2 * coeff[2, i] * T) * (
                    coeff[3, i] + coeff[4, i] * X + coeff[5, i] * X**2 + coeff[6, i] * X**3 + coeff[7, i] * X**4
                )
                # ut
                input_points[i, :, :, 2] = (2 * coeff[2, i]) * (
                    coeff[3, i] + coeff[4, i] * X + coeff[5, i] * X**2 + coeff[6, i] * X**3 + coeff[7, i] * X**4
                )
                # utt
                input_points[i, :, :, 3] = (coeff[0, i] + coeff[1, i] * T + coeff[2, i] * T**2) * (
                    coeff[4, i] + 2 * coeff[5, i] * X + 3 * coeff[6, i] * X**2 + 4 * coeff[7, i] * X**3
                )  # ux
                input_points[i, :, :, 4] = (coeff[0, i] + coeff[1, i] * T + coeff[2, i] * T**2) * (
                    2 * coeff[5, i] + 6 * coeff[6, i] * X + 12 * coeff[7, i] * X**2
                )  # uxx
                input_points[i, :, :, 5] = (coeff[0, i] + coeff[1, i] * T + coeff[2, i] * T**2) * (
                    6 * coeff[6, i] + 24 * coeff[7, i] * X
                )  # uxxx
                input_points[i, :, :, 6] = (coeff[0, i] + coeff[1, i] * T + coeff[2, i] * T**2) * (
                    24 * coeff[7, i]
                )  # uxxxx
                input_points[i, :, :, 7] = X  # x
            # input_points = np.zeros((6, params.t_num,params.x_num ))
            # for i in range( params.t_num):
            #     for j in range(params.x_num):
            #         input_points[0,i,j] = (coeff[0]+coeff[1]*t_grid[i]+coeff[2]*x_grid[j]+coeff[3]*t_grid[i]*x_grid[j]+coeff[4]*t_grid[i]**2+coeff[5]*x_grid[j]**2)
            #         input_points[1,i,j] = (coeff[1]+coeff[3]*x_grid[j]+coeff[4]*t_grid[i]*2)
            #         input_points[2,i,j] = (coeff[2]+coeff[3]*t_grid[i]+coeff[5]*x_grid[j]*2)
            #         input_points[3,i,j] = (coeff[3])
            #         input_points[4,i,j] = (coeff[4])
            #         input_points[5,i,j] = (coeff[5])
            # input_points = np.random.uniform(-5, 5, size=(self.eval_num_input_points, self.space_dim))
            self.input_points = input_points

    def set_env_copies(self, data_types):
        for data_type in data_types:
            setattr(self, "{}_env".format(data_type), deepcopy(self.env))

    def evaluate_in_domain(
        self,
        data_type,
        task,
        save=False,
        save_file=None,
    ):
        """
        Encoding / decoding step on the given evaluation dataset
        """

        params = self.params
        logger.info("====== STARTING EVALUATION (multi-gpu: {}) =======".format(params.multi_gpu))

        if "embedder" in self.modules:
            embedder = self.modules["embedder"].module if params.multi_gpu else self.modules["embedder"]
            embedder.eval()
        else:
            embedder = None

        if "text_encoder" in self.modules:
            text_encoder = self.modules["text_encoder"].module if params.multi_gpu else self.modules["text_encoder"]
            text_encoder.eval()
        else:
            text_encoder = None

        if "text_decoder" in self.modules:
            text_decoder = self.modules["text_decoder"].module if params.multi_gpu else self.modules["text_decoder"]
            text_decoder.eval()
        else:
            text_decoder = None

        if "data_encoder" in self.modules:
            data_encoder = self.modules["data_encoder"].module if params.multi_gpu else self.modules["data_encoder"]
            data_encoder.eval()
        else:
            data_encoder = None

        if "data_decoder" in self.modules:
            data_decoder = self.modules["data_decoder"].module if params.multi_gpu else self.modules["data_decoder"]
            data_decoder.eval()
        else:
            data_decoder = None

        if "fusion" in self.modules:
            fusion = self.modules["fusion"].module if params.multi_gpu else self.modules["fusion"]
            fusion.eval()
        else:
            fusion = None

        if "normalizer" in self.modules:
            normalizer = self.modules["normalizer"].module if params.multi_gpu else self.modules["normalizer"]
            normalizer.eval()
        else:
            normalizer = None

        env = getattr(self, "{}_env".format(data_type))

        if self.params.eval_noise_gamma > 0:
            seed = [self.params.global_rank, self.params.test_env_seed]
            noise_rng = np.random.RandomState(seed)
            gamma = self.params.eval_noise_gamma

        eval_size_per_gpu = params.eval_size // params.n_gpu_per_node
        if self.dataloader is None:
            self.dataloader = env.create_test_iterator(
                data_type,
                task,
                data_path=self.trainer.data_path,
                batch_size=params.batch_size_eval,
                params=params,
                size=eval_size_per_gpu,
                test_env_seed=self.params.test_env_seed,
            )
        iterator = self.dataloader
        colors = ["blue", "orange", "green", "purple", "olive", "red", "magenta", "black"]

        mw = ModelWrapper(
            env=env,
            embedder=embedder,
            text_encoder=text_encoder,
            text_decoder=text_decoder,
            data_encoder=data_encoder,
            data_decoder=data_decoder,
            normalizer=normalizer,
            fusion=fusion,
            beam_length_penalty=params.beam_length_penalty,
            beam_size=params.beam_size,
            max_generated_output_len=params.max_generated_output_len,
            beam_early_stopping=params.beam_early_stopping,
            beam_temperature=params.beam_temperature,
            beam_type=params.beam_type,
            text_only=params.text_only,
            data_only=params.data_only,
            no_text=params.no_text,
            output_dim=self.output_dim,
            use_skeleton=params.use_skeleton,
            input_len=params.input_len,
            input_step=params.input_step,
            output_step=params.eval_output_step,
            amp=params.eval_amp,
            space_dim=self.space_dim,
            # input_tx_grid=self.input_tx_grid,
            # output_tx_grid=self.output_tx_grid,
            x_grid_size=self.x_grid_size,
        )

        total_loss = torch.zeros(len(self.types), dtype=torch.float32)
        best_total_loss = torch.zeros(len(self.types), dtype=torch.float32)
        min_data_loss = torch.ones(len(self.types), dtype=torch.float32)
        min_data_loss_index = torch.zeros(len(self.types), dtype=torch.float32)
        max_data_loss = torch.zeros(len(self.types), dtype=torch.float32)
        max_data_loss_index = torch.zeros(len(self.types), dtype=torch.float32)
        total_abs_loss = torch.zeros(len(self.types), dtype=torch.float32)
        total_count = torch.zeros(len(self.types), dtype=torch.long)
        total_mean_loss = torch.zeros(len(self.types), dtype=torch.float32)
        total_l1_loss = torch.zeros(len(self.types), dtype=torch.float32)
        best_l1_loss = torch.zeros(len(self.types), dtype=torch.float32)
        est_parameter_loss = torch.zeros(len(self.types), dtype=torch.float32)
        init_parameter_loss = torch.zeros(len(self.types), dtype=torch.float32)
        output_list = []
        target_list = []
        type_list = []
        if save:
            if save_file is None:
                save_file = (
                    self.params.eval_dump_path if self.params.eval_dump_path is not None else self.params.dump_path
                )
            if not os.path.exists(save_file):
                os.makedirs(save_file)

            # save_target = os.path.join(save_file, "target.h5")
            save_file = os.path.join(save_file, "eval_in_domain.h5")

        if not self.params.use_wandb:
            pbar = tqdm(total=eval_size_per_gpu)

        input_len = self.params.input_len
        t_eval = self.trainer.t_eval
        output_len = len(t_eval) - input_len
        input_points = self.input_points
        eval_output_start = self.params.eval_output_start
        eval_output_end = len(t_eval)
        text_loss = 0.0
        num_batches =0
        text_valid = 0
        text_total = 0
        eps = 1e-6

        data_loss = []
        #r2_losses = []
        l1_loss = []
        abs_data_loss = []
        data_loss_first_half = []
        data_loss_second_half = []
        data_loss_mean = []
        data_loss_type = [[] for _ in range(len(self.types))]
        abs_data_loss_type = [[] for _ in range(len(self.types))]
        l1_loss_type = [[] for _ in range(len(self.types))]
        est_param_loss_type = [[] for _ in range(len(self.types))]
        init_param_loss_type = [[] for _ in range(len(self.types))]

        if params.text_ode_solve:
            # use text output + ODE solver
            data_loss_valid = 0.0
            text_data_loss = 0.0
            text_valid_output = 0
            output_grid = env.generator.t_eval[eval_output_start : eval_output_end : params.eval_output_step]
        elif params.use_text_refinement:
            #Use text output + Particle Filter
            data_loss_valid = 0.0
            text_data_loss = 0.0
            text_valid_output = 0
            output_grid = env.generator.t_eval[eval_output_start : eval_output_end : params.eval_output_step]

        for samples, _ in iterator:
            data_seqs = samples["data"]  # (bs, data_len, output_dim) or (bs, data_len, x_grid_size, output_dim)

            if self.params.eval_noise_gamma > 0:
                # add noise on data
                if self.params.noise_type == "multiplicative":
                    for i, seq in enumerate(data_seqs):
                        data_seqs[i] = seq + (
                            gamma * torch.abs(seq) * torch.from_numpy(noise_rng.randn(*seq.size()).astype(np.single))
                        )
                else:  # additive
                    for i, seq in enumerate(data_seqs):
                        cur_noise = torch.from_numpy(noise_rng.randn(*seq.size()).astype(np.single))
                        sigma = gamma * torch.linalg.vector_norm(seq) / (torch.linalg.vector_norm(cur_noise) + eps)
                        data_seqs[i] = seq + sigma * cur_noise

            if params.use_skeleton:
                text_seqs = samples["tree_skeleton"]  # (bs, text_len)
            else:
                text_seqs = samples["tree_encoded"]  # (bs, text_len)

            dims = samples["dim"] if "dim" in samples else None

            trees = samples["tree"]  # (bs, )
            bs = len(data_seqs)
            text_total += bs
            num_batches +=1

            # if self.space_dim == 0:
            #     data_input = [seq[: input_len : params.input_step, :] for seq in data_seqs]
            # else:
            #     data_input = [seq[: input_len : params.input_step, :, :] for seq in data_seqs]
            data_input = [seq[: input_len : params.input_step] for seq in data_seqs]
            # norms = [torch.linalg.norm(seq) for seq in data_seqs]

            data_outputs, text_outputs = mw(
                data_input=data_input,  # (bs, input_len, output_dim) or (bs, input_len, x_grid_size, output_dim)
                text_input=text_seqs,  # (bs, text_len)
                dims=dims,
                logger=logger,
                eval_output_start=eval_output_start,
                eval_output_end=eval_output_end,
            )
            # data_outputs: torch tensor of shape (bs, data_len, output_dim) or (bs, data_len, x_grid_size, output_dim)
            # text_outputs: nested list of shape (bs, tree_nums), some inner lists are possibly empty, elements are trees

            # computing eval losses

            data_outputs = data_outputs.to(data_seqs[0].device)
            targets = [seq[eval_output_start : eval_output_end : params.eval_output_step] for seq in samples["data"]]
            if params.plot_comparison:
                plot(data_outputs, targets, samples["type"], params, plot_type=self.types)

                plot(
                    [torch.abs(data_outputs[i] - targets[i]) for i in range(len(data_seqs))],
                    None,
                    samples["type"],
                    params,
                    notes="_diff",
                    plot_type=self.types,
                )
                plot(data_seqs, None, samples["type"], params, notes="all", plot_type=self.types, initial=True)
            cur_output = []
            cur_target = []
            for i in range(bs):
                rel_loss = 0.0
                if not params.text_only:
                    cur_type = samples["type"][i]
                    # data loss
                    if "dim" in samples:
                        dim = samples["dim"][i]
                    else:
                        dim = data_seqs[i].size(-1)
                    if self.space_dim == 0:
                        (abs_loss, rel_loss, rel_loss_first_half, rel_loss_second_half, rel_loss_mean, rel_l1_loss) = (
                            compute_losses(
                                data_outputs[i, :, :dim],
                                data_seqs[i][eval_output_start : eval_output_end : params.eval_output_step, :dim],
                                (params.t_num - input_len) // params.eval_output_step,
                                eps,
                                params.x_range / params.x_num,
                            )
                        )
                        # this_output = np.insert(data_outputs[i, :, :dim].flatten(), 0, self.alltypes_to_idx[cur_type])
                        type_list.append(self.alltypes_to_idx[cur_type])
                        this_output = data_outputs[i, :, :dim]
                        output_list.append(this_output.flatten())
                        cur_output.append(this_output.flatten())
                        this_target = data_seqs[i][
                            eval_output_start : eval_output_end : params.eval_output_step, :dim
                        ]
                        target_list.append(this_target.flatten())
                        cur_target.append(this_target.flatten())
                    else:
                        (abs_loss, rel_loss, rel_loss_first_half, rel_loss_second_half, rel_loss_mean, rel_l1_loss) = (
                            compute_losses(
                                data_outputs[i, :, :, :dim],
                                data_seqs[i][eval_output_start : eval_output_end : params.eval_output_step, ..., :dim],
                                (params.t_num - input_len) // params.eval_output_step,
                                eps,
                                params.x_range / params.x_num,
                            )
                        )
                        type_list.append(self.alltypes_to_idx[cur_type])
                        this_output = data_outputs[i, :, :,:dim]
                        output_list.append(this_output.flatten())
                        cur_output.append(this_output.flatten())
                        this_target = data_seqs[i][
                            eval_output_start : eval_output_end : params.eval_output_step, ..., :dim
                        ]
                        target_list.append(this_target.flatten())
                        cur_target.append(this_target.flatten())

                    abs_data_loss.append(abs_loss)
                    data_loss.append(rel_loss)
                    data_loss_first_half.append(rel_loss_first_half)
                    data_loss_second_half.append(rel_loss_second_half)
                    data_loss_mean.append(rel_loss_mean)
                    l1_loss.append(rel_l1_loss)

                    abs_data_loss_type[self.types_to_idx[cur_type]].append(abs_loss)
                    data_loss_type[self.types_to_idx[cur_type]].append(rel_loss)
                    total_l1_loss[self.types_to_idx[cur_type]] += rel_l1_loss
                    l1_loss_type[self.types_to_idx[cur_type]].append(rel_l1_loss)

                    total_loss[self.types_to_idx[cur_type]] += rel_loss
                    total_abs_loss[self.types_to_idx[cur_type]] += abs_loss
                    total_mean_loss[self.types_to_idx[cur_type]] += rel_loss_mean
                    total_count[self.types_to_idx[cur_type]] += 1

                    if rel_loss < min_data_loss[self.types_to_idx[cur_type]]:
                        min_data_loss_index[self.types_to_idx[cur_type]] = i
                        min_data_loss[self.types_to_idx[cur_type]] = rel_loss
                    if rel_loss > max_data_loss[self.types_to_idx[cur_type]]:
                        max_data_loss_index[self.types_to_idx[cur_type]] = i
                        max_data_loss[self.types_to_idx[cur_type]] = rel_loss
                        if params.plot_worst:
                            for jj in range(self.space_dim):
                                title = "type_{}_dim{}_worst".format(cur_type, jj)
                                plot_sample_output_noerror(
                                    data_outputs[int(i)][:, :, jj],
                                    targets[int(i)][:, :, jj],
                                    self.x_grid[:, jj],
                                    self.t_eval[
                                    eval_output_start: eval_output_end: params.eval_output_step],
                                    params, title=title)
                                plot_sample_output(data_outputs[int(i)][:, :, jj],
                                                   targets[int(i)][:, :, jj],
                                                   self.x_grid[:, jj],
                                                   self.t_eval[
                                                   eval_output_start: eval_output_end: params.eval_output_step],
                                                   params, title=title)

                    # if params.print_outputs:
                    #    plot_dict =  plot([data_outputs[i]], [target[i]], cur_type, params, plot_type=self.types,plot_dict=plot_dict)
                    #     data = data_seqs[i]
                    #     output = data_outputs[i]
                    #     fig = plt.figure()
                    #     for j in range(dim):
                    #         c = colors[j]
                    #         plt.plot(t_eval, data[:, j], "--", linewidth=1.4, alpha=0.8, color=c, label=f"target {j}")
                    #         plt.plot(t_eval[input_len:], output[:, j], "-", linewidth=2, color=c, label=f"output {j}")
                    #     plt.legend(loc="best")
                    #     plt.title("{} | {} | {:.6f}".format(i, cur_type, rel_loss))
                    #     plt.savefig("figures/eval_{}.png".format(i))
                if not params.text_only and params.print_outputs and i%50 ==0:
                    for jj in range(self.space_dim):
                        title = "type_{}_dim{}_{}".format(cur_type, jj, i)
                        plot_sample_output(this_output[:, :, jj], this_target[:, :, jj],
                                           self.x_grid[:, jj],
                                           self.t_eval[
                                           eval_output_start: eval_output_end: params.eval_output_step],
                                           params, title=title)
                        title = "type_{}_dim{}_{}_shortinput".format(cur_type, jj, i)
                        plot_one(data_seqs[i][0: 4, ..., jj], params, title=title, input_size=1)
                        title = "type_{}_dim{}_{}_sampleoutput".format(cur_type, jj, i)
                        plot_one(data_seqs[i][4:: params.eval_output_step, ..., jj], params,
                                 title=title)
                        logger.info(
                            "[{}] Type: {} | Rel loss: {:.4f} ".format(
                                i, cur_type, rel_loss,
                            )
                        )
                        logger.info("Target:    {}".format(trees[i]))
                if not params.data_only and not params.no_text:
                    # text loss
                    tree_list = text_outputs[i]
                    label_outputs = None
                    valid_loss = []
                    valid_loss_less_than_c = []

                    for tree in tree_list:
                        t_grid = np.linspace(0.0, self.params.t_range, self.params.t_num)
                        x_grid = np.linspace(0.0, self.params.x_range, self.params.x_num)
                        coeff = np.random.uniform(-5, 5, size=(8, self.params.max_input_dimension))
                        # Create mesh grids
                        T, X = np.meshgrid(t_grid, x_grid, indexing="ij")

                        x, t = sy.symbols('x t')
                        u = sy.Function('u_0')(x,t)
                        tens_poly = (coeff[0, 0] + coeff[1, 0] * t + coeff[2, 0] * t**2) * (
                                coeff[3, 0] + coeff[4, 0] * x + coeff[5, 0] * x**2 + coeff[6, 0] * x**3 + coeff[7, 0] * x**4)
                        if self.params.use_sympy:
                            try:
                                equation = sy.sympify(tree[0])  
                                expr = equation.subs(u,tens_poly)
                                eval_expr = sy.lambdify([x,t],expr.doit(),"numpy")
                                generated_outputs = eval_expr(X,T)
                            except:
                                continue
                        else:
                            try:
                                generated_outputs = tree.val(input_points, self.space_dim)
                            except:
                                continue

                        if label_outputs is None:
                            # if self.space_dim == 0:
                            #     label_outputs = tree[i].val(input_points)
                            # else:
                            #     label_outputs = tree[i].val(t_grid, x_grid,coeff)
                            if self.params.use_sympy:
                                tree_expr = trees[i]
                                
                                #start of removing systematic parts of the "trees" string of the form "[eqn]" to get eqn
                                original_expr = ""
                                amt_terms = len(tree_expr)
                                terms_to_remove = [0, amt_terms, amt_terms-1]
                                for k in range(len(tree_expr)):
                                    if not k in terms_to_remove:
                                        original_expr = original_expr + tree_expr[k]
                                #original_expr = eqn now.
                                
                                original_expr = sy.sympify(original_expr)     
                                original_expr = original_expr.subs(u,tens_poly)
                                eval_expr = sy.lambdify([x,t],original_expr.doit(),"numpy")
                                label_outputs = eval_expr(X,T)
                            else:
                                label_outputs = trees[i].val(input_points, self.space_dim)
                            assert np.isfinite(label_outputs).all()
                        try:
                            if self.params.use_sympy:
                                if num_batches > 1:
                                    break
                                if not (np.isnan(generated_outputs).all() and np.size(generated_outputs) == 1):
                                    valid_loss.append(
                                    np.sqrt(
                                        np.sum((generated_outputs - label_outputs) ** 2)
                                        / (np.sum(label_outputs**2) + eps)
                                    )
                                )
                            elif np.isfinite(generated_outputs).all():
                                valid_loss.append(
                                    np.sqrt(
                                        np.sum((generated_outputs - label_outputs) ** 2)
                                        / (np.sum(label_outputs**2) + eps)
                                    )
                                )
                        except:
                            continue

                    # logger.info("[{}] Input:     {}".format(i, text_seqs[i]))
                    # logger.info("[{}] Target:    {}".format(i, trees[i]))

                    if len(valid_loss) > 0:
                        # generated tree is valid, compute other metrics
                        min_loss = min(valid_loss)
                        text_valid += 1
                        text_loss += min_loss

                        if params.print_outputs:
                            if i % 50 == 0:
                                logger.info(
                                    "[{}] Type: {} | Rel loss: {:.4f} | Mean Rel loss:{:4f} |Text loss: {:.4f}".format(
                                        i, cur_type, rel_loss,rel_loss_mean, min_loss
                                    )
                                )
                                # logger.info("Input:     {}".format(text_seqs[i]))
                                logger.info("Target:    {}".format(trees[i]))
                                try:
                                    logger.info("Generated: {}\n".format(tree_list[0]))
                                except:
                                    # logger.info("Generated: {}\n".format(tree_list[1]))
                                    pass

                        if params.text_ode_solve:
                            # use tree + ODE solver
                            IC = samples["data"][i][0, :dim]

                            text_data_output = ODE_solver(
                                tree_list[0], IC, output_grid, logger=logger, type=samples["type"][i]
                            )

                            if text_data_output != None:
                                data_loss_valid += rel_loss
                                (
                                    _,
                                    text_rel_loss,
                                    _,
                                    _,
                                    _,
                                    _,
                                ) = compute_losses(
                                    text_data_output,
                                    data_seqs[i][eval_output_start : eval_output_end : params.eval_output_step, :dim],
                                    (params.t_num - input_len) // params.eval_output_step,
                                    eps,
                                    params.x_range / params.x_num,
                                )
                                text_data_loss += text_rel_loss
                                text_valid_output += 1
                        elif params.use_text_refinement:
                            og_data = np.array(data_seqs[i][: output_len : params.eval_output_step, ..., :dim])
                            text_data_output, estimated_coeff_ls, init_coeff_ls = self.refinement(samples["type"][i],
                                                             tree_list[0][0], 
                                                             og_data
                                                             )
                            if text_data_output is not None:
                                text_data_output = text_data_output[:,:,np.newaxis]
                                if "dim" in samples:
                                    dim = samples["dim"][i]
                                else:
                                    dim = data_seqs[i].size(-1)
                                # print(np.shape(text_data_output))
                                text_data_output = torch.from_numpy(text_data_output.astype(np.single))  # (t_num, dim)
                                data_loss_valid += rel_loss
                                (
                                    _,
                                    text_rel_loss,
                                    _,
                                    _,
                                    _,
                                    _,
                                ) = compute_losses(
                                    text_data_output[:,:,:dim],
                                    data_seqs[i][eval_output_start : eval_output_end : params.eval_output_step, ..., :dim],
                                    (params.t_num - input_len) // params.eval_output_step,
                                    eps,
                                    params.x_range / params.x_num,
                                )
                                #computing a different metric between the filtered data and the original
                                try:
                                    error = np.sqrt(np.sum((np.array(text_data_output) - np.array(og_data)) ** 2)/ (np.sum(np.array(og_data) ** 2) + eps))
                                    assert error<2** 200
                                    valid_loss_less_than_c.append(error)
                                except:
                                    pass
                                #compute the L2 error between coefficients
                                tree_expr = trees[i]
                                
                                #start of removing systematic parts of the "trees" string of the form "[eqn]" to get eqn
                                original_expr = ""
                                amt_terms = len(tree_expr)
                                terms_to_remove = [0, amt_terms, amt_terms-1]
                                for k in range(len(tree_expr)):
                                    if not k in terms_to_remove:
                                        original_expr = original_expr + tree_expr[k]

                                coeff_ls = self.get_coeff(samples["type"][i],original_expr)
                                if coeff_ls is not None:
                                    est_coeff_er = 0
                                    init_coeff_er = 0
                                    for i, coefficient in enumerate(coeff_ls):
                                        est_coeff_er += np.sqrt((np.float32(estimated_coeff_ls[i]) - np.float32(coefficient)) ** 2 / (np.float32(coefficient) ** 2 + eps))
                                        init_coeff_er += np.sqrt((np.float32(init_coeff_ls[i]) - np.float32(coefficient)) ** 2 / (np.float32(coefficient) ** 2 + eps))
                                    est_param_loss_type[self.types_to_idx[samples["type"][i]]].append(est_coeff_er)
                                    init_param_loss_type[self.types_to_idx[samples["type"][i]]].append(init_coeff_er)

                                text_data_loss += text_rel_loss
                                text_valid_output += 1

            # r2_losses.append(r2_score(np.stack(cur_target), np.stack(cur_output)[:, 1:]))
           # r2_losses.append(r2_score(np.stack(cur_target), np.stack(cur_output)))

            if not self.params.use_wandb:
                pbar.update(bs)

        if save:
            with h5py.File(save_file, "w") as hf:
                save_output = np.stack(output_list)
                hf.create_dataset("output", data=save_output)
                save_target = np.stack(target_list)
                hf.create_dataset("target", data=save_target)
                save_type = np.array(type_list)
                hf.create_dataset("type", data=save_type)
                logger.info(
                    f"Output ({save_output.shape}), target ({save_target.shape}), types ({save_type.shape}) saved at: {save_file}"
                )

        valid_loss_less_than_c = np.sum(np.array(valid_loss_less_than_c))
        data_loss = np.sum(np.array(data_loss))
        abs_data_loss = np.sum(np.array(abs_data_loss))
        data_loss_first_half = np.sum(np.array(data_loss_first_half))
        data_loss_second_half = np.sum(np.array(data_loss_second_half))
        data_loss_mean = np.sum(np.array(data_loss_mean))
       # print(r2_losses)
       # r2_losses = np.sum(np.array(r2_losses))
        l1_loss = np.sum(np.array(l1_loss))

        best_95perc_data_loss = 0
        best_95perc_l1_loss = 0
        for i in range(len(self.types)):
            cur_len = len(data_loss_type[i])
            best95_cur = np.sort(np.array(data_loss_type[i]))[: int(0.95 * cur_len)]
            best_total_loss[i] += np.sum(best95_cur)

            best_95perc_data_loss += np.sum(best95_cur)

            best95l1_cur = np.sort(np.array(l1_loss_type[i]))[: int(0.95 * cur_len)]
            best_l1_loss[i] += np.sum(best95l1_cur)

            best_95perc_l1_loss += np.sum(best95l1_cur)
        if params.multi_gpu:
            # sync results on all gpus

            lst_sync = torch.Tensor(
                [
                    text_valid,
                    text_total,
                    text_loss,
                    data_loss,
                    best_95perc_data_loss,
                    best_95perc_l1_loss,
                    abs_data_loss,
                    data_loss_first_half,
                    data_loss_second_half,
                    data_loss_mean,
            #        r2_losses,
                    l1_loss,
                ]
            ).cuda()
            total_loss = total_loss.cuda()
            total_abs_loss = total_abs_loss.cuda()
            total_count = total_count.cuda()
            total_mean_loss = total_mean_loss.cuda()
            total_l1_loss = total_l1_loss.cuda()

            dist.barrier()
            dist.all_reduce(lst_sync, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_abs_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_mean_loss, op=dist.ReduceOp.SUM)

            text_valid = lst_sync[0].item()
            text_total = lst_sync[1].item()
            text_loss = lst_sync[2].item()
            data_loss = lst_sync[3].item()
            best_95perc_data_loss = lst_sync[4].item()
            best_95perc_l1_loss = lst_sync[5].item()
            abs_data_loss = lst_sync[6].item()
            data_loss_first_half = lst_sync[7].item()
            data_loss_second_half = lst_sync[8].item()
            data_loss_mean = lst_sync[9].item()
           # r2_losses = lst_sync[9].item()
            l1_loss = lst_sync[10].item()

        if not params.text_only:
            # s = "Rel loss - "

            headers = []
            table = []
            # outputs = np.stack(output_list)[:, 1:]
            # labels = np.stack(output_list)[:, 0]
            outputs = np.stack(output_list)
            labels = np.array(type_list)
            target = np.stack(target_list)
            for i, cur_type in enumerate(self.types):
                cur_loss = total_loss[i].item()
                cur_count = total_count[i].item()
                cur_min_loss = min_data_loss[i].item()
                cur_min_loss_index = min_data_loss_index[i].item()
                cur_max_loss = max_data_loss[i].item()
                cur_max_loss_index = max_data_loss_index[i].item()
                cur_best_loss = best_total_loss[i].item()
                cur_best_l1_loss = best_l1_loss[i].item()
                cur_mean_loss = total_mean_loss[i].item()
                cur_l1_loss = total_l1_loss[i].item()
                cur_est_param_loss = est_param_loss_type[i]
                cur_init_param_loss = init_param_loss_type[i]

                cur_type_indx = self.alltypes_to_idx[cur_type]
                if np.sum(labels == cur_type_indx) != 0:
                    cur_r2_loss = r2_score(target[labels == cur_type_indx, :], outputs[labels == cur_type_indx, :])
                    mean = np.mean(target[labels == cur_type_indx, :], axis=0)
                    cur_mean_pred_loss = np.linalg.norm(target[labels == cur_type_indx, :] - mean) / np.linalg.norm(
                        target[labels == cur_type_indx, :]
                    )
                else:
                    cur_r2_loss = 0

                    cur_mean_pred_loss = 0

                headers.append("Type")
                table.append([cur_type])

                headers.append("Size")
                table[i].append(cur_count)

                headers.append("Rel L2")
                table[i].append(cur_loss / max(cur_count, 1))

                headers.append("Best 95% L2")
                table[i].append(cur_best_loss / max(0.95 * cur_count, 1))

                cur_abs_loss = total_abs_loss[i].item()

                headers.append("MSE")
                table[i].append(cur_abs_loss / max(cur_count, 1))

                headers.append("Mean Rel L2")
                table[i].append(cur_mean_loss / max(cur_count, 1))

                headers.append("L1 ")
                table[i].append(cur_l1_loss / max(cur_count, 1))

                headers.append("Best 95% L1")
                table[i].append(cur_best_l1_loss / max(0.95 * cur_count, 1))

                headers.append("R2 ")
                table[i].append(cur_r2_loss)

                headers.append("prediction by mean ")
                table[i].append(cur_mean_pred_loss)

                if self.params.use_text_refinement:

                    headers.append("Estimated Param Loss")
                    table[i].append(sum(cur_est_param_loss) / max(cur_count, 1))

                    headers.append("Initial Param Loss")
                    table[i].append(sum(cur_init_param_loss) / max(cur_count, 1))

                # s += "{}: {:.6f}/{}  best 95 perc: {:.6f} \n ".format(
                #     cur_type, cur_loss / max(cur_count, 1), cur_count, cur_best_loss / max(0.95 * cur_count, 1)
                # )

                if params.print_outputs:

                    # headers.append("Min")
                    # table[i].append(cur_min_loss)
                    #
                    # headers.append("Min idx")
                    # table[i].append(cur_min_loss_index)

                    headers.append("Max")
                    table[i].append(cur_max_loss)

                    headers.append("Max idx")
                    table[i].append(cur_max_loss_index)

                    # s += "min {} at {}, max {} at {}\t".format(
                    #     cur_min_loss, cur_min_loss_index, cur_max_loss, cur_max_loss_index
                    # )
                    for jj in range(self.space_dim):
                        title = "type_{}_dim{}_worst".format(cur_type, jj)
                        plot_sample_output_noerror(data_outputs[int(cur_max_loss_index)][:,:,jj], targets[int(cur_max_loss_index)][:,:,jj],
                                                   self.x_grid[:, jj],
                                                   self.t_eval[
                                                   eval_output_start: eval_output_end: params.eval_output_step],
                                                   params, title=title)
                        plot_sample_output(data_outputs[int(cur_max_loss_index)][:, :, jj],
                                                   targets[int(cur_max_loss_index)][:, :, jj],
                                                   self.x_grid[:, jj],
                                                   self.t_eval[
                                                   eval_output_start: eval_output_end: params.eval_output_step],
                                                   params, title=title)
                    if params.plot_comparison:
                        plot(
                            [data_outputs[int(cur_max_loss_index)]],
                            [targets[int(cur_max_loss_index)]],
                            [cur_type],
                            params,
                            notes="_worst_",
                            num_choice=1,
                            plot_type=[cur_type],
                        )
                        plot(
                            [torch.abs(data_outputs[int(cur_max_loss_index)] - targets[int(cur_max_loss_index)])],
                            None,
                            [cur_type],
                            params,
                            notes="_worstdiff_",
                            num_choice=1,
                            plot_type=[cur_type],
                        )

                    recomputed_loss = torch.norm(
                        data_outputs[int(cur_max_loss_index)] - targets[int(cur_max_loss_index)]
                    ) / torch.norm(targets[int(cur_max_loss_index)])

                    headers.append("Recomp abs loss")
                    table[i].append(torch.norm(data_outputs[int(cur_max_loss_index)] - targets[int(cur_max_loss_index)]))

                    headers.append("Recomp worst loss")
                    table[i].append(recomputed_loss)

                    headers.append("Recomp norm")
                    table[i].append(torch.norm(targets[int(cur_max_loss_index)]))

                    # s += "recomputed absloss {} , worstloss = {}, norm = {}\n".format(
                    #     torch.norm(data_outputs[int(cur_max_loss_index)] - target[int(cur_max_loss_index)]),
                    #     recomputed_loss,
                    #     torch.norm(target[int(cur_max_loss_index)]),
                    # )
            # logger.info(s)

            # s = "Abs loss - "
            # for i, cur_type in enumerate(self.types):
            #     cur_loss = total_abs_loss[i].item()
            #     cur_count = total_count[i].item()
            #     s += "{}: {:.6f}/{} \t ".format(cur_type, cur_loss / max(cur_count, 1), cur_count)

            # logger.info(s)

            logger.info("Evaluation Stats\n{}".format(tabulate(table, headers=headers, tablefmt="grid")))

        if params.text_ode_solve:
            logger.info(
                "Valid text - Data loss: {:.6f} - Data loss from text: {:.6f} - Text valid: {} - Text valid output: {}".format(
                    data_loss_valid / max(text_valid_output, 1),
                    text_data_loss / max(text_valid_output, 1),
                    text_valid,
                    text_valid_output,
                )
            )
        elif params.use_text_refinement:
            logger.info(
                "Valid text - Data loss: {:.6f} - Data loss from text: {:.6f} - Text valid: {} - Text valid output: {} - Data loss less than a large number: {}".format(
                    data_loss_valid / max(text_valid_output, 1),
                    text_data_loss / max(text_valid_output, 1),
                    text_valid,
                    text_valid_output,
                    valid_loss_less_than_c /  max(text_valid_output, 1),
                )
            )

        # logger.info("text_total: {} | Eval size per gpu: {}".format(text_total, eval_size_per_gpu))
        eval_size_per_gpu = text_total

        data_loss /= eval_size_per_gpu
        # best_95perc_data_loss/= 0.95 * eval_size_per_gpu
        valid_fraction = text_valid / text_total
        text_loss /= max(text_valid, 1)
        abs_data_loss /= eval_size_per_gpu
        data_loss_first_half /= eval_size_per_gpu
        data_loss_second_half /= eval_size_per_gpu
        data_loss_mean /= eval_size_per_gpu
        l1_loss /= eval_size_per_gpu
        r2_losses = r2_score(np.stack(target_list), np.stack(output_list))
        
       # print(r2_losses)
        if self.params.text_only:
            loss = text_loss
            if valid_fraction == 0.0:
                loss = np.inf
        elif self.params.data_only or self.params.no_text:
            loss = data_loss
        else:
            loss = self.params.data_loss_weight * data_loss + text_loss
            if valid_fraction == 0.0:
                loss = np.inf

        output = {
            "valid_fraction": valid_fraction,
            "text_loss": text_loss,
            "data_loss": data_loss,
            "data_loss_abs": abs_data_loss,
            "data_loss_first_half": data_loss_first_half,
            "data_loss_second_half": data_loss_second_half,
            "total_loss": loss,
            "data_loss_mean": data_loss_mean,
            "l1_loss": l1_loss,
            "r2_loss": r2_losses
            # "best_data_loss": best_95perc_data_loss,
        }

        return output

    # def compute_metrics(self,
    #                     data_type,
    #                     task,
    # preds_path,labels_path):
    #     total_loss = torch.zeros(len(self.types), dtype=torch.float32)
    #     best_total_loss = torch.zeros(len(self.types), dtype=torch.float32)
    #     min_data_loss = torch.ones(len(self.types), dtype=torch.float32)
    #     min_data_loss_index = torch.zeros(len(self.types), dtype=torch.float32)
    #     max_data_loss = torch.zeros(len(self.types), dtype=torch.float32)
    #     max_data_loss_index = torch.zeros(len(self.types), dtype=torch.float32)
    #     total_abs_loss = torch.zeros(len(self.types), dtype=torch.float32)
    #     total_count = torch.zeros(len(self.types), dtype=torch.long)
    #     total_mean_loss = torch.zeros(len(self.types), dtype=torch.float32)
    #
    #     input_len = self.params.input_len
    #     t_eval = self.trainer.t_eval
    #     output_len = len(t_eval) - input_len
    #     input_points = self.input_points
    #     eval_output_start = self.params.eval_output_start
    #     eval_output_end = len(t_eval)
    #
    #     text_loss = 0.0
    #     text_valid = 0
    #     text_total = 0
    #     eps = 1e-6
    #
    #     env = getattr(self, "{}_env".format(data_type))
    #     eval_size_per_gpu = self.params.eval_size // self.params.n_gpu_per_node
    #     if self.dataloader is None:
    #         self.dataloader = env.create_test_iterator(
    #             data_type,
    #             task,
    #             data_path=self.trainer.data_path,
    #             batch_size=self.params.batch_size_eval,
    #             params=self.params,
    #             size=eval_size_per_gpu,
    #             test_env_seed=self.params.test_env_seed,
    #         )
    #
    #     iterator = self.dataloader
    #     for samples, _ in iterator:
    #         target =np.array([seq[eval_output_start : eval_output_end : self.params.eval_output_step] for seq in samples["data"]])
    #         prediction = np.array([])[:,1:]
    #         label_indx = np.array([])[:,0]
    #
    # 
    def refinement(self,type,expr,data_input):
        """
        Right now this doesn't work... the errors between generated data and observations is quite high. Mostly just burgers work.
        """

        p = self.params
        input_len = self.params.input_len
        t_eval = self.trainer.t_eval
        eval_output_start = self.params.eval_output_start
        eval_output_end = len(t_eval)
        output_len = (len(t_eval) - input_len) // 2
        t_eval_step = p.eval_output_step
        t_grid = np.linspace(0.0, self.params.t_range, self.params.t_num)
        x_grid = np.linspace(0.0, self.params.x_range, self.params.x_num)
        output_grid = self.env.generator.t_eval[eval_output_start : eval_output_end : self.params.eval_output_step]
        output_grid = np.array(output_grid)
        input_grid = self.env.generator.t_eval[0 : eval_output_start : self.params.eval_output_step]
        input_grid = np.array(input_grid)
        # Number of particles
        M = 5000
        # M = 0 #for PDE solve ideally
        # M = 500 #Need to lower for speed.
        # Time steps
        T = 4
        # Grid points.
        N = p.x_num
        
        x, t = sy.symbols('x t')
        u = sy.Function('u_0')(x,t)

        # Define the initial distribution of particles for alpha (uniform distribution)
        def initial_distribution(init_alpha):
            return np.random.uniform((0.8 * init_alpha), (1.2 * init_alpha), M)
        
        # Propagate particles with noise
        def propagate_particles(particles, process_noise):
            noise = np.random.normal(0, process_noise, M)
            return np.abs(particles + noise)
        
        # Resample particles based on their weights using systematic resampling
        def resample(particles, weights):
            positions = (np.arange(M) + np.random.uniform(0, 1)) / M
            indexes = np.zeros(M, 'i')
            cumulative_sum = np.cumsum(weights)
            i, j = 0, 0
            while i < M:
                if positions[i] < cumulative_sum[j]:
                    indexes[i] = j
                    i += 1
                else:
                    j += 1
            return particles[indexes]        

        if type == "heat":

            # Initial diffusion coefficient (alpha) we want to estimate
            try:
                expr = sy.sympify(expr)
                expr = sy.simplify(expr) 
                expr = sy.expand(expr)
                init_alpha = expr.coeff(sy.diff(u,(x,2)))
            except:
                return None, None, None

            obs_noise = 0.05

            # Process noise
            process_noise = 0.001

            # Compute the next state of the heat equation using finite difference
            def heat_equation_step(u_prev, alpha):
                u_next = np.copy(u_prev)
                for i in range(1, N-1):
                    u_next[i] = u_prev[i] + alpha * (u_prev[i+1] - 2*u_prev[i] + u_prev[i-1])
                return u_next

            # Compute the weights based on the observation
            def compute_weights(particles, observation, u_prev):
                weights = np.zeros(M)
                for i in range(M):
                    u_pred = heat_equation_step(u_prev, particles[i])
                    weights[i] = np.exp(-0.5 * np.sum((observation - u_pred)**2) / obs_noise**2)
                return weights / np.sum(weights)


            # Initial distribution of particles
            particles = initial_distribution(init_alpha)

            # Simulate a sequence of observations
            observations = data_input[:,:]

            # Run the particle filter
            for t in range(1, T):
                # Propagate particles
                particles = propagate_particles(particles, process_noise)

                # Compute weights
                weights = compute_weights(particles, observations[t,:], observations[t-1,:])

                # Resample particles
                particles = resample(particles, weights)

                # Estimate the state
                estimated_alpha = np.mean(particles)

            new_output = np.zeros((output_len + 1, N))
            init_con = observations[-1,:]
            new_output[0,:] = init_con[0,:]
            new_val = heat_equation_step(init_con, estimated_alpha)
            new_output[1,:] = new_val[0,:]
            for cur_t in range(1,len(output_grid)-1):
                new_val = heat_equation_step(new_output[cur_t,:], estimated_alpha)
                new_output[cur_t + 1,:] = new_val
            return new_output, [estimated_alpha], [init_alpha]
        
        elif type == "porous_medium":

            try:
                expr = sy.sympify(expr)
                expr = expr.doit()
                # expr = sy.simplify(expr)        #Just so it is in a form which will work with this way of doing it.
                expr = sy.expand(expr)
            except:
                return None, None, None
            
            if expr.coeff(u * sy.diff(u,(x,2))) != 0:
                init_m = expr.coeff(u*sy.diff(u,(x,2)))
                mode = 2
            elif expr.coeff(u**2 * sy.diff(u,(x,2))) != 0:
                init_m = expr.coeff(u**2 * sy.diff(u,(x,2)))
                mode = 3
            elif expr.coeff(u**3 * sy.diff(u,(x,2))) != 0:
                init_m = expr.coeff(u**3 * sy.diff(u,(x,2)))
                mode = 4
            else:
                return None, None, None

            # Observation noise
            obs_noise = 0.5

            # Process noise
            process_noise = 0.01

            def f_closure(m):
                m = round(m,2)

                def f(u):
                    d2um_dx2 = np.zeros_like(u)
                    dx = p.x_range / p.x_num
                    um = np.power(u, m)
                    um[np.argwhere(np.isnan(um))] = 0.0001
                    # Compute second spatial derivatives using central differences
                    for i in range(1, p.x_num - 1):
                        d2um_dx2[i] = (um[i - 1] - 2 * um[i] + um[i + 1]) / dx**2

                    # Periodic boundary conditions
                    d2um_dx2[0] = (um[-1] - 2 * um[0] + um[1]) / dx**2
                    d2um_dx2[-1] = (um[-2] - 2 * um[-1] + um[0]) / dx**2

                    du_dt = d2um_dx2
                    return du_dt

                return f
        
            def pm_equation_step(u_prev, alpha):   
                dt = input_grid[1] - input_grid[0]
                fun = f_closure(alpha)
                u_next = np.copy(u_prev)
                u_next = u_prev + fun(u_prev) * dt
                return u_next
            
            
            # Compute the weights based on the observation
            def compute_weights(particles, observation, u_prev):
                weights = np.zeros(M)
                for i in range(M):
                    u_pred = pm_equation_step(u_prev, particles[i])
                    weights[i] = np.exp(-0.5 * np.sum((observation - u_pred)**2) / obs_noise**2)
                return weights / np.sum(weights)
            
            # Initial distribution of particles
            particles = initial_distribution(init_m)

            # Simulate a sequence of observations
            observations = data_input[:,:]

            #pm has an issue with taking in weird values..
            try:
                # Run the particle filter
                for t in range(1, T):
                    # Propagate particles
                    particles = propagate_particles(particles,process_noise)
                    
                    # Compute weights
                    weights = compute_weights(particles, observations[t,:], observations[t-1,:])

                    # Resample particles
                    particles = resample(particles, weights)

                    # Estimate the state
                    estimated_m = np.mean(particles)
            except:
                return None, None, None

            output_t = output_grid
            new_output = np.zeros((output_len + 1, N))
            init_con = observations[-1,:]
            new_output[0,:] = init_con[0,:]
            for cur_t in range(len(output_t)):
                if cur_t == 0:
                    new_val = pm_equation_step(init_con, estimated_m)
                    new_output[cur_t + 1,:] = new_val
                else:
                    new_val = pm_equation_step(new_output[cur_t - 1,:], estimated_m)
                    new_output[cur_t + 1,:] = new_val
            return new_output, [estimated_m], [mode]
        
        elif type == "advection":

            try:
                expr = sy.sympify(expr)
                expr = sy.simplify(expr)        #Just so it is in a form which will work with this way of doing it.
                expr = sy.expand(expr)
                init_alpha = expr.coeff(sy.diff(u,x))
            except:
                return None, None, None

            # Observation noise
            obs_noise = 0.25

            # Process noise
            process_noise = 0.01

            base_frequency = 2 * np.pi / p.x_range
            n1, n2 = np.random.randint(1, 3, size=2)
            frequencies = base_frequency * np.array([n1, n2])

            random_phases = np.random.uniform(0, 2 * np.pi, size=2)
            random_amplitudes = np.random.uniform(0, 1, size=2)

            # Composite wave function
            def _func(x):
                # return random_amplitudes[0] * np.sin(
                #     base_frequency * x + random_phases[0])
                wave1 = random_amplitudes[0] * np.sin(frequencies[0] * x + random_phases[0])
                wave2 = random_amplitudes[1] * np.sin(frequencies[1] * x + random_phases[1])
                return wave1 + wave2

            vec = _func(x_grid.flatten())
            slope = vec[-1] - vec[0]
            slope /= p.x_range
            vec = vec - slope * x_grid.flatten()
            min, max = np.min(vec), np.max(vec)

            def func(x):
                    val = _func(x)
                    linear = slope * x
                    val = val - linear
                    val = (val - min) / (max - min)
                    return val
            
            def adv_step(u_prev,beta,t):
                # y0 = func(self.x_grid.flatten())
                # max_y0,min_y0 = np.max(y0),np.min(y0)
                y = [func(u_prev)]
                # t_eval = self.t_eval / coeff
                cur_t = t
                x_adjusted = (x_grid.flatten() - beta * cur_t) % p.x_range
                y.append(func(x_adjusted))
                y_new = np.array(y[0])
            
                return y_new[-1]
            
            # Compute the weights based on the observation
            def compute_weights(particles, observation, u_prev,t):
                weights = np.zeros(M)
                for i in range(M):
                    u_pred = adv_step(u_prev, particles[i],t)
                    weights[i] = np.exp(-0.5 * np.sum((observation - u_pred)**2) / obs_noise**2)
                return weights / np.sum(weights)
            
            # Initial distribution of particles
            particles = initial_distribution(init_alpha)

            # Simulate a sequence of observations
            observations = data_input[:,:]

            # Run the particle filter
            for t in range(1, T):
                obs_t = observations[t,:]
                obs_t_minus_one = observations[t-1,:]
                # Propagate particles
                particles = propagate_particles(particles,process_noise)
                
                # Compute weights
                weights = compute_weights(particles, obs_t, obs_t_minus_one, input_grid[t-1])

                # Resample particles
                particles = resample(particles, weights)

                # Estimate the state
                estimated_alpha = np.mean(particles)

            new_output = np.zeros((output_len + 1, N))
            init_con = observations[-1,:]
            new_output[0,:] = init_con[0,:]
            new_val = adv_step(init_con, estimated_alpha, output_grid[0])
            new_output[1,:] = new_val
            for cur_t in range(1,len(output_grid)-1):
                new_val = adv_step(new_output[cur_t,:], estimated_alpha, output_grid[cur_t])
                new_output[cur_t + 1,:] = new_val
            return new_output, [estimated_alpha], [init_alpha]
        
        elif type == "kdv":
            try:
                expr = sy.sympify(expr)
                expr = sy.simplify(expr)
                expr = sy.expand(expr)
                init_alpha = expr.coeff(sy.diff(u,(x,3)))
            except:
                return None, None, None

            obs_noise = 2.5

            # Process noise
            process_noise = 0.00001
            
            tf = self.env.generator.pde_generator.tfinals["kdv"]
            p = self.params
            coeff = p.t_range / tf
            output_t = output_grid
            dt = output_grid[1] - output_grid[0]

            # Assuming nx is even for simplicity
            kx = np.fft.fftfreq(p.x_num, d=p.x_range / p.x_num)
            kx = 2.0 * np.pi * kx  # Scale by 2*pi for spatial frequency
            def kdv_step(u_prev,delta2,t):
                def uhat2vhat(t, uhat):
                    return np.exp(-1j * (kx**3) * delta2 * t) * uhat

                def vhat2uhat(t, vhat):
                    return np.exp(1j * (kx**3) * delta2 * t) * vhat

                # ----- Define RHS -----
                def uhatprime(t, uhat):
                    u = np.fft.ifft(uhat)
                    return 1j * (kx**2) * delta2 * uhat - 0.5j * kx * np.fft.fft(u**2)

                def vhatprime(t, vhat):
                    u = np.fft.ifft(vhat2uhat(t, vhat))
                    return -0.5j * kx * uhat2vhat(t, np.fft.fft(u**2))
                
                base_frequency = 2 * np.pi / p.x_range
                n1, n2 = np.random.randint(1, 3, size=2)
                frequencies = base_frequency * np.array([n1, n2])

                random_phases = np.random.uniform(0, 2 * np.pi, size=2)
                random_amplitudes = np.random.uniform(0, 1, size=2)

                # Composite wave function
                def _func(x):
                    # return random_amplitudes[0] * np.sin(
                    #     base_frequency * x + random_phases[0])
                    wave1 = random_amplitudes[0] * np.sin(frequencies[0] * x + random_phases[0])
                    wave2 = random_amplitudes[1] * np.sin(frequencies[1] * x + random_phases[1])
                    return wave1 + wave2

                vec = _func(x_grid.flatten())
                slope = vec[-1] - vec[0]
                slope /= p.x_range
                vec = vec - slope * x_grid.flatten()
                min, max = np.min(vec), np.max(vec)

                def func(x):
                    val = _func(x)
                    linear = slope * x
                    val = val - linear
                    val = (val - min) / (2 * (max - min))
                    return val

                u0 = np.zeros(np.size(u_prev, axis = 0))
                if np.shape(u_prev) == (N,1):
                    u0[:] = u_prev[0,:]
                else:
                    u0[:] = u_prev
                uhat0 = np.fft.fft(u0)
                vhat0 = uhat2vhat(t, uhat0)

                # sol = solve_ivp(
                #     vhatprime,
                #     [t / coeff for t in [t, t + dt]],
                #     vhat0,
                #     method="RK45",
                #     t_eval= t_eval / coeff,
                #     rtol=self.env.generator.pde_generator.rtol,
                #     atol=self.env.generator.pde_generator.atol,
                # )

                fun = vhatprime
                if np.shape(u_prev) == (N,1):
                    vhat = np.copy(u_prev[:,0])
                else:
                    vhat = np.copy(u_prev)
                vhat = vhat0 + fun(t,vhat) * dt
                
                # u = np.fft.ifft(vhat2uhat(t + dt, vhat))
                # u = np.zeros((p.x_num), dtype=complex)
                u = np.fft.ifft(vhat2uhat(t + dt, vhat[:]))
                u = np.real(u)
                # if np.all(np.abs(np.imag(u)) < 0.05):
                #     u = np.real(u)
                # else:
                #     raise ValueError
                # except Exception as e:
                #     return None, None, None

                return u
            
            # Compute the weights based on the observation
            def compute_weights(particles, observation, u_prev,t):
                weights = np.zeros(M)
                for i in range(M):
                    u_pred = kdv_step(u_prev, particles[i],t)
                    if u_pred is None:
                        weights[i] = 1/M
                    else:
                        weights[i] = np.exp(-0.5 * np.sum((observation - u_pred)**2) / obs_noise**2)
                return weights / np.sum(weights)
            
            # Initial distribution of particles
            particles = initial_distribution(init_alpha)

            # Simulate a sequence of observations
            observations = data_input[:,:]

            # Run the particle filter
            for t in range(1, T):
                # Propagate particles
                particles = propagate_particles(particles,process_noise)
                
                # Compute weights
                weights = compute_weights(particles, observations[t,:], observations[t-1,:], input_grid[t-1])

                # Resample particles
                particles = resample(particles, weights)

                # Estimate the state
                estimated_alpha = np.mean(particles)

            new_output = np.zeros((output_len + 1, N))
            init_con = observations[-1,:]
            new_output[0,:] = init_con[0,:]
            new_val = kdv_step(init_con, estimated_alpha, output_grid[0])
            new_output[1,:] = new_val
            for cur_t in range(1,len(output_grid)-1):
                new_val = kdv_step(new_output[cur_t,:], estimated_alpha, output_grid[cur_t])
                new_output[cur_t + 1,:] = new_val
            return new_output, [estimated_alpha], [init_alpha]
        
        # elif type == "fplanck":
        #     #this has a lot of terms. Needs more time devoted to it.
        #     um = 1e-6  # micrometer
        #     L = 0.1 * um
        #     c = 5e-21
        #     try:
        #         expr = sy.sympify(expr)
        #         expr = sy.simplify(expr)
        #         init_alpha = expr.coeff(sy.cos(x * (um/L)) * u)
        #         init_beta = expr.coeff(sy.sin(x * (um/L)) * sy.diff(u,x))
        #         init_gamma = expr.coeff(sy.diff(u,(x,2)))
        #     except:
        #         return None, None, None

        #     drag = init_alpha * (L ** 2 / c)                                     #init_alpha = c/(drag * L**2)
        #     drag = 1 / drag       
        #     temperature = init_gamma * (- um ** 2 / (scipy.constants.k * drag))   #init_gamma = -kT/(drag um**2)  
      
        #     obs_noise = 0.05

        #     # Process noise
        #     process_noise = 0.01

        #     output_t = output_grid
        #     dt = output_t[1] - output_t[0]

        #     # Define the potential function U(x) using micrometers
        #     U = lambda x: c * np.cos(x / L)
        #     def fplanck_step(u_prev, temperature, drag, t):
        #         # Setup the fokker_planck simulation with parameters converted to micrometers
        #         sim = fokker_planck(
        #             temperature=temperature,
        #             drag=drag,
        #             extent=2 * um,
        #             # extent converted to micrometers
        #             resolution= self.env.generator.pde_generator.dx * um,  # resolution converted to micrometers
        #             boundary=boundary.periodic,
        #             potential=U,
        #         )

        #         p0 = u_prev
                
        #         time, Pt = sim.propagate_interval(p0, (t + dt) * um, Nsteps=2)
        #         print(np.shape(Pt))
        #         return Pt


        #     # Compute the weights based on the observation
        #     def compute_weights(particles, particles_drag, observation, u_prev, t):
        #         weights = np.zeros(M)
        #         for i in range(M):
        #             u_pred = fplanck_step(u_prev, particles[i], particles_drag[i], t)
        #             weights[i] = np.exp(-0.5 * np.sum((observation - u_pred)**2) / obs_noise**2)
        #         return weights / np.sum(weights)
            
        #     # Initial distribution of particles
        #     particles = initial_distribution(temperature)
        #     particles_drag = initial_distribution(drag)

        #     # Simulate a sequence of observations
        #     observations = data_input[:,:]

        #     # Run the particle filter
        #     for t in range(1, T):
        #         # Propagate particles
        #         particles = propagate_particles(particles, process_noise)
        #         particles_drag = propagate_particles(particles_drag, process_noise)
                
        #         # Compute weights
        #         weights = compute_weights(particles, particles_drag, observations[t,:], observations[t-1,:], t_grid[t-1])

        #         # Resample particles
        #         particles = resample(particles, weights)
        #         particles_b = resample(particles_b, weights)

        #         # Estimate the state
        #         estimated_temperature = np.mean(particles)
        #         estimated_drag = np.mean(particles_drag)

        #     output_t = output_grid
        #     new_output = np.zeros((output_len, N))
        #     init_con = observations[-1,:]
        #     for cur_t in range(len(output_t)):
        #         if cur_t == 0:
        #             new_val = fplanck_step(init_con, estimated_temperature, estimated_drag, output_t[cur_t])
        #             new_output[cur_t,:] = new_val[:,0]
        #         else:
        #             new_val = fplanck_step(new_output[cur_t - 1,:], estimated_temperature,estimated_drag, output_t[cur_t])
        #             new_output[cur_t,:] = new_val
        #     return new_output

        elif type == "diff_logisreact_1D": 
            try:
                expr = sy.sympify(expr)
                init_rho = expr.coeff(u*(1-u))
                init_nu = expr.coeff(sy.diff(u,(x,2)))
            except:
                return None, None, None

            obs_noise = 5

            # Process noise
            process_noise = 0.01
            process_noise_b = 0.01

            IC_train = False
            numbers = 1
            CFL = 0.4
            dt = input_grid[1] - input_grid[0]
            # output_t = np.linspace(eval_output_start, eval_output_end, output_len)
            # t_array = np.arange(0,tf / coeff, dt / coeff)
            # output_t = t_array[eval_output_start:eval_output_end]
            

            def diff_logisreact_step(u_prev, rho, nu, t_eval):
                GivenIC = u_prev
                uu = diff_react_1D_f(
                    p.x_range,
                    0.0,
                    p.x_num,
                    t_eval,
                    t_eval + dt,
                    dt,
                    1,
                    CFL,
                    numbers,
                    20,
                    np.random.randint(100000),
                    rho,
                    nu,
                    IC_train=IC_train,
                    GivenIC=GivenIC
                )
                return uu[-1,:,0]

            # Compute the weights based on the observation
            def compute_weights(particles, particles_k, observation, u_prev, t_eval):
                weights = np.zeros(M)
                for i in range(M):
                    u_pred = diff_logisreact_step(u_prev, particles[i], particles_k[i], t_eval)
                    weights[i] = np.exp(-0.5 * np.sum((observation - u_pred)**2) / obs_noise**2)
                return weights / np.sum(weights)

            # Initial distribution of particles
            particles = initial_distribution(init_rho)
            particles_b = initial_distribution(init_nu)

            # ## Used in solving directly from given k, eps
            # estimated_k = np.float64(init_k)
            # estimated_eps = np.float64(init_eps)
            # ####

            # Simulate a sequence of observations
            observations = data_input[:,:]


            # Run the particle filter
            for t in range(1, T):
                # Propagate particles
                particles = propagate_particles(particles, process_noise)
                particles_b = propagate_particles(particles_b, process_noise_b)
                
                # Compute weights
                weights = compute_weights(particles, particles_b, observations[t,:], observations[t-1,:], input_grid[t-1])

                # Resample particles
                particles = resample(particles, weights)
                particles_b = resample(particles_b, weights)

                # Estimate the state
                estimated_rho = np.mean(particles)
                estimated_nu = np.mean(particles_b)

            new_output = np.zeros((output_len + 1, N))
            init_con = observations[-1,:]
            new_output[0,:] = init_con[0,:]
            new_val = diff_logisreact_step(init_con, estimated_rho, estimated_nu, output_grid[0])
            new_output[1,:] = new_val
            for cur_t in range(1,len(output_grid)-1):
                new_val = diff_logisreact_step(new_output[cur_t,:], estimated_rho, estimated_nu, output_grid[cur_t])
                new_output[cur_t + 1,:] = new_val
            return new_output, [estimated_rho, estimated_nu], [init_rho, init_nu]
        
        elif type == "diff_linearreact_1D": 
            try:
                expr = sy.sympify(expr)
                init_rho = expr.coeff(u)
                init_nu = expr.coeff(sy.diff(u,(x,2)))
            except:
                return None, None, None

            obs_noise = 5

            # Process noise
            process_noise = 0.001
            process_noise_b = 0.0001

            IC_train = False
            numbers = 1
            CFL = 0.35
            dt = input_grid[1] - input_grid[0]
            # output_t = np.linspace(eval_output_start, eval_output_end, output_len)
            # t_array = np.arange(0,tf / coeff, dt / coeff)
            # output_t = t_array[eval_output_start:eval_output_end]
            

            def diff_linearreact_step(u_prev, rho, nu, t_eval):
                GivenIC = u_prev
                uu = diff_react_1D_f(
                    p.x_range,
                    0.0,
                    p.x_num,
                    t_eval,
                    t_eval + dt,
                    dt,
                    1,
                    CFL,
                    numbers,
                    20,
                    np.random.randint(100000),
                    rho,
                    nu,
                    react_term = "linear",
                    IC_train=IC_train,
                    GivenIC=GivenIC
                )
                return uu[-1,:,0]

            # Compute the weights based on the observation
            def compute_weights(particles, particles_k, observation, u_prev, t_eval):
                weights = np.zeros(M)
                for i in range(M):
                    u_pred = diff_linearreact_step(u_prev, particles[i], particles_k[i], t_eval)
                    weights[i] = np.exp(-0.5 * np.sum((observation - u_pred)**2) / obs_noise**2)
                return weights / np.sum(weights)

            # Initial distribution of particles
            particles = initial_distribution(init_rho)
            particles_b = initial_distribution(init_nu)

            # ## Used in solving directly from given k, eps
            # estimated_k = np.float64(init_k)
            # estimated_eps = np.float64(init_eps)
            # ####

            # Simulate a sequence of observations
            observations = data_input[:,:]


            # Run the particle filter
            for t in range(1, T):
                # Propagate particles
                particles = propagate_particles(particles, process_noise)
                particles_b = propagate_particles(particles_b, process_noise_b)
                
                # Compute weights
                weights = compute_weights(particles, particles_b, observations[t,:], observations[t-1,:], input_grid[t-1])

                # Resample particles
                particles = resample(particles, weights)
                particles_b = resample(particles_b, weights)

                # Estimate the state
                estimated_rho = np.mean(particles)
                estimated_nu = np.mean(particles_b)

            new_output = np.zeros((output_len + 1, N))
            init_con = observations[-1,:]
            new_output[0,:] = init_con[0,:]
            new_val = diff_linearreact_step(init_con, estimated_rho, estimated_nu, output_grid[0])
            new_output[1,:] = new_val
            for cur_t in range(1,len(output_grid)-1):
                new_val = diff_linearreact_step(new_output[cur_t,:], estimated_rho, estimated_nu, output_grid[cur_t])
                new_output[cur_t + 1,:] = new_val
            return new_output, [estimated_rho, estimated_nu], [init_rho, init_nu]
        
        elif type == "diff_bistablereact_1D": 
            # Note: It is possible to add a into this.
            try:
                expr = sy.sympify(expr)
                expr_expanded = sy.expand(expr)
                expr_expanded = sy.simplify(expr_expanded)
                init_rho = -1 * expr_expanded.coeff(u**3)       #rho u(1-u)(u-1) = -rho u**3 + rho(1+a)u**2 - a rho u
                init_a = expr_expanded.coeff(u) / (-init_rho)   # will return 0 if it doesn't exist
                init_nu = expr.coeff(sy.diff(u,(x,2)))
            except:
                return None, None, None

            obs_noise = 5

            # Process noise
            process_noise = 0.01
            process_noise_b = 0.0001

            IC_train = False
            numbers = 1
            CFL = 0.35
            dt = input_grid[1] - input_grid[0]
            # output_t = np.linspace(eval_output_start, eval_output_end, output_len)
            # t_array = np.arange(0,tf / coeff, dt / coeff)
            # output_t = t_array[eval_output_start:eval_output_end]
            

            def diff_bistablereact_step(u_prev, rho, nu, t_eval):
                GivenIC = u_prev
                uu = diff_react_1D_f(
                    p.x_range,
                    0.0,
                    p.x_num,
                    t_eval,
                    t_eval + dt,
                    dt,
                    1,
                    CFL,
                    numbers,
                    20,
                    np.random.randint(100000),
                    rho,
                    nu,
                    react_term = "bistable",
                    IC_train=IC_train,
                    GivenIC=GivenIC
                )
                return uu[-1,:,0]

            # Compute the weights based on the observation
            def compute_weights(particles, particles_k, observation, u_prev, t_eval):
                weights = np.zeros(M)
                for i in range(M):
                    u_pred = diff_bistablereact_step(u_prev, particles[i], particles_k[i], t_eval)
                    weights[i] = np.exp(-0.5 * np.sum((observation - u_pred)**2) / obs_noise**2)
                return weights / np.sum(weights)

            # Initial distribution of particles
            particles = initial_distribution(init_rho)
            particles_b = initial_distribution(init_nu)

            # ## Used in solving directly from given k, eps
            # estimated_k = np.float64(init_k)
            # estimated_eps = np.float64(init_eps)
            # ####

            # Simulate a sequence of observations
            observations = data_input[:,:]


            # Run the particle filter
            for t in range(1, T):
                # Propagate particles
                particles = propagate_particles(particles, process_noise)
                particles_b = propagate_particles(particles_b, process_noise_b)
                
                # Compute weights
                weights = compute_weights(particles, particles_b, observations[t,:], observations[t-1,:], input_grid[t-1])

                # Resample particles
                particles = resample(particles, weights)
                particles_b = resample(particles_b, weights)

                # Estimate the state
                estimated_rho = np.mean(particles)
                estimated_nu = np.mean(particles_b)

            new_output = np.zeros((output_len + 1, N))
            init_con = observations[-1,:]
            new_output[0,:] = init_con[0,:]
            new_val = diff_bistablereact_step(init_con, estimated_rho, estimated_nu, output_grid[0])
            new_output[1,:] = new_val
            for cur_t in range(1,len(output_grid)-1):
                new_val = diff_bistablereact_step(new_output[cur_t,:], estimated_rho, estimated_nu, output_grid[cur_t])
                new_output[cur_t + 1,:] = new_val
            return new_output, [estimated_rho, estimated_nu], [init_rho, init_nu]
        
        elif type == "diff_squarelogisticreact_1D": 
            try:
                expr = sy.sympify(expr)
                init_rho = expr.coeff(((u ** 2) * (1 - u) ** 2))
                init_nu = expr.coeff(sy.diff(u,(x,2)))
            except:
                return None, None, None
    
            obs_noise = 5

            # Process noise
            process_noise = 0.01
            process_noise_b = 0.0001

            IC_train = False
            numbers = 1
            CFL = 0.35
            dt = input_grid[1] - input_grid[0]
            # output_t = np.linspace(eval_output_start, eval_output_end, output_len)
            # t_array = np.arange(0,tf / coeff, dt / coeff)
            # output_t = t_array[eval_output_start:eval_output_end]
            

            def diff_squarelogisticreact_step(u_prev, rho, nu, t_eval):
                GivenIC = u_prev
                uu = diff_react_1D_f(
                    p.x_range,
                    0.0,
                    p.x_num,
                    t_eval,
                    t_eval + dt,
                    dt,
                    1,
                    CFL,
                    numbers,
                    20,
                    np.random.randint(100000),
                    rho,
                    nu,
                    react_term = "squarelogistic",
                    IC_train=IC_train,
                    GivenIC=GivenIC
                )
                return uu[-1,:,0]

            # Compute the weights based on the observation
            def compute_weights(particles, particles_k, observation, u_prev, t_eval):
                weights = np.zeros(M)
                for i in range(M):
                    u_pred = diff_squarelogisticreact_step(u_prev, particles[i], particles_k[i], t_eval)
                    weights[i] = np.exp(-0.5 * np.sum((observation - u_pred)**2) / obs_noise**2)
                return weights / np.sum(weights)

            # Initial distribution of particles
            particles = initial_distribution(init_rho)
            particles_b = initial_distribution(init_nu)

            # ## Used in solving directly from given k, eps
            # estimated_k = np.float64(init_k)
            # estimated_eps = np.float64(init_eps)
            # ####

            # Simulate a sequence of observations
            observations = data_input[:,:]


            # Run the particle filter
            for t in range(1, T):
                # Propagate particles
                particles = propagate_particles(particles, process_noise)
                particles_b = propagate_particles(particles_b, process_noise_b)
                
                # Compute weights
                weights = compute_weights(particles, particles_b, observations[t,:], observations[t-1,:], input_grid[t-1])

                # Resample particles
                particles = resample(particles, weights)
                particles_b = resample(particles_b, weights)

                # Estimate the state
                estimated_rho = np.mean(particles)
                estimated_nu = np.mean(particles_b)

            new_output = np.zeros((output_len + 1, N))
            init_con = observations[-1,:]
            new_output[0,:] = init_con[0,:]
            new_val = diff_squarelogisticreact_step(init_con, estimated_rho, estimated_nu, output_grid[0])
            new_output[1,:] = new_val
            for cur_t in range(1,len(output_grid)-1):
                new_val = diff_squarelogisticreact_step(new_output[cur_t,:], estimated_rho, estimated_nu, output_grid[cur_t])
                new_output[cur_t + 1,:] = new_val
            return new_output, [estimated_rho, estimated_nu], [init_rho, init_nu]
        
        elif type == "burgers":
            try:
                expr = sy.sympify(expr)
                expr = expr.doit()
                init_k = expr.coeff(u*sy.diff(u,x))
                init_eps = expr.coeff(sy.diff(u,(x,2)))
            except:
                return None, None, None
    
            obs_noise = 5

            # Process noise
            process_noise = 0.01
            process_noise_b = 0.0001

            IC_train = False
            numbers = 1
            mode = "copy"
            CFL = 0.4
            dt = input_grid[1] - input_grid[0]
            # output_t = np.linspace(eval_output_start, eval_output_end, output_len)
            # t_array = np.arange(0,tf / coeff, dt / coeff)
            # output_t = t_array[eval_output_start:eval_output_end]
            

            def burgers_step(u_prev, eps, k, t_eval):
                GivenIC = u_prev
                uu = burgers_f(
                    p.x_range,
                    0.0,
                    p.x_num,
                    t_eval,
                    t_eval + dt,
                    dt,
                    1,
                    CFL,
                    numbers,
                    20,
                    np.random.randint(100000),
                    eps,
                    k,
                    fluxx="quadratic",
                    IC_train=IC_train,
                    GivenIC=GivenIC,
                    mode=mode
                )
                return uu[-1,:,0]
            
            # Compute the weights based on the observation
            def compute_weights(particles, particles_k, observation, u_prev, t_eval):
                weights = np.zeros(M)
                for i in range(M):
                    u_pred = burgers_step(u_prev, particles[i], particles_k[i], t_eval)
                    weights[i] = np.exp(-0.5 * np.sum((observation - u_pred)**2) / obs_noise**2)
                return weights / np.sum(weights)

            # Initial distribution of particles
            particles = initial_distribution(init_eps)
            particles_b = initial_distribution(init_k)

            # ## Used in solving directly from given k, eps
            # estimated_k = np.float64(init_k)
            # estimated_eps = np.float64(init_eps)
            # ####

            # Simulate a sequence of observations
            observations = data_input[:,:]


            # Run the particle filter
            for t in range(1, T):
                # Propagate particles
                particles = propagate_particles(particles, process_noise)
                particles_b = propagate_particles(particles_b, process_noise_b)
                
                # Compute weights
                weights = compute_weights(particles, particles_b, observations[t,:], observations[t-1,:], input_grid[t-1])

                # Resample particles
                particles = resample(particles, weights)
                particles_b = resample(particles_b, weights)

                # Estimate the state
                estimated_eps = np.mean(particles)
                estimated_k = np.mean(particles_b)

            new_output = np.zeros((output_len + 1, N))
            init_con = observations[-1,:]
            new_output[0,:] = init_con[0,:]
            new_val = burgers_step(init_con, estimated_eps, estimated_k, output_grid[0])
            new_output[1,:] = new_val
            for cur_t in range(1,len(output_grid)-1):
                new_val = burgers_step(new_output[cur_t,:], estimated_eps, estimated_k, output_grid[cur_t])
                new_output[cur_t + 1,:] = new_val
            return new_output, [estimated_eps, estimated_k], [init_eps, init_k]
        
        elif type == "conservation_linearflux":
            try:
                expr = sy.sympify(expr)
                expr = expr.doit()
                init_k = expr.coeff(sy.diff(u,x))
                init_eps = expr.coeff(sy.diff(u,(x,2)))
            except:
                return None, None, None
    
            obs_noise = 5

            # Process noise
            process_noise = 0.01
            process_noise_b = 0.0001

            IC_train = False
            numbers = 1
            mode = "copy"
            CFL = 0.4
            dt = input_grid[1] - input_grid[0]
            # output_t = np.linspace(eval_output_start, eval_output_end, output_len)
            # t_array = np.arange(0,tf / coeff, dt / coeff)
            # output_t = t_array[eval_output_start:eval_output_end]
            

            def cons_linear_step(u_prev, eps, k, t_eval):
                GivenIC = u_prev
                uu = burgers_f(
                    p.x_range,
                    0.0,
                    p.x_num,
                    t_eval,
                    t_eval + dt,
                    dt,
                    1,
                    CFL,
                    numbers,
                    20,
                    np.random.randint(100000),
                    eps,
                    k,
                    fluxx="linear",
                    IC_train=IC_train,
                    GivenIC=GivenIC,
                    mode=mode
                )
                return uu[-1,:,0]
            
            # Compute the weights based on the observation
            def compute_weights(particles, particles_k, observation, u_prev, t_eval):
                weights = np.zeros(M)
                for i in range(M):
                    u_pred = cons_linear_step(u_prev, particles[i], particles_k[i], t_eval)
                    weights[i] = np.exp(-0.5 * np.sum((observation - u_pred)**2) / obs_noise**2)
                return weights / np.sum(weights)

            # Initial distribution of particles
            particles = initial_distribution(init_eps)
            particles_b = initial_distribution(init_k)

            # ## Used in solving directly from given k, eps
            # estimated_k = np.float64(init_k)
            # estimated_eps = np.float64(init_eps)
            # ####

            # Simulate a sequence of observations
            observations = data_input[:,:]


            # Run the particle filter
            for t in range(1, T):
                # Propagate particles
                particles = propagate_particles(particles, process_noise)
                particles_b = propagate_particles(particles_b, process_noise_b)
                
                # Compute weights
                weights = compute_weights(particles, particles_b, observations[t,:], observations[t-1,:], input_grid[t-1])

                # Resample particles
                particles = resample(particles, weights)
                particles_b = resample(particles_b, weights)

                # Estimate the state
                estimated_eps = np.mean(particles)
                estimated_k = np.mean(particles_b)

            new_output = np.zeros((output_len + 1, N))
            init_con = observations[-1,:]
            new_output[0,:] = init_con[0,:]
            new_val = cons_linear_step(init_con, estimated_eps, estimated_k, output_grid[0])
            new_output[1,:] = new_val
            for cur_t in range(1,len(output_grid)-1):
                new_val = cons_linear_step(new_output[cur_t,:], estimated_eps, estimated_k, output_grid[cur_t])
                new_output[cur_t + 1,:] = new_val
            return new_output, [estimated_eps, estimated_k], [init_eps, init_k]

        elif type == "conservation_sinflux":
    
            try:
                expr = sy.sympify(expr)
                expr = expr.doit()
                init_k = expr.coeff(sy.cos(u) * sy.diff(u,x))
                init_eps = expr.coeff(sy.diff(u,(x,2)))
            except:
                return None, None, None
    
            obs_noise = 5

            # Process noise
            process_noise = 0.01
            process_noise_b = 0.0001

            IC_train = False
            numbers = 1
            mode = "copy"
            CFL = 0.4
            dt = input_grid[1] - input_grid[0]
            # output_t = np.linspace(eval_output_start, eval_output_end, output_len)
            # t_array = np.arange(0,tf / coeff, dt / coeff)
            # output_t = t_array[eval_output_start:eval_output_end]
            

            def cons_sinflux_step(u_prev, eps, k, t_eval):
                GivenIC = u_prev
                uu = burgers_f(
                    p.x_range,
                    0.0,
                    p.x_num,
                    t_eval,
                    t_eval + dt,
                    dt,
                    1,
                    CFL,
                    numbers,
                    20,
                    np.random.randint(100000),
                    eps,
                    k,
                    fluxx="sin",
                    IC_train=IC_train,
                    GivenIC=GivenIC,
                    mode=mode
                )
                return uu[-1,:,0]
            
            # Compute the weights based on the observation
            def compute_weights(particles, particles_k, observation, u_prev, t_eval):
                weights = np.zeros(M)
                for i in range(M):
                    u_pred = cons_sinflux_step(u_prev, particles[i], particles_k[i], t_eval)
                    weights[i] = np.exp(-0.5 * np.sum((observation - u_pred)**2) / obs_noise**2)
                return weights / np.sum(weights)

            # Initial distribution of particles
            particles = initial_distribution(init_eps)
            particles_b = initial_distribution(init_k)

            # ## Used in solving directly from given k, eps
            # estimated_k = np.float64(init_k)
            # estimated_eps = np.float64(init_eps)
            # ####

            # Simulate a sequence of observations
            observations = data_input[:,:]


            # Run the particle filter
            for t in range(1, T):
                # Propagate particles
                particles = propagate_particles(particles, process_noise)
                particles_b = propagate_particles(particles_b, process_noise_b)
                
                # Compute weights
                weights = compute_weights(particles, particles_b, observations[t,:], observations[t-1,:], input_grid[t-1])

                # Resample particles
                particles = resample(particles, weights)
                particles_b = resample(particles_b, weights)

                # Estimate the state
                estimated_eps = np.mean(particles)
                estimated_k = np.mean(particles_b)

            new_output = np.zeros((output_len + 1, N))
            init_con = observations[-1,:]
            new_output[0,:] = init_con[0,:]
            new_val = cons_sinflux_step(init_con, estimated_eps, estimated_k, output_grid[0])
            new_output[1,:] = new_val
            for cur_t in range(1,len(output_grid)-1):
                new_val = cons_sinflux_step(new_output[cur_t,:], estimated_eps, estimated_k, output_grid[cur_t])
                new_output[cur_t + 1,:] = new_val
            return new_output, [estimated_eps, estimated_k], [init_eps, init_k]

        elif type == "conservation_cosflux":
    
            try:
                expr = sy.sympify(expr)
                expr = expr.doit()
                init_k = expr.coeff(sy.sin(u) * sy.diff(u,x))
                init_eps = expr.coeff(sy.diff(u,(x,2)))
            except:
                return None, None, None
    
            obs_noise = 5

            # Process noise
            process_noise = 0.01
            process_noise_b = 0.0001

            IC_train = False
            numbers = 1
            mode = "copy"
            CFL = 0.4
            dt = input_grid[1] - input_grid[0]
            # output_t = np.linspace(eval_output_start, eval_output_end, output_len)
            # t_array = np.arange(0,tf / coeff, dt / coeff)
            # output_t = t_array[eval_output_start:eval_output_end]
            

            def cons_cosflux_step(u_prev, eps, k, t_eval):
                GivenIC = u_prev
                uu = burgers_f(
                    p.x_range,
                    0.0,
                    p.x_num,
                    t_eval,
                    t_eval + dt,
                    dt,
                    1,
                    CFL,
                    numbers,
                    20,
                    np.random.randint(100000),
                    eps,
                    k,
                    fluxx="cos",
                    IC_train=IC_train,
                    GivenIC=GivenIC,
                    mode=mode
                )
                return uu[-1,:,0]
            
            # Compute the weights based on the observation
            def compute_weights(particles, particles_k, observation, u_prev, t_eval):
                weights = np.zeros(M)
                for i in range(M):
                    u_pred = cons_cosflux_step(u_prev, particles[i], particles_k[i], t_eval)
                    weights[i] = np.exp(-0.5 * np.sum((observation - u_pred)**2) / obs_noise**2)
                return weights / np.sum(weights)

            # Initial distribution of particles
            particles = initial_distribution(init_eps)
            particles_b = initial_distribution(init_k)

            # ## Used in solving directly from given k, eps
            # estimated_k = np.float64(init_k)
            # estimated_eps = np.float64(init_eps)
            # ####

            # Simulate a sequence of observations
            observations = data_input[:,:]


            # Run the particle filter
            for t in range(1, T):
                # Propagate particles
                particles = propagate_particles(particles, process_noise)
                particles_b = propagate_particles(particles_b, process_noise_b)
                
                # Compute weights
                weights = compute_weights(particles, particles_b, observations[t,:], observations[t-1,:], input_grid[t-1])

                # Resample particles
                particles = resample(particles, weights)
                particles_b = resample(particles_b, weights)

                # Estimate the state
                estimated_eps = np.mean(particles)
                estimated_k = np.mean(particles_b)

            new_output = np.zeros((output_len + 1, N))
            init_con = observations[-1,:]
            new_output[0,:] = init_con[0,:]
            new_val = cons_cosflux_step(init_con, estimated_eps, estimated_k, output_grid[0])
            new_output[1,:] = new_val
            for cur_t in range(1,len(output_grid)-1):
                new_val = cons_cosflux_step(new_output[cur_t,:], estimated_eps, estimated_k, output_grid[cur_t])
                new_output[cur_t + 1,:] = new_val
            return new_output, [estimated_eps, estimated_k], [init_eps, init_k]

        elif type == "conservation_cubicflux":
    
            try:
                expr = sy.sympify(expr)
                expr = expr.doit()
                init_k = expr.coeff(u ** 2 * sy.diff(u,x))
                init_eps = expr.coeff(sy.diff(u,(x,2)))
            except:
                return None, None, None
    
            obs_noise = 5

            # Process noise
            process_noise = 0.01
            process_noise_b = 0.0001

            IC_train = False
            numbers = 1
            mode = "copy"
            CFL = 0.4
            dt = input_grid[1] - input_grid[0]
            # output_t = np.linspace(eval_output_start, eval_output_end, output_len)
            # t_array = np.arange(0,tf / coeff, dt / coeff)
            # output_t = t_array[eval_output_start:eval_output_end]
            

            def cons_cubicflux_step(u_prev, eps, k, t_eval):
                GivenIC = u_prev
                uu = burgers_f(
                    p.x_range,
                    0.0,
                    p.x_num,
                    t_eval,
                    t_eval + dt,
                    dt,
                    1,
                    CFL,
                    numbers,
                    20,
                    np.random.randint(100000),
                    eps,
                    k,
                    fluxx="cubic",
                    IC_train=IC_train,
                    GivenIC=GivenIC,
                    mode=mode
                )
                return uu[-1,:,0]
            
            # Compute the weights based on the observation
            def compute_weights(particles, particles_k, observation, u_prev, t_eval):
                weights = np.zeros(M)
                for i in range(M):
                    u_pred = cons_cubicflux_step(u_prev, particles[i], particles_k[i], t_eval)
                    weights[i] = np.exp(-0.5 * np.sum((observation - u_pred)**2) / obs_noise**2)
                return weights / np.sum(weights)

            # Initial distribution of particles
            particles = initial_distribution(init_eps)
            particles_b = initial_distribution(init_k)

            # ## Used in solving directly from given k, eps
            # estimated_k = np.float64(init_k)
            # estimated_eps = np.float64(init_eps)
            # ####

            # Simulate a sequence of observations
            observations = data_input[:,:]


            # Run the particle filter
            for t in range(1, T):
                # Propagate particles
                particles = propagate_particles(particles, process_noise)
                particles_b = propagate_particles(particles_b, process_noise_b)
                
                # Compute weights
                weights = compute_weights(particles, particles_b, observations[t,:], observations[t-1,:], input_grid[t-1])

                # Resample particles
                particles = resample(particles, weights)
                particles_b = resample(particles_b, weights)

                # Estimate the state
                estimated_eps = np.mean(particles)
                estimated_k = np.mean(particles_b)

            new_output = np.zeros((output_len + 1, N))
            init_con = observations[-1,:]
            new_output[0,:] = init_con[0,:]
            new_val = cons_cubicflux_step(init_con, estimated_eps, estimated_k, output_grid[0])
            new_output[1,:] = new_val
            for cur_t in range(1,len(output_grid)-1):
                new_val = cons_cubicflux_step(new_output[cur_t,:], estimated_eps, estimated_k, output_grid[cur_t])
                new_output[cur_t + 1,:] = new_val
            return new_output, [estimated_eps, estimated_k], [init_eps, init_k]

        elif type == "inviscid_burgers":
    
            try:
                expr = sy.sympify(expr)
                expr = expr.doit()
                init_k = expr.coeff(u*sy.diff(u,x))
                estimated_k = np.float64(init_k)
            except:
                return None, None, None


            obs_noise = 5

            # Process noise
            process_noise = 0.01

            IC_train = False
            numbers = 1
            mode = "copy"
            CFL = 0.4
            dt = input_grid[1] - input_grid[0]

            def inv_burgers_step(u_prev, k, t_eval):
                GivenIC = u_prev
                uu = burgers_f(
                    p.x_range,
                    0.0,
                    p.x_num,
                    t_eval,
                    t_eval + dt,
                    dt,
                    1,
                    CFL,
                    numbers,
                    20,
                    np.random.randint(100000),
                    0,
                    k,
                    viscous = False,
                    fluxx="quadratic",
                    IC_train=IC_train,
                    GivenIC=GivenIC,
                    mode=mode
                )
                return uu[-1,:,0]
            
            # Compute the weights based on the observation
            def compute_weights(particles, observation, u_prev, t_eval):
                weights = np.zeros(M)
                for i in range(M):
                    u_pred = inv_burgers_step(u_prev, particles[i], t_eval)
                    weights[i] = np.exp(-0.5 * np.sum((observation - u_pred)**2) / obs_noise**2)
                return weights / np.sum(weights)

            # Initial distribution of particles
            particles = initial_distribution(init_k)

            # Simulate a sequence of observations
            observations = data_input[:,:]
            # Run the particle filter
            for t in range(1, T):
                # Propagate particles
                particles = propagate_particles(particles, process_noise)
                
                # Compute weights
                weights = compute_weights(particles, observations[t,:], observations[t-1,:], input_grid[t-1])

                # Resample particles
                particles = resample(particles, weights)

                # Estimate the state
                estimated_k = np.mean(particles)

            new_output = np.zeros((output_len + 1, N))
            init_con = observations[-1,:]
            new_output[0,:] = init_con[0,:]
            new_val = inv_burgers_step(init_con, estimated_k, output_grid[0])
            new_output[1,:] = new_val
            for cur_t in range(1,len(output_grid)-1):
                new_val = inv_burgers_step(new_output[cur_t,:], estimated_k, output_grid[cur_t])
                new_output[cur_t + 1,:] = new_val

            return new_output, [estimated_k], [init_k]

        elif type == "inviscid_conservation_sinflux":
    
            try:
                expr = sy.sympify(expr)
                expr = expr.doit()
                init_k = expr.coeff(sy.cos(u)*sy.diff(u,x))
            except:
                return None, None, None
    
            obs_noise = 5

            # Process noise
            process_noise = 0.01

            IC_train = False
            numbers = 1
            mode = "copy"
            CFL = 0.4
            dt = input_grid[1] - input_grid[0]

            def inv_sinflux_step(u_prev, k, t_eval):
                GivenIC = u_prev
                uu = burgers_f(
                    p.x_range,
                    0.0,
                    p.x_num,
                    t_eval,
                    t_eval + dt,
                    dt,
                    1,
                    CFL,
                    numbers,
                    20,
                    np.random.randint(100000),
                    0,
                    k,
                    viscous = False,
                    fluxx="sin",
                    IC_train=IC_train,
                    GivenIC=GivenIC,
                    mode=mode
                )
                return uu[-1,:,0]
            
            # Compute the weights based on the observation
            def compute_weights(particles, observation, u_prev, t_eval):
                weights = np.zeros(M)
                for i in range(M):
                    u_pred = inv_sinflux_step(u_prev, particles[i], t_eval)
                    weights[i] = np.exp(-0.5 * np.sum((observation - u_pred)**2) / obs_noise**2)
                return weights / np.sum(weights)

            # Initial distribution of particles
            particles = initial_distribution(init_k)

            # Simulate a sequence of observations
            observations = data_input[:,:]
            # Run the particle filter
            for t in range(1, T):
                # Propagate particles
                particles = propagate_particles(particles, process_noise)
                
                # Compute weights
                weights = compute_weights(particles, observations[t,:], observations[t-1,:], input_grid[t-1])

                # Resample particles
                particles = resample(particles, weights)

                # Estimate the state
                estimated_k = np.mean(particles)

            new_output = np.zeros((output_len + 1, N))
            init_con = observations[-1,:]
            new_output[0,:] = init_con[0,:]
            new_val = inv_sinflux_step(init_con, estimated_k, output_grid[0])
            new_output[1,:] = new_val
            for cur_t in range(1,len(output_grid)-1):
                new_val = inv_sinflux_step(new_output[cur_t,:], estimated_k, output_grid[cur_t])
                new_output[cur_t + 1,:] = new_val

            return new_output, [estimated_k], [init_k]

        elif type == "inviscid_conservation_cosflux":
    
            try:
                expr = sy.sympify(expr)
                expr = expr.doit()
                init_k = expr.coeff(sy.sin(u)*sy.diff(u,x))
            except:
                return None, None, None
    
            obs_noise = 5

            # Process noise
            process_noise = 0.01

            IC_train = False
            numbers = 1
            mode = "copy"
            CFL = 0.4
            dt = input_grid[1] - input_grid[0]

            def inv_cosflux_step(u_prev, k, t_eval):
                GivenIC = u_prev
                uu = burgers_f(
                    p.x_range,
                    0.0,
                    p.x_num,
                    t_eval,
                    t_eval + dt,
                    dt,
                    1,
                    CFL,
                    numbers,
                    20,
                    np.random.randint(100000),
                    0,
                    k,
                    viscous = False,
                    fluxx="cos",
                    IC_train=IC_train,
                    GivenIC=GivenIC,
                    mode=mode
                )
                return uu[-1,:,0]
            
            # Compute the weights based on the observation
            def compute_weights(particles, observation, u_prev, t_eval):
                weights = np.zeros(M)
                for i in range(M):
                    u_pred = inv_cosflux_step(u_prev, particles[i], t_eval)
                    weights[i] = np.exp(-0.5 * np.sum((observation - u_pred)**2) / obs_noise**2)
                return weights / np.sum(weights)

            # Initial distribution of particles
            particles = initial_distribution(init_k)

            # Simulate a sequence of observations
            observations = data_input[:,:]
            # Run the particle filter
            for t in range(1, T):
                # Propagate particles
                particles = propagate_particles(particles, process_noise)
                
                # Compute weights
                weights = compute_weights(particles, observations[t,:], observations[t-1,:], input_grid[t-1])

                # Resample particles
                particles = resample(particles, weights)

                # Estimate the state
                estimated_k = np.mean(particles)

            new_output = np.zeros((output_len + 1, N))
            init_con = observations[-1,:]
            new_output[0,:] = init_con[0,:]
            new_val = inv_cosflux_step(init_con, estimated_k, output_grid[0])
            new_output[1,:] = new_val
            for cur_t in range(1,len(output_grid)-1):
                new_val = inv_cosflux_step(new_output[cur_t,:], estimated_k, output_grid[cur_t])
                new_output[cur_t + 1,:] = new_val

            return new_output, [estimated_k], [init_k]

        elif type == "inviscid_conservation_cubicflux":
    
            try:
                expr = sy.sympify(expr)
                expr = expr.doit()
                init_k = expr.coeff(u ** 2 * sy.diff(u,x))
            except:
                return None, None, None
    
            obs_noise = 5

            # Process noise
            process_noise = 0.01

            IC_train = False
            numbers = 1
            mode = "copy"
            CFL = 0.4
            dt = input_grid[1] - input_grid[0]

            def inv_cubicflux_step(u_prev, k, t_eval):
                GivenIC = u_prev
                uu = burgers_f(
                    p.x_range,
                    0.0,
                    p.x_num,
                    t_eval,
                    t_eval + dt,
                    dt,
                    1,
                    CFL,
                    numbers,
                    20,
                    np.random.randint(100000),
                    0,
                    k,
                    viscous = False,
                    fluxx="cubic",
                    IC_train=IC_train,
                    GivenIC=GivenIC,
                    mode=mode
                )
                return uu[-1,:,0]
            
            # Compute the weights based on the observation
            def compute_weights(particles, observation, u_prev, t_eval):
                weights = np.zeros(M)
                for i in range(M):
                    u_pred = inv_cubicflux_step(u_prev, particles[i], t_eval)
                    weights[i] = np.exp(-0.5 * np.sum((observation - u_pred)**2) / obs_noise**2)
                return weights / np.sum(weights)

            # Initial distribution of particles
            particles = initial_distribution(init_k)

            # Simulate a sequence of observations
            observations = data_input[:,:]
            # Run the particle filter
            for t in range(1, T):
                # Propagate particles
                particles = propagate_particles(particles, process_noise)
                
                # Compute weights
                weights = compute_weights(particles, observations[t,:], observations[t-1,:], input_grid[t-1])

                # Resample particles
                particles = resample(particles, weights)

                # Estimate the state
                estimated_k = np.mean(particles)

            new_output = np.zeros((output_len + 1, N))
            init_con = observations[-1,:]
            new_output[0,:] = init_con[0,:]
            new_val = inv_cubicflux_step(init_con, estimated_k, output_grid[0])
            new_output[1,:] = new_val
            for cur_t in range(1,len(output_grid)-1):
                new_val = inv_cubicflux_step(new_output[cur_t,:], estimated_k, output_grid[cur_t])
                new_output[cur_t + 1,:] = new_val

            return new_output, [estimated_k], [init_k]

        elif type == "cahnhilliard_1D":
            # DOES NOT INFER 6.
            try:
                expr = sy.sympify(expr)
                init_alpha = expr.coeff(sy.diff(u,(x,4)))
            except:
                return None, None, None

            # Observation noise
            obs_noise = 0.5

            # Process noise
            process_noise = 0.00000001
        
            def f_closure(eps):

                def f(u):
                    d2u_dx2 = np.zeros_like(u)
                    rhs = np.zeros_like(u)
                    dx = p.x_range / p.x_num
                    # Compute second spatial derivatives using central differences
                    for i in range(1, p.x_num - 1):
                        d2u_dx2[i] = (u[i - 1] - 2 * u[i] + u[i + 1]) / dx**2

                    # Periodic boundary conditions
                    d2u_dx2[0] = (u[-1] - 2 * u[0] + u[1]) / dx**2
                    d2u_dx2[-1] = (u[-2] - 2 * u[-1] + u[0]) / dx**2

                    f = u**3 - u
                    fu = 3 * u**2 - 1

                    d2u_dx2af = eps**2 * d2u_dx2 + fu

                    for i in range(1, p.x_num - 1):
                        rhs[i] = (d2u_dx2af[i - 1] - 2 * d2u_dx2af[i] + d2u_dx2af[i + 1]) / dx**2

                    # Periodic boundary conditions
                    rhs[0] = (d2u_dx2af[-1] - 2 * d2u_dx2af[0] + d2u_dx2af[1]) / dx**2
                    rhs[-1] = (d2u_dx2af[-2] - 2 * d2u_dx2af[-1] + d2u_dx2af[0]) / dx**2

                    du_dt = -d2u_dx2af
                    return du_dt

                return f
    
            def ch_equation_step(u_prev, eps):   
                dt = input_grid[1] - input_grid[0]
                fun = f_closure(eps)
                u_next = np.copy(u_prev)
                u_next = u_prev + fun(u_prev) * dt
                return u_next

            # Compute the weights based on the observation
            def compute_weights(particles, observation, u_prev):
                weights = np.zeros(M)
                for i in range(M):
                    u_pred = ch_equation_step(u_prev, particles[i])
                    weights[i] = np.exp(-0.5 * np.sum((observation - u_pred)**2) / obs_noise**2)
                return weights / np.sum(weights)
        
            # Initial distribution of particles
            particles = initial_distribution(init_alpha)

            # Simulate a sequence of observations
            observations = data_input[:,:]

            # Run the particle filter
            for t in range(1, T):
                # Propagate particles
                particles = propagate_particles(particles,process_noise)
                
                # Compute weights
                weights = compute_weights(particles, observations[t,:], observations[t-1,:])

                # Resample particles
                particles = resample(particles, weights)

                # Estimate the state
                estimated_alpha = np.mean(particles)

            new_output = np.zeros((output_len + 1, N))
            init_con = observations[-1,:]
            new_output[0,:] = init_con[0,:]
            new_val = ch_equation_step(init_con, estimated_alpha)
            new_output[1,:] = new_val[0,:]
            for cur_t in range(1,len(output_grid)-1):
                new_val = ch_equation_step(new_output[cur_t,:], estimated_alpha)
                new_output[cur_t + 1,:] = new_val

            return new_output, [estimated_alpha], [init_alpha]


        elif type == "wave":

            try:
                expr = sy.sympify(expr)
                expr = sy.simplify(expr)        #Just so it is in a form which will work with this way of doing it.
                expr = sy.expand(expr)
                init_alpha = expr.coeff(sy.diff(u,(x,2)))
            except:
                return None, None, None
    
            obs_noise = 0.5

            # Process noise
            process_noise = 0.001

            tf = self.env.generator.pde_generator.tfinals["wave"]
            coeff_t = p.t_range / tf
            t_eval = t_grid / coeff_t

            base_frequency = 2 * np.pi / p.x_range
            n1, n2 = np.random.randint(1, 3, size=2)
            frequencies = base_frequency * np.array([n1, n2])

            random_phases = np.random.uniform(0, 2 * np.pi, size=2)
            random_amplitudes = np.random.uniform(0, 1, size=2)

            # Composite wave function
            def _func(x):
                # return random_amplitudes[0] * np.sin(
                #     base_frequency * x + random_phases[0])
                wave1 = random_amplitudes[0] * np.sin(frequencies[0] * x + random_phases[0])
                wave2 = random_amplitudes[1] * np.sin(frequencies[1] * x + random_phases[1])
                return wave1 + wave2

            vec = _func(x_grid.flatten())
            slope = vec[-1] - vec[0]
            slope /= p.x_range
            vec = vec - slope * x_grid.flatten()
            min, max = np.min(vec), np.max(vec)

            def func(x):
                val = _func(x)
                linear = slope * x
                val = val - linear
                val = (val - min) / (max - min)
                return val
            
            def wave_step(u,beta,t):
                y = [func(u)]
                cur_t = t
                x_adjusted1 = (x_grid.flatten() - beta * cur_t) % p.x_range
                x_adjusted2 = (x_grid.flatten() + beta * cur_t) % p.x_range
                y.append(0.5 * func(x_adjusted1) + 0.5 * func(x_adjusted2))
                y = np.array(y[0])
                return y[-1]
            
            # Compute the weights based on the observation
            def compute_weights(particles, observation, u_prev,t):
                weights = np.zeros(M)
                for i in range(M):
                    u_pred = wave_step(u_prev, particles[i],t)
                    weights[i] = np.exp(-0.5 * np.sum((observation - u_pred)**2) / obs_noise**2)
                return weights / np.sum(weights)
            
            # Initial distribution of particles
            particles = initial_distribution(init_alpha)

            # Simulate a sequence of observations
            observations = data_input[:,:]

            # Run the particle filter
            for t in range(1, T):
                # Propagate particles
                particles = propagate_particles(particles,process_noise)
                
                # Compute weights
                weights = compute_weights(particles, observations[t,:], observations[t-1,:],input_grid[t-1])

                # Resample particles
                particles = resample(particles, weights)

                # Estimate the state
                estimated_alpha = np.mean(particles)

            new_output = np.zeros((output_len + 1, N))
            init_con = observations[-1,:]
            new_output[0,:] = init_con[0,:]
            new_val = wave_step(init_con, estimated_alpha, output_grid[0])
            new_output[1,:] = new_val
            for cur_t in range(1,len(output_grid)-1):
                new_val = wave_step(new_output[cur_t,:], estimated_alpha, output_grid[cur_t])
                new_output[cur_t + 1,:] = new_val

            return new_output, [estimated_alpha], [init_alpha]

        elif type == "Klein_Gordon":
            try:
                expr = sy.sympify(expr)
                init_alpha = expr.coeff(sy.diff(u,(x,2)))
                init_beta = expr.coeff(u)
            except:
                return None, None, None
        
            # Observation noise
            obs_noise = 0.5

            # Process noise
            process_noise = 0.00001
        
            def kg_step(u_prev, alpha, beta):
                u_next = np.copy(u_prev)
                for i in range(1,N-1):
                    u_next[i] = alpha*(u_prev[i+1] - 2.0 * u_prev[i] + u_prev[i-1]) - beta * u_prev[i] +  2.0 * u_prev[i] -u_prev[i-1]

                return u_next

            # Compute the weights based on the observation
            def compute_weights(particles,particles_beta, observation, u_prev):
                weights = np.zeros(M)
                for i in range(M):
                    u_pred = kg_step(u_prev, particles[i],particles_beta[i])
                    #right now u_pred = 0 for all entries, so we get 0/0 and nans...
                    weights[i] = np.exp(-0.5 * np.sum((observation - u_pred)**2) / obs_noise**2)
                return weights / np.sum(weights)
            
            # Initial distribution of particles
            particles = initial_distribution(init_alpha)
            particles_b = initial_distribution(init_beta)

            # Simulate a sequence of observations
            observations = data_input[:,:]

            # Run the particle filter
            for t in range(1, T):
                # Propagate particles
                particles = propagate_particles(particles, process_noise)
                particles_b = propagate_particles(particles_b, process_noise)
                
                # Compute weights
                weights = compute_weights(particles, particles_b, observations[t,:], observations[t-1,:])

                # Resample particles
                particles = resample(particles, weights)
                particles_b = resample(particles_b, weights)

                # Estimate the state
                estimated_alpha = np.mean(particles)
                estimated_beta = np.mean(particles_b)

            new_output = np.zeros((output_len + 1, N))
            init_con = observations[-1,:]
            new_output[0,:] = init_con[0,:]
            new_val = kg_step(init_con, estimated_alpha, estimated_beta)
            new_output[1,:] = new_val[0,:]
            for cur_t in range(1,len(output_grid)-1):
                new_val = kg_step(new_output[cur_t,:], estimated_alpha, estimated_beta)
                new_output[cur_t + 1,:] = new_val

            return new_output, [estimated_alpha,estimated_beta], [init_alpha,init_beta]

            

        elif type == "Sine_Gordon":
            try:
                expr = sy.sympify(expr)
                init_alpha = expr.coeff(sy.sin(u))
            except:
                return None, None, None
            tf = self.env.generator.pde_generator.tfinals["Sine_Gordon"]
            coeff = p.t_range/tf
            dt = input_grid[1] - input_grid[0]
            dx = p.x_range/p.x_num
            dt_this = dt / (coeff * 100)
            coeff_a = (dt_this**2) / (dx**2)

            # Observation noise
            obs_noise = 0.5

            # Process noise
            process_noise = 0.00001

            def sg_equation_step(u_prev, alpha):
                u_next = np.copy(u_prev)
                for i in range(1,N-1):
                    u_next[i] = 2.0 * u_prev[i] - u_prev[i-1] + coeff_a * (u_prev[i+1] - 2.0 * u_prev[i] + u_prev[i-1]) - (dt_this**2) * alpha * np.sin(u_prev[i])

                return u_next

            # Compute the weights based on the observation
            def compute_weights(particles, observation, u_prev):
                weights = np.zeros(M)
                for i in range(M):
                    u_pred = sg_equation_step(u_prev, particles[i])
                    weights[i] = np.exp(-0.5 * np.sum((observation - u_pred)**2) / obs_noise**2)
                return weights / np.sum(weights)
        
            # Initial distribution of particles
            particles = initial_distribution(init_alpha)

            # Simulate a sequence of observations
            observations = data_input[:,:]

            # Run the particle filter
            for t in range(1, T):
                # Propagate particles
                particles = propagate_particles(particles,process_noise)
                
                # Compute weights
                weights = compute_weights(particles, observations[t,:], observations[t-1,:])

                # Resample particles
                particles = resample(particles, weights)

                # Estimate the state
                estimated_alpha = np.mean(particles)

            new_output = np.zeros((output_len + 1, N))
            init_con = observations[-1,:]
            new_output[0,:] = init_con[0,:]
            new_val = sg_equation_step(init_con, estimated_alpha)
            new_output[1,:] = new_val[0,:]
            for cur_t in range(1,len(output_grid)-1):
                new_val = sg_equation_step(new_output[cur_t,:], estimated_alpha)
                new_output[cur_t + 1,:] = new_val

            return new_output, [estimated_alpha], [init_alpha]


        else:
            return None, None, None
    
    def get_coeff(self,type,expr):
        x,t = sy.symbols('x t')
        u = sy.Function('u_0')(x,t)

        if type == "heat":

            # Initial diffusion coefficient (alpha) we want to estimate
            try:
                expr = sy.sympify(expr)
                expr = sy.simplify(expr) 
                expr = sy.expand(expr)
                init_alpha = expr.coeff(sy.diff(u,(x,2)))
                return [init_alpha]
            except:
                return None
        elif type == "porous_medium":

            try:
                expr = sy.sympify(expr)
                expr = expr.doit()
                expr = sy.simplify(expr)        #Just so it is in a form which will work with this way of doing it.
                expr = sy.expand(expr)
            except:
                return None
            
            if expr.coeff(u * sy.diff(u,(x,2))) != 0:
                init_m = expr.coeff(u*sy.diff(u,(x,2)))
                mode = 2
                return [mode]
            elif expr.coeff(u**2 * sy.diff(u,(x,2))) != 0:
                init_m = expr.coeff(u**2 * sy.diff(u,(x,2)))
                mode = 3
                return [mode]
            elif expr.coeff(u**3 * sy.diff(u,(x,2))) != 0:
                init_m = expr.coeff(u**3 * sy.diff(u,(x,2)))
                mode = 4
                return [mode]
            else:
                return None
        elif type == "advection":

            try:
                expr = sy.sympify(expr)
                expr = sy.simplify(expr)        #Just so it is in a form which will work with this way of doing it.
                expr = sy.expand(expr)
                init_alpha = expr.coeff(sy.diff(u,x))
                return [init_alpha]
            except:
                return None
        elif type == "kdv":
            try:
                expr = sy.sympify(expr)
                expr = sy.simplify(expr)
                expr = sy.expand(expr)
                init_alpha = expr.coeff(sy.diff(u,(x,3)))
                return [init_alpha]
            except:
                return None
        elif type == "fplanck":
            #this has a lot of terms. Needs more time devoted to it.
            um = 1e-6  # micrometer
            L = 0.1 * um
            c = 5e-21
            try:
                expr = sy.sympify(expr)
                expr = sy.simplify(expr)
                init_alpha = expr.coeff(sy.cos(x * (um/L)) * u)
                init_beta = expr.coeff(sy.sin(x * (um/L)) * sy.diff(u,x))
                init_gamma = expr.coeff(sy.diff(u,(x,2)))
                return [init_beta,init_gamma]
            except:
                return None
        elif type == "diff_logisreact_1D": 
            try:
                expr = sy.sympify(expr)
                init_rho = expr.coeff(u*(1-u))
                init_nu = expr.coeff(sy.diff(u,(x,2)))
                return [init_rho,init_nu]
            except:
                return None
        elif type == "diff_linearreact_1D": 
            try:
                expr = sy.sympify(expr)
                init_rho = expr.coeff(u)
                init_nu = expr.coeff(sy.diff(u,(x,2)))
                return [init_rho,init_nu]
            except:
                return None
        elif type == "diff_bistablereact_1D": 
            # Note: It is possible to add a into this.
            try:
                expr = sy.sympify(expr)
                expr_expanded = sy.expand(expr)
                expr_expanded = sy.simplify(expr_expanded)
                init_rho = -1 * expr_expanded.coeff(u**3)       #rho u(1-u)(u-1) = -rho u**3 + rho(1+a)u**2 - a rho u
                init_a = expr_expanded.coeff(u) / (-init_rho)   # will return 0 if it doesn't exist
                init_nu = expr.coeff(sy.diff(u,(x,2)))
                return [init_rho,init_nu]
            except:
                return None
        elif type == "diff_squarelogisticreact_1D": 
            try:
                expr = sy.sympify(expr)
                init_rho = expr.coeff(((u ** 2) * (1 - u) ** 2))
                init_nu = expr.coeff(sy.diff(u,(x,2)))
                return [init_rho,init_nu]
            except:
                return None
        elif type == "burgers":
            try:
                expr = sy.sympify(expr)
                expr = expr.doit()
                init_k = expr.coeff(u*sy.diff(u,x))
                init_eps = expr.coeff(sy.diff(u,(x,2)))
                return [init_k,init_eps]
            except:
                return None
        elif type == "conservation_linearflux":
            try:
                expr = sy.sympify(expr)
                expr = expr.doit()
                init_k = expr.coeff(sy.diff(u,x))
                init_eps = expr.coeff(sy.diff(u,(x,2)))
                return [init_k,init_eps]
            except:
                return None
        elif type == "conservation_sinflux":
    
            try:
                expr = sy.sympify(expr)
                expr = expr.doit()
                init_k = expr.coeff(sy.cos(u) * sy.diff(u,x))
                init_eps = expr.coeff(sy.diff(u,(x,2)))
                return [init_k,init_eps]
            except:
                return None
        elif type == "conservation_cosflux":
    
            try:
                expr = sy.sympify(expr)
                expr = expr.doit()
                init_k = expr.coeff(sy.sin(u) * sy.diff(u,x))
                init_eps = expr.coeff(sy.diff(u,(x,2)))
                return [init_k,init_eps]
            except:
                return None
        elif type == "conservation_cubicflux":
    
            try:
                expr = sy.sympify(expr)
                expr = expr.doit()
                init_k = expr.coeff(u ** 2 * sy.diff(u,x))
                init_eps = expr.coeff(sy.diff(u,(x,2)))
                return [init_k,init_eps]
            except:
                return None
        elif type == "inviscid_burgers":
    
            try:
                expr = sy.sympify(expr)
                expr = expr.doit()
                init_k = expr.coeff(u*sy.diff(u,x))
                return [init_k]
            except:
                return None
        elif type == "inviscid_conservation_sinflux":
    
            try:
                expr = sy.sympify(expr)
                expr = expr.doit()
                init_k = expr.coeff(sy.cos(u)*sy.diff(u,x))
                return [init_k]
            except:
                return None
        elif type == "inviscid_conservation_cosflux":
    
            try:
                expr = sy.sympify(expr)
                expr = expr.doit()
                init_k = expr.coeff(sy.sin(u)*sy.diff(u,x))
                return [init_k]
            except:
                return None
        elif type == "inviscid_conservation_cubicflux":
    
            try:
                expr = sy.sympify(expr)
                expr = expr.doit()
                init_k = expr.coeff(u ** 2 * sy.diff(u,x))
                return [init_k]
            except:
                return None
        elif type == "cahnhilliard_1D":
            # DOES NOT INFER 6.
            try:
                expr = sy.sympify(expr)
                init_alpha = expr.coeff(sy.diff(u,(x,4)))
                return [init_alpha]
            except:
                return None
        elif type == "wave":

            try:
                expr = sy.sympify(expr)
                expr = sy.simplify(expr)        #Just so it is in a form which will work with this way of doing it.
                expr = sy.expand(expr)
                init_alpha = expr.coeff(sy.diff(u,(x,2)))
                return [init_alpha]
            except:
                return None
        elif type == "Klein_Gordon":
            try:
                expr = sy.sympify(expr)
                init_alpha = expr.coeff(sy.diff(u,(x,2)))
                init_beta = expr.coeff(u)
                return [init_alpha, init_beta]
            except:
                return None
        elif type == "Sine_Gordon":
            try:
                expr = sy.sympify(expr)
                init_alpha = expr.coeff(sy.sin(u))
                return [init_alpha]
            except:
                return None
            
        
            