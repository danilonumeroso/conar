import torch
import math
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from statistics import mean, stdev as std
from utils_execution import compute_tour_cost
from copy import deepcopy
import time

# To replicate COxNAR experiments with BW=1280, uncomment the corresponding line
BEAM_WIDTH = 128
# BEAM_WIDTH = 1280

@jax.jit
def expand_single(beam_vis, beam_last, beam_cost, beam_par, W):
    arranged = jnp.arange(W.shape[0])
    added_cost = W[jnp.resize(beam_last, arranged.shape), arranged]
    new_beam_cost = beam_cost + added_cost # broadcasting (,)+(N,)=(N,)
    new_beam_vis = jnp.repeat(beam_vis[None], arranged.shape[0], axis=0)
    new_beam_vis = new_beam_vis.at[arranged, arranged].set(True)
    new_beam_par = jnp.repeat(beam_par[None], arranged.shape[0], axis=0)
    new_beam_par = new_beam_par.at[arranged, arranged].set(beam_last)
    return new_beam_vis, arranged, new_beam_cost, new_beam_par

expand_multiple = jax.jit(jax.vmap(expand_single, in_axes=(0, 0, 0, 0, 0), out_axes=(0, 0, 0, 0)))

@partial(jax.jit, static_argnames=['beam_width'])
def beam_search_rollout_step(W, beam_width, i, tpl):
    beam_vis, beam_last, beam_cost, beam_par = tpl
    Wrep = jnp.repeat(W[None], beam_vis.shape[0], axis=0)
    beam_vis_rep = beam_vis[:, None, :] & beam_vis[:, :, None]
    Wrep = jnp.where(beam_vis_rep, float('inf'), Wrep)
    new_beam_vis, new_beam_last, new_beam_cost, new_beam_par = expand_multiple(beam_vis, beam_last, beam_cost, beam_par, Wrep)
    beam_vis = new_beam_vis.reshape(-1, new_beam_vis.shape[-1])
    beam_last = new_beam_last.reshape(-1)
    beam_cost = new_beam_cost.reshape(-1)
    beam_par = new_beam_par.reshape(-1, new_beam_par.shape[-1])
    indices = jax.lax.top_k(-beam_cost, min(beam_width, beam_par.shape[0]))[1]
    beam_vis = beam_vis[indices]
    beam_last = beam_last[indices]
    beam_cost = beam_cost[indices]
    beam_par = beam_par[indices]
    return beam_vis, beam_last, beam_cost, beam_par


@partial(jax.jit, static_argnames=['beam_width', 'num_nodes'])
def beam_search_rollout(start_route, W, num_nodes, beam_width):
    snode = start_route.argmax()
    beam_par = jnp.arange(start_route.shape[0])[None, :]
    beam_vis = jnp.array(start_route[None, :], bool)
    beam_last = start_route.argmax()[None]
    beam_cost = jnp.zeros((1,), dtype=float)
    bsrs = partial(beam_search_rollout_step, W, beam_width)
    # The first some steps carry shape changes
    cnt = 0
    while beam_vis.shape[0] < beam_width:
        beam_vis, beam_last, beam_cost, beam_par = bsrs(0, (beam_vis, beam_last, beam_cost, beam_par))
        cnt = cnt + 1



    beam_vis, beam_last, beam_cost, beam_par = jax.lax.fori_loop(cnt, num_nodes-1, bsrs, (beam_vis, beam_last, beam_cost, beam_par))
    best_index = beam_cost.argmin()
    best_par = beam_par[best_index]
    best_par = best_par.at[snode].set(beam_last[best_index])


    return jnp.stack((jnp.arange(best_par.shape[0]), best_par))

vmapped_beam_search_rollout = jax.jit(jax.vmap(beam_search_rollout, in_axes=(0, 0, None, None)), static_argnames=['beam_width', 'num_nodes'])


def beam_search_baseline(data, return_ratio=True):
    num_nodes = data[0].x.shape[0]
    srs = torch.stack(tuple(data[i].start_route for i in range(data.num_samples)))
    eas = torch.stack(tuple(data[i].edge_attr.reshape(num_nodes, num_nodes) for i in range(data.num_samples)))
    tours = vmapped_beam_search_rollout(srs.numpy(), eas.numpy(), num_nodes, BEAM_WIDTH)

    tour_lengths = [compute_tour_cost(np.array(y), x.edge_attr).item() for x, y in zip(data, tours)]
    ratios = [(y/x.optimal_value.item() - 1) for x, y in zip(data, tour_lengths)]

    return (mean(ratios), std(ratios)) if return_ratio else (mean(tour_lengths), std(tour_lengths))
