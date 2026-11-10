"""SMART (CAT-K) as a step-wise latent imagination for the HJ learner.

``SMARTAgentDecoder.inference`` rolls out a whole scene in one call and samples
every agent's token internally. Here the same computation is split into

    encode  -> per-agent feature and next-token logits for the current step
    advance -> apply one token per agent and extend the token history

so that the ego token can be chosen from outside while the other agents keep
being sampled from the model. The split follows ``agent_decoder.inference``
line by line; only the sampling of the ego token differs.

Latent state: the ego's feature after the last attention layer (the input of
``token_predict_head``), dimension ``hidden_dim``.
Actions: the ego's top-K next tokens. Their Q-network feature is the end pose of
the 0.5 s motion token in the ego frame.

This module does not import CAT-K. It takes the already constructed modules,
see ``load_smart`` for building them from a CAT-K checkpoint.

Status: written against the CAT-K sources, not yet executed against a
checkpoint. ``scripts/driving/check_smart_imagination.py`` compares it with
``inference`` and should be run first.
"""

from __future__ import annotations

import sys
from typing import Dict, Iterable, Optional, Sequence

import torch
from torch import Tensor

from ..hj.imagination import Candidates, Imagination
from ..stl.rules import BIG, circle_cover, point_segment_distance

# CAT-K map encoding (src/data_preprocess.py): ``pt_token.pl_type`` indexes
# ["lane", "road_edge", "road_line", "crosswalk"]; ``pt_token.type`` is the
# fine-grained polyline type.
ROAD_EDGE_POLYGON = 1
ROAD_EDGE_BOUNDARY, ROAD_EDGE_MEDIAN = 4, 5


def pad_edge_segments(traj_pos: Tensor, is_edge: Tensor, batch: Tensor, num_graphs: int):
    """Road-edge segments of every scene, padded to a common length.

    Args:
        traj_pos: [n_pl, P, 2] map polyline pieces (CAT-K ``map_save.traj_pos``,
            P = 3 points per piece), is_edge: [n_pl] bool, batch: [n_pl] scene id.
    Returns:
        seg_a, seg_b [num_graphs, M, 2] and mask [num_graphs, M].
    """
    pts = traj_pos[is_edge].float()
    n_seg = pts.shape[1] - 1
    seg_a, seg_b = pts[:, :-1].flatten(0, 1), pts[:, 1:].flatten(0, 1)
    scene = batch[is_edge].long().repeat_interleave(n_seg)
    order = torch.argsort(scene, stable=True)
    scene, seg_a, seg_b = scene[order], seg_a[order], seg_b[order]
    counts = torch.bincount(scene, minlength=num_graphs)
    slot = torch.arange(scene.shape[0], device=scene.device) - (counts.cumsum(0) - counts)[scene]
    m = max(int(counts.max()) if counts.numel() else 0, 1)
    out_a = pts.new_zeros(num_graphs, m, 2)
    out_b = pts.new_zeros(num_graphs, m, 2)
    mask = torch.zeros(num_graphs, m, dtype=torch.bool, device=pts.device)
    out_a[scene, slot], out_b[scene, slot], mask[scene, slot] = seg_a, seg_b, True
    return out_a, out_b, mask


def road_edge_gap(centers: Tensor, radius: Tensor, seg_a: Tensor, seg_b: Tensor,
                  mask: Tensor) -> Tensor:
    """Gap [B] between discs (centers [B, P, 2], radius [B]) and road-edge segments.

    The distance is unsigned: it is negative only while a disc overlaps an edge.
    """
    dist = point_segment_distance(centers, seg_a, seg_b)  # [B, P, M]
    dist = dist.masked_fill(~mask.unsqueeze(1), BIG)
    return dist.amin(dim=(1, 2)) - radius


def angle_between_2d_vectors(ctr_vector: Tensor, nbr_vector: Tensor) -> Tensor:
    return torch.atan2(
        ctr_vector[..., 0] * nbr_vector[..., 1] - ctr_vector[..., 1] * nbr_vector[..., 0],
        (ctr_vector[..., :2] * nbr_vector[..., :2]).sum(dim=-1),
    )


def transform_to_global(pos_local: Tensor, pos_now: Tensor, head_now: Tensor) -> Tensor:
    """Same convention as ``src.smart.utils.rollout.transform_to_global``."""
    cos, sin = head_now.cos(), head_now.sin()
    rot_mat = torch.zeros((head_now.shape[0], 2, 2), device=head_now.device)
    rot_mat[:, 0, 0] = cos
    rot_mat[:, 0, 1] = sin
    rot_mat[:, 1, 0] = -sin
    rot_mat[:, 1, 1] = cos
    return torch.bmm(pos_local, rot_mat) + pos_now.unsqueeze(1)


def token_end_pose(contour: Tensor) -> Tensor:
    """End pose (x, y, heading) of token contours [..., 4, 2] in the local frame.

    Corner order is (left_front, right_front, right_back, left_back).
    """
    center = contour.mean(-2)
    forward = contour[..., 0, :] - contour[..., 3, :]
    return torch.cat([center, torch.atan2(forward[..., 1:2], forward[..., 0:1])], dim=-1)


def load_smart(ckpt_path: str, catk_root: str, device: str = "cuda"):
    """Build the frozen token processor and decoder from a CAT-K checkpoint.

    Uses the ``model_config`` stored in the checkpoint's hyper-parameters, which
    avoids constructing the LightningModule (it needs a live Hydra run).

    Returns:
        (token_processor, decoder) with ``decoder.map_encoder`` and
        ``decoder.agent_encoder``.
    """
    if catk_root not in sys.path:
        sys.path.insert(0, catk_root)
    from src.smart.modules.smart_decoder import SMARTDecoder  # noqa: WPS433
    from src.smart.tokens.token_processor import TokenProcessor  # noqa: WPS433

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_config = ckpt["hyper_parameters"]["model_config"]
    token_processor = TokenProcessor(**model_config.token_processor)
    decoder = SMARTDecoder(
        **model_config.decoder, n_token_agent=token_processor.n_token_agent
    )
    state = ckpt["state_dict"]
    decoder.load_state_dict(
        {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
    )
    token_state = {
        k[len("token_processor."):]: v
        for k, v in state.items()
        if k.startswith("token_processor.")
    }
    if token_state:
        token_processor.load_state_dict(token_state, strict=False)
    token_processor.eval().requires_grad_(False).to(device)
    decoder.eval().requires_grad_(False).to(device)
    return token_processor, decoder


class SmartImagination(Imagination):
    def __init__(
        self,
        token_processor,
        decoder,
        batches: Iterable,
        top_k: int = 16,
        others_top_k: int = 5,
        others_temp: float = 1.0,
        clearance_scale: float = 2.0,
        min_clearance: float = 0.0,
        n_circles: int = 3,
        road_edge: bool = True,
        road_edge_scale: float = 1.0,
        min_road_edge: float = 0.0,
        road_edge_types: Optional[Sequence[int]] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Args:
            token_processor, decoder: frozen CAT-K modules in eval mode.
            batches: iterable of collated ``HeteroData`` batches with exactly
                one ego (``role[:, 0]``) per scene. It is cycled indefinitely.
            top_k: number of ego candidate tokens (the filter's action set).
            others_top_k, others_temp: top-k sampling of the other agents, as in
                CAT-K's ``validation_rollout_sampling``.
            clearance_scale, min_clearance: map the ego's clearance in metres to
                the margin returned by ``true_margin``.
            road_edge: add the road-edge rule to ``true_margin``. The ego body
                is checked against the scene's road-edge polylines at every
                0.1 s pose of the executed token, so an edge crossed inside a
                0.5 s step is not missed.
            road_edge_scale, min_road_edge: as above, for the road-edge gap.
            road_edge_types: fine-grained ``pt_token.type`` values to keep, e.g.
                ``(ROAD_EDGE_BOUNDARY,)`` to ignore medians. None keeps all.
        """
        self.road_edge, self.road_edge_types = road_edge, road_edge_types
        self.road_edge_scale, self.min_road_edge = road_edge_scale, min_road_edge
        self.token_processor = token_processor
        self.map_encoder = decoder.map_encoder
        self.dec = decoder.agent_encoder
        self._batches, self._iter = batches, None
        self.top_k, self.others_top_k, self.others_temp = top_k, others_top_k, others_temp
        self.clearance_scale, self.min_clearance = clearance_scale, min_clearance
        self.n_circles = n_circles
        self.device = device or next(decoder.parameters()).device

        d = self.dec
        self.horizon = d.num_future_steps // d.shift  # 16 tokens of 0.5 s
        self._n_hist = (d.num_historical_steps - 1) // d.shift  # 2 history tokens
        self.latent_dim = d.hidden_dim
        self.action_feat_dim = 4
        self.t = 0

    # ------------------------------------------------------------- interface
    def _next_batch(self):
        if self._iter is None:
            self._iter = iter(self._batches)
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self._batches)
            return next(self._iter)

    @torch.no_grad()
    def reset(self, batch_size: int = 0, data=None) -> Tensor:
        """Start from the next data batch; ``batch_size`` is set by the loader."""
        data = (data if data is not None else self._next_batch()).to(self.device)
        tokenized_map, ta = self.token_processor(data)
        self.ta: Dict[str, Tensor] = ta
        self.map_feature = self.map_encoder(tokenized_map)
        d, n_hist = self.dec, self._n_hist

        self.pos_a = ta["gt_pos"][:, :n_hist].clone()
        self.head_a = ta["gt_heading"][:, :n_hist].clone()
        self.head_vector_a = torch.stack([self.head_a.cos(), self.head_a.sin()], dim=-1)
        (
            self.feat_a,
            self.agent_token_emb,
            self._emb_veh,
            self._emb_ped,
            self._emb_cyc,
            self._veh_mask,
            self._ped_mask,
            self._cyc_mask,
            self._categorical_embs,
        ) = d.agent_token_embedding(
            agent_token_index=ta["gt_idx"][:, :n_hist],
            trajectory_token_veh=ta["trajectory_token_veh"],
            trajectory_token_ped=ta["trajectory_token_ped"],
            trajectory_token_cyc=ta["trajectory_token_cyc"],
            pos_a=self.pos_a,
            head_vector_a=self.head_vector_a,
            agent_type=ta["type"],
            agent_shape=ta["shape"],
            inference=True,
        )
        self.pred_valid = ta["valid_mask"].clone()
        self.pred_idx = ta["gt_idx"].clone()
        self._feat_a_t_dict: Dict[int, Tensor] = {}

        ego_mask = ta["ego_mask"] if "ego_mask" in ta else data["agent"]["role"][:, 0]
        self.ego = ego_mask.nonzero().squeeze(-1)
        if self.ego.numel() != ta["num_graphs"]:
            raise ValueError("expected exactly one ego agent per scene")
        self.t = 0
        self._setup_road_edges(data)
        self._encode()
        return self.latent()

    def _setup_road_edges(self, data) -> None:
        self._ego_sweep = None  # [B, S, 4, 2] contours of the last executed ego token
        self._edges = None
        n_scene = self.ego.numel()
        self.edge_active = torch.zeros(n_scene, dtype=torch.bool, device=self.pos_a.device)
        if not self.road_edge:
            return
        pt = data["pt_token"]
        is_edge = pt["pl_type"].long() == ROAD_EDGE_POLYGON
        if self.road_edge_types is not None:
            keep = torch.as_tensor(list(self.road_edge_types), device=is_edge.device)
            is_edge &= torch.isin(pt["type"].long(), keep)
        self._edges = pad_edge_segments(
            data["map_save"]["traj_pos"], is_edge, pt["batch"], int(self.ta["num_graphs"])
        )
        # A logged start that already touches an edge (curb parking, map noise)
        # makes the rule unsatisfiable; it is switched off for that scene.
        self.edge_active = torch.ones_like(self.edge_active)
        self.edge_active = self.ego_road_edge_gap() >= self.min_road_edge

    @torch.no_grad()
    def candidates(self) -> Candidates:
        logits = self.logits[self.ego]  # [B, n_token]
        top_logits, top_idx = torch.topk(logits, self.top_k, dim=-1)
        contour = self.ta["token_traj"][self.ego]  # [B, n_token, 4, 2], ego frame
        index = top_idx[..., None, None].expand(-1, -1, 4, 2)
        pose = token_end_pose(contour.gather(1, index))  # [B, K, 3]
        feat = torch.stack(
            [pose[..., 0] / 10.0, pose[..., 1] / 2.0, pose[..., 2].cos(), pose[..., 2].sin()],
            dim=-1,
        )
        self._cand = Candidates(idx=top_idx, feat=feat, logp=torch.log_softmax(top_logits, -1))
        return self._cand

    @torch.no_grad()
    def step(self, slot: Tensor) -> Tensor:
        """Apply the ego candidate ``slot`` [B]; other agents are sampled."""
        if self.t >= self.horizon:
            raise RuntimeError("imagination horizon exhausted, call reset()")
        ego_token = self._cand.idx.gather(1, slot.view(-1, 1)).squeeze(1)
        next_token_idx = self._sample_tokens(self.logits)
        next_token_idx[self.ego] = ego_token
        self._advance(next_token_idx)
        self.t += 1
        self._encode()
        return self.latent()

    @torch.no_grad()
    def true_margin(self) -> Tensor:
        """Margin [B] of the ego at the current step: min over the active rules."""
        margin = ((self.ego_clearance() - self.min_clearance) / self.clearance_scale).clamp(-1, 1)
        if self._edges is not None:
            edge = (self.ego_road_edge_gap() - self.min_road_edge) / self.road_edge_scale
            margin = torch.minimum(margin, edge.clamp(-1, 1))
        return margin

    # ------------------------------------------------------------ accessors
    def latent(self) -> Tensor:
        return self.feat_now[self.ego]

    def ego_valid(self) -> Tensor:
        return self.pred_valid[self.ego, self.pos_a.shape[1] - 1]

    def ego_pose(self) -> Tensor:
        """[B, 3] global x, y, heading of the ego at the current step."""
        return torch.cat(
            [self.pos_a[self.ego, -1], self.head_a[self.ego, -1].unsqueeze(-1)], dim=-1
        )

    def ego_clearance(self) -> Tensor:
        """Gap in metres between each ego and the nearest valid agent in its scene."""
        ta = self.ta
        pos, head = self.pos_a[:, -1], self.head_a[:, -1]
        valid = self.pred_valid[:, self.pos_a.shape[1] - 1]
        # token_agent_shape is (width, length); circle_cover expects (length, width)
        size = ta["token_agent_shape"].flip(-1)
        centers, radius = circle_cover(pos, head, size, self.n_circles)  # [A, C, 2], [A]
        n_agent, n_circ = centers.shape[:2]
        dist = torch.cdist(centers[self.ego], centers.flatten(0, 1).unsqueeze(0).expand(
            self.ego.numel(), -1, -1))  # [B, C, A*C]
        dist = dist.view(-1, n_circ, n_agent, n_circ).amin(dim=(1, 3))  # [B, A]
        gap = dist - radius[self.ego].unsqueeze(1) - radius.unsqueeze(0)
        same_scene = ta["batch"][self.ego].unsqueeze(1) == ta["batch"].unsqueeze(0)
        is_self = self.ego.unsqueeze(1) == torch.arange(n_agent, device=pos.device).unsqueeze(0)
        gap = gap.masked_fill(~same_scene | is_self | ~valid.unsqueeze(0), BIG)
        return gap.amin(-1)

    def ego_road_edge_gap(self) -> Tensor:
        """Gap in metres between each ego and the nearest road edge of its scene.

        Minimum over the 0.1 s poses of the token executed in the last step
        (the current pose only, right after ``reset``). ``BIG`` where the rule
        is off.
        """
        n_scene = self.ego.numel()
        if self._edges is None:
            return self.pos_a.new_full((n_scene,), BIG)
        if self._ego_sweep is None:
            pos = self.pos_a[self.ego, -1].unsqueeze(1)  # [B, 1, 2]
            head = self.head_a[self.ego, -1].unsqueeze(1)
        else:
            pos = self._ego_sweep.mean(-2)  # [B, S, 2]
            forward = self._ego_sweep[..., 0, :] - self._ego_sweep[..., 3, :]
            head = torch.atan2(forward[..., 1], forward[..., 0])
        size = self.ta["token_agent_shape"][self.ego].flip(-1).unsqueeze(1)  # (length, width)
        centers, radius = circle_cover(pos, head, size, self.n_circles)  # [B, S, C, 2], [B, S]
        scene = self.ta["batch"][self.ego]
        seg_a, seg_b, mask = (x[scene] for x in self._edges)
        gap = road_edge_gap(centers.flatten(1, 2), radius[:, 0], seg_a, seg_b, mask)
        return torch.where(self.edge_active, gap, torch.full_like(gap, BIG))

    # -------------------------------------------------------------- internals
    def _sample_tokens(self, logits: Tensor) -> Tensor:
        top_logits, top_idx = torch.topk(logits, self.others_top_k, dim=-1, sorted=False)
        sample = torch.distributions.Categorical(logits=top_logits / self.others_temp).sample()
        return top_idx.gather(1, sample.unsqueeze(1)).squeeze(1)

    def _encode(self) -> None:
        """Attention pass for the current step (first half of the CAT-K loop body)."""
        d, ta, mf = self.dec, self.ta, self.map_feature
        t = self.t
        t_now = self._n_hist - 1 + t
        n_step = t_now + 1
        n_agent = self.pos_a.shape[0]
        pos_a, head_a, head_vector_a = self.pos_a, self.head_a, self.head_vector_a
        pred_valid = self.pred_valid

        if t == 0:
            hist_step = self._n_hist
            batch_s = torch.cat(
                [ta["batch"] + ta["num_graphs"] * i for i in range(hist_step)], dim=0
            )
            batch_pl = torch.cat(
                [mf["batch"] + ta["num_graphs"] * i for i in range(hist_step)], dim=0
            )
            inference_mask = pred_valid[:, :n_step]
            edge_index_t, r_t = d.build_temporal_edge(
                pos_a=pos_a, head_a=head_a, head_vector_a=head_vector_a,
                mask=pred_valid[:, :n_step],
            )
        else:
            hist_step = 1
            batch_s = ta["batch"]
            batch_pl = mf["batch"]
            inference_mask = pred_valid[:, :n_step].clone()
            inference_mask[:, :-1] = False
            edge_index_t, r_t = d.build_temporal_edge(
                pos_a=pos_a, head_a=head_a, head_vector_a=head_vector_a,
                mask=pred_valid[:, :n_step], inference_mask=inference_mask,
            )
            edge_index_t[1] = (edge_index_t[1] + 1) // n_step - 1

        edge_index_pl2a, r_pl2a = d.build_map2agent_edge(
            pos_pl=mf["position"],
            orient_pl=mf["orientation"],
            pos_a=pos_a[:, -hist_step:],
            head_a=head_a[:, -hist_step:],
            head_vector_a=head_vector_a[:, -hist_step:],
            mask=inference_mask[:, -hist_step:],
            batch_s=batch_s,
            batch_pl=batch_pl,
        )
        edge_index_a2a, r_a2a = d.build_interaction_edge(
            pos_a=pos_a[:, -hist_step:],
            head_a=head_a[:, -hist_step:],
            head_vector_a=head_vector_a[:, -hist_step:],
            batch_s=batch_s,
            mask=inference_mask[:, -hist_step:],
        )

        feat_a_now = None
        for i in range(d.num_layers):
            feat_temporal = self.feat_a if i == 0 else self._feat_a_t_dict[i]
            if t == 0:
                feat_temporal = d.t_attn_layers[i](
                    feat_temporal.flatten(0, 1), r_t, edge_index_t
                ).view(n_agent, n_step, -1)
                feat_temporal = feat_temporal.transpose(0, 1).flatten(0, 1)
                feat_map = (
                    mf["pt_token"].unsqueeze(0).expand(hist_step, -1, -1).flatten(0, 1)
                )
                feat_temporal = d.pt2a_attn_layers[i](
                    (feat_map, feat_temporal), r_pl2a, edge_index_pl2a
                )
                feat_temporal = d.a2a_attn_layers[i](feat_temporal, r_a2a, edge_index_a2a)
                feat_temporal = feat_temporal.view(n_step, n_agent, -1).transpose(0, 1)
                feat_a_now = feat_temporal[:, -1]
                if i + 1 < d.num_layers:
                    self._feat_a_t_dict[i + 1] = feat_temporal
            else:
                feat_a_now = d.t_attn_layers[i](
                    (feat_temporal.flatten(0, 1), feat_temporal[:, -1]), r_t, edge_index_t
                )
                feat_a_now = d.pt2a_attn_layers[i](
                    (mf["pt_token"], feat_a_now), r_pl2a, edge_index_pl2a
                )
                feat_a_now = d.a2a_attn_layers[i](feat_a_now, r_a2a, edge_index_a2a)
                if i + 1 < d.num_layers:
                    self._feat_a_t_dict[i + 1] = torch.cat(
                        (self._feat_a_t_dict[i + 1], feat_a_now.unsqueeze(1)), dim=1
                    )

        self.feat_now = feat_a_now  # [n_agent, hidden_dim]
        self.logits = d.token_predict_head(feat_a_now)  # [n_agent, n_token]

    def _advance(self, next_token_idx: Tensor) -> None:
        """Apply one token per agent (second half of the CAT-K loop body)."""
        d, ta = self.dec, self.ta
        t_now = self._n_hist - 1 + self.t
        n_step = t_now + 1
        range_a = torch.arange(next_token_idx.shape[0], device=next_token_idx.device)

        next_token_traj_all = ta["token_traj_all"][range_a, next_token_idx]  # [A, 6, 4, 2]
        token_traj_global = transform_to_global(
            next_token_traj_all.flatten(1, 2), self.pos_a[:, t_now], self.head_a[:, t_now]
        ).view(*next_token_traj_all.shape)
        self._ego_sweep = token_traj_global[self.ego]

        pos_a_next = token_traj_global[:, -1].mean(dim=1)
        diff_xy_next = token_traj_global[:, -1, 0] - token_traj_global[:, -1, 3]
        head_a_next = torch.arctan2(diff_xy_next[:, 1], diff_xy_next[:, 0])
        self.pred_idx[:, n_step] = next_token_idx
        self.pred_valid[:, n_step] = self.pred_valid[:, t_now]

        self.pos_a = torch.cat([self.pos_a, pos_a_next.unsqueeze(1)], dim=1)
        self.head_a = torch.cat([self.head_a, head_a_next.unsqueeze(1)], dim=1)
        head_vector_a_next = torch.stack([head_a_next.cos(), head_a_next.sin()], dim=-1)
        self.head_vector_a = torch.cat(
            [self.head_vector_a, head_vector_a_next.unsqueeze(1)], dim=1
        )

        agent_token_emb_next = torch.zeros_like(self.agent_token_emb[:, 0])
        agent_token_emb_next[self._veh_mask] = self._emb_veh[next_token_idx[self._veh_mask]]
        agent_token_emb_next[self._ped_mask] = self._emb_ped[next_token_idx[self._ped_mask]]
        agent_token_emb_next[self._cyc_mask] = self._emb_cyc[next_token_idx[self._cyc_mask]]

        motion_vector_a = self.pos_a[:, -1] - self.pos_a[:, -2]
        x_a = torch.stack(
            [
                torch.norm(motion_vector_a, p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=self.head_vector_a[:, -1], nbr_vector=motion_vector_a
                ),
            ],
            dim=-1,
        )
        x_a = d.x_a_emb(continuous_inputs=x_a, categorical_embs=self._categorical_embs)
        feat_a_next = torch.cat((agent_token_emb_next, x_a), dim=-1).unsqueeze(1)
        feat_a_next = d.fusion_emb(feat_a_next)
        self.feat_a = torch.cat([self.feat_a, feat_a_next], dim=1)
