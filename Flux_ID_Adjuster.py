import torch
import torch.nn.functional as F
import numpy as np
import math

class FluxIDAutoAdjuster:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "layout_blocks": ("STRING", {
                    "default": "3-7", 
                    "tooltip": "Double Blocks. Establishes global layout and integrates the subject into the background safely. Anchors run at 25% strength to prevent Janus artifacts."
                }),
                "identity_blocks": ("STRING", {
                    "default": "8-19",
                    "tooltip": "Single Blocks. Synthesizes high-frequency identity, facial micro-geometry, and photorealism. Hard snapping runs at 100% strength here."
                }),
                "saliency_scan_blocks": ("STRING", {
                    "default": "6-23", 
                    "tooltip": "The specific blocks where the script analyzes the canvas to dynamically isolate the face from the background during Step 1."
                }),
                "photorealistic_smoothing": ("BOOLEAN", {
                    "default": True, 
                    "tooltip": "ON = Mathematically deletes static/noise for photorealistic skin. OFF = Transfers raw reference textures like brushstrokes or film grain."
                }),
                "total_sampling_steps": ("INT", {
                    "default": 4, "min": 1, "max": 100, 
                    "tooltip": "MUST match your KSampler steps! Syncs the internal curve to ensure style and geometry inject at the correct anatomical phases."
                }),
                "boost_fade_curve": ([
                    "Linear", "Smooth", "Ease-In", "Ease-Out"
                ], {"default": "Ease-In", "tooltip": "Controls how the identity injection fades out over time. Ease-In is highly recommended for preserving late-stage text styling."}),
                "identity_strength": ("FLOAT", {
                    "default": 1.50, "min": 0.0, "max": 3.0, "step": 0.05,
                    "tooltip": "Primary multiplier for facial likeness. Higher values force a stronger resemblance but may stiffen the pose."
                }),
                "background_text_strength": ("FLOAT", {
                    "default": 0.60, "min": 0.0, "max": 10.0, "step": 0.05,
                    "tooltip": "Amplifies the text prompt to construct the background. Automatically muted during delicate anatomy phases (Step 2)."
                }),
                "dynamic_text_balancing": ("BOOLEAN", {
                    "default": True, 
                    "tooltip": "ON = Automatically throttles text strength when the face is struggling to form, preventing the prompt from crushing the identity."
                }),
                "target_likeness_metric": ("FLOAT", {
                    "default": 0.35, "min": -1.0, "max": 1.0, "step": 0.01,
                    "tooltip": "The raw cosine goal. 0.35 is mathematically ideal for Flux. Pushing higher forces aggressive pulling; lower allows more stylistic freedom."
                }),
                "soft_blend_k": ("INT", {
                    "default": 1, "min": 1, "max": 10, "step": 1,
                    "tooltip": "Averages the top K matches for smooth skin and soft regions. Committed anchors ignore this and snap exactly to K=1 for sharp eyes/lips."
                }),
                "face_isolation_strictness": ("FLOAT", {
                    "default": 1.00, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "Top % of tokens to lock as the 'Face'. 1.0 pulls the full body/background. Lower values (e.g., 0.35) isolate just the face for hybrids."
                }),
                "confidence_gate": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Minimum confidence required to soft-pull a token. Higher values prevent artifacts but might freeze rendering. 0.0 disables the gate."
                }),
                "hard_anchor_margin": ("FLOAT", {
                    "default": 0.06, "min": 0.0, "max": 0.20, "step": 0.01,
                    "tooltip": "The margin difference required to permanently lock a token (like a pupil). Lower means anchors lock faster; higher requires absolute certainty."
                }),
                "contrast_and_texture_floor": ("FLOAT", {
                    "default": 0.18, "min": -1.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Base similarity cutoff. Increasing this boosts visual contrast and removes noise, but pushing too high creates waxy, over-smoothed skin."
                }),
            },
            "optional": {
                "subject_mask": ("MASK", {
                    "tooltip": "Optional. Restricts the Saliency Radar to only consider tokens inside this drawn area."
                })
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "Advanced/Model Bending"

    def apply(self, model, layout_blocks, identity_blocks, saliency_scan_blocks, photorealistic_smoothing, total_sampling_steps, boost_fade_curve, identity_strength, background_text_strength, dynamic_text_balancing, target_likeness_metric, soft_blend_k, face_isolation_strictness, confidence_gate, hard_anchor_margin, contrast_and_texture_floor, subject_mask=None):
        m = model.clone()
        
        _src_mask = None
        if subject_mask is not None:
            mk = subject_mask
            if mk.dim() == 4: mk = mk[0, 0]
            elif mk.dim() == 3: mk = mk[0]
            _src_mask = mk.detach().float()
            
        _idx_cache = {}
        
        def _get_mask_1d(count, device):
            if _src_mask is None: return None
            if count in _idx_cache: return _idx_cache[count].to(device)
            
            mh, mw = _src_mask.shape[-2:]
            target = mh / max(mw, 1)
            best = (1, count)
            best_err = float("inf")
            limit = int(count ** 0.5) + 3
            for h in range(1, limit):
                if count % h == 0:
                    w = count // h
                    for hh, ww in ((h, w), (w, h)):
                        err = abs(hh / max(ww, 1) - target)
                        if err < best_err:
                            best_err = err
                            best = (hh, ww)
                            
            pooled = F.adaptive_avg_pool2d(_src_mask[None, None], best).view(-1)
            mask_1d = pooled >= 0.5
            _idx_cache[count] = mask_1d
            return mask_1d.to(device)

        if "flux_delta_state" not in m.model_options:            
            m.model_options["flux_delta_state"] = {
                "step_counter": 0,
                "last_ts": -1.0,
                "ref_hits_master": None,
                "step_hits_accum": None,
                "ref_mask": None,
                "last_sim": 0.0,
                "prev_sim_for_vel": 0.0,
                "commit_assign": {},
                "commit_hits": {},
                "expected_steps": total_sampling_steps,
                "persistent_life": None,
                "persistent_anchors": None,
                "heatmap_accum": None,
                "ref_accum": None,
                "raw_sim_accum": None,
                "block_history": {},
                "H": None,
                "W": None
            }

        def parse_to_set(s):
            if not s.strip(): return set()
            blocks = set()
            for part in s.split(','):
                part = part.strip()
                if not part: continue
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    blocks.update(range(start, end + 1))
                else: blocks.add(int(part.strip()))
            return blocks

        d_set = parse_to_set(layout_blocks)
        s_set = parse_to_set(identity_blocks)
        r_set = parse_to_set(saliency_scan_blocks)
        prev_wrapper = m.model_options.get("model_function_wrapper", None)

        def unet_wrapper(apply_model, kwargs):
            state = m.model_options["flux_delta_state"]
            ts_val = kwargs["timestep"][0].item() if torch.is_tensor(kwargs["timestep"]) else kwargs["timestep"]
            
            if state["last_ts"] == -1.0 or ts_val > state["last_ts"]: 
                state["step_counter"] = 1
                state["ref_hits_master"] = None
                state["step_hits_accum"] = None
                state["ref_mask"] = None
                state["last_sim"] = 0.0
                state["prev_sim_for_vel"] = 0.0 
                state["commit_assign"] = {}
                state["commit_hits"] = {}
                state["expected_steps"] = total_sampling_steps
                state["persistent_life"] = None
                state["persistent_anchors"] = None

            elif ts_val < state["last_ts"]: 
                if state.get("step_hits_accum") is not None:
                    if state.get("ref_hits_master") is None:
                        state["ref_hits_master"] = state["step_hits_accum"].clone()
                    else:
                        master = state["ref_hits_master"]
                        curr = state["step_hits_accum"]
                        
                        dynamic_clamp = max(0.02, 0.07 - 0.02 * state["step_counter"])
                        master_norm = master / master.abs().mean(dim=-1, keepdim=True).clamp(min=dynamic_clamp)
                        curr_norm = curr / curr.abs().mean(dim=-1, keepdim=True).clamp(min=dynamic_clamp)
                        
                        state["ref_hits_master"] = (master_norm * 0.70) + (curr_norm * 0.30)
                
                state["step_hits_accum"] = None
                state["ref_mask"] = None 
                
                if state.get("persistent_life") is not None:
                    state["persistent_life"] -= 1
                    state["persistent_life"] = torch.clamp(state["persistent_life"], min=0)
                    state["persistent_anchors"] = state["persistent_life"] > 0
                
                state["step_counter"] += 1

            state["last_ts"] = ts_val

            input_tensor = kwargs.get("input", None)
            if input_tensor is not None:
                state["H"] = input_tensor.shape[2]
                state["W"] = input_tensor.shape[3]

            if prev_wrapper is not None:
                return prev_wrapper(apply_model, kwargs)
            return apply_model(kwargs["input"], kwargs["timestep"], **kwargs.get("c", kwargs))

        def output_patch(attn, extra_options):
            block_type = extra_options.get("block_type", "double")
            block_idx = extra_options.get("block_index", -1)
            state = m.model_options["flux_delta_state"]
            current_step = max(1, state["step_counter"])
            
            is_d_target = (block_type == "double" and block_idx in d_set)
            is_s_target = (block_type == "single" and block_idx in s_set)
            is_radar_block = (block_idx in r_set)

            if not (is_d_target or is_s_target or is_radar_block):
                return attn

            img_slice = extra_options.get("img_slice", None)
            ref_tokens_list = extra_options.get("reference_image_num_tokens", [])
            
            if img_slice is None or not ref_tokens_list:
                return attn

            txt_len = int(img_slice[0])
            total_seq = int(img_slice[1])
            total_ref = sum(ref_tokens_list)
            
            gen_start = txt_len
            gen_end = total_seq - total_ref
            ref_start = total_seq - total_ref
            
            if gen_end <= gen_start or ref_start >= total_seq:
                return attn

            gen_f = attn[:, gen_start:gen_end].float()
            ref_f = attn[:, ref_start:total_seq].float()

            with torch.no_grad():
                gen_norm = F.normalize(gen_f, dim=-1)
                ref_norm = F.normalize(ref_f, dim=-1)
                sim_matrix = torch.bmm(gen_norm, ref_norm.transpose(1, 2)) 
                B, G, R = sim_matrix.shape
                neg_inf = torch.finfo(sim_matrix.dtype).min
                
                margin_conf = torch.zeros((B, G), device=sim_matrix.device, dtype=sim_matrix.dtype)
                safe_weight = torch.zeros((B, G, 1), device=sim_matrix.device, dtype=sim_matrix.dtype)

                expected = state.get("expected_steps", 4)
                is_radar_active = is_radar_block and (current_step <= 2)

                user_mask = _get_mask_1d(R, sim_matrix.device)
                if user_mask is not None:
                    mask_exp = user_mask.unsqueeze(0).unsqueeze(1).expand(B, G, -1)
                    masked_sim_radar = torch.where(mask_exp, sim_matrix, torch.tensor(neg_inf, device=sim_matrix.device, dtype=sim_matrix.dtype))
                else:
                    masked_sim_radar = sim_matrix

                if is_radar_active:
                    def get_causal_hits(matrix):
                        topk_v, topk_i = torch.topk(matrix, k=4, dim=-1)
                        probs = torch.softmax(topk_v * 5.0, dim=-1)
                        local_entropy = -(probs * torch.log(probs + 1e-6)).sum(dim=-1)
                        causal_weight = (1.0 - local_entropy / math.log(4)).clamp(min=0.0)
                        
                        hits = torch.zeros((B, R), device=matrix.device).scatter_add_(1, topk_i[..., 0], causal_weight)
                        return hits
                        
                    global_hits = get_causal_hits(sim_matrix)
                    face_hits = get_causal_hits(masked_sim_radar)
                    
                    ratio = face_hits / (global_hits + 1e-4)
                    identity_gain_hits = torch.log1p(ratio)
                    
                    dynamic_clamp = max(0.02, 0.07 - 0.02 * current_step)
                    identity_gain_hits = identity_gain_hits / identity_gain_hits.abs().mean(dim=-1, keepdim=True).clamp(min=dynamic_clamp)
                    
                    if state.get("step_hits_accum") is None:
                        state["step_hits_accum"] = identity_gain_hits
                    else:
                        # Bulletproof shape validation
                        if state["step_hits_accum"].shape != identity_gain_hits.shape:
                            state["step_hits_accum"] = identity_gain_hits
                        else:
                            state["step_hits_accum"] += identity_gain_hits

                if current_step == 1:
                    if not (is_d_target or is_s_target): 
                        return attn
                    
                    if block_type == "double" and user_mask is not None:
                        soft_subject_mask = F.avg_pool1d(user_mask.float().view(1, 1, -1), kernel_size=17, stride=1, padding=8).squeeze() > 0.05
                        mask_exp_soft = soft_subject_mask.unsqueeze(0).unsqueeze(1).expand(B, G, -1)
                        masked_sim_matrix = torch.where(mask_exp_soft, sim_matrix, sim_matrix * 0.15)
                    elif block_type == "double":
                        masked_sim_matrix = sim_matrix
                    else:
                        masked_sim_matrix = masked_sim_radar if is_radar_active else sim_matrix
                        
                    best_sim, best_idx = masked_sim_matrix.max(dim=-1)
                    sim = state["last_sim"] if state["last_sim"] != 0.0 else best_sim.mean().item()
                    valid_pull_mask = torch.zeros_like(best_sim, dtype=torch.bool) 
                    is_committed = torch.zeros_like(best_sim, dtype=torch.bool)
                    
                else:
                    if not (is_d_target or is_s_target): 
                        return attn

                    if state.get("ref_mask") is None:
                        total_hits = state.get("ref_hits_master")
                        if total_hits is None: total_hits = torch.zeros((B, R), device=sim_matrix.device)
                        
                        mask_size = user_mask.sum().item() if user_mask is not None else R
                        k_tokens = max(1, int(mask_size * face_isolation_strictness))
                        _, top_indices = (torch.where(user_mask.unsqueeze(0), total_hits, torch.zeros_like(total_hits)) if user_mask is not None else total_hits).topk(k_tokens, dim=-1)
                        mask = torch.zeros_like(total_hits, dtype=torch.bool).scatter_(1, top_indices, True)
                        
                        if state.get("persistent_anchors") is not None:
                            safe_anchors = state["persistent_anchors"]
                            if user_mask is not None:
                                safe_anchors = safe_anchors & user_mask.unsqueeze(0)
                            mask = mask | safe_anchors
                            
                        state["ref_mask"] = mask
                        
                    r_mask = state["ref_mask"].unsqueeze(1).expand(-1, G, -1)
                    masked_sim_matrix = torch.where(r_mask, sim_matrix, torch.tensor(neg_inf, device=sim_matrix.device, dtype=sim_matrix.dtype))
                    
                    top2 = torch.topk(masked_sim_matrix, k=2, dim=-1).values
                    best_sim = top2[..., 0]
                    margin = (best_sim - top2[..., 1]).clamp(min=0)
                    best_idx = masked_sim_matrix.max(dim=-1).indices
                    
                    anchor_connected = torch.gather(state["ref_mask"], 1, best_idx)
                    valid_pull_mask = (best_sim > contrast_and_texture_floor) & anchor_connected

                    cache_key = f"{block_type}_{block_idx}"
                    prev_assign = state["commit_assign"].get(cache_key)
                    prev_hits = state["commit_hits"].get(cache_key)
                    
                    if prev_assign is None or prev_assign.shape != best_idx.shape:
                        hits = torch.ones_like(best_idx, dtype=torch.int16)
                    else:
                        if prev_hits is None: prev_hits = torch.zeros_like(best_idx, dtype=torch.int16)
                        is_neighborhood = torch.abs(prev_assign - best_idx) <= 2 
                        hits = torch.where(is_neighborhood & (margin > hard_anchor_margin), (prev_hits + 1).clamp(max=100), torch.zeros_like(prev_hits))
                        
                    state["commit_assign"][cache_key] = best_idx.detach()
                    state["commit_hits"][cache_key] = hits.detach()
                    
                    is_committed = (hits >= 3) & (margin > hard_anchor_margin) & valid_pull_mask
                    margin_conf = torch.sigmoid((margin - 0.08) * 12.0)
                    
                    strong_commit = is_committed & (margin_conf > 0.75) & (best_sim > 0.35)
                    
                    if strong_commit.any():
                        committed_ref_idx = best_idx[strong_commit].unsqueeze(0)
                        if state.get("persistent_life") is None:
                            state["persistent_life"] = torch.zeros((B, R), dtype=torch.int8, device=sim_matrix.device)
                        
                        new_life = torch.full_like(committed_ref_idx, 3, dtype=torch.int8)
                        state["persistent_life"].scatter_reduce_(1, committed_ref_idx, new_life, reduce="amax", include_self=True)
                        state["persistent_anchors"] = state["persistent_life"] > 0

                    weighted = best_sim * margin_conf
                    valid_sims = weighted[valid_pull_mask]
                    sim = valid_sims.mean().item() if valid_sims.numel() > 0 else 0.0
                    state["last_sim"] = sim

            gap = target_likeness_metric - sim 
            prev_sim = state.get("prev_sim_for_vel", sim)
            sim_velocity = sim - prev_sim
            state["prev_sim_for_vel"] = sim

            progress = min(1.0, (current_step - 1) / max(state["expected_steps"] - 1, 1))
            if boost_fade_curve == "Linear": f = progress
            elif boost_fade_curve == "Smooth": f = (1.0 - np.cos(progress * np.pi)) / 2.0
            elif boost_fade_curve == "Ease-In": f = progress ** 2
            elif boost_fade_curve == "Ease-Out": f = 1.0 - (1.0 - progress) ** 2
            else: f = progress
            
            step_attenuator = 1.0 - (f * 0.45)

            boost_r, boost_t = 0.0, 0.0

            if current_step > 1:
                if gap > 0 and identity_strength > 0.0:
                    active_scale = identity_strength
                    if block_type == "double":
                        active_scale *= 0.15
                        
                    drift_mult = 1.0
                    if sim_velocity < 0.0:
                        drift_mult = 1.0 + (abs(sim_velocity) * 15.0)
                    boost_r = min(10.0, active_scale * gap * step_attenuator * drift_mult)
                    
                elif gap < 0 and background_text_strength > 0.0:
                    boost_t = min(10.0, background_text_strength * abs(gap))
                
            if boost_r > 0.0 or boost_t > 0.0:
                attn_out = attn.clone()
                
                if boost_r > 0.0:
                    topk_sims, topk_idx = torch.topk(masked_sim_matrix, k=soft_blend_k, dim=-1)
                    
                    temp = 0.25 
                    attn_w = F.softmax(topk_sims / temp, dim=-1)
                    B_t, G_t, k_val = topk_idx.shape
                    
                    flat_idx_soft = topk_idx.reshape(B_t, G_t * k_val).unsqueeze(-1).expand(-1, -1, ref_f.shape[-1])
                    flat_ref_soft = torch.gather(ref_f, 1, flat_idx_soft)
                    topk_ref_features = flat_ref_soft.reshape(B_t, G_t, k_val, ref_f.shape[-1])
                    soft_ref = (attn_w.unsqueeze(-1) * topk_ref_features).sum(dim=2)

                    flat_idx_hard = best_idx.unsqueeze(-1).expand(-1, -1, ref_f.shape[-1])
                    hard_ref = torch.gather(ref_f, 1, flat_idx_hard)

                    confidence = torch.sigmoid((best_sim - contrast_and_texture_floor) * 8.0)
                    surgical_mask = (valid_pull_mask & (margin_conf > confidence_gate)).to(attn.dtype).unsqueeze(-1)
                    raw_pull_weight = boost_r * confidence.unsqueeze(-1)
                    safe_weight = torch.clamp(raw_pull_weight, max=1.0)
                    uncommitted_delta = (soft_ref - gen_f) * safe_weight * surgical_mask

                    committed_weight = torch.clamp(torch.tensor(boost_r, device=gen_f.device), max=2.5)
                    committed_delta = (hard_ref - gen_f) * committed_weight
                    
                    if photorealistic_smoothing and state.get("H") is not None and state.get("W") is not None:
                        B_d, T_d, D_d = uncommitted_delta.shape
                        s_h, s_w = state["H"], state["W"]
                        p_h, p_w = None, None
                        
                        if T_d == s_h * s_w: p_h, p_w = s_h, s_w
                        elif T_d == (s_h // 2) * (s_w // 2): p_h, p_w = s_h // 2, s_w // 2
                        elif T_d == (s_h // 4) * (s_w // 4): p_h, p_w = s_h // 4, s_w // 4
                        else:
                            side = int(math.sqrt(T_d))
                            if side * side == T_d: p_h, p_w = side, side
                            
                        if p_h is not None and p_w is not None:
                            spatial_delta = uncommitted_delta.view(B_d, p_h, p_w, D_d).permute(0, 3, 1, 2).float()
                            freq = torch.fft.rfft2(spatial_delta, dim=(-2, -1))
                            
                            u = torch.fft.fftfreq(p_h, device=uncommitted_delta.device)
                            v = torch.fft.rfftfreq(p_w, device=uncommitted_delta.device)
                            U, V = torch.meshgrid(u, v, indexing='ij')
                            R_rad = torch.sqrt(U**2 + V**2) 
                            
                            cutoff, taper_width = 0.40, 0.10
                            R_scaled = ((R_rad - (cutoff - taper_width)) / taper_width).clamp(0.0, 1.0)
                            fft_mask = 0.5 * (1.0 + torch.cos(math.pi * R_scaled))
                            freq = freq * fft_mask.unsqueeze(0).unsqueeze(0)
                            
                            spatial_delta = torch.fft.irfft2(freq, s=(p_h, p_w), dim=(-2, -1))
                            uncommitted_delta = spatial_delta.permute(0, 2, 3, 1).reshape(B_d, T_d, D_d).to(uncommitted_delta.dtype)

                    is_committed_exp = is_committed.unsqueeze(-1)
                    delta = torch.where(is_committed_exp, committed_delta, uncommitted_delta)
                    attn_out[:, gen_start:gen_end] += delta.to(attn.dtype)
                    
                is_layout_phase = (block_type == "double")
                is_texture_phase = (block_type == "single" and current_step >= 3)
                
                if dynamic_text_balancing and background_text_strength > 0.0:
                    block_face_exertion = safe_weight.mean().item() if boost_r > 0.0 else 0.0
                    txt_allowance = 1.0 - min(block_face_exertion, 1.0)
                    raw_txt_boost = background_text_strength * txt_allowance
                    
                    if raw_txt_boost > 0.0 and (is_layout_phase or is_texture_phase):
                        active_boost = raw_txt_boost if is_layout_phase else raw_txt_boost * f
                        txt_h = attn_out[:, :txt_len]
                        attn_out[:, :txt_len] += txt_h * active_boost
                        
                elif boost_t > 0.0: 
                    if (is_layout_phase or is_texture_phase):
                        active_boost = boost_t if is_layout_phase else boost_t * f
                        txt_h = attn_out[:, :txt_len]
                        attn_out[:, :txt_len] += txt_h * active_boost

                return attn_out

            return attn

        m.set_model_unet_function_wrapper(unet_wrapper)
        m.set_model_attn1_output_patch(output_patch)
        return (m,)

NODE_CLASS_MAPPINGS = {
    "FluxIDAutoAdjuster": FluxIDAutoAdjuster
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxIDAutoAdjuster": "FLUX Identity Adjuster"
}
