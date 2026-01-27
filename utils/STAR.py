"""
STAR (Structure-Topology Alignment via Residuals) 算法实现

修复版本说明：
1. 增加了 Scale (缩放因子) 的计算，以应对特征空间的伸缩。
2. 实现了严格的"链式对齐"：Task T-1 -> Task T。
3. 配合修改后的 hc_soinn_classifier 使用，移除了与 Plan B 的逻辑冲突。
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Callable, Set, Any
from torch.utils.data import DataLoader
import torch


def _l2_normalize_np(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    if v.ndim == 1:
        n = float(np.linalg.norm(v))
        if n < eps:
            return v.astype(np.float32, copy=True)
        return (v / n).astype(np.float32, copy=True)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return (v / (n + eps)).astype(np.float32, copy=False)


class STARAligner:
    """
    STAR 对齐器：计算特征空间刚性变换并应用到 HC-SOINN 分类器
    """
    
    def __init__(
        self,
        hc_soinn: Any,  # HCSOINNClassifier
        feature_extractor: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device,
        use_full_task_rehearsal: bool = False,
        star_mode: str = "rigid",
        star_lambda: float = 0.3,
    ):
        self.hc_soinn = hc_soinn
        self.feature_extractor = feature_extractor
        self.device = device
        self.use_full_task_rehearsal = use_full_task_rehearsal
        self.star_mode = str(star_mode).lower().strip()
        self.star_lambda = float(star_lambda)
        
        # 锚点存储
        self.anchor_store: Dict[int, Dict[str, Any]] = {}
        
        if self.star_mode not in {"rigid", "trajectory"}:
            logging.warning(f"[STAR] Unknown star_mode='{self.star_mode}', falling back to 'rigid'")
            self.star_mode = "rigid"
        if self.star_lambda <= 0.0 or self.star_lambda > 1.0:
            logging.warning(f"[STAR] star_lambda={self.star_lambda} out of (0,1], clamping.")
            self.star_lambda = float(min(max(self.star_lambda, 1e-3), 1.0))
    
    def compute_rigid_transform(
        self, 
        feats_old: np.ndarray, 
        feats_new: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        计算刚性变换 (旋转 + 平移 + 缩放)
        
        返回:
            R: 旋转矩阵 [D, D]
            mu_old: 旧中心 [D]
            mu_new: 新中心 [D]
            s: 缩放因子 (float)
        """
        feats_old = np.asarray(feats_old, dtype=np.float32)
        feats_new = np.asarray(feats_new, dtype=np.float32)
        
        if feats_old.shape != feats_new.shape:
            raise ValueError(f"Feature shape mismatch: {feats_old.shape} != {feats_new.shape}")
        
        if feats_old.shape[0] < 2:
            # 样本太少无法计算，返回单位变换
            D = feats_old.shape[1]
            return np.eye(D, dtype=np.float32), np.zeros(D), np.zeros(D), 1.0
        
        # 1. 计算均值
        mu_old = feats_old.mean(axis=0)
        mu_new = feats_new.mean(axis=0)
        
        # 2. 去中心化
        X = feats_old - mu_old
        Y = feats_new - mu_new
        
        # ---------------- Subspace-Constrained Procrustes (nullspace = I) ----------------
        # 关键问题：
        # - 在本项目中 D=768 但每类 anchor 常常只有 ~20 个，N << D 时正交 Procrustes 在高维空子空间是欠定的；
        # - 直接求 D×D 的 R 会在 nullspace 中产生任意旋转，随后把 HC-SOINN 的所有原型“乱转”，导致性能下降。
        #
        # 解决：
        # - 仅在 X 的 row-space（由 anchors 张成的子空间，维度 k<=N-1）中求旋转 R_k；
        # - 将 R_k 嵌入到 D 维：R = B R_k B^T + (I - B B^T)，在 nullspace 上保持恒等映射。
        D = X.shape[1]
        N = X.shape[0]
        I = np.eye(D, dtype=np.float32)

        # 通过 SVD 得到 row-space 的正交基 B ∈ R^{D×k}
        # X = Ux Sx Vx^T, Vx^T 的前 k 行张成 X 的 row-space
        Ux, Sx, Vxt = np.linalg.svd(X, full_matrices=False)
        # 以相对阈值估计有效秩，避免数值噪声
        if Sx.size == 0:
            k = 0
        else:
            smax = float(Sx[0])
            # 相对阈值：与最大奇异值相比过小的维度视为噪声
            tol = max(1e-6 * smax, 1e-8)
            k = int(np.sum(Sx > tol))

        # 默认：不做旋转（R=I），仅由平移/缩放决定
        R = I
        trace_S = 0.0
        var_old = 0.0
        error_before = None
        error_after = None

        # 低秩/样本过少时，不做旋转（避免欠定旋转把 nullspace 乱转）
        if k >= 2:
            B = Vxt[:k, :].T.astype(np.float32, copy=False)  # [D, k]
            Xk = np.dot(X, B)  # [N, k]
            Yk = np.dot(Y, B)  # [N, k] (project to same subspace)

            # 在子空间内解 Procrustes：min || s * Xk @ Rk - Yk ||
            Mk = np.dot(Xk.T, Yk)  # [k, k]
            Uk, Sk, Vkt = np.linalg.svd(Mk, full_matrices=False)
            Rk = np.dot(Uk, Vkt).astype(np.float32, copy=False)

            # 确保是 proper rotation（det=+1），避免反射
            if np.linalg.det(Rk) < 0:
                Vkt[-1, :] *= -1
                Rk = np.dot(Uk, Vkt).astype(np.float32, copy=False)

            # 嵌入回 D 维，并在 nullspace 上保持 I
            P = np.dot(B, B.T)  # projector to row-space, [D, D]
            R = (np.dot(np.dot(B, Rk), B.T) + (I - P)).astype(np.float32, copy=False)

            # 用子空间内的量计算 scale（更稳定；也避免把 nullspace 的任意旋转计入）
            var_old = float(np.sum(np.square(Xk)))
            trace_S = float(np.sum(Sk))
        else:
            # k < 2: 无法稳定估计旋转（N<3 或 rank<2），只做平移/缩放（缩放也会做保护）
            var_old = float(np.sum(np.square(X)))
            trace_S = 0.0

        # 4. 计算缩放因子 s（在可辨识子空间内；若不可辨识则退化为 1.0）
        if var_old < 1e-8 or trace_S <= 0.0:
            s = 1.0
        else:
            s = float(trace_S / var_old)

        # 诊断：对齐在可辨识子空间内的残差（仅用于日志分析，不做 gating/skip）
        if k >= 2:
            # 误差按 sqrt(N) 归一化，便于跨类比较
            error_before = float(np.linalg.norm(Yk - Xk, ord='fro') / np.sqrt(N))
            # 用同一 objective 的最优 s、Rk 评估对齐后误差
            Xk_aligned = (np.dot(Xk, Rk) * s)
            error_after = float(np.linalg.norm(Yk - Xk_aligned, ord='fro') / np.sqrt(N))
        # ------------------------------------------------------------------------------
            
        logging.info(f"  > Rotation Strength: {rot_strength:.6f} (Subspace Dim: {subspace_dim}, N={N}, D={D})")
        if error_before is not None and error_after is not None:
            logging.info(f"  > Procrustes Residual (subspace): before={error_before:.6f}, after={error_after:.6f}")
        # ====================================================
        
        return R, mu_old, mu_new, s
    
    def select_anchors_for_current_task(
        self,
        dataset,
        batch_size: int = 128,
        num_workers: int = 4,
        current_task_classes: Optional[Set[int]] = None,
    ) -> None:
        """为当前任务选择锚点并保存参考特征"""
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        all_feats = []
        all_imgs = []
        all_targets = []
        
        with torch.no_grad():
            for _, inputs, targets in loader:
                inputs = inputs.to(self.device)
                # Support for class-conditional feature extractors (e.g., CL-LoRA diagonal segment)
                # - If feature_extractor supports class_id, we pass the batch labels so it can return per-sample features
                try:
                    feats = self.feature_extractor(inputs, class_id=targets)
                except TypeError:
                    feats = self.feature_extractor(inputs)
                if isinstance(feats, tuple): feats = feats[0]
                
                all_feats.append(feats.detach().cpu().numpy())
                all_imgs.append(inputs.cpu())
                all_targets.append(targets.numpy())
        
        if not all_feats: return
        
        all_feats = np.concatenate(all_feats, axis=0)
        all_imgs = torch.cat(all_imgs, dim=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        if current_task_classes is None:
            current_task_classes = set(np.unique(all_targets))
        
        for cls in current_task_classes:
            if cls in self.anchor_store: continue
            
            mask = (all_targets == cls)
            cls_feats = all_feats[mask]
            cls_imgs = all_imgs[mask]
            
            if len(cls_feats) == 0: continue
            
            # --- Anchor Selection ---
            # rigid: 选“覆盖拓扑点”的一组 anchors（允许去重）
            # trajectory: 强制为每个 HC-SOINN 节点保存 1:1 的 anchor（不允许 np.unique 打破对应关系）
            if self.star_mode == "trajectory" and (not self.use_full_task_rehearsal):
                clusters = self.hc_soinn.class_clusters.get(cls, [])
                if len(clusters) == 0:
                    # Fallback: 没有节点则无法做 trajectory，对齐时也无从更新；保存少量样本作为兜底
                    sel_feats = cls_feats[: min(20, cls_feats.shape[0])]
                    sel_imgs = cls_imgs[: sel_feats.shape[0]]
                    self.anchor_store[cls] = {
                        "images": sel_imgs,
                        "feats_ref": sel_feats.copy(),
                        "mode": "trajectory_fallback",
                    }
                    continue

                # 目标：每个 cluster 选一个最相似样本，并尽量避免重复（贪心去冲突）
                targets = np.stack(
                    [(c.center_raw if c.center_raw is not None else c.center) for c in clusters],
                    axis=0,
                ).astype(np.float32, copy=False)  # [M, D]
                t_norm = _l2_normalize_np(targets)  # [M, D]
                f_norm = _l2_normalize_np(cls_feats)  # [N, D]
                sims = np.dot(t_norm, f_norm.T)  # [M, N]

                used = set()
                chosen = []
                # 每个 target 取 argmax；若冲突，取次大直到找到未用样本
                for i in range(sims.shape[0]):
                    order = np.argsort(-sims[i])  # descending
                    pick = int(order[0])
                    if pick in used:
                        for j in order[1:]:
                            jj = int(j)
                            if jj not in used:
                                pick = jj
                                break
                    used.add(pick)
                    chosen.append(pick)

                sel_feats = cls_feats[chosen]
                sel_imgs = cls_imgs[chosen]

                # 记录保存时的 center_raw，用于后续任务做 cluster 匹配（避免 clusters 顺序变化）
                centers_raw_ref = targets.copy()
                self.anchor_store[cls] = {
                    "images": sel_imgs,
                    "feats_ref": sel_feats.copy(),
                    "mode": "trajectory",
                    "centers_raw_ref": centers_raw_ref,
                    "ema_delta": np.zeros_like(sel_feats, dtype=np.float32),  # per-node EMA drift
                }
                continue

            # -------- rigid (original) --------
            targets = []
            if cls in self.hc_soinn.class_clusters:
                for c in self.hc_soinn.class_clusters[cls]:
                    targets.append(c.center_raw if c.center_raw is not None else c.center)
            if cls in self.hc_soinn.class_mu_raw:
                targets.append(self.hc_soinn.class_mu_raw[cls])

            if not targets:
                sel_feats = cls_feats
                sel_imgs = cls_imgs
            else:
                targets = np.array(targets)
                t_norm = targets / (np.linalg.norm(targets, axis=1, keepdims=True) + 1e-8)
                f_norm = cls_feats / (np.linalg.norm(cls_feats, axis=1, keepdims=True) + 1e-8)
                sims = np.dot(t_norm, f_norm.T)
                indices = np.unique(np.argmax(sims, axis=1))
                sel_feats = cls_feats[indices]
                sel_imgs = cls_imgs[indices]

            self.anchor_store[cls] = {
                'images': sel_imgs,
                'feats_ref': sel_feats.copy() # 保存当前任务空间的特征作为参考
            }

    def align_old_classes(
        self,
        cur_task: int,
        current_task_classes: Optional[Set[int]] = None,
    ) -> None:
        """执行链式对齐: Task T-1 -> Task T"""
        if current_task_classes is None: current_task_classes = set()
        
        aligned_cnt = 0
        
        for cls, data in self.anchor_store.items():
            if cls in current_task_classes: continue
            if cls not in self.hc_soinn.class_clusters: continue
            
            try:
                # 1. 准备数据
                imgs = data['images'].to(self.device)
                feats_old = data['feats_ref']  # T-1 时刻的特征
                
                # 2. 提取当前 (Task T) 特征
                with torch.no_grad():
                    # Support for DualPrompt: Pass class_id to feature_extractor if it supports it
                    try:
                        feats_new = self.feature_extractor(imgs, class_id=cls)
                    except TypeError:
                        feats_new = self.feature_extractor(imgs)
                        
                    if isinstance(feats_new, tuple): feats_new = feats_new[0]
                    feats_new = feats_new.detach().cpu().numpy()

                if self.star_mode == "trajectory" and str(data.get("mode", "")).startswith("trajectory"):
                    # -------- Trajectory STAR: pointwise transport with EMA --------
                    clusters = self.hc_soinn.class_clusters.get(cls, [])
                    if len(clusters) == 0:
                        continue

                    feats_old = np.asarray(feats_old, dtype=np.float32)
                    feats_new = np.asarray(feats_new, dtype=np.float32)
                    delta = feats_new - feats_old  # [M, D]

                    ema = data.get("ema_delta", None)
                    if ema is None or np.asarray(ema).shape != delta.shape:
                        ema = np.zeros_like(delta, dtype=np.float32)
                    lam = self.star_lambda
                    ema = (1.0 - lam) * ema + lam * delta
                    self.anchor_store[cls]["ema_delta"] = ema

                    # cluster matching (robust to ordering changes): centers_raw_ref -> current centers_raw
                    centers_raw_ref = data.get("centers_raw_ref", None)
                    if centers_raw_ref is None:
                        # assume 1:1 by index
                        match = list(range(min(len(clusters), ema.shape[0])))
                        ref_indices = list(range(len(match)))
                    else:
                        centers_raw_ref = np.asarray(centers_raw_ref, dtype=np.float32)
                        cur_centers_raw = np.stack([c.center_raw for c in clusters], axis=0).astype(np.float32, copy=False)
                        ref_n = _l2_normalize_np(centers_raw_ref)
                        cur_n = _l2_normalize_np(cur_centers_raw)
                        sim = np.dot(ref_n, cur_n.T)  # [M_ref, M_cur]
                        used = set()
                        match = []
                        ref_indices = []
                        for i in range(sim.shape[0]):
                            j = int(np.argmax(sim[i]))
                            if j in used:
                                order = np.argsort(-sim[i])
                                for cand in order:
                                    jj = int(cand)
                                    if jj not in used:
                                        j = jj
                                        break
                            used.add(j)
                            match.append(j)
                            ref_indices.append(i)

                    # apply EMA drift to matched nodes
                    applied = []
                    for i_ref, j_cur in zip(ref_indices, match):
                        if j_cur >= len(clusters) or i_ref >= ema.shape[0]:
                            continue
                        d = ema[i_ref]
                        new_raw = clusters[j_cur].center_raw + d
                        if np.linalg.norm(new_raw) > 1e-9:
                            clusters[j_cur].center_raw = new_raw.astype(np.float32, copy=False)
                            clusters[j_cur].center = _l2_normalize_np(clusters[j_cur].center_raw)
                            applied.append(d)

                    # update NCM center using mean delta over nodes (as requested)
                    if cls in getattr(self.hc_soinn, "class_mu_raw", {}):
                        if len(applied) > 0:
                            mean_d = np.mean(np.stack(applied, axis=0), axis=0)
                        else:
                            mean_d = np.mean(ema, axis=0)
                        new_mu_raw = self.hc_soinn.class_mu_raw[cls] + mean_d
                        if np.linalg.norm(new_mu_raw) > 1e-9:
                            self.hc_soinn.class_mu_raw[cls] = new_mu_raw.astype(np.float32, copy=False)
                            self.hc_soinn.class_mu[cls] = _l2_normalize_np(self.hc_soinn.class_mu_raw[cls])

                    # chain update: update reference features and ref centers for next task
                    self.anchor_store[cls]["feats_ref"] = feats_new.copy()
                    # refresh centers_raw_ref to current (post-update) centers to keep matching stable
                    self.anchor_store[cls]["centers_raw_ref"] = np.stack(
                        [c.center_raw for c in clusters], axis=0
                    ).astype(np.float32, copy=False)

                    # HC-SOINN nodes/NCM were updated in-place => invalidate inference cache if present
                    if hasattr(self.hc_soinn, "invalidate_cache"):
                        self.hc_soinn.invalidate_cache()

                    aligned_cnt += 1
                else:
                    # -------- Rigid STAR (default) --------
                    R, mu_old, mu_new, s = self.compute_rigid_transform(feats_old, feats_new)
                    self.hc_soinn.apply_rigid_transform(cls, R, mu_old, mu_new, scale=s)
                    self.anchor_store[cls]['feats_ref'] = feats_new.copy()
                    aligned_cnt += 1
                
            except Exception as e:
                logging.error(f"[STAR] Align failed for class {cls}: {e}")
