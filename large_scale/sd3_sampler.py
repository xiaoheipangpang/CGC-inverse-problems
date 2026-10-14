from typing import List, Tuple, Optional
import math
import torch
import os
from tqdm import tqdm
from diffusers import StableDiffusion3Pipeline
import torch.nn.functional as F

# =======================================================================
# Factory
# =======================================================================

__SOLVER__ = {}


def register_solver(name: str):
    def wrapper(cls):
        if __SOLVER__.get(name, None) is not None:
            raise ValueError(f"Solver {name} already registered.")
        __SOLVER__[name] = cls
        return cls

    return wrapper


def get_solver(name: str, **kwargs):
    if name not in __SOLVER__:
        raise ValueError(f"Solver {name} does not exist.")
    return __SOLVER__[name](**kwargs)


# =======================================================================
# Base
# =======================================================================

class StableDiffusion3Base():
    def __init__(self, model_key: str = 'stabilityai/stable-diffusion-3-medium-diffusers',
                 local_model_path: str = None, device='cuda', dtype=torch.float16):
        self.device = device
        self.dtype = dtype

        # 设置默认本地模型路径
        default_local_path = r"D:\yx_code\pretrainmodel\stable-diffusion-3-medium-diffusers"

        if local_model_path is None:
            local_model_path = default_local_path

        if os.path.exists(local_model_path):
            print(f"从本地路径加载模型: {local_model_path}")
            try:
                pipe = StableDiffusion3Pipeline.from_pretrained(
                    local_model_path,
                    torch_dtype=self.dtype,
                    local_files_only=True,
                    safety_checker=None,
                    requires_safety_checker=False
                )
                print("✅ 本地模型加载成功")
            except Exception as e:
                print(f"❌ 本地模型加载失败: {e}")
                raise
        else:
            raise FileNotFoundError(f"本地模型路径不存在: {local_model_path}")

        self.scheduler = pipe.scheduler

        # 保留文本编码器，但创建空嵌入
        self.tokenizer_1 = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.tokenizer_3 = pipe.tokenizer_3
        self.text_enc_1 = pipe.text_encoder
        self.text_enc_2 = pipe.text_encoder_2
        self.text_enc_3 = pipe.text_encoder_3

        self.text_enc_1.to(device)
        self.text_enc_2.to(device)
        self.text_enc_3.to(device)

        self.vae = pipe.vae
        self.transformer = pipe.transformer
        self.transformer.eval()
        self.transformer.requires_grad_(False)

        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )

        # 预计算空文本嵌入
        self.null_prompt_emb, self.null_pooled_emb = self.encode_prompt([""], 1)

        del pipe

    def encode_prompt(self, prompt: List[str], batch_size: int = 1) -> List[torch.Tensor]:

        device = self.device
        # CLIP encode
        text_clip1_ids = self.tokenizer_1(prompt,
                                          padding="max_length",
                                          max_length=77,
                                          truncation=True,
                                          return_tensors='pt').input_ids
        text_clip1_emb = self.text_enc_1(text_clip1_ids.to(device), output_hidden_states=True)
        pool_clip1_emb = text_clip1_emb[0].to(dtype=self.dtype, device=device)
        text_clip1_emb = text_clip1_emb.hidden_states[-2].to(dtype=self.dtype, device=device)

        text_clip2_ids = self.tokenizer_2(prompt,
                                          padding="max_length",
                                          max_length=77,
                                          truncation=True,
                                          return_tensors='pt').input_ids
        text_clip2_emb = self.text_enc_2(text_clip2_ids.to(device), output_hidden_states=True)
        pool_clip2_emb = text_clip2_emb[0].to(dtype=self.dtype, device=device)
        text_clip2_emb = text_clip2_emb.hidden_states[-2].to(dtype=self.dtype, device=device)

        # T5 encode
        text_t5_ids = self.tokenizer_3(prompt,
                                       padding="max_length",
                                       max_length=77,
                                       truncation=True,
                                       add_special_tokens=True,
                                       return_tensors='pt').input_ids
        text_t5_emb = self.text_enc_3(text_t5_ids.to(device))[0]
        text_t5_emb = text_t5_emb.to(dtype=self.dtype, device=device)

        # Merge
        clip_prompt_emb = torch.cat([text_clip1_emb, text_clip2_emb], dim=-1)
        clip_prompt_emb = torch.nn.functional.pad(
            clip_prompt_emb, (0, text_t5_emb.shape[-1] - clip_prompt_emb.shape[-1])
        )
        prompt_emb = torch.cat([clip_prompt_emb, text_t5_emb], dim=-2)
        pooled_prompt_emb = torch.cat([pool_clip1_emb, pool_clip2_emb], dim=-1)

        return prompt_emb, pooled_prompt_emb

    def initialize_latent(self, img_size: Tuple[int], batch_size: int = 1, **kwargs):
        H, W = img_size
        lH, lW = H // self.vae_scale_factor, W // self.vae_scale_factor
        lC = self.transformer.config.in_channels
        latent_shape = (batch_size, lC, lH, lW)

        z = torch.randn(latent_shape, device=self.device, dtype=self.dtype)

        return z

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        z = self.vae.encode(image).latent_dist.sample()
        z = (z - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = (z / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        return self.vae.decode(z, return_dict=False)[0]

    def predict_vector(self, z, t):
        # 使用空文本嵌入
        v = self.transformer(hidden_states=z,
                             timestep=t,
                             pooled_projections=self.null_pooled_emb,
                             encoder_hidden_states=self.null_prompt_emb,
                             return_dict=False)[0]
        return v


class SD3Euler(StableDiffusion3Base):
    def __init__(self, model_key: str = 'stabilityai/stable-diffusion-3-medium-diffusers',
                 local_model_path: str = None, device='cuda'):
        super().__init__(model_key=model_key, local_model_path=local_model_path, device=device)

    def sample(self, measurement, operator, task,
               NFE: int,
               img_shape: Optional[Tuple[int]] = None,
               batch_size: int = 1,
               step_size: float = 30.0,
               latent: Optional[List[torch.Tensor]] = None):

        imgH, imgW = img_shape if img_shape is not None else (1024, 1024)

        if latent is None:
            z = self.initialize_latent((imgH, imgW), batch_size)
        else:
            z = latent

        self.scheduler.set_timesteps(NFE, device=self.device)
        timesteps = self.scheduler.timesteps
        sigmas = timesteps / self.scheduler.config.num_train_timesteps

        pbar = tqdm(timesteps, total=NFE, desc='SD3 Euler')
        for i, t in enumerate(pbar):
            timestep = t.expand(z.shape[0]).to(self.device)
            pred_v = self.predict_vector(z, timestep)

            sigma = sigmas[i]
            sigma_next = sigmas[i + 1] if i + 1 < NFE else 0.0

            z = z + (sigma_next - sigma) * pred_v

        with torch.no_grad():
            img = self.decode(z)
        return img


# =======================================================================
# CGC (Control-Generation Co-design) Solver
# =======================================================================

@register_solver("rfoc")
class SD3RFOC(SD3Euler):


    def __init__(self, model_key: str = 'stabilityai/stable-diffusion-3-medium-diffusers',
                 local_model_path: str = None, device='cuda'):
        super().__init__(model_key=model_key, local_model_path=local_model_path, device=device)
        # CGC 控制参数
        self.control_weight = 1.0
        self.gradient_scale = 50.0
        # iLQR 迭代参数
        self.ilqr_max_iters = 5
        self.ilqr_delta_threshold = 1e-3
        self.show_ilqr = False

    def terminal_gradient(self, z0t, operator, measurement, task):

        z0t = z0t.requires_grad_(True)
        x0t = self.decode(z0t).float()

        if "sr" in task:
            loss = torch.linalg.norm(
                (operator.A_pinv(measurement) - operator.A_pinv(operator.A(x0t))).view(1, -1)
            )

        else:
            loss = torch.linalg.norm(
                (operator.At(measurement) - operator.At(operator.A(x0t))).view(1, -1)
            )

        grad = torch.autograd.grad(loss, z0t)[0].half()
        return grad.detach()

    # 兼容原接口名
    def data_consistency(self, z0t, operator, measurement, task):
        return self.terminal_gradient(z0t, operator, measurement, task)

    def sample(self, measurement, operator, task,
               NFE: int,
               img_shape: Optional[Tuple[int]] = None,
               batch_size: int = 1,
               step_size: float = 50.0,
               latent: Optional[List[torch.Tensor]] = None):

        imgH, imgW = img_shape if img_shape is not None else (1024, 1024)

        # ---------- 初始化潜在变量 ----------
        if latent is None:
            z_init = self.initialize_latent((imgH, imgW), batch_size)
        else:
            z_init = latent

        # ---------- 时间步设置 ----------
        self.scheduler.config.shift = 4.0
        self.scheduler.set_timesteps(NFE, device=self.device)
        timesteps = self.scheduler.timesteps
        sigmas = timesteps / self.scheduler.config.num_train_timesteps


        u_k = [torch.zeros_like(z_init) for _ in range(NFE)]
        alpha = 1.0 / step_size  # 控制权重 α (step_size = 1/α)

        if self.show_ilqr:
            print(f"[CGC-Latent] Starting: max_iters={self.ilqr_max_iters}, "
                  f"alpha={alpha:.6f}, eps={self.ilqr_delta_threshold}")

        # 用于返回的最终图像
        img_final = None

        converged = False
        for iter_idx in range(self.ilqr_max_iters):
            if self.show_ilqr:
                print(f"[iLQR] Iteration {iter_idx + 1}/{self.ilqr_max_iters}")

            # ---------- Step 1: Forward Rollout (第 3-8 行) ----------
            z = z_init.clone()
            u_k_new = [None] * NFE
            g_tilde_list = []

            pbar = tqdm(timesteps, total=NFE,
                        desc=f'CGC-Latent (iter {iter_idx+1}/{self.ilqr_max_iters})',
                        disable=not self.show_ilqr)

            for i, t in enumerate(pbar):
                timestep = t.expand(z.shape[0]).to(self.device)

                # x_k^(i) = h_RF++(x_k^(i) + u_k^(i))
                z_controlled = z + u_k[i]

                # 速度场预测
                with torch.no_grad():
                    pred_v = self.predict_vector(z_controlled, timestep)

                sigma = sigmas[i]
                sigma_next = sigmas[i + 1] if i + 1 < NFE else 0.0

                # x-prediction (Tweedie公式): x̂_0^(k) = x_k - σ_t · v_θ(x_k, t_k)
                z0t = z_controlled - sigma * pred_v
                z1t = z_controlled + (1 - sigma) * pred_v
                delta = sigma - sigma_next

                # ---------- 终端梯度 g̃_0^(k) ----------
                if i < NFE - 1:
                    g_tilde = self.terminal_gradient(z0t, operator, measurement, task)
                    g_tilde_list.append(g_tilde.clone())

                    # 施加数据一致性控制
                    z_after_ctrl = z_controlled - step_size * g_tilde

                    # 记录控制序列 u_k^(i+1) = u_k^(i) - (1/α)·g̃_0^(k))
                    u_k_new[i] = u_k[i] - alpha * g_tilde
                else:
                    z_after_ctrl = z_controlled
                    u_k_new[i] = u_k[i]

                # 状态前进 z = z + v_θ · Δt
                z = z_after_ctrl + (sigma - delta) * (z1t - z_controlled)

                if self.show_ilqr:
                    pbar.set_postfix({
                        'σ': f'{sigma:.3f}',
                        'step': f'{i+1}/{NFE}'
                    })

            delta_u_sq = 0.0
            for i in range(NFE):
                delta_u_i = u_k_new[i] - u_k[i]
                delta_u_sq += torch.sum(delta_u_i ** 2).item()
                u_k[i] = u_k_new[i].detach().clone()
            delta_u = delta_u_sq ** 0.5 if delta_u_sq > 0 else 0.0

            # 解码本次迭代的最终结果
            with torch.no_grad():
                img_iter = self.decode(z)
            img_final = img_iter

            if self.show_ilqr:
                print(f"[iLQR]   Δu = {delta_u:.6e}")


            if iter_idx == 0:
                converged = True
                if self.show_ilqr:
                    print(f"[iLQR] Converged at iteration {iter_idx + 1} "
                          f"(Δu={delta_u:.6e})")
                break
            if delta_u < self.ilqr_delta_threshold:
                converged = True
                if self.show_ilqr:
                    print(f"[iLQR] Converged at iteration {iter_idx + 1} "
                          f"(Δu={delta_u:.6e})")
                break

        if self.show_ilqr and not converged:
            print(f"[iLQR] Reached max iterations ({self.ilqr_max_iters})")

        return img_final