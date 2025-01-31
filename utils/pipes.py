from diffusers import DDIMInverseScheduler
from diffusers.image_processor import VaeImageProcessor
from PIL import Image
import torch
from diffusers.utils.torch_utils import randn_tensor
from dataclasses import dataclass
from typing import Optional, List

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

@dataclass
class CanvasMemory:
    image: Optional[Image.Image]
    latent: Optional[torch.Tensor]  # tensor of shape (1, 8, height, width)
    generation_config: Optional[dict]
    def copy_from(self, other: "CanvasMemory"):
        self.image = other.image
        self.latent = other.latent
        self.generation_config = other.generation_config

@torch.no_grad()
def decode_latent(self, latents: torch.Tensor) -> List[Image.Image]:
    """Decode Latents to Image.
    Derived from https://huggingface.co/Mitsua/mitsua-likes/blob/main/pipeline_likes_base_unet.py#L994
    """
    # make sure the VAE is in float32 mode, as it overflows in float16
    needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

    if needs_upcasting:
        self.upcast_vae()
        latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
    elif latents.dtype != self.vae.dtype:
        if torch.backends.mps.is_available():
            # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
            self.vae = self.vae.to(latents.dtype)

    # unscale/denormalize the latents
    # denormalize with the mean and std if available and not None
    has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
    has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
    if has_latents_mean and has_latents_std:
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.latent_channels, 1, 1).to(latents.device, latents.dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.latent_channels, 1, 1).to(latents.device, latents.dtype)
        )
        latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
    else:
        latents = latents / self.vae.config.scaling_factor

    image = self.vae.decode(latents, return_dict=False)[0]

    # cast back to fp16 if needed
    if needs_upcasting:
        self.vae.to(dtype=torch.float16)
    
    image = self.image_processor.postprocess(image, output_type="pil")
    return image

@torch.no_grad()
def text_to_image(
    self,
    *,
    prompt: str,
    negative_prompt: str,
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    guidance_rescale: float,
    callback_on_step_end = None,
) -> CanvasMemory:
    output = self(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        guidance_rescale=guidance_rescale,
        callback_on_step_end=callback_on_step_end,
        output_type="latent",
    )
    latent = output.images
    image = decode_latent(self, latent)[0]
    return CanvasMemory(
        image,
        latent,
        dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale))
    
def get_timesteps(self, num_inference_steps, strength, device):
    # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
    if hasattr(self.scheduler, "set_begin_index"):
        self.scheduler.set_begin_index(t_start * self.scheduler.order)

    return timesteps, num_inference_steps - t_start

# https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint.py#L881
@torch.no_grad()
def image_to_image(
    self,
    *,
    prompt: str,
    negative_prompt: str,
    height: int,
    width: int,
    latent: torch.Tensor,
    num_inference_steps: int,
    guidance_scale: float,
    guidance_rescale: float,
    strength: float,
    ddim_inversion: bool = False,
    original_prompt: Optional[str] = None,
    original_negative_prompt: Optional[str] = None,
    mask_image: Optional[Image.Image] = None,
    callback_on_step_end = None,
) -> CanvasMemory:
    device = self._execution_device
    generator = None
    eta = 0.0
    crops_coords_top_left = (0, 0)
    callback_on_step_end_tensor_inputs = ["latents"]
    original_size = (height, width)
    target_size = (height, width)
    self._guidance_scale = guidance_scale
    self._guidance_rescale = guidance_rescale
    self._cross_attention_kwargs = None
    self._denoising_end = None
    self._interrupt = False

    def prepare_prompt_latent(prompt: str, negative_prompt: str, do_classifier_free_guidance: bool):
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
        )
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        add_text_embeds = pooled_prompt_embeds
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        negative_add_time_ids = add_time_ids

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device)
        return prompt_embeds, add_text_embeds, add_time_ids, extra_step_kwargs

    latents = latent.clone()

    if mask_image is not None:
        init_latents = latents.clone()
        # prepare masked latents
        mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_normalize=False, do_binarize=True, do_convert_grayscale=True,
        )
        mask_condition = mask_processor.preprocess(
            mask_image, height=height, width=width, resize_mode="default", crops_coords=None
        )
        mask = torch.nn.functional.interpolate(
            mask_condition, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        ).to(device=device, dtype=latents.dtype)

    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps
    if ddim_inversion:
        inv_scheduler = DDIMInverseScheduler.from_config(self.scheduler.config)
        inv_scheduler.set_timesteps(num_inference_steps * 2, device=device)  # 2x denser than original scheduler
        inv_timesteps = inv_scheduler.timesteps
    timesteps, num_inference_steps = get_timesteps(self, num_inference_steps, strength, device)

    if ddim_inversion:
        inv_timesteps = inv_timesteps[inv_timesteps <= timesteps[0]]
        print(inv_timesteps)
        self._num_timesteps = len(timesteps) + len(inv_timesteps)
    else:
        self._num_timesteps = len(timesteps)

    if ddim_inversion:
        noise = randn_tensor(latents.shape, device=device, dtype=latents.dtype)
        latents = self.scheduler.add_noise(latents, noise, inv_timesteps[:1])
        assert original_prompt is not None and original_negative_prompt is not None
        (
            prompt_embeds,
            add_text_embeds,
            add_time_ids,
            extra_step_kwargs
        ) = prepare_prompt_latent(original_prompt, original_negative_prompt, self.do_classifier_free_guidance)
        print("This extra_step_kwargs will be ignored in inversion:", extra_step_kwargs)
        # 9. Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)
        print("cond", timestep_cond)

        with self.progress_bar(total=len(inv_timesteps)) as progress_bar:
            for i, t in enumerate(inv_timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

                latent_model_input = inv_scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # compute the next noisy sample x_t -> x_t+1
                latents_dtype = latents.dtype
                latents = inv_scheduler.step(noise_pred, int(t.item()), latents, return_dict=False)[0]
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_on_step_end(self, i, t, callback_kwargs)

                progress_bar.update()
        print(torch.linalg.norm(latents - latent))
    else:
        noise = randn_tensor(latents.shape, device=device, dtype=latents.dtype)
        latents = self.scheduler.add_noise(latents, noise, timesteps[:1])
        print(torch.linalg.norm(latents - latent))

    (
        prompt_embeds,
        add_text_embeds,
        add_time_ids,
        extra_step_kwargs
    ) = prepare_prompt_latent(prompt, negative_prompt, self.do_classifier_free_guidance)

    # 8. Denoising loop
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

    # 9. Optionally get Guidance Scale Embedding
    timestep_cond = None
    if self.unet.config.time_cond_proj_dim is not None:
        guidance_scale_tensor = torch.tensor(self.guidance_scale - 1)
        timestep_cond = self.get_guidance_scale_embedding(
            guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
        ).to(device=device, dtype=latents.dtype)

    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=self.cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

            # compute the previous noisy sample x_t -> x_t-1
            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    latents = latents.to(latents_dtype)
            if mask_image is not None:
                if i < len(timesteps) - 1:
                    noise_timestep = timesteps[i + 1]
                    lock_latents = self.scheduler.add_noise(
                        init_latents, noise, torch.tensor([noise_timestep])
                    )
                else:
                    lock_latents = init_latents
                latents = (1 - mask) * lock_latents + mask * latents

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                if ddim_inversion:
                    callback_on_step_end(self, len(inv_timesteps)+i, t, callback_kwargs)
                else:
                    callback_on_step_end(self, i, t, callback_kwargs)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

    # Offload all models
    self.maybe_free_model_hooks()

    image = decode_latent(self, latents)[0]
    return CanvasMemory(
        image,
        latents,
        dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale))