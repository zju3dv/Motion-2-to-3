import torch
import torch.nn as nn
from transformers import CLIPVisionModelWithProjection, CLIPTextModelWithProjection, CLIPTextModel, CLIPTokenizer
from transformers.image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from torchvision.transforms import Normalize
from einops import rearrange, reduce
from hmr4d.network.gmd.clip_helper import TransformerReduce
from hmr4d.utils.pylogger import Log


class CLIPLatentEncoder(nn.Module):
    def __init__(self, clip_target, clip_pretrained_path, uncond_strategy="empty_text", pooling_methd="avg"):
        """
        Args:
            uncond_strategy: ["empty_text", "zero_latent"]
        """
        super().__init__()
        if clip_target == "CLIPTextModelWithProjection":
            self.tokenizer = CLIPTokenizer.from_pretrained(clip_pretrained_path)
            self.text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_pretrained_path)
            self.max_length = self.tokenizer.max_model_input_sizes["openai/clip-vit-base-patch32"]
            self.clip_dim = self.text_encoder.config.projection_dim
        elif clip_target == "CLIPVisionModelWithProjection":
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_pretrained_path)
            self.normalize_clip_img = Normalize(OPENAI_CLIP_MEAN, OPENAI_CLIP_STD)
            self.clip_dim = self.image_encoder.config.projection_dim
            self.pooling_method = pooling_methd  # avg | clear | transformer
            if pooling_methd == "transformer":
                self.transformer_reduce = TransformerReduce(embed_dim=512)
        else:
            assert clip_target == "CLIPTextAndVisionModelWithProjection"
            # text
            self.tokenizer = CLIPTokenizer.from_pretrained(clip_pretrained_path)
            self.text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_pretrained_path)
            self.max_length = self.tokenizer.max_model_input_sizes["openai/clip-vit-base-patch32"]
            self.clip_dim = self.text_encoder.config.projection_dim

            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_pretrained_path)
            self.normalize_clip_img = Normalize(OPENAI_CLIP_MEAN, OPENAI_CLIP_STD)
            assert self.clip_dim == self.image_encoder.config.projection_dim
            self.pooling_method = pooling_methd  # avg | clear | transformer

        self.uncond_strategy = uncond_strategy

    def encode_text(self, prompt, num_images_per_prompt=1, enable_cfg=False):
        """
        Args:
            prompt: List of strings or a single string
        """
        assert num_images_per_prompt == 1

        device = self.text_encoder.device
        if isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = 1
            prompt = [prompt]

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids  # [B, L=77]
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.max_length - 1 : -1])
            print(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.max_length} tokens: {removed_text}"
            )

        # Remember to put the data to the same device of the model
        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        # latents (B, L, 768) or (B, 512) in case of WithProjection
        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )[0]
        if len(text_embeddings.shape) == 2:
            text_embeddings = text_embeddings.unsqueeze(1)  # (B, 1, 512)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if not enable_cfg:
            # Let's treat empty text as a special case, which aligns with the later unconditional strategy
            if self.uncond_strategy == "zero_latent":
                copy_ = text_embeddings.clone()  # handle in-place operation in inference mode
                copy_[torch.tensor([p == "" for p in prompt])] = 0
                text_embeddings = copy_

            return text_embeddings

        # get unconditional embeddings for classifier free guidance
        if self.uncond_strategy == "empty_text":
            uncond_tokens = [""] * batch_size

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )[0]
            if len(text_embeddings.shape) == 2:
                text_embeddings = text_embeddings.unsqueeze(1)  # (B, 1, 512)

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_images_per_prompt, seq_len, -1)

        elif self.uncond_strategy == "zero_latent":
            uncond_embeddings = torch.zeros_like(text_embeddings)

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    # def mask_clip_image(self, image, mask, mask_color):
    #     if mask_color == "disco_release":
    #         prompt_fg = image * mask + self.clip_mask_color * (1 - mask)
    #         prompt_fg = F.interpolate(
    #             prompt_fg, size=(224, 224), mode="bilinear", align_corners=False
    #         )
    #     return prompt_fg

    def encode_image_sequence(self, image=None, saved_embeds=None, enable_cfg=False):
        """
        Args:
            image: (B, S, 3, 224, 224) or None
            saved_embeds: (B, S, D) or None
        Returns:
            image_embeddings: ([1/2] * B * num_images_per_prompt, [1/L], D)
        """
        assert image is not None or saved_embeds is not None
        # saved_embeds has higher priority
        if saved_embeds is not None:
            B, S, D = saved_embeds.shape
            image_embeddings = rearrange(saved_embeds, "b s d -> (b s) d")
        else:
            # Input image range should be [0, 1]
            B, S, _, H, W = image.shape
            image_ = self.normalize_clip_img(image)
            image_embeddings = self.image_encoder(rearrange(image_, "b s c h w -> (b s) c h w")).image_embeds

        if self.pooling_method == "avg":
            image_embeddings = reduce(image_embeddings, "(b s) d -> b d", "mean", s=S)
        elif self.pooling_method == "max":
            image_embeddings = reduce(image_embeddings, "(b s) d -> b d", "max", s=S)
        elif self.pooling_method == "transformer":
            image_embeddings = self.transformer_reduce(image_embeddings, S)
        elif self.pooling_method == "clear":
            image_embeddings = reduce(image_embeddings, "(b s) d -> b d", "max", s=S)
            image_embeddings[:, :] = 0
        else:
            raise KeyError

        if enable_cfg:
            assert self.uncond_strategy == "zero_latent"
            negative_prompt_embeds = torch.zeros_like(image_embeddings)
            image_embeddings = torch.cat([negative_prompt_embeds, image_embeddings])

        image_embeddings = image_embeddings.unsqueeze(1)  # (B, 1, D)
        return image_embeddings
