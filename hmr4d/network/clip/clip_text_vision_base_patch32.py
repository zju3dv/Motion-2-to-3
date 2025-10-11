import torch
import torch.nn as nn
from einops import rearrange, reduce

from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from transformers.image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from torchvision.transforms import Normalize

from diffusers.utils import BaseOutput
from dataclasses import dataclass

from hmr4d.utils.pylogger import Log


@dataclass
class CLIPTextOutput(BaseOutput):
    # The feature of 1. empty string "" is regarded as unconditional, the effective length is 2 (SOT & EOT)
    #                2. padding tokens after EOT-token will be set to zero (for safety consideration)
    f_text: torch.FloatTensor = None  # (B or 2B, 77, D)
    f_text_length: torch.LongTensor = None  # (B or 2B), the minimal length is 2


@dataclass
class CLIPImgSeqOutput(BaseOutput):
    # The feature of padding img is set to zero, regarded as unconditional
    f_imgseq: torch.FloatTensor = None  # (B or 2B, I, D)
    f_imgseq_fid: torch.LongTensor = None  # (B or 2B, I), the frame id of each image, -1 indicates padding


class CLIPLatentEncoder(nn.Module):
    clip_pretrained_path = "inputs/checkpoints/huggingface/clip-vit-base-patch32"
    # or from OpenAI https
    # clip_pretrained_path = "openai/clip-vit-base-patch32"

    def __init__(self):
        super().__init__()
        clip_pretrained_path = self.clip_pretrained_path

        # text, use token-wise features for each prompt
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_pretrained_path)
        self.text_encoder = CLIPTextModel.from_pretrained(clip_pretrained_path)
        self.max_length = self.tokenizer.max_model_input_sizes["openai/clip-vit-base-patch32"]
        self.clip_dim = self.text_encoder.config.projection_dim  # 512 in base-patch32

        # image, use class-token feature for each image
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_pretrained_path)
        self.normalize_clip_img = Normalize(OPENAI_CLIP_MEAN, OPENAI_CLIP_STD)
        assert self.clip_dim == self.image_encoder.config.projection_dim

    def encode_text(self, prompt, enable_cfg=False, with_projection=False):
        """
        Args:
            prompt: List of strings or a single string
            enable_cfg: the output will be (2*B, 77, D), [uncond, cond]
        Returns: CLIPTextOutput
        """
        device = self.text_encoder.device
        if isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = 1
            prompt = [prompt]

        # Tokenize the prompt
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids  # [B, L=77]
        f_text_length = text_inputs.attention_mask.sum(dim=-1)  # [B]
        assert f_text_length.min() >= 2, "The minimal length of text input should be 2 (SOT & EOT)"

        # Warn the user if the input was truncated
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.max_length - 1 :], skip_special_tokens=True
            )
            removed_text = [t_ for t_ in removed_text if t_ != ""]
            Log.warn(f"Up to {self.max_length} tokens can be processed. Remove: {removed_text}")

        # In fact, CLIP uses causal mask and do not use attention_mask
        attention_mask = None
        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)

        # Network Forward, the output is (B, 77, 512)
        #   if without-projection: the last_hidden_states is used (B, 77, 512)
        #   if with-projection: the projected pooled_outputs is used (B, 512)
        if with_projection:
            f_text = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)[1]
        else:
            f_text = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)[0]

        # Add feature of unconditional text if needed
        if enable_cfg:
            # get unconditional embeddings for classifier free guidance
            uncond_tokens = [""] * batch_size
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=text_input_ids.shape[-1],
                truncation=True,
                return_tensors="pt",
            )
            f_text_length_uncond = uncond_input.attention_mask.sum(dim=-1)  # [B]
            assert (f_text_length_uncond == 2).all()

            attention_mask = None
            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            if with_projection:
                f_text_uncond = self.text_encoder(uncond_input.input_ids.to(device), attention_mask=attention_mask)[1]
            else:
                f_text_uncond = self.text_encoder(uncond_input.input_ids.to(device), attention_mask=attention_mask)[0]

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            f_text = torch.cat([f_text_uncond, f_text])
            f_text_length = torch.cat([f_text_length_uncond, f_text_length])

        # make sure f_text_length is on the same device as f_text
        f_text_length = f_text_length.to(f_text.device)

        return CLIPTextOutput(f_text=f_text, f_text_length=f_text_length)

    def encode_imgseq(self, imgseq, imgseq_fid, enable_cfg=False):
        """
        Args:
            imgseq: (B, I, 3, 224, 224), in range [0, 1]
            imgseq_fid: (B, I), -1 indicates padding
        Returns: CLIPImgSeqOutput
        """
        B, I, _, H, W = imgseq.shape

        # Network forward
        img_ = self.normalize_clip_img(imgseq)  # normalize from [0, 1]
        img_ = rearrange(img_, "b i c h w -> (b i) c h w")
        f_imgseq = self.image_encoder(img_).image_embeds  # （B*I, D)
        f_imgseq = f_imgseq.reshape(B, I, -1)  # (B, I, D)

        # Set padding frames to zero
        mask = imgseq_fid == -1  # (B, I)
        f_imgseq[mask] = 0
        f_imgseq_fid = imgseq_fid.clone()

        # Add feature of unconditional image if needed
        if enable_cfg:
            f_imgseq_uncond = torch.zeros_like(f_imgseq)
            f_imgseq = torch.cat([f_imgseq_uncond, f_imgseq])
            f_imgseq_fid_uncond = torch.full_like(f_imgseq_fid, -1)
            f_imgseq_fid = torch.cat([f_imgseq_fid_uncond, f_imgseq_fid])

        return CLIPImgSeqOutput(f_imgseq=f_imgseq, f_imgseq_fid=f_imgseq_fid)
