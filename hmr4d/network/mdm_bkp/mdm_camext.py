import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hmr4d.network.gmd.mdm_unet import MdmUnetOutput
from hmr4d.network.mdm.mdm import MDM, lengths_to_mask
from einops import einsum, rearrange, repeat


class MDMcamext(MDM):
    def _build_condition(self):
        self.embed_cond = nn.Linear(self.clip_dim, self.latent_dim)
        self.embed_ext = nn.Linear(4, self.latent_dim)

    def mask_cond(self, prompt, cam_ext, force_mask=False):
        """_summary_

        Args:
            prompt (_tensor_): [B, 1, d]
            cond_x: [bs, d], camera extrinsic
            force_mask (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        prompt = prompt.permute(1, 0, 2)  # [bs, 1, d] -> [1, bs, d]
        if force_mask:
            return prompt, cam_ext
        elif self.training and self.cond_mask_prob > 0.0:
            # To support classifier-free guidance, randomly drop out only text conditioning 5%, only image conditioning 5%, and both 5%.
            rand_num = torch.rand(prompt.shape[1], device=prompt.device)
            prompt_mask = 1.0 - rearrange(rand_num < self.cond_mask_prob, "n -> 1 n 1").float()
            cam_mask = 1.0 - rearrange(
                (rand_num >= self.cond_mask_prob).float() * (rand_num < 3 * self.cond_mask_prob).float(), "n -> n 1"
            )
            prompt = prompt * prompt_mask
            cam_ext = cam_ext * cam_mask
            return prompt, cam_ext
        else:
            return prompt, cam_ext

    def forward(self, x, timesteps, prompt_latent, cam_ext, length, uncond=False, **kwargs):
        """
        x: [batch_size, c_input, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        prompt_latent: [bs, 1, d]
        d_T: [bs, d_]
        cond_x: [batch_size, c_input, max_frames], another view of motion
        length: [bs] (int) tensor
        uncond: bool
        """
        # inference t might be a int
        if len(timesteps.shape) == 0:
            timesteps = timesteps.reshape([1]).to(x.device).expand(x.shape[0])
        # inference length is fewer as x has mutliview
        if length.shape[0] != x.shape[0]:
            length = length.expand(x.shape[0])

        emb = self.embed_timestep(timesteps)  # [1, bs, d]

        prompt_emb, cam_ext = self.mask_cond(prompt_latent, cam_ext, uncond)
        emb += self.embed_cond(prompt_emb)
        emb += self.embed_ext(cam_ext)

        x = self.input_process(x)  # [seqlen, bs, d]

        # adding the timestep embed
        xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        maskseq = lengths_to_mask(length + 1, xseq.shape[0])  # [bs, seqlen+1]
        output = self.seqTransEncoder(xseq, src_key_padding_mask=~maskseq)[1:]  # [seqlen, bs, d]

        output = self.output_process(output)  # [bs, feats, nframes]
        return MdmUnetOutput(sample=output, mask=maskseq[:, None, 1:])
