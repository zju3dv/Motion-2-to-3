from hmr4d.network.gmd.mdm_unet_utils import Conv1dAdaGNBlock


def finetune_all(net):
    """the entire unet"""
    net.unet.requires_grad_(True)


def finetune_unet_adagn_linear(net):
    """adagn processes the condition and then scale+shift the latent"""
    net.requires_grad_(False)
    for n, m in net.unet.named_modules():
        if isinstance(m, Conv1dAdaGNBlock):
            m.requires_grad_(True)


def finetune_unet_mid(net):
    """mid part in unet"""
    net.requires_grad_(False)
    for n, p in net.named_parameters():
        # "mid_block1", "mid_block2"
        if "mid_block" in n:
            p.requires_grad_(True)


FreezeUnetFuncs = {
    "all": finetune_all,
    "adagn": finetune_unet_adagn_linear,
    "unet-mid": finetune_unet_mid,
}
