def archive_sliding_window():
    B = batch["B"]
    assert B == 1
    # Sliding window inference
    seq_length = batch["length"]
    L = 300
    interval = 50
    device = seq_length.device

    # start = torch.tensor([0] * B).to(device)
    # end = torch.stack([start + L, seq_length], dim=-1).min(-1)[0]
    start = int(0)
    end = int(min(start + L, seq_length))

    T_ayfznew2c = None
    # T_ayfznew2c = batch["gt_T_ayfz2c"][:, None]
    T_to_start = rearrange(torch.eye(4), f"c d -> {B} c d").to(device)
    pred_ayfz_motions = []
    pred_meta_start_end = []
    pred_T_to_starts = []

    while True:
        length_ = torch.tensor([end - start] * B).to(device)
        assert B == 1
        obs_c_p2d_ = pad_to_max_len(batch["pred_c_p2d"][0, start:end], L)[None]
        obs_cr_p3d_ = pad_to_max_len(batch["pred_cr_motion3d"][0, start:end], L)[None]
        f_imgseq_ = pad_to_max_len(batch["f_imgseq"][0, start:end], L)[None]

        batch_window = {
            "B": 1,
            "generator": batch["generator"],
            "length": length_,
            "obs_c_p2d": obs_c_p2d_,
            "obs_cr_p3d": obs_cr_p3d_,
            "text": batch["text"],
            "f_imgseq": f_imgseq_,
            "T_ayfz2c": T_ayfznew2c,
        }

        outputs_window = self.pipeline.forward_mocap(batch_window)
        pred_ayfz_motion_ = outputs_window["pred_ayfz_motion"]

        # Save to the full motion
        pred_ayfz_motions.append(pred_ayfz_motion_)  # (B, L, 22, 3)
        pred_meta_start_end.append((start, end))
        pred_T_to_starts.append(T_to_start)

        if end == seq_length:
            break

        # Next window index
        start_in_last_window = end - interval - start
        start = end - interval
        end = min(start + L, int(seq_length))

        # Compute T_ayfz2c for the later windows
        T_ayfzlast2c = outputs_window["T_ayfz2c"]  # (B, 1, 4, 4)
        ay_pose_ = pred_ayfz_motion_[0, start_in_last_window][None]  # (22, 3)
        T_ayfznew2ay = compute_T_ayfz2ay(ay_pose_, inverse=False)  # (B, 4, 4)
        T_ayfznew2c = T_ayfzlast2c @ T_ayfznew2ay  # (B, 4, 4) this will be used as next T_ayfz2c
        T_to_start = T_to_start @ T_ayfznew2ay  # (B, 4, 4)

    # outputs
    outputs = {}
    outputs["pred_ayfz_motions"] = torch.stack(pred_ayfz_motions, dim=1)  # (B, W, L, 22, 3)
    outputs["pred_meta_start_end"] = pred_meta_start_end
    outputs["pred_T_to_starts"] = torch.stack(pred_T_to_starts, dim=1)  # (B, W, 4, 4)

    # wis3d = make_wis3d(name="debug")
    mid = batch["meta"][0][0].replace("/", "-")
    wis3d = make_wis3d(output_dir="outputs/wis3d_debug_seq", name=mid)
    add_motion_as_lines(batch["gt_ayfz_motion"][0], wis3d, name="gt_motion_full", const_color="green")

    assert B == 1
    b = 0
    # (W, L, J, 3)
    preds = apply_T_on_points(outputs["pred_ayfz_motions"][b], outputs["pred_T_to_starts"][b][:, None])

    start_end_list = outputs["pred_meta_start_end"]  # FIXME: not batched
    full_motion = torch.zeros(start_end_list[-1][-1], 22, 3).to(device)  # (L, J, 3)
    last_e = 0
    for w, (s, e) in enumerate(start_end_list):
        if s < last_e:
            # do linear blending
            old = full_motion[s:last_e]
            new = preds[w, : last_e - s]

            overlap = last_e - s
            alpha = torch.linspace(0, 1, overlap).to(device)
            alpha = alpha[:, None, None]
            full_motion[s:last_e] = alpha * new + (1 - alpha) * old
            full_motion[last_e:e] = preds[w, last_e - s : e - s]
        else:
            full_motion[s:e] = preds[w, : e - s]
        last_e = e

    # add_motion_as_lines(full_motion, wis3d, name=f"pred_motion_full_linear_blending")
    add_motion_as_lines(full_motion, wis3d, name=f"pred")

    # for w in range(outputs["pred_ayfz_motions"].shape[1]):
    #     add_motion_as_lines(preds[w, : end - start], wis3d, name=f"pred_motion_full", offset=start)
    #     add_motion_as_lines(preds[w, : end - start], wis3d, name=f"pred_motion_full_gtcam_gtp2d", offset=start)

    print(f"{mid}")
