import numpy as np

def mask_scheduler(soft_mask_rate, epochs, niter_per_ep, cutoff_epoch=0, mode="standard", schedule="constant"):
    assert mode in ["standard", "soft"]
    if mode == "standard":
        return np.full(epochs * niter_per_ep, soft_mask_rate)

    early_iters = cutoff_epoch * niter_per_ep
    late_iters = (epochs - cutoff_epoch) * niter_per_ep

    if mode == "soft":
        assert schedule in ["constant", "linear"]
        if schedule == 'constant':
            early_schedule = np.full(early_iters, soft_mask_rate)
        elif schedule == 'linear':
            early_schedule = np.linspace(soft_mask_rate, 0, early_iters)
        final_schedule = np.concatenate((early_schedule, np.full(late_iters, 0)))

    assert len(final_schedule) == epochs * niter_per_ep
    return final_schedule