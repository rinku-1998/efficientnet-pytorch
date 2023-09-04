import torch


def auto_device() -> torch.device:
    """自動決定運行裝置()

    Returns:
        torch.device: Torch Device
    """

    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)

    return device
