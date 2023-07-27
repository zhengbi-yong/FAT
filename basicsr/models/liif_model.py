from basicsr.models.sr_model import SRModel
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.utils.registry import ARCH_REGISTRY
from torch.nn import L1Loss, MSELoss
@MODEL_REGISTRY.register()
class LIIFModel(SRModel):
    def __init__(self, opt):
        super(LIIFModel, self).__init__(opt)
        self.net_g = ARCH_REGISTRY.get('LIIF')(encoder_spec=opt['network_g']['encoder_spec'],
                                        imnet_spec=opt['network_g']['imnet_spec'])
        # Initialize pixel criterion
        pixel_criterion_type = opt['train'].get('pixel_criterion')
        if pixel_criterion_type == 'l1':
            self.pixel_criterion = L1Loss().to(self.device)
        elif pixel_criterion_type == 'mse':
            self.pixel_criterion = MSELoss().to(self.device)
        else:
            raise ValueError(f'Unsupported pixel criterion type {pixel_criterion_type}')

    def init_net(self, model_type):
        """Initialize the LIIF network.

        Note that the LIIF network should already be registered in the ARCH_REGISTRY.
        """
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

    def feed_data(self, data):
        """Unpack data from the data loader and perform necessary pre-processing steps.

        Args:
            data (dict): Input data.

        The option 'scale' controls whether to multiply the LR images by 1 / scale.
        """
        # get LQ and GT images
        self.lq = data['lq'].to(self.device)  # LQ
        self.gt = data['gt'].to(self.device)  # GT

    def optimize_parameters(self, current_iter):
        # compute SR images
        self.fake_H = self.net_g(self.lq)

        # compute and optimize losses
        self.optimizer_G.zero_grad()

        # compute losses
        l_g_total = 0

        # pixel loss
        l_g_pix = self.pixel_criterion(self.fake_H, self.gt)
        l_g_total += l_g_pix

        # calculate final loss
        l_g_total.backward()
        self.optimizer_G.step()

    # Implement other necessary functions like `validation` and `get_current_log`.
