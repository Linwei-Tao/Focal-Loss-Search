import torch
import torch.nn as nn
import torch.utils
from Metrics.metrics import ECELoss, AdaptiveECELoss, ClasswiseECELoss, test_classification_net_logits, \
    ModelWithTemperature
import hfai.nccl.distributed as dist


def model_valid(test_queue, valid_queue, model):
    logits, labels = get_logits_labels(test_queue, model)
    if dist.get_rank() == 0:
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = ECELoss().cuda()
        adaece_criterion = AdaptiveECELoss().cuda()
        cece_criterion = ClasswiseECELoss().cuda()
        conf_matrix, p_accuracy, _, _, _ = test_classification_net_logits(logits, labels)

        p_ece = ece_criterion(logits, labels).item()
        p_adaece = adaece_criterion(logits, labels).item()
        p_cece = cece_criterion(logits, labels).item()
        p_nll = nll_criterion(logits, labels).item()

        scaled_model = ModelWithTemperature(model)
        scaled_model.set_temperature(valid_queue, cross_validate="ece")
        T_opt = scaled_model.get_temperature()
    logits, labels = get_logits_labels(test_queue, scaled_model)
    if dist.get_rank() == 0:
        conf_matrix, accuracy, _, _, _ = test_classification_net_logits(logits, labels)

        ece = ece_criterion(logits, labels).item()
        adaece = adaece_criterion(logits, labels).item()
        cece = cece_criterion(logits, labels).item()
        nll = nll_criterion(logits, labels).item()
    else:
        p_accuracy = None
        p_ece = None
        p_adaece = None
        p_cece = None
        p_nll = None
        T_opt = None
        ece = None
        adaece = None
        cece = None
        nll = None

    return p_accuracy, p_ece, p_adaece, p_cece, p_nll, T_opt, ece, adaece, cece, nll


def get_logits_labels(dataloader, model):
    logits_list = []
    labels_list = []
    loss, correct1, correct5, total = torch.zeros(4).cuda()
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            samples, labels = [x.cuda(non_blocking=True) for x in batch]
            logits = model(samples)
            total += samples.size(0)
            gathered_logits = [torch.zeros_like(logits) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_logits, logits)  # gather not supported with NCCL
            gathered_labels = [torch.zeros_like(labels) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_labels, labels)  # gather not supported with NCCL
            logits_list.extend([logits.cpu().numpy() for logits in gathered_logits])
            labels_list.extend([labels.cpu().numpy() for labels in gathered_labels])
    return logits, labels
