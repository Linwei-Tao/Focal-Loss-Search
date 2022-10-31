import torch
import torch.nn as nn
import torch.utils
from Metrics.metrics import ECELoss, AdaptiveECELoss, ClasswiseECELoss, test_classification_net_logits, ModelWithTemperature


def model_valid(test_queue, valid_queue, model):
    nll_criterion = nn.CrossEntropyLoss().cuda()
    ece_criterion = ECELoss().cuda()
    adaece_criterion = AdaptiveECELoss().cuda()
    cece_criterion = ClasswiseECELoss().cuda()

    logits, labels = get_logits_labels(test_queue, model)
    conf_matrix, p_accuracy, _, _, _ = test_classification_net_logits(logits, labels)

    p_ece = ece_criterion(logits, labels).item()
    p_adaece = adaece_criterion(logits, labels).item()
    p_cece = cece_criterion(logits, labels).item()
    p_nll = nll_criterion(logits, labels).item()

    scaled_model = ModelWithTemperature(model)
    scaled_model.set_temperature(valid_queue, cross_validate="ece")
    T_opt = scaled_model.get_temperature()
    logits, labels = get_logits_labels(test_queue, scaled_model)
    conf_matrix, accuracy, _, _, _ = test_classification_net_logits(logits, labels)

    ece = ece_criterion(logits, labels).item()
    adaece = adaece_criterion(logits, labels).item()
    cece = cece_criterion(logits, labels).item()
    nll = nll_criterion(logits, labels).item()

    return p_accuracy, p_ece, p_adaece, p_cece, p_nll, T_opt, ece, adaece, cece, nll


def get_logits_labels(data_loader, net):
    logits_list = []
    labels_list = []
    net.eval()
    with torch.no_grad():
        for data, label in data_loader:
            data = data.cuda()
            logits = net(data)
            logits_list.append(logits)
            labels_list.append(label)
        logits = torch.cat(logits_list).cuda()
        labels = torch.cat(labels_list).cuda()
    return logits, labels