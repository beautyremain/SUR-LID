import torch
import torch.nn.functional as F
import torch.nn as nn


class SupConLoss(torch.nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        if len(features.shape) != 2:
            raise ValueError('`features`需要是 [batch_size, feature_dim] 形式')

        batch_size = features.shape[0]
        if labels is not None and mask is None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('`labels` 维度不匹配 `features` 的 batch size')
            mask = torch.eq(labels, labels.T).float().to(device)
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif mask is not None:
            mask = mask.float().to(device)

        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

def loss_ConSup(fc_features,labels):
    criterion_supcon = SupConLoss()
    fc_features = F.adaptive_avg_pool2d(fc_features, (1, 1))
    fc_features = fc_features.view(fc_features.size(0), -1)
    loss = criterion_supcon(fc_features,labels)

    return loss

def loss_fn_kd(y, labels, teacher_scores, T, alpha):
    return nn.KLDivLoss()(F.log_softmax(y/T,dim=1), F.log_softmax(teacher_scores/T,dim=1)) * (T*T * 2.0 * alpha) + F.cross_entropy(y, labels.long()) * (1. - alpha)

def loss_FD(Student_feature, Teacher_feature):
    Student_feature = F.adaptive_avg_pool2d(Student_feature, (1, 1))
    Student_feature = Student_feature.view(Student_feature.size(0), -1)
    Student_feature = F.normalize(Student_feature, dim=1)

    Teacher_feature = F.adaptive_avg_pool2d(Teacher_feature, (1, 1))
    Teacher_feature = Teacher_feature.view(Teacher_feature.size(0), -1)
    Teacher_feature = F.normalize(Teacher_feature, dim=1)

    # loss = torch.nn.functional.mse_loss(Student_feature, Teacher_feature, reduction="none")
    # return loss.sum()

    loss = torch.nn.functional.mse_loss(Student_feature, Teacher_feature, reduction="mean")
    return loss