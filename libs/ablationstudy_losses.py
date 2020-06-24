
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def _compute_cross_entropy_loss(input, target):
    input = torch.log_softmax(input, dim=1)
    loss = torch.sum(input * target, dim=1, keepdim=False)
    loss = -torch.mean(loss, dim=0, keepdim=False)
    return loss


def _compute_soft_cross_entropy_loss(input, target):
    input = torch.log_softmax(input, dim=1)
    target = torch.nn.Sigmoid()(target)
    loss = torch.sum(input * target, dim=1, keepdim=False)
    loss = -torch.mean(loss, dim=0, keepdim=False)
    return loss


# distillation loss as described by Hinton et. al
def _compute_hinton_loss(input, target):
    T = 2
    input = torch.log_softmax(input / T, dim=1)
    target = torch.softmax(target / T, dim=1)
    loss = torch.sum(input * target, dim=1, keepdim=False)
    loss = -torch.mean(loss, dim=0, keepdim=False)
    return T * T * loss


# bce with hard target: classification loss in iCaRL
def _compute_bce_loss(input, target):
    crit = nn.BCEWithLogitsLoss(reduction="mean")
    return crit(input, target)


# bce with soft targets: distillation loss as described in iCaRL
def _compute_soft_bce_loss(input, target):
    crit = nn.BCEWithLogitsLoss(reduction="mean")
    target = nn.Sigmoid()(target)
    return crit(input, target)


def _compute_kldiv_loss(input, target):
    crit = nn.KLDivLoss(reduction="mean")
    input = torch.log_softmax(input, dim=1)
    target = torch.softmax(target, dim=1)
    return crit(input, target)


def _compute_l2_loss(input, target):
    input = nn.Sigmoid()(input)
    target = nn.Sigmoid()(target)
    crit = nn.MSELoss(reduction='mean')
    return crit(input, target)


def _compute_l2_hard_loss(input, target):
    input = nn.Sigmoid()(input)
    crit = nn.MSELoss(reduction='mean')
    return crit(input, target)


# loss described in the "Learning a Unified Classifier Incrementally via Rebalancing"
# measures the cosine similarity of the previous and new features representation (normalized)
# lfc = less forget constraint
def _compute_lfc_loss(input, target):
    input = F.normalize(input, p=2)
    target = F.normalize(target, p=2)
    crit = nn.CosineEmbeddingLoss()
    loss = crit(input, target, torch.ones(input.shape[0], ).cuda())
    return loss


# the CustomizedLoss compute a loss made by 2 terms:
# a [classification] and a [distillation] term
class _Classification_Distillation_Losses():
    def __init__(self, classification, distillation):
        self.classification = classification
        self.distillation = distillation
        self.loss_computer = {
            "bce": _compute_bce_loss,
            "icarl": _compute_soft_bce_loss,
            "ce": _compute_cross_entropy_loss,
            "icarl_ce": _compute_soft_cross_entropy_loss,
            "hinton": _compute_hinton_loss,
            "kldiv": _compute_kldiv_loss,
            "lfc": _compute_lfc_loss,
            "l2": _compute_l2_loss,
            "l2_hard": _compute_l2_hard_loss,
        }

    def __call__(self, class_input, class_target, dist_input, dist_target, class_ratio):
        # need this to handle the variation case, when normally the class_loss is "icarl"
        # but in the first step is "bce"
        if self.classification == "icarl" and dist_input is None:
            class_loss = self.loss_computer["bce"](class_input, class_target)
        else:
            class_loss = self.loss_computer[self.classification](class_input, class_target)

        if self.distillation is not None and dist_input is not None and dist_target is not None:
            if self.distillation == "lfc":
                # lfc requires its own class-dist ratio
                dist_ratio = 5 * math.sqrt(class_ratio / (1 - class_ratio))
                class_ratio = 1
            else:
                dist_ratio = 1 - class_ratio

            dist_loss = self.loss_computer[self.distillation](dist_input, dist_target)
            tot_loss = class_ratio * class_loss + dist_ratio * dist_loss
            return tot_loss
        else:
            return class_loss
