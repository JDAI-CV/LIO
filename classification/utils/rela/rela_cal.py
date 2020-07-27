import torch
import torch.nn as nn
import torch.nn.functional as F


def calc_loss(all_objects, all_masks):
    N, H, W = all_objects.shape
    all_objects_input = all_objects.view(N, H * W, 1)
    all_objects_target = all_masks.view(N, H * W, 1)
    all_loss = cosine_loss(all_objects_input, all_objects_target)
    return all_loss

def calc_mse_loss(all_objects, all_masks):
    return (all_objects - all_masks)**2

def calc_mask(object_1, object_2, num_positive):
    N_1, S_1, C_1 = object_1.shape
    N_2, S_2, C_2 = object_2.shape
    object_1 = object_1.unsqueeze(1).expand((N_1, num_positive, S_1, C_1)).contiguous().view(N_1 * num_positive, S_1, C_1)
    relation = torch.matmul(object_1, object_2.transpose(1, 2)) / C_1  # (N_2, S_1, S_2)
    object_1_target = torch.max(relation, dim=2)[0].unsqueeze(-1).view(N_1, num_positive, S_1)
    object_1_target = torch.mean(object_1_target, dim=1)
    return object_1_target

def get_mask(all_objects, num_positive):
    N, C, H, W = all_objects.shape
    N_1 = N // (num_positive + 1)
    N_2 = N_1 * num_positive
    S = H * W
    all_objects = all_objects.view(N, C, S).transpose(1, 2)
    all_objects = all_objects.view(N_1, num_positive + 1, S, C)
    all_masks = torch.zeros(N_1, num_positive + 1, H, W).cuda()
    for i in range(num_positive + 1):
        main_index = torch.Tensor([i]).cuda().long()
        main_object = torch.index_select(all_objects, 1, main_index).view(N_1, S, C)
        sub_indexs = torch.Tensor([k for k in range(num_positive + 1) if k != i]).cuda().long()
        sub_objects = torch.index_select(all_objects, 1, sub_indexs).view(N_1 * num_positive, S, C)
        main_mask = calc_mask(main_object, sub_objects, num_positive)
        all_masks[:, i, :, :] = main_mask.view(N_1, H, W)
    all_masks = all_masks.view(N_1 + N_2, H, W)
    return all_masks

def calc_rela(all_objects, pred_masks, num_positive):
    all_masks = get_mask(all_objects, num_positive)
    return calc_mse_loss(pred_masks, all_masks)

