import torch

def area(box) -> torch.Tensor:
    """
    Computes the area of all the boxes.

    Returns:
        torch.Tensor: a vector with areas of each box.
    """
    area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
    return area

# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def pairwise_iou(boxes1, boxes2) -> torch.Tensor:
    area1, area2 = area(boxes1), area(boxes2)
    width_height = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(boxes1[:, None, :2], boxes2[:, :2])
    width_height.clamp_(min=0)
    inter = width_height.prod(dim=2)
    del width_height

    iou = torch.where(inter > 0, inter / (area1[:, None] + area2 - inter), torch.zeros(1, dtype=inter.dtype, device=inter.device))
    return iou

def pairwise_oais(boxes1, boxes2, depth1, depth2) -> torch.Tensor:
    area1, area2 = area(boxes1), area(boxes2)
    width_height = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(boxes1[:, None, :2], boxes2[:, :2]); width_height.clamp_(min=0)

    inter = width_height.prod(dim=2); del width_height
    iou = torch.zeros(boxes1.size(0), boxes2.size(0))
    for i in range(boxes1.size(0)):
        for j in range(boxes2.size(0)):
            if inter[i,j] != 0:
                iou[i,j] = inter[i,j] / area1[i] if depth1[i] > depth2[j] else inter[i,j] / area2[j]
    return iou