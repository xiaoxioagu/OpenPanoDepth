import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_berhu_loss(pred, target, mask):
	threshold = 0.2
	assert pred.dim() == target.dim(), "inconsistent dimensions"
	valid_mask = (target > 0).detach()
	mask = mask.bool()
	if mask is not None:
		valid_mask *= mask.detach()

	diff = torch.abs(target - pred)
	diff = diff[valid_mask]
	delta = threshold * torch.max(diff).data.cpu().numpy()

	part1 = -F.threshold(-diff, -delta, 0.)
	part2 = F.threshold(diff ** 2 + delta ** 2, 2.0 * delta ** 2, 0.)
	part2 = part2 / (2. * delta)
	diff = part1 + part2
	loss = diff.mean()
	return loss

def compute_silog_loss(pred, target, mask):
	assert pred.dim() == target.dim(), "inconsistent dimensions"
	if mask is not None:
		pred = pred[mask]
		target = target[mask]
	g = torch.log(pred) - torch.log(target)

	Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
	return 10 * torch.sqrt(Dg)

def compute_igev_loss(pred, target, mask):

	n = len(pred[1])
	init_pred = pred[0]
	preds = pred[1]
	loss_gamma = 0.9

	init_loss = compute_berhu_loss(init_pred, target, mask)
	for i in range(n):
		adjusted_loss = loss_gamma**(15/(n-1))
		i_gamma = adjusted_loss**(n - i - 1)
		i_loss = compute_berhu_loss(preds[i], target, mask)
		loss = init_loss + i_gamma * i_loss


	return loss
