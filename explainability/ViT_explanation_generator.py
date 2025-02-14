import argparse
import torch
import numpy as np
from numpy import *

# compute rollout between attention layers
def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration- code adapted from https://github.com/samiraabnar/attention_flow
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                          for i in range(len(all_layer_matrices))]
    joint_attention = matrices_aug[start_layer]
    propagated_layerwise_attention = []
    for i in range(start_layer+1, len(matrices_aug)):
        joint_attention = matrices_aug[i].bmm(joint_attention)
        propagated_layerwise_attention.append(joint_attention)

    #print(joint_attention.shape)
    return joint_attention, propagated_layerwise_attention

class LRP:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_LRP(self, input, index=None, method="transformer_attribution", is_ablation=False, start_layer=0):
        output = self.model(input)
        kwargs = {"alpha": 1}
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        return self.model.relprop(torch.tensor(one_hot_vector).to(input.device), method=method, is_ablation=is_ablation,
                                  start_layer=start_layer, **kwargs)



class Baselines:
    def __init__(self, model):
        self.model = model
        self.model.eval()
    
    def generate_cam_attn(self, input, index=None):
        output = self.model(input.cuda(), register_hook=True)
        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        #################### attn
        grad = self.model.blocks[-1].attn.get_attn_gradients()
        cam = self.model.blocks[-1].attn.get_attention_map()
        cam = cam[0, :, 0, 1:].reshape(-1, 14, 14)
        grad = grad[0, :, 0, 1:].reshape(-1, 14, 14)
        grad = grad.mean(dim=[1, 2], keepdim=True)
        cam = (cam * grad).mean(0).clamp(min=0)
        cam = (cam - cam.min()) / (cam.max() - cam.min())

        #print(cam.shape)
        return cam
        #################### attn

    def generate_rollout(self, input, start_layer=0):
        self.model(input)
        blocks = self.model.blocks
        all_layer_attentions = []
        all_head_attentions = []
        for blk in blocks:
            attn_heads = blk.attn.get_attn()
            avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
            all_layer_attentions.append(avg_heads)
            all_head_attentions.append(attn_heads.detach())
        rollout = compute_rollout_attention(all_layer_attentions, start_layer=start_layer)
        return rollout[:,0, 1:]

    def generate_grad_rollout(self, input, index = None, start_layer=0):
        output = self.model(input.cuda())
        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        #################### attn

        blocks = self.model.blocks
        all_layer_attentions = []
        all_head_grad_attentions = []
        all_head_attentions = []

        # cams, grads = [], []

        for blk in blocks:
            cam = blk.attn.get_attn()
            grad = blk.attn.get_attn_gradients()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])

            # cams.append(cam)
            # grads.append(grad)

            # log, cam = all_head_attentions
            all_head_attentions.append(cam.detach())

            cam = grad * cam
            cam2 = cam
            all_head_grad_attentions.append(cam2.detach()) # change to all_head_grad_attentions
            cam = cam.clamp(min=0).mean(dim=0)
            all_layer_attentions.append(cam.unsqueeze(0))

        # print(len(all_layer_attentions)) 
        # print(len(all_head_attentions))        
        # print(all_layer_attentions[0].shape)
        # print(all_head_attentions[0].shape)
        rollout, prop_lw_attn = compute_rollout_attention(all_layer_attentions, start_layer=start_layer)  
        # all_layer_rolled = []
        # for i in (0, 12):
        #     rolled = compute_rollout_attention(all_layer_attentions, start_layer=i)
        #     all_layer_rolled.append(rolled)

        # torch.save({'cams': cams, 'grads':grads, 'attention_maps': all_layer_attentions, 'final_map': rollout}, 'grad_rollout_maps.pt')
        # import pdb; pdb.set_trace()
        return rollout[:,0, 1:], all_head_attentions, all_head_grad_attentions, all_layer_attentions, prop_lw_attn
