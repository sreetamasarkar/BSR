###################################

# heavily borrow from

import copy
import torch
import torch.nn as nn
# import mesa as ms
# from ofa.utils import Hswish, Hsigmoid, MyConv2d

# from ofa.utils.layers import ResidualBlock
from torchvision.models.resnet import BasicBlock, Bottleneck
# from torchvision.models.mobilenet import InvertedResidual

from timm.models.vision_transformer import Attention, Mlp
from models.reprogram import MainAttention
# from ViT.models.modeling_new_prune import AttentionActPrune, MlpActPrune
from models.reprogram import ReprogramViT

from pdb import set_trace
# from custom_functions.custom_fc import LinearSparse
# from custom_functions.custom_softmax import SoftmaxSparse
# from custom_functions.custom_gelu import GELUSparse
# from custom_functions.custom_layer_norm import LayerNormSparse
from functools import partial
from timm.models.vision_transformer import VisionTransformer
from torchprofile import profile_macs
import models.reprogram as reprogram

__all__ = ['count_model_size', 'count_activation_size', 'profile_memory_cost']


def count_model_size(net, trainable_param_bits=32, frozen_param_bits=8, print_log=True):
	frozen_param_bits = 32 if frozen_param_bits is None else frozen_param_bits

	trainable_param_size = 0
	frozen_param_size = 0
	for p in net.parameters():
		if p.requires_grad:
			trainable_param_size += trainable_param_bits / 8 * p.numel()
		else:
			frozen_param_size += frozen_param_bits / 8 * p.numel()
	model_size = trainable_param_size + frozen_param_size
	if print_log:
		print('Total: %d' % model_size,
		      '\tTrainable: %d (data bits %d)' % (trainable_param_size, trainable_param_bits),
		      '\tFrozen: %d (data bits %d)' % (frozen_param_size, frozen_param_bits))
	# Byte
	return model_size


def is_leaf(m_):
	return len(list(m_.children())) == 0 or (len(list(m_.children())) == 1 and isinstance(next(m_.children()), nn.Identity))
		# isinstance(m_, LinearSparse) or isinstance(m_, SoftmaxSparse) or \
		#    isinstance(m_, GELUSparse) or isinstance(m_, LayerNormSparse) or \
		   

def count_activation_size(net, input_size=(1, 3, 224, 224), require_backward=True, activation_bits=32, head_only=False):
	act_byte = activation_bits / 8
	model = copy.deepcopy(net)

	# noinspection PyArgumentList
	def count_convNd(m, x, y):
		# count activation size required by backward
		if m.weight is not None and m.weight.requires_grad:
			m.grad_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes
		else:
			m.grad_activations = torch.Tensor([0])
		# temporary memory footprint required by inference
		m.tmp_activations = torch.Tensor([x[0].numel() * act_byte + y.numel() * act_byte // m.groups])  # bytes

	# noinspection PyArgumentList
	def count_linear(m, x, y):
		# print("count_linear")
		# count activation size required by backward
		if m.weight is not None and m.weight.requires_grad:
			m.grad_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes
		else:
			m.grad_activations = torch.Tensor([0])

		# if isinstance(m, LinearSparse) and m.masker is not None:
		# 	if m.half:
		# 		ratio = 0.5
		# 	else:
		# 		ratio = 1

		# 	mask = m.masker(x[0])
		# 	# print("mlp density is {}".format(mask.float().mean().cpu()))
		# 	m.grad_activations *= mask.float().mean().cpu() * ratio
		# 	if m.quantize:
		# 		m.grad_activations *= 0.25
		# 	m.grad_activations += (mask.numel() / 8)

		# temporary memory footprint required by inference
		m.tmp_activations = torch.Tensor([x[0].numel() * act_byte + y.numel() * act_byte])  # bytes

	# noinspection PyArgumentList
	def count_quantized_linear(m, x, y):
		# count activation size required by backward
		if m.weight is not None and m.weight.requires_grad:
			if m.enable:
				ratio = 0.25
			else:
				ratio = 1.0
			# print("count_quantized_linear, enable {}".format(m.enable))
			m.grad_activations = torch.Tensor([x[0].numel() * act_byte * ratio])  # bytes
		else:
			m.grad_activations = torch.Tensor([0])

		# temporary memory footprint required by inference
		m.tmp_activations = torch.Tensor([x[0].numel() * act_byte + y.numel() * act_byte])  # bytes

	# noinspection PyArgumentList
	def count_bn(m, x, _):
		# print("count LN")
		# count activation size required by backward
		if m.weight is not None and m.weight.requires_grad:
			m.grad_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes
		else:
			m.grad_activations = torch.Tensor([0])

		# if isinstance(m, LayerNormSparse) and m.masker is not None:
		# 	mask = m.masker(x[0])
		# 	# print("layer norm density is {}".format(mask.float().mean().cpu()))
		# 	m.grad_activations *= mask.float().mean().cpu()
		# 	if m.quantize:
		# 		m.grad_activations *= 0.25
		# 	m.grad_activations += (mask.numel() / 8)

		# temporary memory footprint required by inference
		m.tmp_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes

	# noinspection PyArgumentList
	def count_quantized_bn(m, x, _):
		# count activation size required by backward
		if m.weight is not None and m.weight.requires_grad:
			if m.enable:
				ratio = 0.25
			else:
				ratio = 1.0
			# print("count quantized LN, enable {}".format(m.enable))
			m.grad_activations = torch.Tensor([x[0].numel() * act_byte * ratio])  # bytes
		else:
			m.grad_activations = torch.Tensor([0])

		# temporary memory footprint required by inference
		m.tmp_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes

	# noinspection PyArgumentList
	def count_relu(m, x, _):
		# count activation size required by backward
		if require_backward:
			m.grad_activations = torch.Tensor([x[0].numel() / 8])  # bytes
		else:
			m.grad_activations = torch.Tensor([0])
		# temporary memory footprint required by inference
		m.tmp_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes

	# noinspection PyArgumentList
	def count_smooth_act(m, x, _):
		# print("count gelu")
		# count activation size required by backward
		if require_backward:
			m.grad_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes
		else:
			m.grad_activations = torch.Tensor([0])
		if hasattr(m, 'requires_backward'):
			if not m.requires_backward:
				m.grad_activations = torch.Tensor([0])

		# temporary memory footprint required by inference
		m.tmp_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes

	def add_hooks(m_):
		# if isinstance(m_, nn.GELU):
		# 	set_trace()

		if not is_leaf(m_):
			return

		m_.register_buffer('grad_activations', torch.zeros(1))
		m_.register_buffer('tmp_activations', torch.zeros(1))

		if type(m_) in [nn.Conv1d, nn.Conv2d, nn.Conv3d]:
			fn = count_convNd
		elif type(m_) in [nn.Linear]:
			fn = count_linear
		elif type(m_) in [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm, nn.LayerNorm]:
			fn = count_bn
		elif type(m_) in [nn.ReLU, nn.ReLU6, nn.LeakyReLU]:
			fn = count_relu
		elif type(m_) in [nn.Sigmoid, nn.Tanh, nn.GELU]:
			fn = count_smooth_act
		else:
			fn = None

		if fn is not None:
			_handler = m_.register_forward_hook(fn)

	model.train()
	model.apply(add_hooks)

	x = torch.randn(input_size).to(model.parameters().__next__().device)
	with torch.no_grad():
		model(x)

	memory_info_dict = {
		'peak_activation_size': torch.zeros(1),
		'grad_activation_size': torch.zeros(1),
		'residual_size': torch.zeros(1),
	}

	for m in model.modules():
		if is_leaf(m):
			def new_forward(_module):
				def lambda_forward(*args, **kwargs):
					current_act_size = _module.tmp_activations + memory_info_dict['grad_activation_size'] + \
					                   memory_info_dict['residual_size']
					memory_info_dict['peak_activation_size'] = max(
						current_act_size, memory_info_dict['peak_activation_size']
					)
					memory_info_dict['grad_activation_size'] += _module.grad_activations
					return _module.old_forward(*args, **kwargs)

				return lambda_forward

			m.old_forward = m.forward
			m.forward = new_forward(m)

		# if isinstance(m, Attention) or isinstance(m, AttentionActPrune):
		if isinstance(m, Attention) or isinstance(m, MainAttention):
			def new_forward(_module):
				def lambda_forward(_x, *args, **kwargs):
					# print("count Attention")
					memory_info_dict['residual_size'] = _x[0].numel() * act_byte
					# save key, query, value
					# print("_x.shape is {}".format(_x.shape))
					n_tokens = _x.shape[1]
					# set_trace()

					# if isinstance(m, AttentionActPrune) and _module.query.enable:
					# 	backward_act_byte = 1
					# elif isinstance(_module, AttentionActPrune) and _module.query.quantize:
					# 	backward_act_byte = 1
					# else:
					backward_act_byte = act_byte

					# print("attention, backward_act_byte is {}".format(backward_act_byte))
					# if isinstance(_module, AttentionActPrune):
					# 	attn_prune_ratio = _module.masker.prune_ratio
					# 	# print("attn_prune_ratio is {}".format(attn_prune_ratio))
					# 	# attn matrix
					# 	if _module.query.half:
					# 		assert backward_act_byte == 4
					# 		ratio = 0.5
					# 	else:
					# 		ratio = 1

					# 	# attn_matrix
					# 	memory_info_dict['grad_activation_size'] += _module.num_attention_heads * n_tokens * n_tokens * (backward_act_byte) * (1 - attn_prune_ratio) * ratio
					# 	memory_info_dict['grad_activation_size'] += _module.num_attention_heads * n_tokens * n_tokens * 1/8
					# 	# key, query, value
					# 	memory_info_dict['grad_activation_size'] += 3 * _module.all_head_size * n_tokens * (backward_act_byte) * (1 - attn_prune_ratio) * ratio
					# 	memory_info_dict['grad_activation_size'] += 3 * _module.all_head_size * n_tokens * 1/8

					# else:
					# attn_matrix
					if hasattr(_module, 'requires_backward'):
						if not _module.requires_backward:
							memory_info_dict['grad_activation_size'] += 0
					else:
						memory_info_dict['grad_activation_size'] += model.num_attention_heads * n_tokens * n_tokens * backward_act_byte
						memory_info_dict['grad_activation_size'] += 3 * model.all_head_size * n_tokens * backward_act_byte

					if head_only:
						memory_info_dict['grad_activation_size'] *= 0
					# check if requires_backward is an attribute of the module
					# if not, we assume that the module requires backward
											
					result = _module.old_forward(_x, *args, **kwargs)
					memory_info_dict['residual_size'] = 0
					return result

				return lambda_forward

			m.old_forward = m.forward
			m.forward = new_forward(m)

		# if isinstance(m, Mlp) or isinstance(m, MlpActPrune):
		# if isinstance(m, Mlp):
		# 	def new_forward(_module):
		# 		def lambda_forward(_x, *args, **kwargs):
		# 			# print("count Mlp")
		# 			memory_info_dict['residual_size'] = _x[0].numel() * act_byte

		# 			# gelu function
		# 			# print("_x.shape is {}".format(_x.shape))
		# 			n_tokens = _x.shape[1]
		# 			# set_trace()

		# 			# if isinstance(m, MlpActPrune) and _module.act_fn.enable:
		# 			# 	backward_act_byte = 1
		# 			# elif isinstance(_module, MlpActPrune) and _module.act_fn.quantize:
		# 			# 	backward_act_byte = 1
		# 			# else:
		# 			backward_act_byte = act_byte

		# 			# print("mlp, backward_act_byte is {}".format(backward_act_byte))

		# 			# if isinstance(_module, MlpActPrune):
		# 			# 	# if isinstance(_module.act_fn, ms.GELU):
		# 			# 	if _module.act_fn.masker is not None:
		# 			# 		ratio = _module.act_fn.masker.prune_ratio
		# 			# 		memory_info_dict['grad_activation_size'] += _module.fc1.out_features * n_tokens / 8
		# 			# 	else:
		# 			# 		ratio = 1

		# 			# 	if _module.act_fn.half and backward_act_byte == 4:
		# 			# 		ratio *= 0.5

		# 			# 	memory_info_dict['grad_activation_size'] += _module.fc1.out_features * n_tokens * backward_act_byte * ratio
		# 			# else:
		# 			memory_info_dict['grad_activation_size'] += _module.fc1.out_features * n_tokens * backward_act_byte

		# 			if head_only:
		# 				memory_info_dict['grad_activation_size'] *= 0

		# 			result = _module.old_forward(_x, *args, **kwargs)
		# 			memory_info_dict['residual_size'] = 0
		# 			return result

		# 		return lambda_forward

		# 	m.old_forward = m.forward
		# 	m.forward = new_forward(m)

	with torch.no_grad():
		model(x)

	return memory_info_dict['peak_activation_size'].item(), memory_info_dict['grad_activation_size'].item()


def profile_memory_cost(net, input_size=(1, 3, 224, 224), require_backward=True, head_only=False,
                        activation_bits=32, trainable_param_bits=32, frozen_param_bits=8, batch_size=8):
	param_size = count_model_size(net, trainable_param_bits, frozen_param_bits, print_log=True)
	activation_size, grad_activation_size = count_activation_size(net, input_size, require_backward, activation_bits, head_only)

	MB = 1024 * 1024
	print("grad activation size is {:.1f} MB".format(grad_activation_size / MB))
	memory_cost = activation_size * batch_size + param_size
	return memory_cost, {'param_size': param_size, 'act_size': activation_size}

if __name__ == '__main__':
	# reprogram_network_params = None
	reprogram_network_params = {'num_heads':6, 'mlp_ratio':4, 'qkv_bias':True, 'act_layer':nn.GELU, 'main_branch_ratio':1.0}
	# # print('reprogram_network_params', reprogram_network_params)
	# reprogram_index = [2]
	reprogram_index = None
	main_branch_index = [3, 7, 11]
	drop_loc = [3, 6, 9]
	base_keep_rate = 0.5
	num_classes = 10
	model = ReprogramViT(
	        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
	        norm_layer=partial(nn.LayerNorm, eps=1e-6), reprogram_index=reprogram_index, drop_loc=drop_loc, reprogram_network_params=reprogram_network_params, base_keep_rate=base_keep_rate, num_classes=num_classes)
	setattr(model, "num_attention_heads", 6)
	setattr(model, "all_head_size", 384) # H * L//H

	num_classes = 100
	model_type = 'deit_small_patch16'
	reprogram_index = None
	base_keep_rate = 0.5
	drop_loc = [3, 6, 9]
	main_branch_index = [3, 7, 11]
	fuse_token = True
	model = reprogram.__dict__[model_type](
        num_classes=num_classes,
        reprogram_index= reprogram_index,
        base_keep_rate= base_keep_rate,
        drop_loc = drop_loc,
        fuse_token = fuse_token,
        # global_pool=args.global_pool
    )

	preserve_layers = ['reprogram', 'head'] + [str(x) for x in main_branch_index]
	for name, param in model.named_parameters():
		if not any(x in name for x in preserve_layers):
			print('Setting gradients false for layer: {}'.format(name))
			param.requires_grad = False
	for name, param in model.named_parameters():
		if 'norm.weight' in name or 'norm.bias' in name:
			print('Setting gradients false for layer: {}'.format(name))
			param.requires_grad = True

	no_grad_blocks = [model.blocks[i] for i in range(12) if i < main_branch_index[0]]
	for block in no_grad_blocks:
		for m in block.modules():
			if isinstance(m, Attention) or isinstance(m, MainAttention) or isinstance(m, nn.GELU):
				setattr(m, 'requires_backward', False)
				print(f'Setting grad for module {m} as False')
	# small
	# model = VisionTransformer(
	# 		patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
	# 		norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=10)
	# setattr(model, "num_attention_heads", 6)
	# setattr(model, "all_head_size", 384) # H * L//H
	# base
	# model = VisionTransformer(
	#         img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
	#         norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=100)
	# setattr(model, "num_attention_heads", 12)
	# setattr(model, "all_head_size", 768) # H * L//H
	

	activation_bits = 32
	total_memory_cost, memory_cost_dict = profile_memory_cost(model, input_size=(1, 3, 224, 224), require_backward=True,
														activation_bits=activation_bits, trainable_param_bits=32,
														head_only=False,
														frozen_param_bits=8, batch_size=128)
	MB = 1024 * 1024

	# total_memory_cost, memory_cost_dict = profile_memory_cost(model, input_size=(128, 3, 224, 224))
	print('Total train memory:', total_memory_cost/MB)
	print('Param size:', memory_cost_dict['param_size'] / MB)
	print('Act size:', memory_cost_dict['act_size'] / MB)

	# input_size=(1, 3, 224, 224)
	# sample_input = torch.randn(input_size)
	# macs = profile_macs(model, sample_input)
	# print("MACS: {} G".format(macs / 1e9))