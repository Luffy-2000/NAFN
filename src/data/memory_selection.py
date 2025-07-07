import torch
import numpy as np
import time
from copy import deepcopy
from typing import Iterable, Optional, Union, List
from torch.utils.data import Dataset, DataLoader
from torch import nn
import math
from networks.network import LLL_Net

class ExemplarsSelector:
	"""Exemplar selector for approaches with an interface of Dataset"""

	def __init__(self, exemplars_dataset, max_num_exemplars=100, max_num_exemplars_per_class=10):
		self.exemplars_dataset = exemplars_dataset
		self.already_added_classes = ()
		self.max_num_exemplars = max_num_exemplars
		self.max_num_exemplars_per_class = max_num_exemplars_per_class

	def __call__(self, model: LLL_Net, trn_loader: DataLoader, transform, t=None, from_inputs=False,
				 clean_memory=False, alternate=False, step_selection=False):
		tmp_model = deepcopy(model)
		# Management of the pre-allocated-output: the non used heads info are removed
		if t is not None:
			tmp_model._modules['heads'] = tmp_model._modules['heads'][:t + 1]
			tmp_model.task_cls = tmp_model.task_cls[:t + 1]
			tmp_model.task_offset = tmp_model.task_offset[:t + 1]
		clock0 = time.time()
		exemplars_per_class = self._exemplars_per_class_num(tmp_model)

		sel_loader = DataLoader(trn_loader.dataset, batch_size=trn_loader.batch_size, shuffle=False,
							num_workers=trn_loader.num_workers, pin_memory=trn_loader.pin_memory)
		selected_indices = self._select_indices(
			tmp_model, sel_loader, exemplars_per_class, transform, from_inputs, clean_memory, alternate, step_selection)
		# self.already_added_classes = set(_get_labels(sel_loader))

		x, y = zip(*(trn_loader.dataset[idx] for idx in selected_indices))
		x = torch.tensor(np.array(x))
		y = torch.stack(y)
		clock1 = time.time()
		print('| Selected {:d} train exemplars, time={:5.1f}s'.format(len(x), clock1 - clock0))
		return x, y


	def _exemplars_per_class_num(self, model: LLL_Net):
		if self.max_num_exemplars_per_class:
			return self.max_num_exemplars_per_class
		
		num_cls = model.task_cls.sum().item()
		num_exemplars = self.max_num_exemplars
		exemplars_per_class = int(np.ceil(num_exemplars / num_cls))
		assert exemplars_per_class > 0, \
			"Not enough exemplars to cover all classes!\n" \
			"Number of classes so far: {}. " \
			"Limit of exemplars: {}".format(num_cls,
											num_exemplars)
		return exemplars_per_class 

	def _select_indices(self, model: LLL_Net, sel_loader: DataLoader, exemplars_per_class: int, transform,
						from_inputs=None, clean_memory=False) -> Iterable:
		pass


class HerdingExemplarsSelector(ExemplarsSelector):
	"""Selection of new samples. This is based on herding selection, which produces a sorted list of samples of one
	class based on the distance to the mean sample of that class. From iCaRL algorithm 4 and 5:
	https://openaccess.thecvf.com/content_cvpr_2017/papers/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.pdf
	"""
	def __init__(self, exemplars_dataset, max_num_exemplars=1000, max_num_exemplars_per_class=10):
		super().__init__(
			exemplars_dataset, 
			max_num_exemplars=max_num_exemplars, 
			max_num_exemplars_per_class=max_num_exemplars_per_class
			)

	def _select_indices(self, model: LLL_Net, sel_loader: DataLoader, exemplars_per_class: int, transform,
						from_inputs=False, clean_memory=False, alternate=False, step_selection=False) -> Iterable:
		def format_inputs(device, x):
			'''transform correct format and move to specified device'''
			if isinstance(x, (list, tuple)):
				return [format_inputs(device, xi) for xi in x]
			return x.to(device)
		model_device = next(model.parameters()).device  # Assume model is on a single device

		extracted_features = []
		extracted_targets = []
		clean_index = []
		
		with torch.no_grad():
			model.eval()
			for images, targets in sel_loader:
				images, targets = format_inputs(model_device, images), format_inputs(model_device, targets)
				if not from_inputs:
					logits, feats, recon_x = model(images, return_features=True)
					if clean_memory:
						preds = torch.argmax(logits, dim=1)
						clean_index.extend((targets == preds).cpu().numpy().tolist())
					feats = feats / feats.norm(dim=1, keepdim=True) 
				else:
					feats = images
				extracted_features.append(feats)
				extracted_targets.extend(targets)
		
		extracted_features = torch.cat(extracted_features).cpu()
		extracted_targets = torch.tensor(extracted_targets)
		
		if not clean_index:
			clean_index = torch.ones(len(extracted_targets), dtype=torch.bool)
		else:
			clean_index = torch.tensor(clean_index, dtype=torch.bool)
			# Ensure at least some samples are kept for each class
			for curr_cls in torch.unique(extracted_targets):
				cls_ind = torch.where(extracted_targets == curr_cls)[0]
				cls_clean = clean_index[cls_ind]
				if torch.sum(cls_clean) == 0:  # If all samples of this class are filtered out
					# Select samples with highest prediction probabilities
					with torch.no_grad():
						cls_feats = extracted_features[cls_ind]
						cls_feats = cls_feats.to(model_device)  # Move to same device as model
						# directly use features, no need to pass through model
						print(cls_feats.shape)
						cls_probs = torch.softmax(model.head(cls_feats), dim=1)
						# Select top-k samples with highest prediction probabilities
						k = min(exemplars_per_class, len(cls_ind))
						_, topk_indices = torch.topk(cls_probs.max(dim=1)[0], k)
						# Move indices to CPU before indexing
						topk_indices = topk_indices.cpu()
						clean_index[cls_ind[topk_indices]] = True
		
		result = []
		
		for curr_cls in torch.unique(extracted_targets):
			cls_ind = torch.where((extracted_targets == curr_cls) & clean_index)[0]
			assert len(cls_ind) > 0, f"No samples to choose from for class {curr_cls}"
			
			if exemplars_per_class < len(cls_ind):
				cls_feats = extracted_features[cls_ind]
				cls_feats = cls_feats.to(model_device)  # Move to same device as model
				cls_mu = cls_feats.mean(0)
				
				selected = []
				selected_feat = []
				for k in range(exemplars_per_class):
					if k == 0:
						distances = torch.norm(cls_feats - cls_mu, dim=1)
						newone = torch.argmin(distances).item()
					else:
						sum_others = torch.stack(selected_feat).sum(0) / (k + 1)
						distances = torch.norm(cls_feats - (cls_mu - sum_others), dim=1)
						distances[selected] = float('inf')  # This avoids exemplars reselection
						newone = torch.argmin(distances).item()
					
					newonefeat = cls_feats[newone]
					selected_feat.append(newonefeat)
					selected.append(newone)
				
				result.extend(cls_ind[selected].tolist())
			else:
				print(f'WARNING: Not enough samples to store for class {curr_cls}: selected ALL.')
				result.extend(cls_ind.tolist())
		
		return result


class UncertaintyExemplarsSelector(ExemplarsSelector):
	"""Selection of new samples. This is based on Rainbow Memory proposal (https://arxiv.org/pdf/2103.17230.pdf)
	Implements a sampling based on uncertainty of model to perturbation of input (by employing translating and jittering of IAT field)
	To adopt Rainbow Memory, use this exemplars selector strategy with "step" option activated
	"""

	def __init__(self, exemplars_dataset, max_num_exemplars=1000, max_num_exemplars_per_class=10):
		super().__init__(
			exemplars_dataset, 
			max_num_exemplars=max_num_exemplars, 
			max_num_exemplars_per_class=max_num_exemplars_per_class
			)

	def _select_indices(self, model: LLL_Net, sel_loader: DataLoader, exemplars_per_class: int, transform,
						from_inputs=None, clean_memory=False, alternate=False, step_selection=False) -> Iterable:
		def format_inputs(device, x):
			'''transform correct format and move to specified device'''
			if isinstance(x, (list, tuple)):
				return [format_inputs(device, xi) for xi in x]
			return x.to(device)

		model_device = next(model.parameters()).device  # we assume here that whole model is on a single device
		
		# extract outputs from the model for all train samples
		extracted_logits = []
		extracted_targets = []
		extracted_images = np.array(sel_loader.dataset.images)
		clean_index = []

		with torch.no_grad():
			model.eval()
			for images, targets in sel_loader:
				images = format_inputs(model_device, images)
				targets = format_inputs(model_device, targets)
				logits = model(images)
				if clean_memory:
					preds = torch.argmax(logits, dim=1)
					clean_index.extend((targets == preds).detach().cpu().numpy().tolist())
				extracted_logits.append(logits.cpu())
				extracted_targets.extend(targets.cpu().numpy())
			
		extracted_logits = torch.cat(extracted_logits, dim=0).cpu()
		extracted_targets = np.array(extracted_targets)
		
		if not len(clean_index):
			clean_index = [True] * len(extracted_targets)
		else:
			for i, t in enumerate(extracted_targets):
				clean_index[i] |= (t in self.already_added_classes)
		result = []
		# iterate through all classes
		for curr_cls in np.unique(extracted_targets):
			# get all indices from current class
			cls_ind = np.where((extracted_targets == curr_cls) & clean_index)[0]
			assert (len(cls_ind) > 0), "No samples to choose from for class {:d}".format(curr_cls)
#             assert (exemplars_per_class <= len(cls_ind)), "Not enough samples to store"
			if exemplars_per_class < len(cls_ind):
				# get all extracted features for current class
				cls_logits = extracted_logits[cls_ind]   
				cls_img = extracted_images[cls_ind]
				
				sel_dataset = deepcopy(sel_loader.dataset)
				sel_dataset.images = cls_img
				sel_dataset.labels = [curr_cls] * len(cls_ind)
				
				augmented_images, multiplier = augment_data(sel_dataset.images)
				augmented_dataset = deepcopy(sel_loader.dataset)
				augmented_dataset.images = augmented_images
				augmented_dataset.labels = sel_dataset.labels * (multiplier-1)

				aug_loader = torch.utils.data.DataLoader(augmented_dataset,
									batch_size=len(cls_ind),
									shuffle=False,
									num_workers=0,
									pin_memory=sel_loader.pin_memory)
				
				uncertainty_scores = [0] * len(cls_ind)
				
				for img,lbl in aug_loader:
					aug_logits = torch.cat(model(img), dim=1)
					aug_pred = torch.argmax(aug_logits, dim=1)
					
					uncertainty_scores = np.add(uncertainty_scores,
						[int(x) for x in torch.tensor((aug_pred==lbl)).cpu().numpy()])

				uncertainty_scores = [(1-(1/6)*x) for x in uncertainty_scores]
				sort_index = np.argsort(uncertainty_scores)
				selected = []
				
				if step_selection:
					step = math.ceil(len(uncertainty_scores)/exemplars_per_class)
					selected = [cls_ind[sort_index[t]] for t in range(0, len(uncertainty_scores), step)]
				
					iterator=1
					while len(selected) < exemplars_per_class:
						selected.extend([cls_ind[sort_index[t]] for t in range(iterator, len(uncertainty_scores), step)][:exemplars_per_class-len(selected)])
						iterator=iterator+1
					
					result.extend(selected)
				else:
					selected = cls_ind[sort_index[:exemplars_per_class]]
					result.extend(selected)
			else:
				print('WARNING: Not enough samples to store for class {:d}: selected ALL.'.format(curr_cls))
				result.extend(list(cls_ind))
		return result
