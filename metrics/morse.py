import torch
import torch.nn as nn
import torch.autograd as ag


def eikonal_loss(non_manifold_grad, manifold_grad, eikonal_type='abs'):
	# Compute the eikonal loss that penalises when ||grad(f)|| != 1 for points on and off the manifold
	# shape is (bs, num_points, dim=3) for both grads
	# Eikonal
	all_grads = None
	if non_manifold_grad is not None and manifold_grad is not None:
		all_grads = torch.cat([non_manifold_grad, manifold_grad], dim=-2)
	elif non_manifold_grad is not None:
		all_grads = non_manifold_grad
	elif manifold_grad is not None:
		all_grads = manifold_grad

	if eikonal_type == 'abs':
		eikonal_term = ((all_grads.norm(2, dim=2) - 1).abs()).mean()
	else:
		eikonal_term = ((all_grads.norm(2, dim=2) - 1).square()).mean()

	return eikonal_term


def relax_eikonal_loss(non_manifold_grad, manifold_grad, minim=.8,  eikonal_type='abs'):
	# Compute the eikonal loss that penalises when ||grad(f)|| != 1 for points on and off the manifold
	# shape is (bs, num_points, dim=3) for both grads
	# Eikonal
	all_grads = None
	if non_manifold_grad is not None and manifold_grad is not None:
		all_grads = torch.cat([non_manifold_grad, manifold_grad], dim=-2)
	elif non_manifold_grad is not None:
		all_grads = non_manifold_grad
	elif manifold_grad is not None:
		all_grads = manifold_grad

	grad_norm = all_grads.norm(2, dim=-1) + 1e-12
	term = torch.relu(-(grad_norm - torch.tensor(minim)))
	if eikonal_type == 'abs':
		eikonal_term = term.abs().mean()
	else:
		eikonal_term = term.square().mean()
	return eikonal_term


def latent_rg_loss(latent_reg, device):
	# compute the VAE latent representation regularization loss
	if latent_reg is not None:
		reg_loss = latent_reg.mean()
	else:
		reg_loss = torch.tensor([0.0], device=device)

	return reg_loss


def gradient(inputs, outputs, create_graph=True, retain_graph=True):
	d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
	points_grad = ag.grad(
		outputs=outputs,
		inputs=inputs,
		grad_outputs=d_points,
		create_graph=create_graph,
		retain_graph=retain_graph,
		only_inputs=True)[0]  # [:, -3:]
	return points_grad


class MorseLoss(nn.Module):
	def __init__(self, weights=None, loss_type='siren_wo_n_w_morse', div_decay='none', div_type='l1',
				bidirectional_morse=True, udf=False):
		super().__init__()
		if weights is None:
			weights = [3e3, 1e2, 1e2, 5e1, 1e2, 1e1]
		self.weights = weights  # sdf, intern, normal, eikonal, div
		self.loss_type = loss_type
		self.div_decay = div_decay
		self.div_type = div_type
		self.use_morse = True if 'morse' in self.loss_type else False
		self.bidirectional_morse = bidirectional_morse
		self.udf = udf
		self.decay_params_list = None

	def forward(self, output_pred, manifold_points, non_manifold_points, manifold_n_gt=None, near_points=None):
		dims = manifold_points.shape[-1]
		device = manifold_points.device

		#########################################
		# Compute required terms
		#########################################

		non_manifold_pred = output_pred["non_manifold_pts_pred"]
		manifold_pred = output_pred["manifold_pts_pred"]
		latent_reg = output_pred["latent_reg"]

		div_loss = torch.tensor([0.0], device=manifold_points.device)
		morse_loss = torch.tensor([0.0], device=manifold_points.device)
		curv_term = torch.tensor([0.0], device=manifold_points.device)
		# latent_reg_term = torch.tensor([0.0], device=manifold_points.device)
		normal_term = torch.tensor([0.0], device=manifold_points.device)

		# compute gradients for div (divergence), curl and curv (curvature)
		if manifold_pred is not None:
			manifold_grad = gradient(manifold_points, manifold_pred)
		else:
			manifold_grad = None

		non_manifold_grad = gradient(non_manifold_points, non_manifold_pred)

		morse_non_manifold_points = None
		morse_non_manifold_grad = None
		if self.use_morse and near_points is not None:
			morse_non_manifold_points = near_points
			morse_non_manifold_grad = gradient(near_points, output_pred['near_pts_pred'])
		elif self.use_morse and near_points is None:
			morse_non_manifold_points = non_manifold_points
			morse_non_manifold_grad = non_manifold_grad

		if self.use_morse:
			non_manifold_dx = gradient(morse_non_manifold_points, morse_non_manifold_grad[:, :, 0])
			non_manifold_dy = gradient(morse_non_manifold_points, morse_non_manifold_grad[:, :, 1])

			manifold_dx = gradient(manifold_points, manifold_grad[:, :, 0])
			manifold_dy = gradient(manifold_points, manifold_grad[:, :, 1])
			if dims == 3:
				non_manifold_dz = gradient(morse_non_manifold_points, morse_non_manifold_grad[:, :, 2])
				non_manifold_hessian_term = torch.stack((non_manifold_dx, non_manifold_dy, non_manifold_dz), dim=-1)

				manifold_dz = gradient(manifold_points, manifold_grad[:, :, 2])
				manifold_hessian_term = torch.stack((manifold_dx, manifold_dy, manifold_dz), dim=-1)
			else:
				non_manifold_hessian_term = torch.stack((non_manifold_dx, non_manifold_dy), dim=-1)
				manifold_hessian_term = torch.stack((manifold_dx, manifold_dy), dim=-1)

			non_manifold_det = torch.det(non_manifold_hessian_term)
			manifold_det = torch.det(manifold_hessian_term)

			morse_manifold = torch.tensor([0.0], device=manifold_points.device)
			morse_non_manifold = torch.tensor([0.0], device=manifold_points.device)
			if self.div_type == 'l2':
				morse_non_manifold = non_manifold_det.square().mean()
				if self.bidirectional_morse:
					morse_manifold = manifold_det.square().mean()
			elif self.div_type == 'l1':
				morse_non_manifold = non_manifold_det.abs().mean()
				# morse_non_manifold = morse_non_manifold_grad.norm(dim=-1).square().mean()
				# morse_non_manifold = non_manifold_hessian_term.norm(dim=[-1, -2]).square().mean()
				# non_manifold_divergence = non_manifold_dx[:, :, 0] + non_manifold_dy[:, :, 1] + non_manifold_dz[:, :, 2]
				# morse_non_manifold = torch.clamp(torch.abs(non_manifold_divergence), 0.1, 50).mean()
				if self.bidirectional_morse:
					morse_manifold = manifold_det.abs().mean()

			morse_loss = 0.5 * (morse_non_manifold + morse_manifold)

		# latent regularization for multiple shape learning
		latent_reg_term = latent_rg_loss(latent_reg, device)

		# normal term
		if manifold_n_gt is not None:
			if 'igr' in self.loss_type:
				normal_term = ((manifold_grad - manifold_n_gt).abs()).norm(2, dim=1).mean()
			else:
				normal_term = (
						1 - torch.abs(torch.nn.functional.cosine_similarity(manifold_grad, manifold_n_gt, dim=-1))).mean()

		# signed distance function term
		sdf_term = torch.abs(manifold_pred).mean()
		# sdf_term = (torch.abs(manifold_pred) * torch.exp(manifold_pred.abs())).mean()

		# eikonal term
		# eikonal_term = eikonal_loss(non_manifold_grad, manifold_grad=manifold_grad, eikonal_type='abs')
		# Sometimes > relax may lead to bad results, use another type relax
		# eikonal_term = relax_eikonal_loss(None, manifold_grad=manifold_grad, udf=self.udf)
		eikonal_term = eikonal_loss(None, manifold_grad=manifold_grad, eikonal_type='abs')

		# inter term
		inter_term = torch.exp(-1e2 * torch.abs(non_manifold_pred)).mean()
		#########################################
		# Losses
		#########################################

		# losses used in the paper
		if self.loss_type == 'siren':  # SIREN loss
			loss = (self.weights[0] * sdf_term + self.weights[1] * inter_term +
					self.weights[2] * normal_term + self.weights[3] * eikonal_term)
		elif self.loss_type == 'siren_w_morse':
			loss = (self.weights[0] * sdf_term + self.weights[1] * inter_term +
					self.weights[2] * normal_term + self.weights[3] * eikonal_term +
					self.weights[4] * morse_loss)
		elif self.loss_type == 'siren_wo_n':  # SIREN loss without normal constraint
			self.weights[2] = 0
			loss = self.weights[0] * sdf_term + self.weights[1] * inter_term + self.weights[3] * eikonal_term
		elif self.loss_type == 'siren_wo_n_w_morse':
			self.weights[2] = 0
			loss = (self.weights[0] * sdf_term + self.weights[1] * inter_term + self.weights[3] * eikonal_term +
					self.weights[5] * morse_loss)
		elif self.loss_type == 'siren_wo_n_wo_e_wo_morse':
			loss = self.weights[0] * sdf_term + self.weights[1] * inter_term
		elif self.loss_type == 'igr':  # IGR loss
			self.weights[1] = 0
			loss = self.weights[0] * sdf_term + self.weights[2] * normal_term + self.weights[3] * eikonal_term
		elif self.loss_type == 'igr_wo_n':  # IGR without normals loss
			self.weights[1] = 0
			self.weights[2] = 0
			loss = self.weights[0] * sdf_term + self.weights[3] * eikonal_term
		elif self.loss_type == 'igr_wo_n_w_morse':
			self.weights[1] = 0
			self.weights[2] = 0
			loss = self.weights[0] * sdf_term + self.weights[3] * eikonal_term + self.weights[5] * morse_loss
		elif self.loss_type == 'siren_w_div':  # SIREN loss with divergence term
			loss = (self.weights[0] * sdf_term + self.weights[1] * inter_term +
					self.weights[2] * normal_term + self.weights[3] * eikonal_term +
					self.weights[4] * div_loss)
		elif self.loss_type == 'siren_wo_e_w_morse':
			self.weights[3] = 0
			self.weights[4] = 0
			loss = (self.weights[0] * sdf_term + self.weights[1] * inter_term +
					self.weights[2] * normal_term + self.weights[5] * morse_loss)
		elif self.loss_type == 'siren_wo_e_wo_n_w_morse':
			self.weights[2] = 0
			self.weights[3] = 0
			self.weights[4] = 0
			loss = self.weights[0] * sdf_term + self.weights[1] * inter_term + self.weights[5] * morse_loss
		elif self.loss_type == 'siren_wo_n_w_div':  # SIREN loss without normals and with divergence constraint
			loss = (self.weights[0] * sdf_term + self.weights[1] * inter_term + self.weights[3] * eikonal_term +
					self.weights[4] * div_loss)
		else:
			print(self.loss_type)
			raise Warning("unrecognized loss type")

		# If multiple surface reconstruction, then latent and latent_reg are defined so reg_term need to be used
		if latent_reg is not None:
			loss += self.weights[6] * latent_reg_term

		return {"loss": loss, 'sdf_term': sdf_term, 'inter_term': inter_term, 'latent_reg_term': latent_reg_term,
				'eikonal_term': eikonal_term, 'normals_loss': normal_term, 'div_loss': div_loss,
				'curvature_loss': curv_term.mean(), 'morse_term': morse_loss}, manifold_grad

	def update_morse_weight(self, current_iteration, n_iterations, params=None):
		if not hasattr(self, 'decay_params_list'):
			assert len(params) >= 2, params
			assert len(params[1:-1]) % 2 == 0
			self.decay_params_list = list(zip([params[0], *params[1:-1][1::2], params[-1]], [0, *params[1:-1][::2], 1]))

		curr = current_iteration / n_iterations
		we, e = min([tup for tup in self.decay_params_list if tup[1] >= curr], key=lambda tup: tup[1])
		w0, s = max([tup for tup in self.decay_params_list if tup[1] <= curr], key=lambda tup: tup[1])

		# Divergence term annealing functions
		if self.div_decay == 'linear':  # linearly decrease weight from iter s to iter e
			if current_iteration < s * n_iterations:
				self.weights[5] = w0
			elif e * n_iterations > current_iteration >= s * n_iterations:
				self.weights[5] = w0 + (we - w0) * (current_iteration / n_iterations - s) / (e - s)
			else:
				self.weights[5] = we
		elif self.div_decay == 'quintic':  # linearly decrease weight from iter s to iter e
			if current_iteration < s * n_iterations:
				self.weights[5] = w0
			elif e * n_iterations > current_iteration >= s * n_iterations:
				self.weights[5] = w0 + (we - w0) * (1 - (1 - (current_iteration / n_iterations - s) / (e - s)) ** 5)
			else:
				self.weights[5] = we
		elif self.div_decay == 'step':  # change weight at s
			if current_iteration < s * n_iterations:
				self.weights[5] = w0
			else:
				self.weights[5] = we
		elif self.div_decay == 'none':
			pass
		else:
			raise Warning("unsupported div decay value")


class HessianSimpleLoss(nn.Module):
	def __init__(self, weights=None, div_decay='none', div_type='l1', bidirectional_morse=True):
		super().__init__()
		if weights is None:
			weights = [7e3, 6e2, 5e1, 3, 1]
		self.weights = weights  # sdf_manifold, sdf_non_manifold, eikonal, hessian, latent_reg
		self.div_decay = div_decay
		self.div_type = div_type
		self.bidirectional_morse = bidirectional_morse
		self.decay_params_list = None

	def forward(self, output_pred, manifold_points, non_manifold_points, near_points=None):
		dims = manifold_points.shape[-1]
		device = manifold_points.device

		#########################################
		# Compute required terms
		#########################################

		manifold_pred = output_pred["manifold_pts_pred"]
		non_manifold_pred = output_pred["non_manifold_pts_pred"]
		near_pred = output_pred["near_pts_pred"]
		latent_reg = output_pred["latent_reg"]

		# signed distance function term for points on surface
		sdf_term_manifold = torch.abs(manifold_pred).mean()

		# signed distance function term for points not on surface
		sdf_term_non_manifold = torch.exp(-1e2 * torch.abs(non_manifold_pred)).mean()

		# compute gradients for div (divergence), curl and curv (curvature)
		manifold_grad = gradient(manifold_points, manifold_pred)
		non_manifold_grad = gradient(non_manifold_points, non_manifold_pred)
		near_grad = gradient(near_points, near_pred)

		# eikonal term
		eikonal_term = eikonal_loss(non_manifold_grad=non_manifold_grad, manifold_grad=manifold_grad, eikonal_type='abs')

		# hessian term
		near_dx = gradient(near_points, near_grad[:, :, 0])
		near_dy = gradient(near_points, near_grad[:, :, 1])

		manifold_dx = gradient(manifold_points, manifold_grad[:, :, 0])
		manifold_dy = gradient(manifold_points, manifold_grad[:, :, 1])
		if dims == 3:
			near_dz = gradient(near_points, near_grad[:, :, 2])
			near_hessian = torch.stack((near_dx, near_dy, near_dz), dim=-1)

			manifold_dz = gradient(manifold_points, manifold_grad[:, :, 2])
			manifold_hessian = torch.stack((manifold_dx, manifold_dy, manifold_dz), dim=-1)
		else:
			near_hessian = torch.stack((near_dx, near_dy), dim=-1)
			manifold_hessian = torch.stack((manifold_dx, manifold_dy), dim=-1)

		near_det = torch.det(near_hessian)
		manifold_det = torch.det(manifold_hessian)

		manifold_hessian_term = torch.tensor([0.0], device=device)
		near_hessian_term = torch.tensor([0.0], device=device)
		if self.div_type == 'l2':
			near_hessian_term = near_det.square().mean()
			if self.bidirectional_morse:
				manifold_hessian_term = manifold_det.square().mean()
		elif self.div_type == 'l1':
			near_hessian_term = near_det.abs().mean()
			if self.bidirectional_morse:
				manifold_hessian_term = manifold_det.abs().mean()

		hessian_term = 0.5 * (manifold_hessian_term + near_hessian_term)

		# If multiple surface reconstruction, then latent and latent_reg are defined so reg_term need to be used
		# latent regularization for multiple shape learning
		latent_reg_term = latent_rg_loss(latent_reg, device)

		#########################################
		# combined losses
		#########################################

		loss = self.weights[0] * sdf_term_manifold + self.weights[1] * sdf_term_non_manifold + self.weights[
			2] * eikonal_term + self.weights[3] * hessian_term + self.weights[4] * latent_reg_term

		return ({"loss": loss, 'sdf_term_manifold': sdf_term_manifold, 'sdf_term_non_manifold': sdf_term_non_manifold,
				'eikonal_term': eikonal_term, 'hessian_term': hessian_term, 'latent_reg_term': latent_reg_term},
				manifold_grad)

	def update_morse_weight(self, current_iteration, n_iterations, params=None):
		if not hasattr(self, 'decay_params_list'):
			assert len(params) >= 2, params
			assert len(params[1:-1]) % 2 == 0
			self.decay_params_list = list(zip([params[0], *params[1:-1][1::2], params[-1]], [0, *params[1:-1][::2], 1]))

		curr = current_iteration / n_iterations
		we, e = min([tup for tup in self.decay_params_list if tup[1] >= curr], key=lambda tup: tup[1])
		w0, s = max([tup for tup in self.decay_params_list if tup[1] <= curr], key=lambda tup: tup[1])

		# Divergence term annealing functions
		if self.div_decay == 'linear':  # linearly decrease weight from iter s to iter e
			if current_iteration < s * n_iterations:
				self.weights[3] = w0
			elif e * n_iterations > current_iteration >= s * n_iterations:
				self.weights[3] = w0 + (we - w0) * (current_iteration / n_iterations - s) / (e - s)
			else:
				self.weights[3] = we
		elif self.div_decay == 'quintic':  # linearly decrease weight from iter s to iter e
			if current_iteration < s * n_iterations:
				self.weights[3] = w0
			elif e * n_iterations > current_iteration >= s * n_iterations:
				self.weights[3] = w0 + (we - w0) * (1 - (1 - (current_iteration / n_iterations - s) / (e - s)) ** 5)
			else:
				self.weights[3] = we
		elif self.div_decay == 'step':  # change weight at s
			if current_iteration < s * n_iterations:
				self.weights[3] = w0
			else:
				self.weights[3] = we
		elif self.div_decay == 'none':
			pass
		else:
			raise Warning("unsupported div decay value")
