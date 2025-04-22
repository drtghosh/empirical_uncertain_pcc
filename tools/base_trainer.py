import os
import torch
import torch.optim as optim
import torch.nn as nn
from abc import abstractmethod
from tensorboardX import SummaryWriter


class TrainWatcher(object):
	"""
		Class of object to track epoch and step during training
	"""
	def __init__(self):
		self.epoch = 1
		self.minibatch = 0
		self.step = 0

	def within_epoch(self):
		self.minibatch += 1
		self.step += 1

	def new_epoch(self):
		self.epoch += 1
		self.minibatch = 0

	def make_checkpoint(self):
		return {
			'epoch': self.epoch,
			'minibatch': self.minibatch,
			'step': self.step
		}

	def restore_checkpoint(self, clock_dict):
		self.epoch = clock_dict['epoch']
		self.minibatch = clock_dict['minibatch']
		self.step = clock_dict['step']


class TrainerCommon(object):
	"""
		Base trainer that provides common training process with single models and loss function.
		All customized trainer for training solitary models should be a subclass of this class.
	"""
	def __init__(self, config):
		# paths/ directories
		self.log_dir = config.log_dir
		self.model_dir = config.model_dir

		# device
		self.device = config.device

		# watcher
		self.watcher = TrainWatcher()

		# training batch size
		self.batch_size = config.batch_size

		# build network
		self.model = self.build_model(config)

		# set loss function
		self.criterion = None
		self.set_loss_function()

		# set optimizer
		self.base_lr = None
		self.optimizer = None
		self.set_optimizer(config)

		# set lr scheduler
		self.scheduler = None
		self.set_scheduler(config)

		# store predicted points
		self.predicted_pts = None

		# store loss values
		self.loss = None

		# set tensorboard writer
		self.train_tbw = SummaryWriter(os.path.join(self.log_dir, 'train.events'))
		self.val_tbw = SummaryWriter(os.path.join(self.log_dir, 'val.events'))

	@abstractmethod
	def build_model(self, config):
		raise NotImplementedError

	def set_loss_function(self):
		"""
			set loss function used in training
		"""
		self.criterion = nn.MSELoss().to(self.device)

	@abstractmethod
	def collect_loss(self):
		"""
			collect all losses into a dict
		"""
		raise NotImplementedError

	def set_optimizer(self, config):
		"""
			set optimizer used in training
		"""
		self.base_lr = config.lr
		self.optimizer = optim.Adam(self.model.parameters(), config.lr)

	def set_scheduler(self, config):
		"""
			set lr scheduler used in training
		"""
		# self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, config.lr_step_size)
		self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, config.lr_decay)

	def save_ckpt(self, name=None):
		"""
			save checkpoint during training for future reload
		"""
		if name is None:
			save_path = os.path.join(self.model_dir, "ckpt_epoch{}.pth".format(self.watcher.epoch))
			print("Saving checkpoint epoch {}...".format(self.watcher.epoch))
		else:
			save_path = os.path.join(self.model_dir, "{}.pth".format(name))

		if isinstance(self.model, nn.DataParallel):
			model_state_dict = self.model.module.cpu().state_dict()
		else:
			model_state_dict = self.model.cpu().state_dict()

		torch.save({
			'watcher': self.watcher.make_checkpoint(),
			'model_state_dict': model_state_dict,
			'optimizer_state_dict': self.optimizer.state_dict(),
			'scheduler_state_dict': self.scheduler.state_dict(),
		}, save_path)

		self.model.to(self.device)

	def load_ckpt(self, epoch, name=None):
		"""
			load checkpoint from saved checkpoint
		"""
		name = name if name == 'latest' else "ckpt_epoch{}".format(epoch)
		load_path = os.path.join(self.model_dir, "{}.pth".format(name))
		if not os.path.exists(load_path):
			raise ValueError("Checkpoint {} not exists.".format(load_path))

		checkpoint = torch.load(load_path)
		print("Loading checkpoint from {} ...".format(load_path))
		if isinstance(self.model, nn.DataParallel):
			self.model.module.load_state_dict(checkpoint['model_state_dict'])
		else:
			self.model.load_state_dict(checkpoint['model_state_dict'])
		self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		self.watcher.restore_checkpoint(checkpoint['watcher'])

	@abstractmethod
	def forward(self, data):
		"""
			forward method for the network
		"""
		raise NotImplementedError

	def update_network(self, loss_dict):
		"""
			update network by back propagation
		"""
		loss = sum(loss_dict.values())
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

	def update_learning_rate(self):
		"""
			record and update learning rate
		"""
		self.train_tbw.add_scalar('learning_rate', self.optimizer.param_groups[-1]['lr'], self.watcher.epoch)
		if not self.optimizer.param_groups[-1]['lr'] < self.base_lr / 10.0:
			self.scheduler.step(self.watcher.epoch)

	def record_losses(self, loss_dict, mode='train'):
		"""
			record loss to tensorboard
		"""
		losses_values = {k: v.item() for k, v in loss_dict.items()}

		tbw = self.train_tbw if mode == 'train' else self.val_tbw
		for k, v in losses_values.items():
			tbw.add_scalar(k, v, self.watcher.step)

	def train_func(self, data):
		"""
			one step of training
		"""
		self.model.train()
		self.forward(data)

		losses = self.collect_loss()
		self.update_network(losses)
		self.record_losses(losses, 'train')

	def val_func(self, data):
		"""
			one step of validation
		"""
		self.model.eval()

		with torch.no_grad():
			self.forward(data)

		losses = self.collect_loss()
		self.record_losses(losses, 'validation')

	def visualize_batch(self, data, tbw, num, **kwargs):
		"""
			write visualization results to tensorboard writer
		"""
		raise NotImplementedError


class TrainerCommonMulti(object):
	"""
		Base trainer that provides common training process with multiple models and loss function.
		All customized trainer for training combined models should be a subclass of this class.
	"""
	def __init__(self, config):
		# paths/ directories
		self.log_dir = config.log_dir
		self.model_dir = config.model_dir

		# device
		self.device = config.device

		# watcher
		self.watcher = TrainWatcher()

		# training batch size
		self.batch_size = config.batch_size

		# build network
		self.pointAE = None
		self.model = self.build_model(config)

		# set loss function
		self.criterionMSE = nn.MSELoss().to(self.device)
		self.criterionL1 = nn.L1Loss().to(self.device)
		self.criterionLatent = None
		self.criterionRecon = None
		self.set_loss_function()

		# set optimizer
		self.base_lr = None
		self.optimizer_gen = None
		self.set_optimizer(config)

		# store predicted points
		self.predicted_pts = None

		# store loss values
		self.loss = None

		# set tensorboard writer
		self.train_tbw = SummaryWriter(os.path.join(self.log_dir, 'train.events'))
		self.val_tbw = SummaryWriter(os.path.join(self.log_dir, 'val.events'))

	@abstractmethod
	def build_model(self, config):
		raise NotImplementedError

	def set_loss_function(self):
		"""
			set loss function used in training
		"""
		raise NotImplementedError

	@abstractmethod
	def collect_loss(self):
		"""
			collect all losses into a dict
		"""
		raise NotImplementedError

	def set_optimizer(self, config):
		"""
			set optimizer used in training
		"""
		self.base_lr = config.lr
		self.optimizer_gen = optim.Adam(self.model.parameters(), config.lr, betas=(config.beta1_gen, 0.999))

	def save_ckpt(self, name=None):
		"""
			save checkpoint during training for future reload
		"""
		if name is None:
			save_path = os.path.join(self.model_dir, "ckpt_epoch{}.pth".format(self.watcher.epoch))
			print("Saving checkpoint epoch {}...".format(self.watcher.epoch))
		else:
			save_path = os.path.join(self.model_dir, "{}.pth".format(name))

		if isinstance(self.model, nn.DataParallel):
			model_state_dict = self.model.module.cpu().state_dict()
		else:
			model_state_dict = self.model.cpu().state_dict()

		torch.save({
			'watcher': self.watcher.make_checkpoint(),
			'generator_state_dict': model_state_dict,
			'optimizer_gen_state_dict': self.optimizer_gen.state_dict(),
		}, save_path)

		self.model.to(self.device)

	def load_ckpt(self, epoch, name=None):
		"""
			load checkpoint from saved checkpoint
		"""
		name = name if name == 'latest' else "ckpt_epoch{}".format(epoch)
		load_path = os.path.join(self.model_dir, "{}.pth".format(name))
		if not os.path.exists(load_path):
			raise ValueError("Checkpoint {} not exists.".format(load_path))

		checkpoint = torch.load(load_path)
		print("Loading checkpoint from {} ...".format(load_path))
		self.model.load_state_dict(checkpoint['generator_state_dict'])
		self.optimizer_gen.load_state_dict(checkpoint['optimizer_gen_state_dict'])
		self.watcher.restore_checkpoint(checkpoint['watcher'])

	@abstractmethod
	def forward(self, data):
		"""
			forward method for the network
		"""
		raise NotImplementedError

	def update_generator(self):
		"""
			update network by back propagation
		"""
		raise NotImplementedError

	def update_learning_rate(self):
		"""
			record and update learning rate
		"""
		self.train_tbw.add_scalar('learning_rate', self.optimizer_gen.param_groups[-1]['lr'], self.watcher.epoch)

	def record_losses(self, loss_dict, mode='train'):
		"""
			record loss to tensorboard
		"""
		losses_values = {k: v.item() for k, v in loss_dict.items()}

		tbw = self.train_tbw if mode == 'train' else self.val_tbw
		for k, v in losses_values.items():
			tbw.add_scalar(k, v, self.watcher.step)

	def train_func(self, data):
		"""
			one step of training
		"""
		self.model.train()
		self.forward(data)

		losses = self.collect_loss()
		self.update_generator()
		self.record_losses(losses, 'train')

	def val_func(self, data):
		"""
			one step of validation
		"""
		self.model.eval()

		with torch.no_grad():
			self.forward(data)

		losses = self.collect_loss()
		self.record_losses(losses, 'validation')

	def visualize_batch(self, data, tbw, num, **kwargs):
		"""
			write visualization results to tensorboard writer
		"""
		raise NotImplementedError


class TrainerCommonEnsemble(object):
	"""
		Base trainer that provides common training process with ensemble of models and loss function.
		All customized trainer for training combined models should be a subclass of this class.
	"""

	def __init__(self, config):
		# paths/ directories
		self.log_dir = config.log_dir
		self.model_dir = config.model_dir

		# device
		self.device = config.device

		# watcher
		self.watcher = TrainWatcher()

		# training batch size
		self.batch_size = config.batch_size

		# build network#
		self.n_models = config.n_models
		self.pointAE = None
		self.models = self.build_models(config)

		# set loss function
		self.criterionMSE = nn.MSELoss().to(self.device)
		self.criterionL1 = nn.L1Loss().to(self.device)
		self.criterionLatent = None
		self.criterionRecon = None
		self.set_loss_function()

		# set optimizer
		self.base_lr = None
		self.optimizers_gen = self.set_optimizers(config)

		# store predicted points
		self.predicted_pts = None

		# store loss values for each model
		self.losses = []

		# set tensorboard writer
		self.train_tbw = SummaryWriter(os.path.join(self.log_dir, 'train.events'))
		self.val_tbw = SummaryWriter(os.path.join(self.log_dir, 'val.events'))

	@abstractmethod
	def build_models(self, config):
		raise NotImplementedError

	def set_loss_function(self):
		"""
			set loss function used in training
		"""
		raise NotImplementedError

	@abstractmethod
	def collect_losses(self):
		"""
			collect all losses into a dict
		"""
		raise NotImplementedError

	def set_optimizers(self, config):
		"""
			set optimizer used in training
		"""
		raise NotImplementedError

	def save_ckpt(self, name=None):
		"""
			save checkpoint during training for future reload
		"""
		for i in range(self.n_models):
			if name is None:
				save_path = os.path.join(self.model_dir, "model{}_ckpt_epoch{}.pth".format(i, self.watcher.epoch))
				print("Saving model {} checkpoint epoch {}...".format(i, self.watcher.epoch))
			else:
				save_path = os.path.join(self.model_dir, "model{}_{}.pth".format(i, name))

			if isinstance(self.models[i], nn.DataParallel):
				model_state_dict = self.models[i].module.cpu().state_dict()
			else:
				model_state_dict = self.models[i].cpu().state_dict()

			torch.save({
				'watcher': self.watcher.make_checkpoint(),
				'generator_state_dict': model_state_dict,
				'optimizer_gen_state_dict': self.optimizers_gen[i].state_dict(),
			}, save_path)

			self.models[i].to(self.device)

	def load_ckpt(self, epoch, name=None):
		"""
			load checkpoint from saved checkpoint
		"""
		name = name if name == 'latest' else "ckpt_epoch{}".format(epoch)
		load_path = os.path.join(self.model_dir, "{}.pth".format(name))
		if not os.path.exists(load_path):
			raise ValueError("Checkpoint {} not exists.".format(load_path))

		checkpoint = torch.load(load_path)
		print("Loading checkpoint from {} ...".format(load_path))
		self.model.load_state_dict(checkpoint['generator_state_dict'])
		self.optimizer_gen.load_state_dict(checkpoint['optimizer_gen_state_dict'])
		self.watcher.restore_checkpoint(checkpoint['watcher'])

	@abstractmethod
	def forward(self, data):
		"""
			forward method for the network
		"""
		raise NotImplementedError

	def update_generators(self):
		"""
			update network by back propagation
		"""
		raise NotImplementedError

	def update_learning_rates(self):
		"""
			record and update learning rate
		"""
		for i in range(self.n_models):
			self.train_tbw.add_scalar(f'learning_rate_{i}', self.optimizers_gen[i].param_groups[-1]['lr'],
									  self.watcher.epoch)

	def record_losses(self, loss_dict, mode='train'):
		"""
			record loss to tensorboard
		"""
		losses_values = {k: v.item() for k, v in loss_dict.items()}

		tbw = self.train_tbw if mode == 'train' else self.val_tbw
		for k, v in losses_values.items():
			tbw.add_scalar(k, v, self.watcher.step)

	def train_func(self, data):
		"""
			one step of training
		"""
		for model in self.models:
			model.train()
		self.forward(data)

		losses = self.collect_losses()
		self.update_generators()
		self.record_losses(losses, 'train')

	def val_func(self, data):
		"""
			one step of validation
		"""
		for model in self.models:
			model.eval()

		with torch.no_grad():
			self.forward(data)

		losses = self.collect_losses()
		self.record_losses(losses, 'validation')


class TrainerCommonEBM(object):
	"""
		Base trainer that provides common training process with Energy-based models and multiple loss functions.
		All customized trainer for training combined Energy-based models should be a subclass of this class.
	"""

	def __init__(self, config):
		# paths/ directories
		self.log_dir = config.log_dir
		self.model_dir = config.model_dir

		# device
		self.device = config.device

		# watcher
		self.watcher = TrainWatcher()

		# training batch size
		self.batch_size = config.batch_size

		# build network
		self.model = self.build_model(config)

		# set loss function
		self.criterionMSE = nn.MSELoss().to(self.device)
		self.criterionL1 = nn.L1Loss().to(self.device)
		self.criterionLatent = None
		self.criterionRecon = None
		self.set_loss_function()

		# set optimizer
		self.base_lr = None
		self.optimizer_ed = None
		self.optimizer_ebm = None
		self.set_optimizer(config)

		# set scheduler
		self.scheduler_ed = None
		self.scheduler_ebm = None
		self.set_scheduler(config)

		# store predicted points
		self.predicted_pts = None

		# store loss values
		self.encoder_decoder_loss = None
		self.ebm_loss = None

		# set tensorboard writer
		self.train_tbw = SummaryWriter(os.path.join(self.log_dir, 'train.events'))
		self.val_tbw = SummaryWriter(os.path.join(self.log_dir, 'val.events'))

	@abstractmethod
	def build_model(self, config):
		raise NotImplementedError

	def set_loss_function(self):
		"""
			set loss function used in training
		"""
		raise NotImplementedError

	@abstractmethod
	def collect_loss(self):
		"""
			collect all losses into a dict
		"""
		raise NotImplementedError

	def set_optimizer(self, config):
		"""
			set optimizer used in training
		"""
		self.base_lr = config.lr
		self.optimizer_ed = optim.Adam(list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()),
									   config.lr, betas=(config.beta1_ed, 0.999))
		self.optimizer_ebm = optim.Adam(self.model.ebm.parameters(), config.lr, betas=(config.beta1_ebm, 0.999))

	def set_scheduler(self, config):
		"""
			set optimizer used in training
		"""
		if config.decay_step is not None:
			lr_lamb = lambda e: max(config.lr_decay ** (e / config.decay_step), config.lowest_decay)
		else:
			raise NotImplementedError()
		self.scheduler_ed = optim.lr_scheduler.LambdaLR(self.optimizer_ed, lr_lamb)
		self.scheduler_ebm = optim.lr_scheduler.LambdaLR(self.optimizer_ebm, lr_lamb)

	def save_ckpt(self, name=None):
		"""
			save checkpoint during training for future reload
		"""
		if name is None:
			save_path = os.path.join(self.model_dir, "ckpt_epoch{}.pth".format(self.watcher.epoch))
			print("Saving checkpoint epoch {}...".format(self.watcher.epoch))
		else:
			save_path = os.path.join(self.model_dir, "{}.pth".format(name))

		if isinstance(self.model, nn.DataParallel):
			model_state_dict = self.model.module.cpu().state_dict()
		else:
			model_state_dict = self.model.cpu().state_dict()

		torch.save({
			'watcher': self.watcher.make_checkpoint(),
			'model_state_dict': model_state_dict,
			'ed_optimizer_state_dict': self.optimizer_ed.state_dict(),
			'ebm_optimizer_state_dict': self.optimizer_ebm.state_dict(),
			'ed_scheduler_state_dict': self.scheduler_ed.state_dict(),
			'ebm_scheduler_state_dict': self.scheduler_ebm.state_dict(),
		}, save_path)

		self.model.to(self.device)

	def load_ckpt(self, epoch, name=None):
		"""
			load checkpoint from saved checkpoint
		"""
		name = name if name == 'latest' else "ckpt_epoch{}".format(epoch)
		load_path = os.path.join(self.model_dir, "{}.pth".format(name))
		if not os.path.exists(load_path):
			raise ValueError("Checkpoint {} not exists.".format(load_path))

		checkpoint = torch.load(load_path)
		print("Loading checkpoint from {} ...".format(load_path))
		self.model.load_state_dict(checkpoint['model_state_dict'])
		self.optimizer_ed.load_state_dict(checkpoint['ed_optimizer_state_dict'])
		self.optimizer_ebm.load_state_dict(checkpoint['ebm_optimizer_state_dict'])
		self.scheduler_ed.load_state_dict(checkpoint['ed_scheduler_state_dict'])
		self.scheduler_ebm.load_state_dict(checkpoint['ebm_scheduler_state_dict'])
		self.watcher.restore_checkpoint(checkpoint['watcher'])

	@abstractmethod
	def forward(self, data):
		"""
			forward method for the network
		"""
		raise NotImplementedError

	def update_models(self):
		"""
			update models by back propagation
		"""
		raise NotImplementedError

	def update_learning_rate(self):
		"""
			record and update learning rate
		"""
		self.train_tbw.add_scalar('learning_rate_ed', self.optimizer_ed.param_groups[-1]['lr'], self.watcher.epoch)
		self.train_tbw.add_scalar('learning_rate_ebm', self.optimizer_ebm.param_groups[-1]['lr'], self.watcher.epoch)

	def record_losses(self, loss_dict, mode='train'):
		"""
			record loss to tensorboard
		"""
		losses_values = {k: v.item() for k, v in loss_dict.items()}

		tbw = self.train_tbw if mode == 'train' else self.val_tbw
		for k, v in losses_values.items():
			tbw.add_scalar(k, v, self.watcher.step)

	def train_func(self, data):
		"""
			one step of training
		"""
		self.model.train()
		self.forward(data)

		losses = self.collect_loss()
		self.update_models()
		self.record_losses(losses, 'train')

	def val_func(self, data):
		"""
			one step of validation
		"""
		self.model.eval()

		with torch.no_grad():
			self.forward(data)

		losses = self.collect_loss()
		self.record_losses(losses, 'validation')

	def visualize_batch(self, data, tbw, num, **kwargs):
		"""
			write visualization results to tensorboard writer
		"""
		raise NotImplementedError


class TrainerContrastive(object):
	"""
		Base trainer that provides common training process for contrastive learning.
		All customized trainer for contrastive learning should be a subclass of this class.
	"""
	def __init__(self, config):
		# paths/ directories
		self.log_dir = config.log_dir
		self.model_dir = config.model_dir

		# device
		self.device = config.device

		# watcher
		self.watcher = TrainWatcher()

		# training batch size
		self.batch_size = config.batch_size

		# build network
		self.pointAE = None
		self.vAE = None
		self.vqvAE = None
		self.model = self.build_model(config)

		# set loss function and related stuff
		self.loss_criterion = None
		self.criterionContrast = None
		self.set_loss_function(config)
		self.loss_batch = config.loss_batch

		# set optimizer
		self.base_lr = None
		self.optimizer_con = None
		self.set_optimizer(config)

		# store predicted points
		self.predicted_pts = None

		# store loss values
		self.loss = None

		# set tensorboard writer
		self.train_tbw = SummaryWriter(os.path.join(self.log_dir, 'train.events'))
		self.val_tbw = SummaryWriter(os.path.join(self.log_dir, 'val.events'))

	@abstractmethod
	def build_model(self, config):
		raise NotImplementedError

	def set_loss_function(self, config):
		"""
			set loss function used in training
		"""
		raise NotImplementedError

	@abstractmethod
	def collect_loss(self):
		"""
			collect all losses into a dict
		"""
		raise NotImplementedError

	def set_optimizer(self, config):
		"""
			set optimizer used in training
		"""
		self.base_lr = config.lr
		self.optimizer_con = optim.Adam(self.model.parameters(), config.lr, betas=(config.beta1_con, 0.999))

	def save_ckpt(self, name=None):
		"""
			save checkpoint during training for future reload
		"""
		if name is None:
			save_path = os.path.join(self.model_dir, "ckpt_epoch{}.pth".format(self.watcher.epoch))
			print("Saving checkpoint epoch {}...".format(self.watcher.epoch))
		else:
			save_path = os.path.join(self.model_dir, "{}.pth".format(name))

		if isinstance(self.model, nn.DataParallel):
			model_state_dict = self.model.module.cpu().state_dict()
		else:
			model_state_dict = self.model.cpu().state_dict()

		torch.save({
			'watcher': self.watcher.make_checkpoint(),
			'con_state_dict': model_state_dict,
			'optimizer_con_state_dict': self.optimizer_con.state_dict(),
		}, save_path)

		self.model.to(self.device)

	def load_ckpt(self, epoch, name=None):
		"""
			load checkpoint from saved checkpoint
		"""
		name = name if name == 'latest' else "ckpt_epoch{}".format(epoch)
		load_path = os.path.join(self.model_dir, "{}.pth".format(name))
		if not os.path.exists(load_path):
			raise ValueError("Checkpoint {} not exists.".format(load_path))

		checkpoint = torch.load(load_path)
		print("Loading checkpoint from {} ...".format(load_path))
		self.model.load_state_dict(checkpoint['con_state_dict'])
		self.optimizer_con.load_state_dict(checkpoint['optimizer_con_state_dict'])
		self.watcher.restore_checkpoint(checkpoint['watcher'])

	@abstractmethod
	def forward(self, data):
		"""
			forward method for the network
		"""
		raise NotImplementedError

	def update_network(self, loss_dict):
		"""
			update network by back propagation
		"""
		loss = sum(loss_dict.values())
		self.optimizer_con.zero_grad()
		loss.backward()
		self.optimizer_con.step()

	def update_learning_rate(self):
		"""
			record and update learning rate
		"""
		self.train_tbw.add_scalar('learning_rate', self.optimizer_con.param_groups[-1]['lr'], self.watcher.epoch)

	def record_losses(self, loss_dict, mode='train'):
		"""
			record loss to tensorboard
		"""
		losses_values = {k: v.item() for k, v in loss_dict.items()}

		tbw = self.train_tbw if mode == 'train' else self.val_tbw
		for k, v in losses_values.items():
			tbw.add_scalar(k, v, self.watcher.step)

	def train_func(self, data):
		"""
			one step of training
		"""
		self.model.train()
		self.forward(data)

		losses = self.collect_loss()
		self.update_network(losses)
		self.record_losses(losses, 'train')

	def val_func(self, data):
		"""
			one step of validation
		"""
		self.model.eval()

		with torch.no_grad():
			self.forward(data)

		losses = self.collect_loss()
		self.record_losses(losses, 'validation')
