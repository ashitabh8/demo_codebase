"""
Temperature Schedulers for Gumbel-Softmax
==========================================

Similar to PyTorch's LR schedulers, but for temperature annealing in Gumbel-Softmax.

Usage:
------
    from models.Temperature_Scheduler import CosineAnnealingTemp, build_temp_scheduler
    from models.QuantModules import QuanConvImportance
    
    # Create model with QuanConvImportance layers (or any Conv class with set_temperature method)
    model = create_your_model(...)
    
    # Create optimizer and LR scheduler as usual
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    # Create temperature scheduler - pass the Conv class to target
    temp_scheduler = CosineAnnealingTemp(
        model=model,
        conv_class=QuanConvImportance,  # Any class with set_temperature() method
        temp_start=1.0,      # High temperature = more exploration
        temp_min=0.1,        # Low temperature = more exploitation  
        num_epochs=100
    )
    
    # Or use the factory function with config:
    # temp_scheduler = build_temp_scheduler(model, QuanConvImportance, config, num_epochs=100)
    
    # Training loop
    for epoch in range(100):
        train_one_epoch(model, train_loader, optimizer)
        validate(model, val_loader)
        
        # Step schedulers at end of epoch
        lr_scheduler.step()
        temp_scheduler.step()  # Updates temperature in all conv_class layers
        
        print(f"Epoch {epoch}: LR={lr_scheduler.get_last_lr()}, Temp={temp_scheduler.get_last_temp()}")

    # Checkpointing
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'temp_scheduler': temp_scheduler.state_dict(),  # Save temperature scheduler state
    }
    torch.save(checkpoint, 'checkpoint.pth')
    
    # Resuming
    checkpoint = torch.load('checkpoint.pth')
    temp_scheduler.load_state_dict(checkpoint['temp_scheduler'])
"""

import math
from abc import ABC, abstractmethod


class TemperatureScheduler(ABC):
    """
    Base class for Gumbel-Softmax temperature scheduling.
    
    Similar to PyTorch's LR schedulers, but for temperature annealing.
    Automatically updates the temperature in all layers of the specified conv class.
    """
    
    def __init__(self, model, conv_class, temp_start=1.0, temp_min=0.1, num_epochs=100, last_epoch=-1):
        """
        Args:
            model: PyTorch model containing quantized conv layers
            conv_class: The Conv class type to update (e.g., QuanConvImportance).
                        Must have a set_temperature(temperature) method.
            temp_start: Initial temperature (high = more exploration)
            temp_min: Minimum temperature (low = more exploitation)
            num_epochs: Total number of training epochs
            last_epoch: The index of last epoch (for resuming training)
        """
        self.model = model
        self.conv_class = conv_class
        self.temp_start = temp_start
        self.temp_min = temp_min
        self.num_epochs = num_epochs
        self.last_epoch = last_epoch
        self.current_temp = temp_start
        
        # Initialize temperature in all layers
        self._update_model_temperature(temp_start)
    
    def _update_model_temperature(self, temperature):
        """Update temperature in all layers of the specified conv class"""
        for module in self.model.modules():
            if isinstance(module, self.conv_class):
                module.set_temperature(temperature)
        
        self.current_temp = temperature
    
    @abstractmethod
    def _compute_temperature(self, epoch):
        """Compute temperature for given epoch (to be implemented by subclasses)"""
        pass
    
    def step(self, epoch=None):
        """
        Update temperature for the next epoch.
        
        Args:
            epoch: Optional epoch number. If None, uses self.last_epoch + 1
        
        Returns:
            float: The new temperature value
        """
        if epoch is None:
            epoch = self.last_epoch + 1
        
        self.last_epoch = epoch
        temp = self._compute_temperature(epoch)
        self._update_model_temperature(temp)
        
        return temp
    
    def get_last_temp(self):
        """Get the current temperature"""
        return self.current_temp
    
    def state_dict(self):
        """Return state dict for checkpointing"""
        return {
            'temp_start': self.temp_start,
            'temp_min': self.temp_min,
            'num_epochs': self.num_epochs,
            'last_epoch': self.last_epoch,
            'current_temp': self.current_temp
        }
    
    def load_state_dict(self, state_dict):
        """Load state dict from checkpoint"""
        self.temp_start = state_dict['temp_start']
        self.temp_min = state_dict['temp_min']
        self.num_epochs = state_dict['num_epochs']
        self.last_epoch = state_dict['last_epoch']
        self.current_temp = state_dict['current_temp']
        self._update_model_temperature(self.current_temp)


class CosineAnnealingTemp(TemperatureScheduler):
    """
    Cosine annealing temperature schedule.
    
    Temperature smoothly decreases following a cosine curve from temp_start to temp_min.
    
    Formula: temp = temp_min + 0.5 * (temp_start - temp_min) * (1 + cos(π * progress))
    
    Advantages:
    - Smooth decay (starts slow, speeds up in middle, slows at end)
    - Widely used in deep learning (similar to cosine LR schedules)
    - Recommended for most use cases
    
    Example:
        >>> from models.QuantModules import QuanConvImportance
        >>> scheduler = CosineAnnealingTemp(model, QuanConvImportance, temp_start=1.0, temp_min=0.1, num_epochs=100)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     scheduler.step()
    """
    
    def _compute_temperature(self, epoch):
        if self.num_epochs <= 1:
            return self.temp_min
        
        progress = min(epoch / (self.num_epochs - 1), 1.0)
        temp = self.temp_min + 0.5 * (self.temp_start - self.temp_min) * \
               (1 + math.cos(math.pi * progress))
        
        return temp


class ExponentialTemp(TemperatureScheduler):
    """
    Exponential decay temperature schedule.
    
    Temperature decays exponentially each epoch: temp = temp_start * (decay_rate ^ epoch)
    
    The decay_rate is automatically computed to reach temp_min at the final epoch.
    
    Formula: decay_rate = (temp_min / temp_start) ^ (1 / (num_epochs - 1))
             temp = max(temp_start * decay_rate^epoch, temp_min)
    
    Advantages:
    - More aggressive early cooling compared to cosine
    - Good when you want to quickly converge to exploitation
    
    Example:
        >>> from models.QuantModules import QuanConvImportance
        >>> scheduler = ExponentialTemp(model, QuanConvImportance, temp_start=1.0, temp_min=0.1, num_epochs=100)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     scheduler.step()
    """
    
    def __init__(self, model, conv_class, temp_start=1.0, temp_min=0.1, num_epochs=100, 
                 decay_rate=None, last_epoch=-1):
        """
        Args:
            model: PyTorch model containing quantized conv layers
            conv_class: The Conv class type to update (must have set_temperature method)
            temp_start: Initial temperature
            temp_min: Minimum temperature
            num_epochs: Total number of training epochs
            decay_rate: Optional decay rate per epoch. If None, computed automatically
                        to reach temp_min at final epoch.
            last_epoch: The index of last epoch (for resuming training)
        """
        self.decay_rate = decay_rate
        super().__init__(model, conv_class, temp_start, temp_min, num_epochs, last_epoch)
        
        # Compute decay rate if not provided
        if self.decay_rate is None and num_epochs > 1:
            self.decay_rate = (temp_min / temp_start) ** (1.0 / (num_epochs - 1))
        elif self.decay_rate is None:
            self.decay_rate = 1.0
    
    def _compute_temperature(self, epoch):
        if self.num_epochs <= 1:
            return self.temp_min
        
        temp = self.temp_start * (self.decay_rate ** epoch)
        return max(temp, self.temp_min)
    
    def state_dict(self):
        """Return state dict for checkpointing"""
        state = super().state_dict()
        state['decay_rate'] = self.decay_rate
        return state
    
    def load_state_dict(self, state_dict):
        """Load state dict from checkpoint"""
        self.decay_rate = state_dict.get('decay_rate', self.decay_rate)
        super().load_state_dict(state_dict)


class LinearTemp(TemperatureScheduler):
    """
    Linear decay temperature schedule.
    
    Temperature decreases linearly from temp_start to temp_min.
    
    Formula: temp = temp_start - (temp_start - temp_min) * (epoch / (num_epochs - 1))
    
    Advantages:
    - Simple and predictable
    - Constant rate of cooling
    
    Example:
        >>> from models.QuantModules import QuanConvImportance
        >>> scheduler = LinearTemp(model, QuanConvImportance, temp_start=1.0, temp_min=0.1, num_epochs=100)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     scheduler.step()
    """
    
    def _compute_temperature(self, epoch):
        if self.num_epochs <= 1:
            return self.temp_min
        
        progress = min(epoch / (self.num_epochs - 1), 1.0)
        temp = self.temp_start - (self.temp_start - self.temp_min) * progress
        
        return temp


class StepTemp(TemperatureScheduler):
    """
    Step decay temperature schedule.
    
    Temperature drops by a factor at specified epochs (milestones).
    
    Example:
        >>> from models.QuantModules import QuanConvImportance
        >>> scheduler = StepTemp(model, QuanConvImportance, temp_start=1.0, temp_min=0.1, 
        ...                      milestones=[30, 60, 80], gamma=0.5)
        >>> # temp = 1.0 for epochs 0-29
        >>> # temp = 0.5 for epochs 30-59
        >>> # temp = 0.25 for epochs 60-79
        >>> # temp = 0.125 for epochs 80+
    """
    
    def __init__(self, model, conv_class, temp_start=1.0, temp_min=0.1, num_epochs=100,
                 milestones=None, gamma=0.5, last_epoch=-1):
        """
        Args:
            model: PyTorch model containing quantized conv layers
            conv_class: The Conv class type to update (must have set_temperature method)
            temp_start: Initial temperature
            temp_min: Minimum temperature
            num_epochs: Total number of training epochs
            milestones: List of epoch indices to drop temperature
            gamma: Multiplicative factor of temperature decay at each milestone
            last_epoch: The index of last epoch (for resuming training)
        """
        self.milestones = sorted(milestones) if milestones else [num_epochs // 3, 2 * num_epochs // 3]
        self.gamma = gamma
        super().__init__(model, conv_class, temp_start, temp_min, num_epochs, last_epoch)
    
    def _compute_temperature(self, epoch):
        # Count how many milestones have passed
        num_drops = sum(1 for m in self.milestones if epoch >= m)
        temp = self.temp_start * (self.gamma ** num_drops)
        return max(temp, self.temp_min)
    
    def state_dict(self):
        """Return state dict for checkpointing"""
        state = super().state_dict()
        state['milestones'] = self.milestones
        state['gamma'] = self.gamma
        return state
    
    def load_state_dict(self, state_dict):
        """Load state dict from checkpoint"""
        self.milestones = state_dict.get('milestones', self.milestones)
        self.gamma = state_dict.get('gamma', self.gamma)
        super().load_state_dict(state_dict)


class CyclicTemp(TemperatureScheduler):
    """
    Cyclic temperature schedule with warm restarts.
    
    Temperature follows cosine annealing but restarts at regular intervals.
    Similar to CosineAnnealingWarmRestarts in PyTorch.
    
    This can help escape local minima by periodically increasing exploration.
    
    Example:
        >>> from models.QuantModules import QuanConvImportance
        >>> scheduler = CyclicTemp(model, QuanConvImportance, temp_start=1.0, temp_min=0.1, 
        ...                        cycle_length=20, num_epochs=100)
        >>> # Temperature cycles through [1.0 -> 0.1] every 20 epochs
    """
    
    def __init__(self, model, conv_class, temp_start=1.0, temp_min=0.1, num_epochs=100,
                 cycle_length=20, last_epoch=-1):
        """
        Args:
            model: PyTorch model containing quantized conv layers
            conv_class: The Conv class type to update (must have set_temperature method)
            temp_start: Initial temperature (peak of each cycle)
            temp_min: Minimum temperature (trough of each cycle)
            num_epochs: Total number of training epochs
            cycle_length: Number of epochs per cycle
            last_epoch: The index of last epoch (for resuming training)
        """
        self.cycle_length = cycle_length
        super().__init__(model, conv_class, temp_start, temp_min, num_epochs, last_epoch)
    
    def _compute_temperature(self, epoch):
        # Position within current cycle
        cycle_position = epoch % self.cycle_length
        
        if self.cycle_length <= 1:
            return self.temp_min
        
        progress = cycle_position / (self.cycle_length - 1)
        temp = self.temp_min + 0.5 * (self.temp_start - self.temp_min) * \
               (1 + math.cos(math.pi * progress))
        
        return temp
    
    def state_dict(self):
        """Return state dict for checkpointing"""
        state = super().state_dict()
        state['cycle_length'] = self.cycle_length
        return state
    
    def load_state_dict(self, state_dict):
        """Load state dict from checkpoint"""
        self.cycle_length = state_dict.get('cycle_length', self.cycle_length)
        super().load_state_dict(state_dict)


class ConstantTemp(TemperatureScheduler):
    """
    Constant temperature (no annealing).
    
    Useful for ablation studies or when you want fixed exploration level.
    
    Example:
        >>> from models.QuantModules import QuanConvImportance
        >>> scheduler = ConstantTemp(model, QuanConvImportance, temp_start=0.5)
        >>> # Temperature stays at 0.5 throughout training
    """
    
    def _compute_temperature(self, epoch):
        return self.temp_start


def build_temp_scheduler(model, config, num_epochs=None):
    """
    Factory function to build temperature scheduler from config dict.
    
    Args:
        model: PyTorch model containing quantized conv layers
        config: Configuration dict with keys:
            - name: Scheduler type ('cosine', 'exponential', 'linear', 'step', 'cyclic', 'constant')
            - temp_start: Initial temperature (default: 1.0)
            - temp_min: Minimum temperature (default: 0.1)
            - num_epochs: Override for num_epochs (optional)
            - Additional scheduler-specific params
        num_epochs: Total epochs (can also be in config)
    
    Returns:
        TemperatureScheduler instance
    
    Example:
        >>> from models.QuantModules import QuanConvImportance
        >>> config = {
        ...     'name': 'cosine',
        ...     'temp_start': 1.0,
        ...     'temp_min': 0.1
        ... }
        >>> scheduler = build_temp_scheduler(model, QuanConvImportance, config, num_epochs=100)
    """
    scheduler_name = config.get('name', 'cosine').lower()
    temp_start = config.get('temp_start', 1.0)
    temp_min = config.get('temp_min', 0.1)
    epochs = config.get('num_epochs', num_epochs)
    conv_class = model.get_conv_class()
    
    if epochs is None:
        raise ValueError("num_epochs must be provided either in config or as argument")
    
    if scheduler_name == 'cosine':
        return CosineAnnealingTemp(model, conv_class, temp_start, temp_min, epochs)
    
    elif scheduler_name == 'exponential':
        decay_rate = config.get('decay_rate', None)
        return ExponentialTemp(model, conv_class, temp_start, temp_min, epochs, decay_rate)
    
    elif scheduler_name == 'linear':
        return LinearTemp(model, conv_class, temp_start, temp_min, epochs)
    
    elif scheduler_name == 'step':
        milestones = config.get('milestones', None)
        gamma = config.get('gamma', 0.5)
        return StepTemp(model, conv_class, temp_start, temp_min, epochs, milestones, gamma)
    
    elif scheduler_name == 'cyclic':
        cycle_length = config.get('cycle_length', 20)
        return CyclicTemp(model, conv_class, temp_start, temp_min, epochs, cycle_length)
    
    elif scheduler_name == 'constant':
        return ConstantTemp(model, conv_class, temp_start, temp_min, epochs)
    
    else:
        raise ValueError(f"Unknown temperature scheduler: {scheduler_name}. "
                        f"Choose from: cosine, exponential, linear, step, cyclic, constant")

