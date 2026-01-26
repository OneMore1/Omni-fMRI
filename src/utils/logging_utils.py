import torch
import torch.distributed as dist

import time
import datetime

import yaml

class MetricLogger:
    """Metric logger for training"""
    def __init__(self, delimiter="\t"):
        self.meters = {}
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if k not in self.meters:
                self.meters[k] = SmoothedValue()
            self.meters[k].update(v)

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {meter}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        log_msg = self.delimiter.join(log_msg)
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available() and dist.get_rank() == 0:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f'{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)')


class SmoothedValue:
    """Track a series of values and provide access to smoothed values"""
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = []
        self.total = 0.0
        self.count = 0
        self.fmt = fmt
        self.window_size = window_size

    def update(self, value, n=1):
        self.deque.append(value)
        if len(self.deque) > self.window_size:
            self.deque.pop(0)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """Synchronize across all processes"""
        if not dist.is_available() or not dist.is_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = sorted(self.deque)
        n = len(d)
        if n == 0:
            return 0
        if n % 2 == 0:
            return (d[n // 2 - 1] + d[n // 2]) / 2
        return d[n // 2]

    @property
    def avg(self):
        if len(self.deque) == 0:
            return 0
        return sum(self.deque) / len(self.deque)

    @property
    def global_avg(self):
        if self.count == 0:
            return 0
        return self.total / self.count

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=max(self.deque) if len(self.deque) > 0 else 0,
            value=self.deque[-1] if len(self.deque) > 0 else 0
        )
    

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def log_to_file(log_file, message):
    """Write message to log file"""
    if log_file is not None:
        with open(log_file, 'a') as f:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{timestamp}] {message}\n")
            f.flush()



def count_parameters(model, verbose=True):
    """Count model parameters"""
    def count_params(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    def format_number(num):
        if num >= 1e9:
            return f"{num/1e9:.2f}B"
        elif num >= 1e6:
            return f"{num/1e6:.2f}M"
        elif num >= 1e3:
            return f"{num/1e3:.2f}K"
        else:
            return str(num)

    # If DDP model, get original model
    if hasattr(model, 'module'):
        model = model.module

    total_params = count_params(model)

    if verbose:
        print("\n" + "="*80)
        print("Model Parameter Statistics")
        print("="*80)

        # Count encoder parameters
        encoder_params = 0
        for name in ['patch_embed', 'layers', 'pos_embeds', 'last_layer']:
            if hasattr(model, name):
                module = getattr(model, name)
                params = count_params(module)
                encoder_params += params
                print(f"{name:.<35} {params:>15,} ({format_number(params):>8})")

        # Count head parameters
        if hasattr(model, 'head'):
            head_params = count_params(model.head)
            print(f"{'Classification/Regression Head':.<35} {head_params:>15,} ({format_number(head_params):>8})")

        print("\n" + "="*80)
        print(f"{'Encoder Parameters':.<35} {encoder_params:>15,} ({format_number(encoder_params):>8})")
        print(f"{'TOTAL TRAINABLE PARAMETERS':.<35} {total_params:>15,} ({format_number(total_params):>8})")
        print("="*80 + "\n")

    return total_params