import csv
import sys
import datetime
import os
import psutil
from collections import defaultdict

import numpy as np
import torch
from termcolor import colored
from torch.utils.tensorboard import SummaryWriter
import wandb

COMMON_TRAIN_FORMAT = [('epoch', 'E', 'int'), ('step', 'S', 'int'), 
                       ('loss', 'L', 'float'), ('accuracy', 'A', 'float'), 
                       ('total_time', 'T', 'time')]

COMMON_EVAL_FORMAT = [('epoch', 'E', 'int'), ('step', 'S', 'int'), 
                       ('loss', 'L', 'float'),  ('accuracy', 'A', 'float'),
                       ('total_time', 'T', 'time')]


class AverageMeter(object):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        self._sum += value
        self._count += n

    def value(self):
        return self._sum / max(1, self._count)


class MeterDict(object):
    def __init__(self):
        self._meters = defaultdict(AverageMeter)

    def update(self, metrics):
        for key, value in metrics.items():
            self._meters[key].update(value)

    def log(self, key, value, n=1):
        self._meters[key].update(value, n)

    def clear(self):
        self._meters.clear()

    def dump(self):
        return {key: meter.value() for key, meter in self._meters.items()}


class MetersGroup(object):
    def __init__(self, csv_file_name, formating, use_wandb):
        self._csv_file_name = csv_file_name
        self._formating = formating
        self._meters = defaultdict(AverageMeter)
        self._csv_file = None
        self._csv_writer = None
        self.use_wandb = use_wandb

    def log(self, key, value, n=1):
        self._meters[key].update(value, n)

    def _prime_meters(self):
        data = dict()
        for key, meter in self._meters.items():
            if key.startswith('train'):
                key = key[len('train') + 1:]
            elif key.startswith('eval'):
                key = key[len('eval') + 1:]
            elif key.startswith('retrieval'):
                key = key[len('retrieval') + 1:]
            else:
                raise NotImplementedError
            key = key.replace('/', '_')
            data[key] = meter.value()
        return data

    def _remove_old_entries(self, data):
        rows = []
        with self._csv_file_name.open('r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if float(row['step']) >= data['step']:
                    break
                rows.append(row)
        with self._csv_file_name.open('w') as f:
            writer = csv.DictWriter(f,
                                    fieldnames=sorted(data.keys()),
                                    restval=0.0)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _dump_to_csv(self, input_data):
        data = dict()
        for k, v in input_data.items():
            if k.startswith('hist'):
                continue
            else:
                data[k] = v
        if self._csv_writer is None:
            should_write_header = True
            if self._csv_file_name.exists():
                self._remove_old_entries(data)
                should_write_header = False

            self._csv_file = self._csv_file_name.open('a')
            self._csv_writer = csv.DictWriter(self._csv_file,
                                              fieldnames=sorted(data.keys()),
                                              restval=0.0)
            if should_write_header:
                self._csv_writer.writeheader()

        self._csv_writer.writerow(data)
        self._csv_file.flush()

    def _format(self, key, value, ty):
        if ty == 'int':
            value = int(value)
            return f'{key}: {value}'
        elif ty == 'float':
            return f'{key}: {value:.04f}'
        elif ty == 'time':
            value = str(datetime.timedelta(seconds=int(value)))
            return f'{key}: {value}'
        else:
            raise f'invalid format type: {ty}'

    def _dump_to_console(self, data, prefix):
        if prefix == 'train':
            prefix = colored(prefix, 'yellow')
        elif prefix == 'eval':
            prefix = colored(prefix, 'green')
        elif prefix == 'retrieval':
            prefix = colored(prefix, 'red')
        else:
            raise NotImplementedError
        pieces = [f'| {prefix: <14}']
        for key, disp_key, ty in self._formating:
            value = data.get(key, 0)
            pieces.append(self._format(disp_key, value, ty))
        print(' | '.join(pieces))

    def _dump_to_wandb(self, data):
        wandb.log(data)

    def dump(self, step, prefix):
        if len(self._meters) == 0:
            return
        data = self._prime_meters()
        data['step'] = step
        if self.use_wandb:
            wandb_data = {prefix + '/' + key: val for key,val in data.items()}
            self._dump_to_wandb(data=wandb_data)
        self._dump_to_csv(data)
        self._dump_to_console(data, prefix)
        self._meters.clear()


class Logger(object):
    def __init__(self, log_dir, use_tb, use_wandb, name=None, project=None, cfg=None,
                save_stdout=True):
        if save_stdout:
            self._log_dir = log_dir
            self.terminal = sys.stdout
            sys.stdout, sys.stderr = self, self
            self.log_file = open(log_dir / 'log.txt', "a")

        self._train_mg = MetersGroup(log_dir / 'train.csv',
                                     formating=COMMON_TRAIN_FORMAT,
                                     use_wandb=use_wandb)
        self._eval_mg = MetersGroup(log_dir / 'eval.csv',
                                    formating=COMMON_EVAL_FORMAT,
                                    use_wandb=use_wandb)
        if use_tb:
            self._sw = SummaryWriter(str(log_dir / 'tb'))
        else:
            self._sw = None
        self.use_wandb = use_wandb

    def _try_sw_log(self, key, value, step):
        if self._sw is not None:
            is_hist = key.split('/')[1].startswith('hist')
            if is_hist:
                self._sw.add_histogram(key, value, step)
            else:
                self._sw.add_scalar(key, value, step)

    def log(self, key, value, step):
        assert key.startswith('train') or key.startswith('eval') or key.startswith('retrieval')
        if type(value) == torch.Tensor:
            value = value.item()
        self._try_sw_log(key, value, step)
        if key.startswith('train'):
            mg = self._train_mg
        elif key.startswith('eval'):
            mg = self._eval_mg
        elif key.startswith('retrieval'):
            mg = self._retrieval_mg
        mg.log(key, value)

    def plot(self, name, path):
        wandb.log({name: wandb.Image(path)})

    def log_metrics(self, metrics, step, ty):
        if ty == 'train':
            process = psutil.Process(os.getpid())
            metrics['RAM_GB'] = float(process.memory_info().rss / (1024 ** 3))
        for key, value in metrics.items():
            self.log(f'{ty}/{key}', value, step)

    def dump(self, step, ty=None):
        if ty is None or ty == 'eval':
            self._eval_mg.dump(step, 'eval')
        if ty is None or ty == 'train':
            self._train_mg.dump(step, 'train')
        if ty == 'retrieval':
            self._retrieval_mg.dump(step, 'retrieval')

    def log_and_dump_ctx(self, step, ty):
        return LogAndDumpCtx(self, step, ty)

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        # needed for python3 compatibility
        pass


class LogAndDumpCtx:
    def __init__(self, logger, step, ty):
        self._logger = logger
        self._step = step
        self._ty = ty

    def __enter__(self):
        return self

    def __call__(self, key, value):
        self._logger.log(f'{self._ty}/{key}', value, self._step)

    def __exit__(self, *args):
        self._logger.dump(self._step, self._ty)