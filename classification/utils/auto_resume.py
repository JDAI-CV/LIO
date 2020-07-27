import os
import torch
import logging

from .warmup_scheduler import WarmupMultiStepLR


class AutoResumer(object):
    def __init__(self, scheduler, model_dir):
        super().__init__()
        self.milestones = scheduler.milestones
        self.model_dir = model_dir
        
    def analyze(self):
        model_names = [e for e in os.listdir(self.model_dir) if e.endswith('.pth')]
        epoch_with_acc = []
        for idx, model_name in enumerate(model_names):
            _, epoch, batch, suffix = model_name.split('-')
            epoch = int(epoch)
            acc = suffix.replace('[', '').replace('].pth', '')
            acc = float(acc)
            epoch_with_acc.append([idx, epoch, acc])
        return model_names, epoch_with_acc
    
    def get_model_file(self, end):
        model_names, epoch_with_acc = self.analyze()
        epoch_with_acc = [ewa for ewa in epoch_with_acc
                              if  ewa[1] <= end]
        max_acc_index = sorted(epoch_with_acc, key=lambda e: -e[2])[0][0]
        return os.path.join(self.model_dir, model_names[max_acc_index])
    
    def resume(self, model, model_file):
        logging.info("[auto_resume] resuming best accuracy model: %s" % model_file)
        state_dicts = torch.load(model_file)
        [m.load_state_dict(d) for m, d in zip(model, state_dicts)]
        return

    def step(self, model, epoch):
        if epoch in self.milestones:
            model_file = self.get_model_file(epoch)
            self.resume(model, model_file)
        return
