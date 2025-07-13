# build.py
import copy
from losses import DBLoss
from neck_fpn import FPN
from neck_fpem_ffm import FPEM_FFM
from head_DBHead import DBHead
from backbone_mobilenetv3 import MobileNetV3

def build_model(config):
    """
    get architecture model class
    """
    support_model = ['Model']
    copy_config = copy.deepcopy(config)
    arch_type = copy_config.pop('type')
    assert arch_type in support_model, f'{arch_type} is not developed yet!, only {support_model} are support now'
    
    # Import Model here to avoid circular import
    from model import Model
    arch_model = Model(copy_config)
    return arch_model

def build_loss(config):
    support_loss = ['DBLoss']
    copy_config = copy.deepcopy(config)
    loss_type = copy_config.pop('type')
    assert loss_type in support_loss, f'all support loss is {support_loss}'
    criterion = eval(loss_type)(**copy_config)
    return criterion

def build_neck(neck_name, **kwargs):
    support_neck = ['FPN', 'FPEM_FFM']
    assert neck_name in support_neck, f'all support neck is {support_neck}'
    neck = eval(neck_name)(**kwargs)
    return neck

def build_head(head_name, **kwargs):
    support_head = ['ConvHead', 'DBHead']
    assert head_name in support_head, f'all support head is {support_head}'
    head = eval(head_name)(**kwargs)
    return head

def build_backbone(backbone_name, **kwargs):
    support_backbone = ['MobileNetV3']
    assert backbone_name in support_backbone, f'all support backbone is {support_backbone}'
    backbone = eval(backbone_name)(**kwargs)
    return backbone