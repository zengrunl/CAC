import torch
import torch.nn as nn


def handle_scp(scp_path):
    '''
    Read scp file script
    input: 
          scp_path: .scp file's file path
    output: 
          scp_dict: {'key':'wave file path'}
    '''
    scp_dict = dict()
    line = 0
    lines = open(scp_path, 'r').readlines()
    for l in lines:
        scp_parts = l.strip().split()
        line += 1
        if len(scp_parts) != 2:
            raise RuntimeError("For {}, format error in line[{:d}]: {}".format(
                scp_path, line, scp_parts))
        if len(scp_parts) == 2:
            key, value = scp_parts
        # if key in scp_dict:
        #     raise ValueError("Duplicated key \'{0}\' exists in {1}".format(
        #         key, scp_path))

        scp_dict[key] = value

    return scp_dict


def check_parameters(net):
    '''
        Returns module parameters. Mb
    '''
    parameters = sum(param.numel() for param in net.parameters())    # param.numel() 统计模型参数量
    return parameters / 10**6


def check_parameters_amount(net):
    # 参数量   百万级M
    total = sum([param.nelement() for param in net.parameters()])
    return total / 1e6
    # print("Number of parameter: %.2fM" % (total / 1e6))