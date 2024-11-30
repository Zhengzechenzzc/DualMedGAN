#!/usr/bin/python3

import argparse
import os
from trainer import Cyc_Trainer,Nice_Trainer,P2p_Trainer,Munit_Trainer,Unit_Trainer,cycle_info_Trainer,\
    Siam_Trainer,fu_Trainer,DA_Trainer, fu_eloss_par_Trainer, fu_eloss_par_info_Trainer,\
    fu_eloss_par512_Trainer, fu_eloss_par_caca_Trainer,fu_eloss_par_info_s2_Trainer,fu_eloss_info_SGCA_Trainer,\
    fu_eloss_info_MCVA_Trainer, fu_eloss_info_jump1_DMCVA_Trainer,fuS2_eloss_info_jump1_DESA_Trainer,\
    fuS2_eloss_info_jump1S2_DMCVA_Trainer, fuS2_Trainer, fuCBAM_DESA_Trainer, fuCBAM_DESA_jump1CBAM_Trainer, \
fuS2_eloss_info_jump1_DMCVA_Trainer, SLS2_Trainer, baseline_Trainer, eloss_Trainer, fuS2_loss_Trainer, fuS2_SLS2_eloss_Trainer
import yaml

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_config(config):
    with open(config, 'r', encoding='utf-8') as stream:
        return yaml.safe_load(stream)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='Yaml/fuCBAM_DESA_jump1CBAM.yaml', help='Path to the config file.')
    opts = parser.parse_args()
    config = get_config(opts.config)
    # print("Validation data root:", config['val_dataroot'])
    if config['name'] == 'name':
        trainer =name_Trainer(config)
    elif config['name'] == 'fuS2_eloss_info_jump1S2_DMCVA':
        trainer = fuS2_eloss_info_jump1S2_DMCVA_Trainer(config)


    trainer.train()
    
    



###################################
if __name__ == '__main__':
    main()