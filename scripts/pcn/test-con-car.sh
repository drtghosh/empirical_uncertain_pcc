python test_contrastive.py -l proj_log_con \
                          -x pcn_car_con \
                          -m contrast_ae \
                          -n pcn \
                          -r data/PCN \
                          -c car \
                          -b 1 \
                          -a proj_log_ae/pcn_car_ae/ae/model/ckpt_epoch100.pth \
                          --ckpt 100 \
                          --gp_batch 100