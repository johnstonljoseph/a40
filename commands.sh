

python -m a40.main \
--steps 4000 \
--batch-size 32 \
--seq-len 256 \
--checkpoint-interval 500 \
--train-layers 2,8,15


python -m a40.weight_calibration.main --layers 
