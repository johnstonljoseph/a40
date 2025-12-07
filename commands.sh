
python -m a40.weight_calibration.main --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

python -m a40.activation_calibration.main --layers 0,1,2,3,4,5,6,7,9,10,11,12,13,14,15

python -m a40.main --train-layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15


# Train on cpu
python -m a40.main --device cpu --train-layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15



# Eval
python -m a40.eval --checkpoint-step 1000 --train-layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --batch-size 16 --seq-len 512 --tasks gsm8k


