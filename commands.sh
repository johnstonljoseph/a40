

python -m a40.main \
--steps 4000 \
--batch-size 32 \
--seq-len 256 \
--checkpoint-interval 500 \
--train-layers 2,8,15

python -m a40.weight_calibration.main --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

python -m a40.activation_calibration.main --layers 0,1,3,4,5,6,7,9,10,11,12,13,14,15


python -m a40.main --train-layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

pyhon -m a40.eval 



# Train on cpu
python -m a40.main --device cpu --train-layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15



# Eval
python -m a40.eval --checkpoint-step 1000 --train-layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --batch-size 16 --seq-len 512 --tasks gsm8k











# ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJCfdt1bZQlcOOxKQAsNCZy50c/OrOdwnXXXIbho6dnf h200
# -----BEGIN OPENSSH PRIVATE KEY-----
# b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gtZW
# QyNTUxOQAAACCQn3bdW2UJXDjsSkALDQmcudHPzqzncJ111yG4aOnZ3wAAAIi6EWWWuhFl
# lgAAAAtzc2gtZWQyNTUxOQAAACCQn3bdW2UJXDjsSkALDQmcudHPzqzncJ111yG4aOnZ3w
# AAAECNiGr23qXNmF07duCjTIV+swUy2bBA+AECBTKGquukLJCfdt1bZQlcOOxKQAsNCZy5
# 0c/OrOdwnXXXIbho6dnfAAAABGgyMDAB
# -----END OPENSSH PRIVATE KEY-----

