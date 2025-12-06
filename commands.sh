

python -m a40.main \
--steps 4000 \
--batch-size 32 \
--seq-len 256 \
--checkpoint-interval 500 \
--train-layers 2,8,15


python -m a40.weight_calibration.main --layers 

python -m a40.activation_calibration.main --layers 2 --batch_count 1


python -m a40.main \
--steps 40 \
--batch-size 2 \
--seq-len 32 \
--checkpoint-interval 5 \
--train-layers 2,8,15



# ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJCfdt1bZQlcOOxKQAsNCZy50c/OrOdwnXXXIbho6dnf h200
# -----BEGIN OPENSSH PRIVATE KEY-----
# b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gtZW
# QyNTUxOQAAACCQn3bdW2UJXDjsSkALDQmcudHPzqzncJ111yG4aOnZ3wAAAIi6EWWWuhFl
# lgAAAAtzc2gtZWQyNTUxOQAAACCQn3bdW2UJXDjsSkALDQmcudHPzqzncJ111yG4aOnZ3w
# AAAECNiGr23qXNmF07duCjTIV+swUy2bBA+AECBTKGquukLJCfdt1bZQlcOOxKQAsNCZy5
# 0c/OrOdwnXXXIbho6dnfAAAABGgyMDAB
# -----END OPENSSH PRIVATE KEY-----

