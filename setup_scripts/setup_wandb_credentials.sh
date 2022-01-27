# This is secret and shouldn't be checked into version control
WANDB_API_KEY=$MY_WANDB_API_KEY
# Name and notes optional
#WANDB_NAME="My first run"
#WANDB_NOTES="Smaller learning rate, more regularization."

# Only needed if you don't checkin the wandb/settings file
WANDB_ENTITY=evolvingfungus
WANDB_PROJECT=tali-gcp-sweep-1

# If you don't want your script to sync to the cloud
# os.environ['WANDB_MODE'] = 'offline'