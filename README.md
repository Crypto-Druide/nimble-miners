# Nimble Miners

This repo provides a set of miners for anyone to mine Nimble native tokens. It enables single machine model training and distributed training to increase your chances of successful token mining.

# Development

### Code Downloading

```bash
# clone the code repo to local
git clone https://github.com/nimble-technology/nimble-miners.git
cd nimble-miners
```

### Virtual Env Setup

```bash
# install python3.9 or higher if not yet.
brew install python3.9

# create virtualenv
python3 -m venv nimble-env

# activate virtualenv
source nimble-env/bin/activate

# install nimble miners dependencies
python3 -m pip install -r requirements.txt

# deactivate virtualenv
deactivate

# clean virtualenv
rm -rf nimble-env
```
