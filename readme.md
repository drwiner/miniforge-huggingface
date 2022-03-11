Steps to reproduce environment on Mac OS
```
  pip install tensorflow-macos
  arch -arm64 brew install hdf5  
  export HDF5_DIR=/opt/homebrew/opt/hdf5
  pip install --no-binary=h5py h5py
  pip install tensorflow-macos
  pip install tensorflow-metal
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  git clone https://github.com/huggingface/tokenizers
  cd tokenizers/bindings/python/
  pip install setuptools_rust
  python setup.py install
  cd ../../../
  pip install git+https://github.com/huggingface/transformers

```