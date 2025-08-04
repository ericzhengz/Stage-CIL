# When Classes Evolve: A Benchmark and Framework for Stage-Aware Class-Incremental Learning

## 🔧 Requirements

**Environment**

1 [torch 1.11.0](https://github.com/pytorch/pytorch)

2 [torchvision 0.12.0](https://github.com/pytorch/vision)

3 [open-clip 2.17.1](https://github.com/mlfoundations/open_clip/releases/tag/v2.17.1)


## 💡 Running scripts

To prepare your JSON files, refer to the settings in the `exps` folder and run the following command. All main experiments from the paper are already provided in the `exps` folder, you can simply execute them to reproduce the results found in the `logs` folder.

```
python main.py --config ./exps/[configname].json
```
