# AAAI 2022 Paper
## [Event-Aware Multimodal Mobility Nowcasting](https://arxiv.org/pdf/2112.08443.pdf)

## Usage
- Training mode example (to train ST-Net on JONAS-NYC)
```
python Main.py -data=JONAS-NYC -model=ST-Net
```
- Testing mode example (to test EAST-Net on COVID-US)
```
python Main.py -data=COVID-US -model=EAST-Net -test
```

## Citation
If you find anything in this repository useful to your research, please cite our paper :) We sincerely appreciate it.
```
@inproceedings{wang2022event,
  title={Event-Aware Multimodal Mobility Nowcasting},
  author={Wang, Zhaonan and Jiang, Renhe and Xue, Hao and Salim, Flora D and Song, Xuan and Shibasaki, Ryosuke},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  number={4},
  pages={4228--4236},
  year={2022}
}
```
