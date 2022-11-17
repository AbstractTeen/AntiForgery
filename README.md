## Anti-Forgery
An example of **[Anti-Forgery: Towards a Stealthy and Robust DeepFake Disruption Attack via Adversarial Perceptual-aware Perturbations](https://arxiv.org/abs/2206.00477)** (to be presented at the **IJCAI-ECAI 2022**). This repository contains code for crafting perceptual-aware perturbation in the Lab color space to attack an image-to-image translation network. 

## Preparation
**CelebA Dataset**

```
bash download.sh celeba
```
**StarGAN Model**

```
bash download.sh pretrained-celeba-256x256
```

More information about the CelebA dataset can be found [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). 

## Attack Testing

Here is a simple example of  testing our method to attack StarGAN on the CelebA dataset.
```
# Test
python main.py --mode test --image_size 256 --c_dim 5 --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young --model_save_dir='stargan_celeba_256/models' --result_dir='./results' --test_iters 200000 --attack_iters 100 --batch_size 1
```

## Related Work

We use some code from the original [Disrupting-Deepfakes](https://github.com/natanielruiz/disrupting-deepfakes), which does a good work.

## Citation
If you find this work useful, please cite our [paper](https://arxiv.org/abs/2206.00477):

```
@article{wang2022anti,
  title={Anti-Forgery: Towards a Stealthy and Robust DeepFake Disruption Attack via Adversarial Perceptual-aware Perturbations},
  author={Wang, Run and Huang, Ziheng and Chen, Zhikai and Liu, Li and Chen, Jing and Wang, Lina},
  journal={arXiv preprint arXiv:2206.00477},
  year={2022}
}

```
The IJCAI camera-ready version (pdf) is available [here](https://www.ijcai.org/proceedings/2022/0107.pdf)
