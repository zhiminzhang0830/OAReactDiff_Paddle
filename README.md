OA-ReactDiff_Paddle
==============================

*This is a PaddlePaddle implementation of OA-ReactDiff. For PyTorch version, please check [OAReactDiff](https://github.com/chenruduan/OAReactDiff).*

### <ins>O</ins>bject-<ins>a</ins>ware SE(3) GNN for generating chemical <ins>react</ins>ions under <ins>diff</ins>usion model (OA-ReactDiff)


*OA-ReactDiff* is the first diffusion-based generative model for generating  **3D chemical reactions**, which not only accelerates the search for 3D transition state in chemical reactions by **a factor of 1000**, but also generates and explores **new and unknown** chemical reactions.

Recently, the framework of DDPM has been progressively applied to tasks related to chemical molecules. Examples include generating [3D structures of organic small molecules](https://arxiv.org/abs/2203.17003), [protein-small molecule docking](https://arxiv.org/abs/2210.01776), and [protein structure-based drug design](https://arxiv.org/abs/2210.13695) (Figure 1a). Here we present OA-ReactDiff, a model directly generates the core concept of chemistry, elementary chemical reactions. OA-ReactDiff does so by diffusing and denoising three 3D structures in a reaction, i.e., reactant, transition state, and product, all together (Figure 1b).

<div>
    <img src="./figures/F1_ReactDiff_v2.png" alt="image_diffusion" width="750" title="image_diffusion">
    <p style='font-size:1rem; font-weight:bold'>Figure 1｜DDPM for generating a molecule and a reaction</p>
</div>

Sounds easy? There is one caveat! Conventional SE(3) won't apply here. Why? Three structures in a reaction follows a higher **"object-wise"** SE(3) symmetry instead of a "system-wise" SE(3) symmetry (Figure 2a). For example, any SE(3) transformation on a object (e.g., a molecule in reactant) should not change the nature of the chemical reaction of study. To tackle this problem, we develop a simple way to realize "object-wise" SE(3) symmetry by mixing a conventional SE(3) update and scalar message passing layer (Figure 2b). This protocol is general to many SE(3) graph neural networks taht are still under active development. At this point, we use [LEFTNet](https://arxiv.org/abs/2304.04757), a SOTA-level SE(3) graph neural network in OA-ReactDiff.

<div>
    <img src="./figures/F2_OASE3_v2.png" alt="image_diffusion" width="750" title="image_diffusion">
    <p style='font-size:1rem; font-weight:bold'>Figure 2｜Object-wise SE(3) and our implementation</p>
</div>

Since OA-ReactDiff maintains **all symmetries and constraints** in chemical reactions, it does not require any pre-processing (e.g., atom mapping, fragment alignment) and post-processting (e.g., coverting distance matrix to 3D geometry), which sometimes are infeasible in exploring unknown reactions. OA-ReactDiff reduces the transition state search cost **from days in using density functional theory to seconds**. Due to the stochastic nature of DDPM, OA-ReactDiff can generate unintended reactions "beyond design", **complementing the chemical intuition-based reaction exploration** that people have been using for decades.



### Installation
1. install dependencies
```
    pip install -r requirements.txt
```
2. install [PaddlePaddle](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)
``` 
    # for PaddlePaddle==2.6.2 and cuda==11.6
    python -m pip install paddlepaddle-gpu==2.6.2.post116 -i https://www.paddlepaddle.org.cn/packages/stable/cu116/
```

### Usage
```
    python demo.py
```


### Citation
```
@misc{Duan2023OAReactDiff,
    title={Accurate transition state generation with an object-aware equivariant elementary reaction diffusion model}, 
    author={Chenru Duan and Yuanqi Du and Haojun Jia and Heather J. Kulik},
    year={2023},
    eprint={2304.06174},
    archivePrefix={arXiv},
}

@Article{OA-ReactDiff,
    author={Duan, Chenru
    and Du, Yuanqi
    and Jia, Haojun
    and Kulik, Heather J.},
    title={Accurate transition state generation with an object-aware equivariant elementary reaction diffusion model},
    journal={Nature Computational Science},
    year={2023},
    month={Dec},
    day={01},
    volume={3},
    number={12},
    pages={1045-1055},
    doi={10.1038/s43588-023-00563-7},
    url={https://doi.org/10.1038/s43588-023-00563-7}
}
```



### Acknowledgements

Code development based on [OAReactDiff](https://github.com/chenruduan/OAReactDiff)
