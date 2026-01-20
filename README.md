# Awesome-Deep-Learning-based-Flare-Removal-Methods
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome) ![GitHub stars](https://img.shields.io/github/stars/cranbs/Awesome-Flare-Removal?color=green)  ![GitHub forks](https://img.shields.io/github/forks/cranbs/Awesome-Flare-Removal?color=9cf)

<p align="center">
  <img src="asset/flareRemoval.png" />
</p>

<details>
<summary>:loudspeaker:<strong>Last updated: 2025.10.27</strong></summary>
  
- [01/2026] Update with AAAI2026 papers. 
- [10/2025] Update with NeurlPS2025, EAAI papers. 
- [09/2025] Update with ACMMM2025, TPAMI papers. 
</details>

------

## 📚 Table of Contents:

<table style="margin-left: auto; margin-right: auto;">
  <li><a href="#1-Latest">1. Latest Work</a></li>
  <li><a href="#2-Survey-Papers">2. Survey Papers</a></li>
  <li><a href="#3-Datasets">3. Datasets</a>
  <li><a href="#4-Flare-Removal">4. Flare Removal</a>
      <ul>
          <li><a href="#41-unsupervised">4.1. Unsupervised Flare Removal </a></li>
          <li><a href="#42-selfsupervised">4.2. Self-supervised Flare Removal</a></li>
          <li><a href="#43-supervised">4.3. Supervised Flare Removal</a></li>
      </ul>
  </li><li><a href="#5-Other-Related">5. Other Related</a>
</table>

-------

<h2 id="1-Latest">
    <span>:fire: 1. Latest Work </span>
</h2>

| Year |       Publication        |                            Title                             |                            Paper                             |                        Project                        |
| :--: | :----------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :---------------------------------------------------: |
| 2026 | `AAAI` `CCF A` | **Nighttime Flare Removal via Wavelet-Guided and Gated-Enhanced Spatial-Frequency Fusion Network** |  | [[code](https://github.com/gyang666/WGSF-Net)]   |
| 2026 | `AAAI` `CCF A` | **CAST-LUT: Tokenizer-Guided HSV Look-Up Tables for Purple Flare Removal** | **[[paper](https://arxiv.org/pdf/2511.06764)]** | [[code](https://github.com/Pu-Wang-alt/Reduce-Purple-Flare/)]   |
| 2025 | `NIPS` `CCF A` | **FlareX: A Physics-Informed Dataset for Lens Flare Removal via 2D Synthesis and 3D Rendering** |      **[[paper](https://arxiv.org/pdf/2510.09995)]**       |   **[[dataset](https://github.com/qulishen/FlareX)]**   |
| 2025 |       `ACM MM` `CCF A`       | **DeflareMamba: Hierarchical Vision Mamba for Contextually Consistent Lens Flare Removal** |       **[[paper](https://arxiv.org/pdf/2508.02113)]**        |  [[code](https://github.com/BNU-ERC-ITEA/DeflareMamba)]       |
| 2025 |       `ICCV` `CCF A`       | **Removing Out-of-Focus Reflective Flares via Color Alignment** |   **[[paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Lan_Removing_Out-of-Focus_Reflective_Flares_via_Color_Alignment_ICCV_2025_paper.pdf)]**         |     |
| 2025 |       `ICCV` `CCF A`       | **LightsOut: Diffusion-based Outpainting for Enhanced Lens Flare Removal** |         **[[paper](https://arxiv.org/pdf/2510.15868)]**     |    [[code](https://github.com/Ray-1026/LightsOut-official)]    |
| 2025 | `ICCV` `CCF A` | **PBFG: A New Physically-Based Dataset and Removal of Lens Flares and Glares** |      **[[paper](https://cg.skku.edu/pub/papers/2025-zhu-iccv-pbfg-cam.pdf)]**       |   **[[dataset](https://github.com/cgskku/pbfg)]**   |
| 2025 | `ICCVW` `CCF A` | **FlareGS: 4D Flare Removal using Gaussian Splatting for Urban Scenes** |      **[[paper](https://openaccess.thecvf.com/content/ICCV2025W/2COOOL/papers/Chandak_FlareGS_4D_Flare_Removal_using_Gaussian_Splatting_for_Urban_Scenes_ICCVW_2025_paper.pdf)]**       |     |
| 2025 |        `AAAI` `CCF A`         | **Disentangle Nighttime Lens Flares: Self-supervised Generation-based Lens Flare Removal** |       **[[paper](https://arxiv.org/pdf/2502.10714)]**        |   **[[code](https://github.com/xhnshui/Flare-Removal)]**        |
| 2025 |        `TPAMI` `CCF A`       | **Image Lens Flare Removal Using Adversarial Curve Learning** |       **[[paper](https://ieeexplore.ieee.org/document/10989553)]**        | [[code](https://github.com/YuyanZhou1/Improving-Lens-Flare-Removal)] |

-------
<details open>
  <summary>
      <h2 id="2-Survey-Papers">
          <span>2. Survey Papers</span>
      </h2>
  </summary>

| Year |       Publication        |                            Title                             |                            Paper                             |                        Project                        |
| :--: | :----------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :---------------------------------------------------: |
| 2023 | `CVPRW` `CCF A Workshop` | **Mipi 2023 challenge on nighttime flare removal: Methods and results** | **[[paper](https://openaccess.thecvf.com/content/CVPR2023W/MIPI/papers/Dai_MIPI_2023_Challenge_on_Nighttime_Flare_Removal_Methods_and_Results_CVPRW_2023_paper.pdf)]** | **[[project](https://mipi-challenge.org/MIPI2023/)]** |
| 2024 | `CVPRW` `CCF A Workshop` | **Mipi 2024 challenge on nighttime flare removal: Methods and results** | **[[paper](https://ieeexplore.ieee.org/document/10678229)]** | **[[project](https://mipi-challenge.org/MIPI2024/)]** |
| 2023 |         `ArXiv`          |            **Toward flare-free images: A survey**            |       **[[paper](https://arxiv.org/abs/2310.14354)]**        |                                                       |
| 2023 |         `ArXiv`          | **Toward Real Flare Removal: A Comprehensive Pipeline and A New Benchmark** |       **[[paper](https://arxiv.org/pdf/2306.15884)]**                                          |

</details>

-------

<details open>
  <summary>
      <h2 id="3-Datasets">
          <span>3. Datasets</span>
      </h2>
  </summary>

| Year |   Publication   |                            Title                             |                            Paper                             |                      Dataset                      |
| :--: | :-------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :-----------------------------------------------: |
| 2025 | `NIPS` `CCF A` | **FlareX: A Physics-Informed Dataset for Lens Flare Removal via 2D Synthesis and 3D Rendering** |      **[[paper](https://arxiv.org/pdf/2510.09995)]**       |   **[[dataset](https://github.com/qulishen/FlareX)]**   |
| 2025 | `ICCV` `CCF A` | **PBFG: A New Physically-Based Dataset and Removal of Lens Flares and Glares** |      **[[paper](https://cg.skku.edu/pub/papers/2025-zhu-iccv-pbfg-cam.pdf)]**       |   **[[dataset](https://github.com/cgskku/pbfg)]**   |
| 2024 | `TPAMI` `CCF A` | **Flare7k++: Mixing synthetic and real datasets for nighttime flare removal and beyond** |       **[[paper](https://arxiv.org/pdf/2306.04236)]**        | **[[dataset](https://github.com/ykdai/Flare7K)]** |
| 2022 | `NIPS` `CCF A`  | **Flare7k: A phenomenological nighttime flare removal dataset** | **[[paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/1909ac72220bf5016b6c93f08b66cf36-Paper-Datasets_and_Benchmarks.pdf)]** | **[[dataset](https://github.com/ykdai/Flare7K)]** |
| 2023 |     `ArXiv`     | **Tackling scattering and reflective flare in mobile camera systems: A raw image dataset for enhanced flare removal** | **[[paper](https://ui.adsabs.harvard.edu/abs/2023arXiv230714180L/abstract)]** |                                                   |
| 2023 |     `ArXiv`     | **Toward Real Flare Removal: A Comprehensive Pipeline and A New Benchmark** |       **[[paper](https://arxiv.org/pdf/2306.15884)]**        |     

</details>

-------
<details open>
  <summary>
      <h2 id="4-Flare-Removal">
          <span>4. Flare removal</span>
      </h2>
  </summary>
  
<details open>
  <summary><h3 id="41-unsupervised"><span>4.1. Unsupervised Flare Removal</span></h3></summary>

| Year |  Publication   |                            Title                             |                            Paper                             |                             Code                             |
| :--: | :------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2021 | `ICCV` `CCF A` | **Light source guided single image flare removal from unpaired data** | **[[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Qiao_Light_Source_Guided_Single-Image_Flare_Removal_From_Unpaired_Data_ICCV_2021_paper.pdf)]** | **[[code](https://github.com/tanmayj2020/LightSourceGuideSingleImageFlareRemoval-ICCV2021)]** |

</details>

<details open>
  <summary><h3 id="42-selfsupervised"><span>4.2. Self-supervised  Flare Removal</span></h3></summary>

| Year |  Publication   |                            Title                             |                            Paper                             |                             Code                             |
| :--: | :------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2025 |        `AAAI` `CCF A`         | **Disentangle Nighttime Lens Flares: Self-supervised Generation-based Lens Flare Removal** |       **[[paper](https://arxiv.org/pdf/2502.10714)]**        |   **[[code](https://github.com/xhnshui/Flare-Removal)]**        |

</details>

<details open>
  <summary><h3 id="43-supervised"><span>4.3. Supervised  Flare Removal</span></h3></summary>


| Year |          Publication          |                            Title                             |                            Paper                             |                             Code                             |
| :--: | :---------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2026 | `Pattern Recognition Letter` | **Nighttime flare removal via frequency decoupling** |  **[[paper](https://www.sciencedirect.com/science/article/pii/S0167865526000188)]** |  |
| 2026 | `AAAI` `CCF A` | **Nighttime Flare Removal via Wavelet-Guided and Gated-Enhanced Spatial-Frequency Fusion Network** |  | [[code](https://github.com/gyang666/WGSF-Net)]   |
| 2025 | `EAAI` `CCF C` | **Flare detection and detail compensation for nighttime flare removal** |      **[[paper](https://www.sciencedirect.com/science/article/pii/S0952197625029574)]**       |     |
| 2025 |       `ICCV` `CCF A`       | **LightsOut: Diffusion-based Outpainting for Enhanced Lens Flare Removal** |         **[[paper](https://arxiv.org/pdf/2510.15868)]**   |    [[code](https://github.com/Ray-1026/LightsOut-official)]    |
| 2025 |       `ICCV` `CCF A`       | **Removing Out-of-Focus Reflective Flares via Color Alignment** |  **[[paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Lan_Removing_Out-of-Focus_Reflective_Flares_via_Color_Alignment_ICCV_2025_paper.pdf)]**       |      |
| 2025 | `ICCVW` `CCF A` | **FlareGS: 4D Flare Removal using Gaussian Splatting for Urban Scenes** |      **[[paper](https://openaccess.thecvf.com/content/ICCV2025W/2COOOL/papers/Chandak_FlareGS_4D_Flare_Removal_using_Gaussian_Splatting_for_Urban_Scenes_ICCVW_2025_paper.pdf)]**       |     |
| 2025 |       `ACM MM` `CCF A`       | **DeflareMamba: Hierarchical Vision Mamba for Contextually Consistent Lens Flare Removal** |       **[[paper](https://arxiv.org/pdf/2508.02113)]**        |  [[code](https://github.com/BNU-ERC-ITEA/DeflareMamba)]       |
| 2025 |       `IEEE TCSVT` `CCF B`       | **SAFAformer: Scale-Aware Frequency-Adaptive Guidance for Nighttime Flare Removal** |       **[[paper](https://ieeexplore.ieee.org/document/11113259)]**        |         |
| 2025 |       `Neural Networks` `CCF B`       | **LUFormer : A luminance-informed localized transformer with frequency augmentation for nighttime flare removal** |       **[[paper](https://www.sciencedirect.com/science/article/pii/S0893608025005404)]**        |   [[code](https://github.com/HeZhao0725/LUFormer)]|
| 2025 |        `TCE`       | **Nighttime Glare Removal for Consumer Electronics via Latent Space Transformation and Feature Enhanced Attention Mechanism** |       **[[paper](https://ieeexplore.ieee.org/abstract/document/11006158)]**        |   |
| 2025 |        `TPAMI` `CCF A`       | **Image Lens Flare Removal Using Adversarial Curve Learning** |       **[[paper](https://ieeexplore.ieee.org/document/10989553)]**        | [[code](https://github.com/YuyanZhou1/Improving-Lens-Flare-Removal)] |
| 2025 |        `Expert Systems with Applications` `CCF C`       | **IllumiNet: A two-stage model for effective flare removal and light enhancement under complex lighting conditions** |       **[[paper](https://www.sciencedirect.com/science/article/pii/S0957417425012606)]**        |  
| 2025 |       `ICASSP` `CCF C`        |            **Flare-Aware RWKV for Flare Removal**            | **[[paper](https://ieeexplore.ieee.org/document/10888487)]** |                                                              |
| 2025 |   `Neural Networks` `CCF B`   | **When low-light meets flares: Towards Synchronous Flare Removal and Brightness Enhancement** | **[[paper](https://www.sciencedirect.com/science/article/abs/pii/S0893608025000280)]** |                                                              |
| 2025 |   `Neurocomputing` `CCF C`    |        **Mask-Q attention network for flare removal**        | **[[paper](https://www.sciencedirect.com/science/article/abs/pii/S0952197625001034)]** |                                                              |
| 2025 |        `EAAI` `CCF C`         | **A self-prompt based dual-domain network for nighttime flare removal** | **[[paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231225007726)]** |                                                              |
| 2025 |      `IEEE TASE` `CCF B`      | **Self-prior Guided Spatial and Fourier Transformer  for Nighttime Flare Removal** | **[[paper](https://ieeexplore.ieee.org/abstract/document/10877847)]** |        **[[code](https://github.com/cranbs/SGSFT)]**         |
| 2025 |      `IEEE TCSVT` `CCF B`      | **LPFSformer: Location Prior Guided Frequency and Spatial Interactive Learning for Nighttime Flare Removal** | **[[paper](https://ieeexplore.ieee.org/document/10777570)]** |                                                              |
| 2024 |       `ICASSP` `CCF C`        | **Flare-free vision: Empowering uformer with depth insights** | **[[paper](https://www.researchgate.net/profile/Marwan-Torki/publication/376586936_FLARE-FREE_VISION_EMPOWERING_UFORMER_WITH_DEPTH_INSIGHTS_ICASSP2024/links/657ea3058e2401526dde1e84/FLARE-FREE-VISION-EMPOWERING-UFORMER-WITH-DEPTH-INSIGHTS-ICASSP2024.pdf)]** | **[[code](https://github.com/yousefkotp/Flare-Free-Vision-Empowering-Uformer-with-Depth-Insights)]** |
| 2024 | `The Visual Computer` `CCF C` | **Mfdnet: Multifrequency deflare network for efficient nighttime flare removal** | **[[paper](https://link.springer.com/article/10.1007/s00371-024-03540-x)]** | **[[code](https://github.com/Jiang-maomao/flare-removal)]**                                                             |
| 2024 |            `ArXiv`            | **Harmonizing Light and Darkness: A Symphony of Prior-guided Data Synthesis and Adaptive Focus for Nighttime Flare Removal** |       **[[paper](https://arxiv.org/pdf/2404.00313)]**        | **[[code](https://github.com/qulishen/Harmonizing-Light-and-Darkness)]** |
| 2024 | `Pattern Recognition` `CCF B` | **Gr-gan: A unified adversarial framework for single image glare removal and denoising** | **[[paper](https://www.sciencedirect.com/science/article/pii/S0031320324005661)]** |                                                              |
| 2024 |        `BMVC` `CCF C`         | **Difflare: Removing image lens flare with latent diffusion model** |       **[[paper](https://arxiv.org/pdf/2407.14746)]**        |    **[[code](https://github.com/TianwenZhou/Difflare)]**     |
| 2024 |         `ACM MM` `CCF A`       | **Understanding and tackling scattering and reflective flare for mobile camera systems** |       **[[paper](https://dl.acm.org/doi/pdf/10.1145/3664647.3681306)]**        |      |
| 2024 | `BMVC` `CCF C` | **GN-FR: Generalizable Neural Radiance Fields for Flare Removal** | **[[paper](https://arxiv.org/pdf/2412.08200)]** |  |
| 2024 |      `IEEE TIP` `CCF A`       | **Towards blind flare removal using knowledge-driven flare-level estimator** | **[[paper](https://ieeexplore.ieee.org/abstract/document/10726687)]** |  
| 2023 |        `ICCV` `CCF A`         | **Improving lens flare removal with general-purpose pipeline and multiple light sources recovery** | **[[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhou_Improving_Lens_Flare_Removal_with_General-Purpose_Pipeline_and_Multiple_Light_ICCV_2023_paper.pdf)]** | **[[code](https://github.com/YuyanZhou1/Improving-Lens-Flare-Removal)]** |
| 2023 |   `CVPRW` `CCF A Workshop`    | **Hard-negative sampling with cascaded fine-tuning network to boost flare removal performance in the nighttime images** | **[[paper](https://openaccess.thecvf.com/content/CVPR2023W/MIPI/papers/Song_Hard-Negative_Sampling_With_Cascaded_Fine-Tuning_Network_To_Boost_Flare_Removal_CVPRW_2023_paper.pdf)]** |                                                              |
| 2023 |   `CVPRW` `CCF A Workshop`    | **Ff-former: Swin fourier transformer for nighttime flare removal** | **[[paper](https://openaccess.thecvf.com/content/CVPR2023W/MIPI/papers/Zhang_FF-Former_Swin_Fourier_Transformer_for_Nighttime_Flare_Removal_CVPRW_2023_paper.pdf)]** |                                                              |
| 2023 |        `CVPR` `CCF A`         | **Nighttime smartphone reflective flare removal using optical center symmetry prior** | **[[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Dai_Nighttime_Smartphone_Reflective_Flare_Removal_Using_Optical_Center_Symmetry_Prior_CVPR_2023_paper.pdf)]** |     **[[code](https://github.com/ykdai/BracketFlare)]**      |
| 2022 |        `NIPS` `CCF A`         | **Flare7k: A phenomenological nighttime flare removal dataset** | **[[paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/1909ac72220bf5016b6c93f08b66cf36-Paper-Datasets_and_Benchmarks.pdf)]** |        **[[code](https://github.com/ykdai/Flare7K)]**        |
| 2021 |        `ICCV` `CCF A`         |      **How to train neural networks for flare removal**      | **[[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_How_To_Train_Neural_Networks_for_Flare_Removal_ICCV_2021_paper.pdf)]** | **[[code](https://github.com/budui/flare_removal_pytorch)]** |

</details>

</details>

-------

<details open>
  <summary><h2 id="5-Other-Related"><span>5. Other Related</span></h2></summary>


                                                        
| Year |  Publication   |                            Title                             |                            Paper                             |                             Code                             |
| :--: | :------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2026 | `AAAI` `CCF A` | **CAST-LUT: Tokenizer-Guided HSV Look-Up Tables for Purple Flare Removal** | **[[paper](https://arxiv.org/pdf/2511.06764)]** | [[code](https://github.com/Pu-Wang-alt/Reduce-Purple-Flare/)]   |
| 2025 |`ICCVW`| **Learning to See Through Flare** | **[[paper](https://openaccess.thecvf.com/content/ICCV2025W/Responsible-Imaging/papers/Peng_Learning_to_See_Through_Flare_ICCVW_2025_paper.pdf)]** |   |
| 2025 |`IEEE IoTJ`| **Adversarial Lens Flares: A Threat to Camera-Based Systems in Smart Devices** | **[[paper](https://ieeexplore.ieee.org/abstract/document/10806811/)]** |   |
| 2025 |`ArXiv`| **Integrating Spatial and Frequency Information for Under-Display Camera Image Restoration** | **[[paper](https://arxiv.org/pdf/2501.18517)]** |   |

</details>

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=cranbs/Awesome-Flare-Removal&type=date&legend=top-left)](https://www.star-history.com/#c%20ran%20b%20s/c%20ran%20b%20s&cranbs/Awesome-Flare-Removal&type=date&legend=top-left)
