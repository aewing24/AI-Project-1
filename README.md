<br />
<div align="center">
  <h3 align="center">AI-Project-1</h3>

  <p align="center">
    TCSS 435A Spring 2025 Project 1 using DQN algorithm to learn to play Lunar Lander-v3 gym game.
    <br />
    <a href="https://github.com/aewing24/AI-Project-1"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/aewing24/AI-Project-1">View Demo</a>
    &middot;
    <a href="https://github.com/aewing24/AI-Project-1/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/aewing24/AI-Project-1/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

## Contributors

- [Nazarii Revitskyi](https://github.com/NazariiR)
- [Mathew Belmont](https://github.com/belmontmat)
- [Alexander Ewing](https://github.com/aewing24)
- [Lucas Jeong](https://github.com/ljeong072)
- [Nathan Wanjongkhum](https://github.com/NathanWanjongkhum)

<!-- ABOUT THE PROJECT -->

## About The Project

This project is a DQN agent that learns to play Lunar Lander-v3 gym game. Which can be optionally extended to DDQN or Double DQN.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

- Python 3.11
- Pytorch 1.13.1
- Gymnasium 1.0.0
- Swig 4.3.0

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->

## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

If you are on a Windows machine, you will need to install the following:

- Python 3.11 or higher
- Anaconda or Miniconda
- Microsoft C++ Build Tools

### Installation

If you are on a linux machine, you can setup the environment using the following command.

```bash
conda install python==3.10

conda install conda-forge::gymnasium==1.0.0

conda install swig (v4.3.0)

conda install conda-forge::gymnasium-box2d
```

- `python==3.10`: For gymnasium-box2d that supports python <3.13
- `conda install conda-forge::gymnasium==1.0.0`
  : this gymnasium version supports LunarLander-v3 the 'conda install gymnasium' installs 0.2x version which only has LunarLander-v2
- `conda install conda-forge::gymnasium-box2d ` for v1.0.0

If you are using Windows, you can install the environment using Anaconda Prompt. Either watch the video or follow the steps below.
https://www.youtube.com/watch?v=gMgj4pSHLww

1. Install Anaconda Prompt
2. Open Anaconda Prompt
3. Create a new environment

```bash
conda create -n gymenv
```

4. Activate the environment

```bash
conda activate gymenv
```

5. Install python

```bash
conda install python==3.11
```

6. Install gymnasium

```bash
pip install gymnasium[classic-control]
```

6. Install toy-text

```bash
pip install gymnasium[toy-text]
```

7. Install mujoco

```bash
pip install gymnasium[mujoco]
```

8. Install atari

```bash
pip install gymnasium[atari]
pip install gymnasium[accept-rom-license]
```

9. Install swig

```bash
conda install swig
```

10. Install gymnasium-box2d

```bash
pip install gymnasium[box2d]
```

11. Install pytorch

```bash
pip install torch
```

12. Install matplotlib

```bash
pip install matplotlib
```

13. Install Pandas

```
conda install anaconda::pandas
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->

## Usage

You can run the code using the following command.

```bash
python3 -m main
```

or by running the main.py file in your IDE.

## License

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Sources
- https://www.youtube.com/watch?v=EUrWGTCGzlA
- https://gymnasium.farama.org/environments/box2d/lunar_lander/
- https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html?highlight=parameter+grad+data
- https://medium.com/data-science/double-deep-q-networks-905dd8325412
- https://www.youtube.com/watch?v=FKOQTdcKkN4
- https://intuitivetutorial.com/2020/11/15/discount-factor/#:~:text=The%20discounted%20sum%20of%20rewards,%2B%20%E2%80%A6%20is%20a%20geometric%20series

