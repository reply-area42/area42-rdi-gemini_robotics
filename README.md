# Google Gemini Robotics Setup Guide

This guide walks you through setting up an environment for working with **Google Gemini Robotics**, including creating a Conda environment with Python 3.12 and installing the required dependencies.

---

## 📌 Prerequisites

- Conda installed (Anaconda or Miniconda)
- Git installed
- Access to the repository

---

## Step 1: Create a Conda Environment
```bash
conda create -n env_google python=3.12 -y
```
---

## Step 2: Activate the Environment
```bash
conda activate env_google
```
---

## Step 3: Clone the Repository
```bash
git clone https://github.com/reply-area42/area42-rdi-gemini_robotics.git
cd area area42-rdi-gemini_robotics
```
---

## Step 4: Install Dependencies
```bash
pip install -U -q google-genai
```
---

## Step 5: Environment Variables

To run the following cells, your API key must be stored in a Colab Secret named GEMINI_API_KEY. If you don't already have an API key, or you're not sure how to create a Colab Secret, see [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) for an example.
```bash
export GEMINI_API_KEY="your_api_key_here"
```
---

## Step 6: Initialize SDK client


```bash
from google import genai
from google.genai import types

client = genai.Client(api_key=GEMINI_API_KEY)
```   
## Example

You can look at the example given from [Google](https://github.com/google-gemini/robotics-samples/blob/main/Getting%20Started/gemini_robotics_er.ipynb)

---

## Optional Tools

pip install black flake8 pytest

---

## Troubleshooting

- Check Python version: python --version  
- Upgrade pip if needed: pip install --upgrade pip  

---

## 📄 License

See repository for details.
