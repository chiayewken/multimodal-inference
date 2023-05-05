## Multimodal Inference

### Setup

```
conda create -n multimodal-inference python=3.10 -y
conda activate multimodal-inference
pip install -r requirements.txt
```

### Usage

Perform visual question answering on two images using the [Fromage](https://jykoh.com/fromage) model:

```
python main.py test_fromage

# Output:
{'prompt': 'Q: What is this image?\nA:', 'outputs': ' This is a photo of a woman in a car.'}
{'prompt': 'Q: What color is the dress?\nA:', 'outputs': " It's a black dress with a white collar and a black bow."}
{'prompt': 'Q: When do you think it was taken?\nA:', 'outputs': " I think it was taken in the late '80s or early '90s."}
{'prompt': 'Q: What color is the sofa?\nA:', 'outputs': " It's a brown sofa."}
{'prompt': 'Q: What is the difference between the two images?\nA:', 'outputs': ' The first image is a photograph of a woman in a car.'}
```