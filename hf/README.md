
## Inference demo

This subrepo hosts cleaned code that are necessary for inference.

-----

First install dependencies.
```
conda create --name marginalsemlink python=3.9
conda activate marginalsemlink
pip install -r requirements.txt
```

### Run the inference model

Then, run sequence classification:

```
tokenizer = AutoTokenizer.from_pretrained('tli8hf/roberta-base-marginal-semlink')
model = RobertaForSRL.from_pretrained('tli8hf/roberta-base-marginal-semlink')
model = model.to('cuda:0')
config = model.config

orig_toks = 'The company said it will continue to pursue a lifting of the suspension .'.split()
v_idx = [2, 7]
vnclasses = ['B-37.7', 'B-35.6']
senses = ['say.01', 'pursue.01']
batch = preprocess_input(config, tokenizer, orig_toks, v_idx, vnclasses, senses)
batch = batch_to_device(batch, 'cuda:0')
outputs = model.classify(batch)
print(outputs)

```

The output should look like the below. For each predicate (vnclass, sense) pair, there will be a corresponding sequence of VN tags and SRL tags.
```
{
	'vn_output': [[
		'(AGENT* *) (V*) (TOPIC* * * * * * * * * *) *',
		'* * * (AGENT*) * * * (V*) (THEME* * * * *) *'
		]],
	'srl_output': [[
		'(ARG0* *) (V*) (ARG1* * * * * * * * * *) *',
		'* * * (ARG0*) * * * (V*) (ARG1* * * * *) *'
		]]
}
```
