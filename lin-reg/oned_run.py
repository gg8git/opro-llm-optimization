from prompts import oned_meta_prompt
from data_gen import oned_data_gen
from keys import OPENAI_API_KEY
import pandas as pd
from langchain import PromptTemplate
import openai

data = oned_data_gen(10,20,50)
w = data['weights']
b = data['intercept']
coords = data['coords']

performance_df = pd.DataFrame({'input':[], 'value':[]})

def init_instructions(performance_df,init_value=15,n_training_samples=5):
    init_pair = f'[{init_value},{init_value}]'
    training_examples = coords.sample(n_training_samples)
    score_pairs([init_pair],training_examples,performance_df)

def rank_instructions(performance_df,num_scores):
    performance_df = performance_df.sort_values(by='value')
    if len(performance_df) > num_scores:
        performance_df = performance_df.tail(num_scores)
    return performance_df

def build_input(pair):
    return f'w={pair[0]}, b={pair[1]}'

def build_pairs_and_scores(performance_df):
    return ''.join([f"input:\n{build_input(performance_df.iloc[i]['input'])}\nvalue:\n{performance_df.iloc[i]['value']}\n" for i in range(len(performance_df))])

def generate_prompts(prompt_template,pairs_and_scores,n_prompts=8,temperature=1):
    prompt = PromptTemplate.from_template(template=prompt_template).format(pairs_and_scores=pairs_and_scores)
    return [((openai.Completion.create(model='text-davinci-003',prompt=prompt,max_tokens=20,temperature=temperature,api_key=OPENAI_API_KEY))['choices'][0]['text']) for _ in range(n_prompts)]

def pair_vals(npair):
    pair = npair.replace("[","").replace("]","").replace(" ","").split(',')
    w = int(pair[0])
    b = int(pair[1])
    if isinstance(w, int) and isinstance(b, int):
        return (w,b)
    else:
        return True

def error(y,y_pred):
    return (y-y_pred)**2

def score_pairs(npairs,training_examples,performance_df):
    for npair in npairs:
        pair_vals = pair_vals(npair)
        if pair_vals:
            break
        values = []
        for _, example in training_examples.iterrows():
            x = example['xs']
            y = example['ys']
            y_pred = x*pair_vals[0] + pair_vals[1]
            values.append(error(y,y_pred))
        value = int(100*sum(values)/len(values))
        performance_df = performance_df._append({'input':pair_vals,'value':value},ignore_index=True)
    return performance_df

def opro(prompt_template,performance_df,n_scores=20,n_prompts=8,n_training_samples=5,max_iterations=12):
    performance_df = rank_instructions(performance_df,n_scores)
    for _ in range(max_iterations):
        pairs_and_scores = build_pairs_and_scores(performance_df)
        npairs = generate_prompts(prompt_template,pairs_and_scores,n_prompts,temperature=1)
        training_examples = coords.sample(n_training_samples)
        performance_df = score_pairs(npairs,training_examples,performance_df)
        performance_df = rank_instructions(performance_df,n_scores)
    return performance_df

performance_df = init_instructions(performance_df,init_value=15,n_training_samples=5)

# iterate through this a bunch of times (implement later)
performance_df = opro(oned_data_gen,performance_df,n_scores=20,n_prompts=8,n_training_samples=5,max_iterations=12)