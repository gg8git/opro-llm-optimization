from prompts import multid_meta_prompt
from data_gen import multid_data_gen
from keys import OPENAI_API_KEY
import pandas as pd
import numpy as np
from langchain.prompts import PromptTemplate
import openai

dim = 5
data = multid_data_gen(dim,10,20,50)
ws = data['weights']
b = data['intercept']
coords = data['coords']

performance_df = pd.DataFrame({'input':[], 'error':[]})

def init_pairs(performance_df,dim,init_value=15,n_training_samples=12):
    init_pairs = [f'[{[init_value for _ in range(dim)]},{init_value}]',f'[{[init_value+5 for _ in range(dim)]},{init_value+5}]',f'[{[init_value-5 for _ in range(dim)]},{init_value-5}]',f'[{[init_value+5 for _ in range(dim)]},{init_value-5}]',f'[{[init_value-5 for _ in range(dim)]},{init_value+5}]']
    return score_pairs(init_pairs,dim,coords,n_training_samples,performance_df)

def rank_pairs(performance_df,num_scores):
    performance_df = performance_df.sort_values(by='error')
    performance_df = performance_df.drop_duplicates(subset=['input'])
    performance_df = performance_df[::-1]
    if len(performance_df) > num_scores:
        performance_df = performance_df.tail(num_scores)
    return performance_df

def build_input(pair):
    return f'w={pair[0]}, b={pair[1]}'

def build_pairs_and_scores(performance_df):
    return ''.join([f"input:\n{build_input(performance_df.iloc[i]['input'])}\nerror:\n{performance_df.iloc[i]['error']}\n\n" for i in range(len(performance_df))])

def generate_prompts(prompt_template,dim,pairs_and_scores,n_prompts=12,temperature=0):
    prompt = PromptTemplate.from_template(template=prompt_template).format(dim=dim,tdim=(dim+1),pairs_and_scores=pairs_and_scores)
    return [((openai.Completion.create(model='text-davinci-003',prompt=prompt,max_tokens=20,temperature=temperature,api_key=OPENAI_API_KEY))['choices'][0]['text']) for _ in range(n_prompts)]

def pair_vals(npair,dim):
    pair = npair.translate(str.maketrans({'[':None, ']':None, '(':None, ')':None, ' ':None, 'w':None, 'b':None, '=':None, '\n':None})).split(',')
    try:
        ws = [int(float(pair[i])) for i in range(dim)]
        b = int(float(pair[dim]))
        if isinstance(ws, list) and all(isinstance(w, int) for w in ws) and isinstance(b, int):
            return (ws,b)
        else:
            return True
    except ValueError or IndexError or TypeError or AttributeError:
        return True

def msq_error(y,y_pred):
    return (y-y_pred)**2

def score_pairs(npairs,dim,training_data,n_training_samples,performance_df):
    for npair in npairs:
        p_vals = pair_vals(npair,dim)
        if (p_vals==True):
            break
        training_examples = training_data.sample(n_training_samples)
        errors = []
        for _, example in training_examples.iterrows():
            xs = example['xs']
            y = example['ys']
            y_pred = np.dot(xs,p_vals[0]) + p_vals[1]
            errors.append(msq_error(y,y_pred))
        error = int(100*sum(errors)/len(errors))
        performance_df = performance_df._append({'input':p_vals,'error':error},ignore_index=True)
    return performance_df

def opro(performance_df,dim,prompt_template,n_scores=32,n_prompts=12,n_training_samples=12,max_iterations=15):
    performance_df = rank_pairs(performance_df,n_scores)
    for _ in range(max_iterations):
        pairs_and_scores = build_pairs_and_scores(performance_df)
        npairs = generate_prompts(prompt_template,dim,pairs_and_scores,n_prompts,temperature=0)
        print(f'opro loop generated: {npairs}')
        performance_df = score_pairs(npairs,dim,coords,n_training_samples,performance_df)
        performance_df = rank_pairs(performance_df,n_scores)
    return performance_df

performance_df = init_pairs(performance_df,dim,init_value=15,n_training_samples=12)

performance_df = opro(performance_df,dim,multid_meta_prompt,n_scores=12,n_prompts=3,n_training_samples=12,max_iterations=20)

results = f'\n{performance_df}\n\npredicted values: (w,b) = {performance_df.iloc[11]["input"]}\nactual values: w = {ws.tolist()}, b = {b}'
print(results)