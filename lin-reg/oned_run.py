from prompts import oned_meta_prompt
from data_gen import oned_data_gen
from keys import OPENAI_API_KEY
import pandas as pd
from langchain.prompts import PromptTemplate
import openai

data = oned_data_gen(10,20,50)
w = data['weights']
b = data['intercept']
coords = data['coords']

performance_df = pd.DataFrame({'input':[], 'error':[]})

def init_instructions(performance_df,init_value=15,n_training_samples=12):
    init_pairs = [f'[{init_value},{init_value}]',f'[{init_value+5},{init_value+5}]',f'[{init_value-5},{init_value-5}]',f'[{init_value+5},{init_value-5}]',f'[{init_value-5},{init_value+5}]']
    return score_pairs(init_pairs,coords,n_training_samples,performance_df)

def rank_instructions(performance_df,num_scores):
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

def generate_prompts(prompt_template,pairs_and_scores,n_prompts=12,temperature=0):
    prompt = PromptTemplate.from_template(template=prompt_template).format(pairs_and_scores=pairs_and_scores)
    return [((openai.Completion.create(model='text-davinci-003',prompt=prompt,max_tokens=20,temperature=temperature,api_key=OPENAI_API_KEY))['choices'][0]['text']) for _ in range(n_prompts)]

def pair_vals(npair):
    pair = npair.replace("[","").replace("]","").replace(" ","").replace("\n","").split(',')
    try:
        w = int(float(pair[0]))
        b = int(float(pair[1]))
        if isinstance(w, int) and isinstance(b, int):
            return (w,b)
        else:
            return True
    except ValueError:
        return True

def msq_error(y,y_pred):
    return (y-y_pred)**2

def score_pairs(npairs,training_data,n_training_samples,performance_df):
    for npair in npairs:
        p_vals = pair_vals(npair)
        if (p_vals==True):
            break
        training_examples = training_data.sample(n_training_samples)
        errors = []
        for _, example in training_examples.iterrows():
            x = example['xs']
            y = example['ys']
            y_pred = x*p_vals[0] + p_vals[1]
            errors.append(msq_error(y,y_pred))
        error = int(100*sum(errors)/len(errors))
        performance_df = performance_df._append({'input':p_vals,'error':error},ignore_index=True)
    return performance_df

def opro(prompt_template,performance_df,n_scores=32,n_prompts=12,n_training_samples=12,max_iterations=15):
    performance_df = rank_instructions(performance_df,n_scores)
    for _ in range(max_iterations):
        pairs_and_scores = build_pairs_and_scores(performance_df)
        npairs = generate_prompts(prompt_template,pairs_and_scores,n_prompts,temperature=0)
        print(f'opro loop generated: {npairs}')
        performance_df = score_pairs(npairs,coords,n_training_samples,performance_df)
        performance_df = rank_instructions(performance_df,n_scores)
    return performance_df

performance_df = init_instructions(performance_df,init_value=15,n_training_samples=12)

# iterate through this a bunch of times (implement later)
performance_df = opro(oned_meta_prompt,performance_df,n_scores=12,n_prompts=3,n_training_samples=12,max_iterations=20)

results = f'\n{performance_df}\n\npredicted values: w = {performance_df.iloc[11]}, b = {performance_df.iloc[11]}\nactual values: w = {w}, b = {b}'
print(results)
