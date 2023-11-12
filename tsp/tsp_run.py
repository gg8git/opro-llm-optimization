from prompts import tsp_meta_prompt
from data_gen import tsp_data_gen, tsp_ans
from keys import OPENAI_API_KEY
import pandas as pd
import numpy as np
from langchain.prompts import PromptTemplate
import openai
import math

dim = 20
(coords,coordinates) = tsp_data_gen(dim,50)
(length_ans,trace_ans) = tsp_ans(coords)
performance_df = pd.DataFrame({'trace':[], 'length':[]})

def init_traces(performance_df,dim):
    init_trace = f'<trace> {str(", ".join([str(i) for i in range(dim)]))} </trace>'
    return score_traces([init_trace],dim,coords,performance_df)

def rank_traces(performance_df,num_scores):
    performance_df = performance_df.sort_values(by='length')
    performance_df = performance_df.drop_duplicates(subset=['trace'])
    performance_df = performance_df[::-1]
    if len(performance_df) > num_scores:
        performance_df = performance_df.tail(num_scores)
    return performance_df

def build_input(trace):
    return f'{str(trace)}'.replace("[","").replace("]","")

def build_traces_and_lengths(performance_df):
    return ''.join([f"<trace> {build_input(performance_df.iloc[i]['trace'])} </trace>\nlength:\n{performance_df.iloc[i]['length']}\n\n" for i in range(len(performance_df))])

def generate_prompts(prompt_template,dim,coordinates,traces_and_lengths,n_prompts=12,temperature=0):
    prompt = PromptTemplate.from_template(template=prompt_template).format(dim=dim,coordinates=coordinates,traces_and_lengths=traces_and_lengths)
    return [((openai.Completion.create(model='text-davinci-003',prompt=prompt,max_tokens=20,temperature=temperature,api_key=OPENAI_API_KEY))['choices'][0]['text']) for _ in range(n_prompts)]

def trace_list(ntrace,dim):
    trace = (ntrace.replace("<trace>","").replace("</trace>","")).translate(str.maketrans({'<':None, '>':None, '[':None, ']':None, '(':None, ')':None, ' ':None, 'w':None, 'b':None, '=':None, '\n':None})).split(',')
    try:
        trace = [int(float(trace[i])) for i in range(dim)]
        if isinstance(trace, list) and all(isinstance(c, int) for c in trace):
            return trace
        else:
            return True
    except ValueError or IndexError or TypeError or AttributeError:
        return True

def coords_dist(pt1, pt2):
    return math.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)

def score_traces(ntraces,dim,coords,performance_df):
    for ntrace in ntraces:
        trace = trace_list(ntrace,dim)
        if (trace==True):
            break
        length = 0
        for i in range(len(trace)-1):
            pt1 = (coords.iloc[trace[i]]['xs'],coords.iloc[trace[i]]['ys'])
            pt2 = (coords.iloc[trace[i+1]]['xs'],coords.iloc[trace[i+1]]['ys'])
            length += coords_dist(pt1,pt2)
        performance_df = performance_df._append({'trace':trace,'length':length},ignore_index=True)
    return performance_df

def opro(performance_df,dim,prompt_template,coordinates,n_scores=32,n_prompts=12,max_iterations=15):
    performance_df = rank_traces(performance_df,n_scores)
    for _ in range(max_iterations):
        traces_and_lengths = build_traces_and_lengths(performance_df)
        ntraces = generate_prompts(prompt_template,dim,coordinates,traces_and_lengths,n_prompts,temperature=0)
        print(f'opro loop generated: {ntraces}')
        performance_df = score_traces(ntraces,dim,coords,performance_df)
        performance_df = rank_traces(performance_df,n_scores)
    return performance_df

performance_df = init_traces(performance_df,dim)

performance_df = opro(performance_df,dim,tsp_meta_prompt,coordinates,n_scores=12,n_prompts=3,max_iterations=20)

# edit
# results = f'\n{performance_df}\n\npredicted values: (w,b) = {performance_df.iloc[11]["input"]}\nactual values: w = {ws.tolist()}, b = {b}'
# print(results)