oned_meta_prompt = '''
You will help me minimize a function with two input variables w, b. I have some (w, b) pairs
and the function values at those points. The pairs are arranged in descending order based on their
function values, where lower values are better.

pairs:
{pairs_and_scores}

Give me a new (w, b) pair that is different from all pairs above, and has a function value lower than
any of the above. Do not write code. The output must end with a pair [w, b], where w and b are
numerical values.
'''

multid_meta_prompt = '''
You will help me minimize a function with {tdim} input variables. I have some {tdim}-vectors
and the function values at those points. The vectors are arranged in descending order based on their
function values, where lower values are better.

pairs:
{vectors_and_scores}

Give me a new {tdim}-vector that is different from all vectors above, and has a function value lower than
any of the above. Do not write code. The output must end with a {tdim}-vector contained within [] brackets, 
where all entries of the vector are numerical values.
'''