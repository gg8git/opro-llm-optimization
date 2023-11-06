oned_meta_prompt = '''
You will help me minimize a function with two input variables w, b. I have some (w, b) pairs
and the function errors at those points. The pairs are arranged in ascending order based on their
function errors, where lower errors are better, and the best error is 0.

pairs:
{pairs_and_scores}
Give me a new (w, b) pair, that is different from all pairs above, and has a function error 
lower than any of the above. Do not write code or a sentence. The output must end with a pair 
[w, b], where w and b can be any numerical values.
'''

multid_meta_prompt = '''
You will help me minimize a function with {tdim} input variables. These input variables will be
represented as {dim}-vector and scalar pairs (w, b). I have some (w, b) {dim}-vector and scalar pairs 
and the function errors at those points. The pairs are arranged in ascending order based on their 
function values, where lower values are better, and the best value is 0.

pairs:
{vectors_and_scores}
Give me a new {dim}-vector and scalar pair (w, b), that is different from all pairs above, and has a 
function error lower than any of the above. Do not write code or a sentence. The output must end with 
a {dim}-vector and scalar pair [w, b], where w can be any numerical valued {dim}-vector contained 
within [] brackets, and b can be any numerical value.
'''