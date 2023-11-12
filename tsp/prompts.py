tsp_meta_prompt = '''
You are given a list of {dim} points with coordinates below: {coordinates}.
Below are some previous traces and their lengths. The traces are arranged in ascending order based
on their lengths, where shorter trace lengths are better.

{traces_and_lengths}

Give me a new trace that is different from all traces above, and has a length shorter than any of the
above. The trace should traverse all points exactly once. The trace should start with <trace> and end
with </trace>.
'''