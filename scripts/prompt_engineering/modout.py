import re
# The output of your python script
output_string = ''' '''
pattern = re.compile(r"Action Input: (.*)")

chain_marker = "> Entering new AgentExecutor chain..."

matched_lines = []
current_chain_lines = []

for line in output_string.split('\n'):
    if chain_marker in line:
        if current_chain_lines:
            matched_lines.append(current_chain_lines[-1])
        current_chain_lines = []
    match = pattern.search(line)
    if match:
        current_chain_lines.append(match.group(1))

if current_chain_lines:
    matched_lines.append(current_chain_lines[-1])

if matched_lines:
    with open("output_vicuna.txt", "a") as output_file:
        for line in matched_lines:
            output_file.write(line + '\n')
else:
    print("No matching line was found.")