def compare_files(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        queries1 = f1.readlines()
        queries2 = f2.readlines()

    matches = sum(q1.lower().replace(' ', '') == q2.lower().replace(' ', '') for q1, q2 in zip(queries1, queries2))
    total = len(queries1) 

    return matches / total * 100  
percentage = compare_files('output_openai_spider.txt', 'sql.txt')
print(f"The percentage of exact matching queries is: {percentage}%")
