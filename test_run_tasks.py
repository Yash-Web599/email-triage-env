from tasks.task_easy import run_easy_task
from tasks.task_medium import run_medium_task
from tasks.task_hard import run_hard_task

print("Running task functions...")

easy_score = run_easy_task()
print(f"Easy score: {easy_score}")

medium_score = run_medium_task()
print(f"Medium score: {medium_score}")

hard_score = run_hard_task()
print(f"Hard score: {hard_score}")

print("All scores should be strictly between 0 and 1.")