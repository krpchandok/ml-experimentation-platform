import time

for step in range(1, 61):
    print(f"Training step {step}/60", flush=True)
    time.sleep(1)

print("Job finished!", flush=True)
