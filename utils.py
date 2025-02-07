import time
from datetime import datetime

def time_execution(script_name, func, *args, **kwargs):
    start_time = time.time()  # Record the start time
    result = func(*args, **kwargs)  # Execute the function
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate the elapsed time

    # Get the current time and date in a human-readable format
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Write the elapsed time and current time to a file
    with open("execution_time.txt", "a") as f:
        f.write(f"{current_time} - {script_name}: {elapsed_time} seconds\n")

    return result