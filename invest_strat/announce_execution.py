def announce_execution(func):
    def wrapper(*args, **kwargs):
        print(f"Executing {func.__name__}...")
        result = func(*args, **kwargs)
        #print(f"{func.__name__} returned {result}")
        return result
    return wrapper