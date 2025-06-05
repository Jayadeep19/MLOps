from prefect import task, flow

@task(log_prints=True)
def display(message:str):
    print(f"The message from the main is: {message}")
    return None

@flow
def main(message:str):
    display(message= message)
    return None

if __name__=="__main__":
    main("This is an example run")