import random
import time

from hydraflow.progress import progress


def task(total):
    def func():
        for i in range(total):
            yield i, total
            time.sleep(random.random())

    return func()


def main():
    tasks = [task(random.randint(10, 20)) for _ in range(12)]
    progress(*tasks, n_jobs=4)


if __name__ == "__main__":
    main()
