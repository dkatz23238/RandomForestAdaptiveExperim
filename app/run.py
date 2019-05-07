from axtrainer import main


if __name__ == "__main__":
    try:
        b,m = main()
        print(f"Best Parameter: {b}")
        print(f"Metrics: {m}")
    except Exception as e:
        raise e