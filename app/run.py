from axtrainer.xgboost import main


if __name__ == "__main__":

    ax, b , m = main()

    
    print(f"Best Parameter: {b}")
    print(f"Metrics: {m}")
    # print(ax.get_report())