from src.preprocessing import load_california_data


def main():
    df = load_california_data()
    print("First 5 rows:")
    print(df.head())
    print("\nShape:", df.shape)


if __name__ == "__main__":
    main()
