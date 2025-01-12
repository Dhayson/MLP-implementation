from src.problems import classification_problem, regression_problem
import sys

def main():
    if sys.argv[1] == "Classification":
        classification_problem()
    elif sys.argv[1] == "Regression":
        regression_problem()

if __name__ == "__main__":
    main()