"""Score two synthetic three-round trajectories without inference."""

from sauce import sauce, sauce_score


def main() -> None:
    entropy = [0.4, 0.2, 0.1]
    strong_agreement = [2 / 3, 1.0, 1.0]
    weak_agreement = [1 / 3, 1 / 3, 2 / 3]

    result = sauce(strong_agreement, entropy, protocol="debate")
    weak_score = sauce_score(weak_agreement, entropy, protocol="debate")
    print(f"Strong agreement: {result['score']:.6f}")
    print(f"Weak agreement:   {weak_score:.6f}")
    print("Higher score means greater uncertainty.")
    print("Posterior means:", [round(r["mu"], 6) for r in result["trace"]["rounds"]])


if __name__ == "__main__":
    main()
