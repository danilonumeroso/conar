from statistics import mean, stdev as std


def optimal_baseline(data):
    tour_lengths = [d.optimal_value.item() for d in data]
    return mean(tour_lengths), std(tour_lengths)
