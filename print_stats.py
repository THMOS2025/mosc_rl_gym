import pstats
p= pstats.Stats("sts.prof")
p.sort_stats("cumulative").print_stats(100)