# -*- coding: utf-8 -*-
"""
SER501 Assignment 4 scaffolding code
created by: Xiangyu Guo
updated by: James Smith, Fall 2022
author: Sai Swaroop Reddy V
"""

# ======================== Longest Ordered Subsequence ========================


def longest_ordered_subsequence(L):
    if not L:
        return 0

    dp = [1] * len(L)

    for i in range(len(L)):
        for j in range(i):
            if L[i] > L[j]:
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)

# ============================== Counting Pond ================================


def count_ponds(grid):
    G = [list(row) for row in grid]

    def dfs(x, y):
        if x < 0 or x >= len(G) or y < 0 or y >= len(G[0]) or G[x][y] != '#':
            return
        G[x][y] = '.'
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            dfs(x + dx, y + dy)

    count = 0
    for i in range(len(G)):
        for j in range(len(G[i])):
            if G[i][j] == '#':
                dfs(i, j)
                count += 1

    return count

# =============================== Supermarket =================================


def supermarket(Items):
    sorted_items = sorted(Items, key=lambda x: (x[1], -x[0]))

    max_deadline = max(deadline for _, deadline in sorted_items)
    time_slots = [0] * (max_deadline + 1)

    total_profit = 0
    for profit, deadline in sorted_items:
        for t in range(deadline, 0, -1):
            if time_slots[t] == 0:
                time_slots[t] = profit
                total_profit += profit
                break

    return total_profit

# =============================== Unit tests ==================================


def test_suite():

    if longest_ordered_subsequence([1, 7, 3, 5, 9, 4, 8]) == 4:
        print('passed')
    else:
        print('failed')

    if count_ponds(["#--------##-",
                    "-###-----###",
                    "----##---##-",
                    "---------##-",
                    "---------#--",
                    "--#------#--",
                    "-#-#-----##-",
                    "#-#-#-----#-",
                    "-#-#------#-",
                    "--#-------#-"]) == 3:
        print('passed')
    else:
        print('failed')

    if supermarket([(50, 2), (10, 1), (20, 2), (30, 1)]) == 80:
        print('passed')
    else:
        print('failed')

    # More Test cases

    # if longest_ordered_subsequence([10, 22, 9, 33, 21, 50, 41, 60, 80]) == 6:
    #     print('Test case 1 for LOS passed')
    # else:
    #     print('Test case 1 for LOS failed')

    # if longest_ordered_subsequence([3, 10, 2, 1, 20]) == 2:
    #     print('Test case 2 for LOS passed')
    # else:
    #     print('Test case 2 for LOS failed')

    # if longest_ordered_subsequence([]) == 0:
    #     print('Test case 3 for LOS passed')
    # else:
    #     print('Test case 3 for LOS failed')

    # if count_ponds(["##-##",
    #                 "#----",
    #                 "--#--",
    #                 "-##--",
    #                 "##-##"]) == 3:
    #     print('Test case 1 for counting ponds passed')
    # else:
    #     print('Test case 1 for counting ponds failed')

    # if count_ponds(["-#",
    #                 "#-"]) == 2:
    #     print('Test case 2 for counting ponds passed')
    # else:
    #     print('Test case 2 for counting ponds failed')

    # if count_ponds(["---",
    #                 "---",
    #                 "---"]) == 0:
    #     print('Test case 3 for counting ponds passed')
    # else:
    #     print('Test case 3 for counting ponds failed')

    # if count_ponds(["###",
    #                 "###",
    #                 "###"]) == 1:
    #     print('Test case 4 for counting ponds passed')
    # else:
    #     print('Test case 4 for counting ponds failed')

    # if supermarket([(100, 2), (10, 1), (15, 2), (20, 1), (1, 3)]) == 130:
    #     print('Test case 1 for max profit passed')
    # else:
    #     print('Test case 1 for max profit failed')

    # if supermarket([(5, 1), (7, 1), (8, 1)]) == 8:
    #     print('Test case 2 for max profit passed')
    # else:
    #     print('Test case 2 for max profit failed')

    # if supermarket([(1, 5), (10, 3), (1, 3)]) == 11:
    #     print('Test case 3 for max profit passed')
    # else:
    #     print('Test case 3 for max profit failed')


if __name__ == '__main__':
    test_suite()
