import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates141 : List Nat := [857, 1019, 1049, 1301, 1721, 1811, 1847, 1871, 2267, 2273, 2351, 2711, 2819, 2909, 3299, 3527, 3593, 3719, 4463, 5303, 5417, 5471, 5717, 6143]
set_option maxRecDepth 4096 in
theorem rootCandidates141_checked : candidates edge 5 0 827 tail142 best.bound = rootCandidates141 := by
  unfold candidates
  have row : edge 827 = PrimePairCache.smallRow827.contains := by funext q; rfl
  rw [row]
  decide

def innerTail141_24 : List Nat := []
theorem innerScan141_24 : scan edge (search edge 3) 4 [827] 827 innerTail141_24 best = best := rfl

def innerTail141_23 : List Nat := 6143 :: innerTail141_24
def innerCandidates141_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates141_23_checked : candidates edge 4 827 6143 innerTail141_24 best.bound = innerCandidates141_23 := by
  unfold candidates
  have row : edge 6143 = SearchData.row6143.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan141_23 : scan edge (search edge 3) 4 [827] 827 innerTail141_23 best = best := by
  rw [innerTail141_23, scan, if_neg (by decide), innerCandidates141_23_checked]
  change scan edge (search edge 3) 4 [827] 827 innerTail141_24
    (if 3 ≤ innerCandidates141_23.length then search edge 3 [827, 6143] innerCandidates141_23 6970 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan141_24

def innerTail141_22 : List Nat := 5717 :: innerTail141_23
def innerCandidates141_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates141_22_checked : candidates edge 4 827 5717 innerTail141_23 best.bound = innerCandidates141_22 := by
  unfold candidates
  have row : edge 5717 = SearchData.row5717.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan141_22 : scan edge (search edge 3) 4 [827] 827 innerTail141_22 best = best := by
  rw [innerTail141_22, scan, if_neg (by decide), innerCandidates141_22_checked]
  change scan edge (search edge 3) 4 [827] 827 innerTail141_23
    (if 3 ≤ innerCandidates141_22.length then search edge 3 [827, 5717] innerCandidates141_22 6544 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan141_23

def innerTail141_21 : List Nat := 5471 :: innerTail141_22
def innerCandidates141_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates141_21_checked : candidates edge 4 827 5471 innerTail141_22 best.bound = innerCandidates141_21 := by
  unfold candidates
  have row : edge 5471 = SearchData.row5471.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan141_21 : scan edge (search edge 3) 4 [827] 827 innerTail141_21 best = best := by
  rw [innerTail141_21, scan, if_neg (by decide), innerCandidates141_21_checked]
  change scan edge (search edge 3) 4 [827] 827 innerTail141_22
    (if 3 ≤ innerCandidates141_21.length then search edge 3 [827, 5471] innerCandidates141_21 6298 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan141_22

def innerTail141_20 : List Nat := 5417 :: innerTail141_21
def innerCandidates141_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates141_20_checked : candidates edge 4 827 5417 innerTail141_21 best.bound = innerCandidates141_20 := by
  unfold candidates
  have row : edge 5417 = SearchData.row5417.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan141_20 : scan edge (search edge 3) 4 [827] 827 innerTail141_20 best = best := by
  rw [innerTail141_20, scan, if_neg (by decide), innerCandidates141_20_checked]
  change scan edge (search edge 3) 4 [827] 827 innerTail141_21
    (if 3 ≤ innerCandidates141_20.length then search edge 3 [827, 5417] innerCandidates141_20 6244 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan141_21

def innerTail141_19 : List Nat := 5303 :: innerTail141_20
def innerCandidates141_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates141_19_checked : candidates edge 4 827 5303 innerTail141_20 best.bound = innerCandidates141_19 := by
  unfold candidates
  have row : edge 5303 = SearchData.row5303.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan141_19 : scan edge (search edge 3) 4 [827] 827 innerTail141_19 best = best := by
  rw [innerTail141_19, scan, if_neg (by decide), innerCandidates141_19_checked]
  change scan edge (search edge 3) 4 [827] 827 innerTail141_20
    (if 3 ≤ innerCandidates141_19.length then search edge 3 [827, 5303] innerCandidates141_19 6130 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan141_20

def innerTail141_18 : List Nat := 4463 :: innerTail141_19
def innerCandidates141_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates141_18_checked : candidates edge 4 827 4463 innerTail141_19 best.bound = innerCandidates141_18 := by
  unfold candidates
  have row : edge 4463 = SearchData.row4463.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan141_18 : scan edge (search edge 3) 4 [827] 827 innerTail141_18 best = best := by
  rw [innerTail141_18, scan, if_neg (by decide), innerCandidates141_18_checked]
  change scan edge (search edge 3) 4 [827] 827 innerTail141_19
    (if 3 ≤ innerCandidates141_18.length then search edge 3 [827, 4463] innerCandidates141_18 5290 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan141_19

def innerTail141_17 : List Nat := 3719 :: innerTail141_18
def innerCandidates141_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates141_17_checked : candidates edge 4 827 3719 innerTail141_18 best.bound = innerCandidates141_17 := by
  unfold candidates
  have row : edge 3719 = SearchData.row3719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan141_17 : scan edge (search edge 3) 4 [827] 827 innerTail141_17 best = best := by
  rw [innerTail141_17, scan, if_neg (by decide), innerCandidates141_17_checked]
  change scan edge (search edge 3) 4 [827] 827 innerTail141_18
    (if 3 ≤ innerCandidates141_17.length then search edge 3 [827, 3719] innerCandidates141_17 4546 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan141_18

def innerTail141_16 : List Nat := 3593 :: innerTail141_17
def innerCandidates141_16 : List Nat := [3719]
set_option maxRecDepth 4096 in
theorem innerCandidates141_16_checked : candidates edge 4 827 3593 innerTail141_17 best.bound = innerCandidates141_16 := by
  unfold candidates
  have row : edge 3593 = SearchData.row3593.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan141_16 : scan edge (search edge 3) 4 [827] 827 innerTail141_16 best = best := by
  rw [innerTail141_16, scan, if_neg (by decide), innerCandidates141_16_checked]
  change scan edge (search edge 3) 4 [827] 827 innerTail141_17
    (if 3 ≤ innerCandidates141_16.length then search edge 3 [827, 3593] innerCandidates141_16 4420 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan141_17

def innerTail141_15 : List Nat := 3527 :: innerTail141_16
def innerCandidates141_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates141_15_checked : candidates edge 4 827 3527 innerTail141_16 best.bound = innerCandidates141_15 := by
  unfold candidates
  have row : edge 3527 = SearchData.row3527.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan141_15 : scan edge (search edge 3) 4 [827] 827 innerTail141_15 best = best := by
  rw [innerTail141_15, scan, if_neg (by decide), innerCandidates141_15_checked]
  change scan edge (search edge 3) 4 [827] 827 innerTail141_16
    (if 3 ≤ innerCandidates141_15.length then search edge 3 [827, 3527] innerCandidates141_15 4354 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan141_16

def innerTail141_14 : List Nat := 3299 :: innerTail141_15
def innerCandidates141_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates141_14_checked : candidates edge 4 827 3299 innerTail141_15 best.bound = innerCandidates141_14 := by
  unfold candidates
  have row : edge 3299 = SearchData.row3299.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan141_14 : scan edge (search edge 3) 4 [827] 827 innerTail141_14 best = best := by
  rw [innerTail141_14, scan, if_neg (by decide), innerCandidates141_14_checked]
  change scan edge (search edge 3) 4 [827] 827 innerTail141_15
    (if 3 ≤ innerCandidates141_14.length then search edge 3 [827, 3299] innerCandidates141_14 4126 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan141_15

def innerTail141_13 : List Nat := 2909 :: innerTail141_14
def innerCandidates141_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates141_13_checked : candidates edge 4 827 2909 innerTail141_14 best.bound = innerCandidates141_13 := by
  unfold candidates
  have row : edge 2909 = SearchData.row2909.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan141_13 : scan edge (search edge 3) 4 [827] 827 innerTail141_13 best = best := by
  rw [innerTail141_13, scan, if_neg (by decide), innerCandidates141_13_checked]
  change scan edge (search edge 3) 4 [827] 827 innerTail141_14
    (if 3 ≤ innerCandidates141_13.length then search edge 3 [827, 2909] innerCandidates141_13 3736 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan141_14

def innerTail141_12 : List Nat := 2819 :: innerTail141_13
def innerCandidates141_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates141_12_checked : candidates edge 4 827 2819 innerTail141_13 best.bound = innerCandidates141_12 := by
  unfold candidates
  have row : edge 2819 = SearchData.row2819.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan141_12 : scan edge (search edge 3) 4 [827] 827 innerTail141_12 best = best := by
  rw [innerTail141_12, scan, if_neg (by decide), innerCandidates141_12_checked]
  change scan edge (search edge 3) 4 [827] 827 innerTail141_13
    (if 3 ≤ innerCandidates141_12.length then search edge 3 [827, 2819] innerCandidates141_12 3646 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan141_13

def innerTail141_11 : List Nat := 2711 :: innerTail141_12
def innerCandidates141_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates141_11_checked : candidates edge 4 827 2711 innerTail141_12 best.bound = innerCandidates141_11 := by
  unfold candidates
  have row : edge 2711 = SearchData.row2711.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan141_11 : scan edge (search edge 3) 4 [827] 827 innerTail141_11 best = best := by
  rw [innerTail141_11, scan, if_neg (by decide), innerCandidates141_11_checked]
  change scan edge (search edge 3) 4 [827] 827 innerTail141_12
    (if 3 ≤ innerCandidates141_11.length then search edge 3 [827, 2711] innerCandidates141_11 3538 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan141_12

def innerTail141_10 : List Nat := 2351 :: innerTail141_11
def innerCandidates141_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates141_10_checked : candidates edge 4 827 2351 innerTail141_11 best.bound = innerCandidates141_10 := by
  unfold candidates
  have row : edge 2351 = PrimePairCache.smallRow2351.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan141_10 : scan edge (search edge 3) 4 [827] 827 innerTail141_10 best = best := by
  rw [innerTail141_10, scan, if_neg (by decide), innerCandidates141_10_checked]
  change scan edge (search edge 3) 4 [827] 827 innerTail141_11
    (if 3 ≤ innerCandidates141_10.length then search edge 3 [827, 2351] innerCandidates141_10 3178 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan141_11

def innerTail141_9 : List Nat := 2273 :: innerTail141_10
def innerCandidates141_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates141_9_checked : candidates edge 4 827 2273 innerTail141_10 best.bound = innerCandidates141_9 := by
  unfold candidates
  have row : edge 2273 = PrimePairCache.smallRow2273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan141_9 : scan edge (search edge 3) 4 [827] 827 innerTail141_9 best = best := by
  rw [innerTail141_9, scan, if_neg (by decide), innerCandidates141_9_checked]
  change scan edge (search edge 3) 4 [827] 827 innerTail141_10
    (if 3 ≤ innerCandidates141_9.length then search edge 3 [827, 2273] innerCandidates141_9 3100 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan141_10

def innerTail141_8 : List Nat := 2267 :: innerTail141_9
def innerCandidates141_8 : List Nat := [3527]
set_option maxRecDepth 4096 in
theorem innerCandidates141_8_checked : candidates edge 4 827 2267 innerTail141_9 best.bound = innerCandidates141_8 := by
  unfold candidates
  have row : edge 2267 = PrimePairCache.smallRow2267.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan141_8 : scan edge (search edge 3) 4 [827] 827 innerTail141_8 best = best := by
  rw [innerTail141_8, scan, if_neg (by decide), innerCandidates141_8_checked]
  change scan edge (search edge 3) 4 [827] 827 innerTail141_9
    (if 3 ≤ innerCandidates141_8.length then search edge 3 [827, 2267] innerCandidates141_8 3094 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan141_9

def innerTail141_7 : List Nat := 1871 :: innerTail141_8
def innerCandidates141_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates141_7_checked : candidates edge 4 827 1871 innerTail141_8 best.bound = innerCandidates141_7 := by
  unfold candidates
  have row : edge 1871 = PrimePairCache.smallRow1871.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan141_7 : scan edge (search edge 3) 4 [827] 827 innerTail141_7 best = best := by
  rw [innerTail141_7, scan, if_neg (by decide), innerCandidates141_7_checked]
  change scan edge (search edge 3) 4 [827] 827 innerTail141_8
    (if 3 ≤ innerCandidates141_7.length then search edge 3 [827, 1871] innerCandidates141_7 2698 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan141_8

def innerTail141_6 : List Nat := 1847 :: innerTail141_7
def innerCandidates141_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates141_6_checked : candidates edge 4 827 1847 innerTail141_7 best.bound = innerCandidates141_6 := by
  unfold candidates
  have row : edge 1847 = PrimePairCache.smallRow1847.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan141_6 : scan edge (search edge 3) 4 [827] 827 innerTail141_6 best = best := by
  rw [innerTail141_6, scan, if_neg (by decide), innerCandidates141_6_checked]
  change scan edge (search edge 3) 4 [827] 827 innerTail141_7
    (if 3 ≤ innerCandidates141_6.length then search edge 3 [827, 1847] innerCandidates141_6 2674 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan141_7

def innerTail141_5 : List Nat := 1811 :: innerTail141_6
def innerCandidates141_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates141_5_checked : candidates edge 4 827 1811 innerTail141_6 best.bound = innerCandidates141_5 := by
  unfold candidates
  have row : edge 1811 = PrimePairCache.smallRow1811.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan141_5 : scan edge (search edge 3) 4 [827] 827 innerTail141_5 best = best := by
  rw [innerTail141_5, scan, if_neg (by decide), innerCandidates141_5_checked]
  change scan edge (search edge 3) 4 [827] 827 innerTail141_6
    (if 3 ≤ innerCandidates141_5.length then search edge 3 [827, 1811] innerCandidates141_5 2638 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan141_6

def innerTail141_4 : List Nat := 1721 :: innerTail141_5
def innerCandidates141_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates141_4_checked : candidates edge 4 827 1721 innerTail141_5 best.bound = innerCandidates141_4 := by
  unfold candidates
  have row : edge 1721 = PrimePairCache.smallRow1721.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan141_4 : scan edge (search edge 3) 4 [827] 827 innerTail141_4 best = best := by
  rw [innerTail141_4, scan, if_neg (by decide), innerCandidates141_4_checked]
  change scan edge (search edge 3) 4 [827] 827 innerTail141_5
    (if 3 ≤ innerCandidates141_4.length then search edge 3 [827, 1721] innerCandidates141_4 2548 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan141_5

def innerTail141_3 : List Nat := 1301 :: innerTail141_4
def innerCandidates141_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates141_3_checked : candidates edge 4 827 1301 innerTail141_4 best.bound = innerCandidates141_3 := by
  unfold candidates
  have row : edge 1301 = PrimePairCache.smallRow1301.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan141_3 : scan edge (search edge 3) 4 [827] 827 innerTail141_3 best = best := by
  rw [innerTail141_3, scan, if_neg (by decide), innerCandidates141_3_checked]
  change scan edge (search edge 3) 4 [827] 827 innerTail141_4
    (if 3 ≤ innerCandidates141_3.length then search edge 3 [827, 1301] innerCandidates141_3 2128 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan141_4

def innerTail141_2 : List Nat := 1049 :: innerTail141_3
def innerCandidates141_2 : List Nat := [1847]
set_option maxRecDepth 4096 in
theorem innerCandidates141_2_checked : candidates edge 4 827 1049 innerTail141_3 best.bound = innerCandidates141_2 := by
  unfold candidates
  have row : edge 1049 = PrimePairCache.smallRow1049.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan141_2 : scan edge (search edge 3) 4 [827] 827 innerTail141_2 best = best := by
  rw [innerTail141_2, scan, if_neg (by decide), innerCandidates141_2_checked]
  change scan edge (search edge 3) 4 [827] 827 innerTail141_3
    (if 3 ≤ innerCandidates141_2.length then search edge 3 [827, 1049] innerCandidates141_2 1876 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan141_3

def innerTail141_1 : List Nat := 1019 :: innerTail141_2
def innerCandidates141_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates141_1_checked : candidates edge 4 827 1019 innerTail141_2 best.bound = innerCandidates141_1 := by
  unfold candidates
  have row : edge 1019 = PrimePairCache.smallRow1019.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan141_1 : scan edge (search edge 3) 4 [827] 827 innerTail141_1 best = best := by
  rw [innerTail141_1, scan, if_neg (by decide), innerCandidates141_1_checked]
  change scan edge (search edge 3) 4 [827] 827 innerTail141_2
    (if 3 ≤ innerCandidates141_1.length then search edge 3 [827, 1019] innerCandidates141_1 1846 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan141_2

def innerTail141_0 : List Nat := 857 :: innerTail141_1
def innerCandidates141_0 : List Nat := [1301, 4463]
set_option maxRecDepth 4096 in
theorem innerCandidates141_0_checked : candidates edge 4 827 857 innerTail141_1 best.bound = innerCandidates141_0 := by
  unfold candidates
  have row : edge 857 = PrimePairCache.smallRow857.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan141_0 : scan edge (search edge 3) 4 [827] 827 innerTail141_0 best = best := by
  rw [innerTail141_0, scan, if_neg (by decide), innerCandidates141_0_checked]
  change scan edge (search edge 3) 4 [827] 827 innerTail141_1
    (if 3 ≤ innerCandidates141_0.length then search edge 3 [827, 857] innerCandidates141_0 1684 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan141_1

theorem rootBranch141 : search edge 4 [827] rootCandidates141 827 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [827] 827 innerTail141_0 best = best
  exact innerScan141_0

def rootCandidates140 : List Nat := [877, 967, 1303, 1459, 1777, 1933, 2017, 2113, 2179, 2281, 2311, 2347, 2557, 2953, 3541, 4243, 4789, 4909, 4951, 5209, 5281, 5323, 5827, 5881, 6337, 6733, 7237, 7699, 8101]
set_option maxRecDepth 4096 in
theorem rootCandidates140_checked : candidates edge 5 0 823 tail141 best.bound = rootCandidates140 := by
  unfold candidates
  have row : edge 823 = PrimePairCache.smallRow823.contains := by funext q; rfl
  rw [row]
  decide

def innerTail140_29 : List Nat := []
theorem innerScan140_29 : scan edge (search edge 3) 4 [823] 823 innerTail140_29 best = best := rfl

def innerTail140_28 : List Nat := 8101 :: innerTail140_29
def innerCandidates140_28 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates140_28_checked : candidates edge 4 823 8101 innerTail140_29 best.bound = innerCandidates140_28 := by
  unfold candidates
  have row : edge 8101 = SearchData.row8101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan140_28 : scan edge (search edge 3) 4 [823] 823 innerTail140_28 best = best := by
  rw [innerTail140_28, scan, if_neg (by decide), innerCandidates140_28_checked]
  change scan edge (search edge 3) 4 [823] 823 innerTail140_29
    (if 3 ≤ innerCandidates140_28.length then search edge 3 [823, 8101] innerCandidates140_28 8924 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan140_29

def innerTail140_27 : List Nat := 7699 :: innerTail140_28
def innerCandidates140_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates140_27_checked : candidates edge 4 823 7699 innerTail140_28 best.bound = innerCandidates140_27 := by
  unfold candidates
  have row : edge 7699 = SearchData.row7699.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan140_27 : scan edge (search edge 3) 4 [823] 823 innerTail140_27 best = best := by
  rw [innerTail140_27, scan, if_neg (by decide), innerCandidates140_27_checked]
  change scan edge (search edge 3) 4 [823] 823 innerTail140_28
    (if 3 ≤ innerCandidates140_27.length then search edge 3 [823, 7699] innerCandidates140_27 8522 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan140_28

def innerTail140_26 : List Nat := 7237 :: innerTail140_27
def innerCandidates140_26 : List Nat := [7699]
set_option maxRecDepth 4096 in
theorem innerCandidates140_26_checked : candidates edge 4 823 7237 innerTail140_27 best.bound = innerCandidates140_26 := by
  unfold candidates
  have row : edge 7237 = SearchData.row7237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan140_26 : scan edge (search edge 3) 4 [823] 823 innerTail140_26 best = best := by
  rw [innerTail140_26, scan, if_neg (by decide), innerCandidates140_26_checked]
  change scan edge (search edge 3) 4 [823] 823 innerTail140_27
    (if 3 ≤ innerCandidates140_26.length then search edge 3 [823, 7237] innerCandidates140_26 8060 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan140_27

def innerTail140_25 : List Nat := 6733 :: innerTail140_26
def innerCandidates140_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates140_25_checked : candidates edge 4 823 6733 innerTail140_26 best.bound = innerCandidates140_25 := by
  unfold candidates
  have row : edge 6733 = SearchData.row6733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan140_25 : scan edge (search edge 3) 4 [823] 823 innerTail140_25 best = best := by
  rw [innerTail140_25, scan, if_neg (by decide), innerCandidates140_25_checked]
  change scan edge (search edge 3) 4 [823] 823 innerTail140_26
    (if 3 ≤ innerCandidates140_25.length then search edge 3 [823, 6733] innerCandidates140_25 7556 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan140_26

def innerTail140_24 : List Nat := 6337 :: innerTail140_25
def innerCandidates140_24 : List Nat := [8101]
set_option maxRecDepth 4096 in
theorem innerCandidates140_24_checked : candidates edge 4 823 6337 innerTail140_25 best.bound = innerCandidates140_24 := by
  unfold candidates
  have row : edge 6337 = SearchData.row6337.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan140_24 : scan edge (search edge 3) 4 [823] 823 innerTail140_24 best = best := by
  rw [innerTail140_24, scan, if_neg (by decide), innerCandidates140_24_checked]
  change scan edge (search edge 3) 4 [823] 823 innerTail140_25
    (if 3 ≤ innerCandidates140_24.length then search edge 3 [823, 6337] innerCandidates140_24 7160 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan140_25

def innerTail140_23 : List Nat := 5881 :: innerTail140_24
def innerCandidates140_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates140_23_checked : candidates edge 4 823 5881 innerTail140_24 best.bound = innerCandidates140_23 := by
  unfold candidates
  have row : edge 5881 = SearchData.row5881.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan140_23 : scan edge (search edge 3) 4 [823] 823 innerTail140_23 best = best := by
  rw [innerTail140_23, scan, if_neg (by decide), innerCandidates140_23_checked]
  change scan edge (search edge 3) 4 [823] 823 innerTail140_24
    (if 3 ≤ innerCandidates140_23.length then search edge 3 [823, 5881] innerCandidates140_23 6704 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan140_24

def innerTail140_22 : List Nat := 5827 :: innerTail140_23
def innerCandidates140_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates140_22_checked : candidates edge 4 823 5827 innerTail140_23 best.bound = innerCandidates140_22 := by
  unfold candidates
  have row : edge 5827 = SearchData.row5827.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan140_22 : scan edge (search edge 3) 4 [823] 823 innerTail140_22 best = best := by
  rw [innerTail140_22, scan, if_neg (by decide), innerCandidates140_22_checked]
  change scan edge (search edge 3) 4 [823] 823 innerTail140_23
    (if 3 ≤ innerCandidates140_22.length then search edge 3 [823, 5827] innerCandidates140_22 6650 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan140_23

def innerTail140_21 : List Nat := 5323 :: innerTail140_22
def innerCandidates140_21 : List Nat := [7237]
set_option maxRecDepth 4096 in
theorem innerCandidates140_21_checked : candidates edge 4 823 5323 innerTail140_22 best.bound = innerCandidates140_21 := by
  unfold candidates
  have row : edge 5323 = SearchData.row5323.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan140_21 : scan edge (search edge 3) 4 [823] 823 innerTail140_21 best = best := by
  rw [innerTail140_21, scan, if_neg (by decide), innerCandidates140_21_checked]
  change scan edge (search edge 3) 4 [823] 823 innerTail140_22
    (if 3 ≤ innerCandidates140_21.length then search edge 3 [823, 5323] innerCandidates140_21 6146 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan140_22

def innerTail140_20 : List Nat := 5281 :: innerTail140_21
def innerCandidates140_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates140_20_checked : candidates edge 4 823 5281 innerTail140_21 best.bound = innerCandidates140_20 := by
  unfold candidates
  have row : edge 5281 = SearchData.row5281.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan140_20 : scan edge (search edge 3) 4 [823] 823 innerTail140_20 best = best := by
  rw [innerTail140_20, scan, if_neg (by decide), innerCandidates140_20_checked]
  change scan edge (search edge 3) 4 [823] 823 innerTail140_21
    (if 3 ≤ innerCandidates140_20.length then search edge 3 [823, 5281] innerCandidates140_20 6104 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan140_21

def innerTail140_19 : List Nat := 5209 :: innerTail140_20
def innerCandidates140_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates140_19_checked : candidates edge 4 823 5209 innerTail140_20 best.bound = innerCandidates140_19 := by
  unfold candidates
  have row : edge 5209 = SearchData.row5209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan140_19 : scan edge (search edge 3) 4 [823] 823 innerTail140_19 best = best := by
  rw [innerTail140_19, scan, if_neg (by decide), innerCandidates140_19_checked]
  change scan edge (search edge 3) 4 [823] 823 innerTail140_20
    (if 3 ≤ innerCandidates140_19.length then search edge 3 [823, 5209] innerCandidates140_19 6032 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan140_20

def innerTail140_18 : List Nat := 4951 :: innerTail140_19
def innerCandidates140_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates140_18_checked : candidates edge 4 823 4951 innerTail140_19 best.bound = innerCandidates140_18 := by
  unfold candidates
  have row : edge 4951 = SearchData.row4951.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan140_18 : scan edge (search edge 3) 4 [823] 823 innerTail140_18 best = best := by
  rw [innerTail140_18, scan, if_neg (by decide), innerCandidates140_18_checked]
  change scan edge (search edge 3) 4 [823] 823 innerTail140_19
    (if 3 ≤ innerCandidates140_18.length then search edge 3 [823, 4951] innerCandidates140_18 5774 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan140_19

def innerTail140_17 : List Nat := 4909 :: innerTail140_18
def innerCandidates140_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates140_17_checked : candidates edge 4 823 4909 innerTail140_18 best.bound = innerCandidates140_17 := by
  unfold candidates
  have row : edge 4909 = SearchData.row4909.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan140_17 : scan edge (search edge 3) 4 [823] 823 innerTail140_17 best = best := by
  rw [innerTail140_17, scan, if_neg (by decide), innerCandidates140_17_checked]
  change scan edge (search edge 3) 4 [823] 823 innerTail140_18
    (if 3 ≤ innerCandidates140_17.length then search edge 3 [823, 4909] innerCandidates140_17 5732 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan140_18

def innerTail140_16 : List Nat := 4789 :: innerTail140_17
def innerCandidates140_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates140_16_checked : candidates edge 4 823 4789 innerTail140_17 best.bound = innerCandidates140_16 := by
  unfold candidates
  have row : edge 4789 = SearchData.row4789.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan140_16 : scan edge (search edge 3) 4 [823] 823 innerTail140_16 best = best := by
  rw [innerTail140_16, scan, if_neg (by decide), innerCandidates140_16_checked]
  change scan edge (search edge 3) 4 [823] 823 innerTail140_17
    (if 3 ≤ innerCandidates140_16.length then search edge 3 [823, 4789] innerCandidates140_16 5612 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan140_17

def innerTail140_15 : List Nat := 4243 :: innerTail140_16
def innerCandidates140_15 : List Nat := [7699]
set_option maxRecDepth 4096 in
theorem innerCandidates140_15_checked : candidates edge 4 823 4243 innerTail140_16 best.bound = innerCandidates140_15 := by
  unfold candidates
  have row : edge 4243 = SearchData.row4243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan140_15 : scan edge (search edge 3) 4 [823] 823 innerTail140_15 best = best := by
  rw [innerTail140_15, scan, if_neg (by decide), innerCandidates140_15_checked]
  change scan edge (search edge 3) 4 [823] 823 innerTail140_16
    (if 3 ≤ innerCandidates140_15.length then search edge 3 [823, 4243] innerCandidates140_15 5066 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan140_16

def innerTail140_14 : List Nat := 3541 :: innerTail140_15
def innerCandidates140_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates140_14_checked : candidates edge 4 823 3541 innerTail140_15 best.bound = innerCandidates140_14 := by
  unfold candidates
  have row : edge 3541 = SearchData.row3541.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan140_14 : scan edge (search edge 3) 4 [823] 823 innerTail140_14 best = best := by
  rw [innerTail140_14, scan, if_neg (by decide), innerCandidates140_14_checked]
  change scan edge (search edge 3) 4 [823] 823 innerTail140_15
    (if 3 ≤ innerCandidates140_14.length then search edge 3 [823, 3541] innerCandidates140_14 4364 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan140_15

def innerTail140_13 : List Nat := 2953 :: innerTail140_14
def innerCandidates140_13 : List Nat := [6733]
set_option maxRecDepth 4096 in
theorem innerCandidates140_13_checked : candidates edge 4 823 2953 innerTail140_14 best.bound = innerCandidates140_13 := by
  unfold candidates
  have row : edge 2953 = SearchData.row2953.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan140_13 : scan edge (search edge 3) 4 [823] 823 innerTail140_13 best = best := by
  rw [innerTail140_13, scan, if_neg (by decide), innerCandidates140_13_checked]
  change scan edge (search edge 3) 4 [823] 823 innerTail140_14
    (if 3 ≤ innerCandidates140_13.length then search edge 3 [823, 2953] innerCandidates140_13 3776 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan140_14

def innerTail140_12 : List Nat := 2557 :: innerTail140_13
def innerCandidates140_12 : List Nat := [4243]
set_option maxRecDepth 4096 in
theorem innerCandidates140_12_checked : candidates edge 4 823 2557 innerTail140_13 best.bound = innerCandidates140_12 := by
  unfold candidates
  have row : edge 2557 = PrimePairCache.smallRow2557.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan140_12 : scan edge (search edge 3) 4 [823] 823 innerTail140_12 best = best := by
  rw [innerTail140_12, scan, if_neg (by decide), innerCandidates140_12_checked]
  change scan edge (search edge 3) 4 [823] 823 innerTail140_13
    (if 3 ≤ innerCandidates140_12.length then search edge 3 [823, 2557] innerCandidates140_12 3380 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan140_13

def innerTail140_11 : List Nat := 2347 :: innerTail140_12
def innerCandidates140_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates140_11_checked : candidates edge 4 823 2347 innerTail140_12 best.bound = innerCandidates140_11 := by
  unfold candidates
  have row : edge 2347 = PrimePairCache.smallRow2347.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan140_11 : scan edge (search edge 3) 4 [823] 823 innerTail140_11 best = best := by
  rw [innerTail140_11, scan, if_neg (by decide), innerCandidates140_11_checked]
  change scan edge (search edge 3) 4 [823] 823 innerTail140_12
    (if 3 ≤ innerCandidates140_11.length then search edge 3 [823, 2347] innerCandidates140_11 3170 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan140_12

def innerTail140_10 : List Nat := 2311 :: innerTail140_11
def innerCandidates140_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates140_10_checked : candidates edge 4 823 2311 innerTail140_11 best.bound = innerCandidates140_10 := by
  unfold candidates
  have row : edge 2311 = PrimePairCache.smallRow2311.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan140_10 : scan edge (search edge 3) 4 [823] 823 innerTail140_10 best = best := by
  rw [innerTail140_10, scan, if_neg (by decide), innerCandidates140_10_checked]
  change scan edge (search edge 3) 4 [823] 823 innerTail140_11
    (if 3 ≤ innerCandidates140_10.length then search edge 3 [823, 2311] innerCandidates140_10 3134 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan140_11

def innerTail140_9 : List Nat := 2281 :: innerTail140_10
def innerCandidates140_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates140_9_checked : candidates edge 4 823 2281 innerTail140_10 best.bound = innerCandidates140_9 := by
  unfold candidates
  have row : edge 2281 = PrimePairCache.smallRow2281.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan140_9 : scan edge (search edge 3) 4 [823] 823 innerTail140_9 best = best := by
  rw [innerTail140_9, scan, if_neg (by decide), innerCandidates140_9_checked]
  change scan edge (search edge 3) 4 [823] 823 innerTail140_10
    (if 3 ≤ innerCandidates140_9.length then search edge 3 [823, 2281] innerCandidates140_9 3104 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan140_10

def innerTail140_8 : List Nat := 2179 :: innerTail140_9
def innerCandidates140_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates140_8_checked : candidates edge 4 823 2179 innerTail140_9 best.bound = innerCandidates140_8 := by
  unfold candidates
  have row : edge 2179 = PrimePairCache.smallRow2179.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan140_8 : scan edge (search edge 3) 4 [823] 823 innerTail140_8 best = best := by
  rw [innerTail140_8, scan, if_neg (by decide), innerCandidates140_8_checked]
  change scan edge (search edge 3) 4 [823] 823 innerTail140_9
    (if 3 ≤ innerCandidates140_8.length then search edge 3 [823, 2179] innerCandidates140_8 3002 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan140_9

def innerTail140_7 : List Nat := 2113 :: innerTail140_8
def innerCandidates140_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates140_7_checked : candidates edge 4 823 2113 innerTail140_8 best.bound = innerCandidates140_7 := by
  unfold candidates
  have row : edge 2113 = PrimePairCache.smallRow2113.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan140_7 : scan edge (search edge 3) 4 [823] 823 innerTail140_7 best = best := by
  rw [innerTail140_7, scan, if_neg (by decide), innerCandidates140_7_checked]
  change scan edge (search edge 3) 4 [823] 823 innerTail140_8
    (if 3 ≤ innerCandidates140_7.length then search edge 3 [823, 2113] innerCandidates140_7 2936 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan140_8

def innerTail140_6 : List Nat := 2017 :: innerTail140_7
def innerCandidates140_6 : List Nat := [7237]
set_option maxRecDepth 4096 in
theorem innerCandidates140_6_checked : candidates edge 4 823 2017 innerTail140_7 best.bound = innerCandidates140_6 := by
  unfold candidates
  have row : edge 2017 = PrimePairCache.smallRow2017.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan140_6 : scan edge (search edge 3) 4 [823] 823 innerTail140_6 best = best := by
  rw [innerTail140_6, scan, if_neg (by decide), innerCandidates140_6_checked]
  change scan edge (search edge 3) 4 [823] 823 innerTail140_7
    (if 3 ≤ innerCandidates140_6.length then search edge 3 [823, 2017] innerCandidates140_6 2840 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan140_7

def innerTail140_5 : List Nat := 1933 :: innerTail140_6
def innerCandidates140_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates140_5_checked : candidates edge 4 823 1933 innerTail140_6 best.bound = innerCandidates140_5 := by
  unfold candidates
  have row : edge 1933 = PrimePairCache.smallRow1933.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan140_5 : scan edge (search edge 3) 4 [823] 823 innerTail140_5 best = best := by
  rw [innerTail140_5, scan, if_neg (by decide), innerCandidates140_5_checked]
  change scan edge (search edge 3) 4 [823] 823 innerTail140_6
    (if 3 ≤ innerCandidates140_5.length then search edge 3 [823, 1933] innerCandidates140_5 2756 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan140_6

def innerTail140_4 : List Nat := 1777 :: innerTail140_5
def innerCandidates140_4 : List Nat := [2017]
set_option maxRecDepth 4096 in
theorem innerCandidates140_4_checked : candidates edge 4 823 1777 innerTail140_5 best.bound = innerCandidates140_4 := by
  unfold candidates
  have row : edge 1777 = PrimePairCache.smallRow1777.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan140_4 : scan edge (search edge 3) 4 [823] 823 innerTail140_4 best = best := by
  rw [innerTail140_4, scan, if_neg (by decide), innerCandidates140_4_checked]
  change scan edge (search edge 3) 4 [823] 823 innerTail140_5
    (if 3 ≤ innerCandidates140_4.length then search edge 3 [823, 1777] innerCandidates140_4 2600 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan140_5

def innerTail140_3 : List Nat := 1459 :: innerTail140_4
def innerCandidates140_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates140_3_checked : candidates edge 4 823 1459 innerTail140_4 best.bound = innerCandidates140_3 := by
  unfold candidates
  have row : edge 1459 = PrimePairCache.smallRow1459.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan140_3 : scan edge (search edge 3) 4 [823] 823 innerTail140_3 best = best := by
  rw [innerTail140_3, scan, if_neg (by decide), innerCandidates140_3_checked]
  change scan edge (search edge 3) 4 [823] 823 innerTail140_4
    (if 3 ≤ innerCandidates140_3.length then search edge 3 [823, 1459] innerCandidates140_3 2282 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan140_4

def innerTail140_2 : List Nat := 1303 :: innerTail140_3
def innerCandidates140_2 : List Nat := [2953, 8101]
set_option maxRecDepth 4096 in
theorem innerCandidates140_2_checked : candidates edge 4 823 1303 innerTail140_3 best.bound = innerCandidates140_2 := by
  unfold candidates
  have row : edge 1303 = PrimePairCache.smallRow1303.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan140_2 : scan edge (search edge 3) 4 [823] 823 innerTail140_2 best = best := by
  rw [innerTail140_2, scan, if_neg (by decide), innerCandidates140_2_checked]
  change scan edge (search edge 3) 4 [823] 823 innerTail140_3
    (if 3 ≤ innerCandidates140_2.length then search edge 3 [823, 1303] innerCandidates140_2 2126 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan140_3

def innerTail140_1 : List Nat := 967 :: innerTail140_2
def innerCandidates140_1 : List Nat := [2557]
set_option maxRecDepth 4096 in
theorem innerCandidates140_1_checked : candidates edge 4 823 967 innerTail140_2 best.bound = innerCandidates140_1 := by
  unfold candidates
  have row : edge 967 = PrimePairCache.smallRow967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan140_1 : scan edge (search edge 3) 4 [823] 823 innerTail140_1 best = best := by
  rw [innerTail140_1, scan, if_neg (by decide), innerCandidates140_1_checked]
  change scan edge (search edge 3) 4 [823] 823 innerTail140_2
    (if 3 ≤ innerCandidates140_1.length then search edge 3 [823, 967] innerCandidates140_1 1790 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan140_2

def innerTail140_0 : List Nat := 877 :: innerTail140_1
def innerCandidates140_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates140_0_checked : candidates edge 4 823 877 innerTail140_1 best.bound = innerCandidates140_0 := by
  unfold candidates
  have row : edge 877 = PrimePairCache.smallRow877.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan140_0 : scan edge (search edge 3) 4 [823] 823 innerTail140_0 best = best := by
  rw [innerTail140_0, scan, if_neg (by decide), innerCandidates140_0_checked]
  change scan edge (search edge 3) 4 [823] 823 innerTail140_1
    (if 3 ≤ innerCandidates140_0.length then search edge 3 [823, 877] innerCandidates140_0 1700 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan140_1

theorem rootBranch140 : search edge 4 [823] rootCandidates140 823 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [823] 823 innerTail140_0 best = best
  exact innerScan140_0

end ProjectEulerSolutions.PrimePairUpper.Search
