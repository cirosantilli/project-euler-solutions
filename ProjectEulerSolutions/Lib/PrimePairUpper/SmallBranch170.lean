import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates172 : List Nat := [1279, 1663, 1999, 2467, 2659, 2917, 3457, 3637, 3967, 4297, 4723, 5233, 5821, 5851, 6073, 6709, 7993, 8161]
set_option maxRecDepth 4096 in
theorem rootCandidates172_checked : candidates edge 5 0 1039 tail173 best.bound = rootCandidates172 := by
  unfold candidates
  have row : edge 1039 = PrimePairCache.smallRow1039.contains := by funext q; rfl
  rw [row]
  decide

def innerTail172_18 : List Nat := []
theorem innerScan172_18 : scan edge (search edge 3) 4 [1039] 1039 innerTail172_18 best = best := rfl

def innerTail172_17 : List Nat := 8161 :: innerTail172_18
def innerCandidates172_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates172_17_checked : candidates edge 4 1039 8161 innerTail172_18 best.bound = innerCandidates172_17 := by
  unfold candidates
  have row : edge 8161 = SearchData.row8161.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan172_17 : scan edge (search edge 3) 4 [1039] 1039 innerTail172_17 best = best := by
  rw [innerTail172_17, scan, if_neg (by decide), innerCandidates172_17_checked]
  change scan edge (search edge 3) 4 [1039] 1039 innerTail172_18
    (if 3 ≤ innerCandidates172_17.length then search edge 3 [1039, 8161] innerCandidates172_17 9200 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan172_18

def innerTail172_16 : List Nat := 7993 :: innerTail172_17
def innerCandidates172_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates172_16_checked : candidates edge 4 1039 7993 innerTail172_17 best.bound = innerCandidates172_16 := by
  unfold candidates
  have row : edge 7993 = SearchData.row7993.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan172_16 : scan edge (search edge 3) 4 [1039] 1039 innerTail172_16 best = best := by
  rw [innerTail172_16, scan, if_neg (by decide), innerCandidates172_16_checked]
  change scan edge (search edge 3) 4 [1039] 1039 innerTail172_17
    (if 3 ≤ innerCandidates172_16.length then search edge 3 [1039, 7993] innerCandidates172_16 9032 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan172_17

def innerTail172_15 : List Nat := 6709 :: innerTail172_16
def innerCandidates172_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates172_15_checked : candidates edge 4 1039 6709 innerTail172_16 best.bound = innerCandidates172_15 := by
  unfold candidates
  have row : edge 6709 = SearchData.row6709.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan172_15 : scan edge (search edge 3) 4 [1039] 1039 innerTail172_15 best = best := by
  rw [innerTail172_15, scan, if_neg (by decide), innerCandidates172_15_checked]
  change scan edge (search edge 3) 4 [1039] 1039 innerTail172_16
    (if 3 ≤ innerCandidates172_15.length then search edge 3 [1039, 6709] innerCandidates172_15 7748 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan172_16

def innerTail172_14 : List Nat := 6073 :: innerTail172_15
def innerCandidates172_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates172_14_checked : candidates edge 4 1039 6073 innerTail172_15 best.bound = innerCandidates172_14 := by
  unfold candidates
  have row : edge 6073 = SearchData.row6073.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan172_14 : scan edge (search edge 3) 4 [1039] 1039 innerTail172_14 best = best := by
  rw [innerTail172_14, scan, if_neg (by decide), innerCandidates172_14_checked]
  change scan edge (search edge 3) 4 [1039] 1039 innerTail172_15
    (if 3 ≤ innerCandidates172_14.length then search edge 3 [1039, 6073] innerCandidates172_14 7112 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan172_15

def innerTail172_13 : List Nat := 5851 :: innerTail172_14
def innerCandidates172_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates172_13_checked : candidates edge 4 1039 5851 innerTail172_14 best.bound = innerCandidates172_13 := by
  unfold candidates
  have row : edge 5851 = SearchData.row5851.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan172_13 : scan edge (search edge 3) 4 [1039] 1039 innerTail172_13 best = best := by
  rw [innerTail172_13, scan, if_neg (by decide), innerCandidates172_13_checked]
  change scan edge (search edge 3) 4 [1039] 1039 innerTail172_14
    (if 3 ≤ innerCandidates172_13.length then search edge 3 [1039, 5851] innerCandidates172_13 6890 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan172_14

def innerTail172_12 : List Nat := 5821 :: innerTail172_13
def innerCandidates172_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates172_12_checked : candidates edge 4 1039 5821 innerTail172_13 best.bound = innerCandidates172_12 := by
  unfold candidates
  have row : edge 5821 = SearchData.row5821.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan172_12 : scan edge (search edge 3) 4 [1039] 1039 innerTail172_12 best = best := by
  rw [innerTail172_12, scan, if_neg (by decide), innerCandidates172_12_checked]
  change scan edge (search edge 3) 4 [1039] 1039 innerTail172_13
    (if 3 ≤ innerCandidates172_12.length then search edge 3 [1039, 5821] innerCandidates172_12 6860 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan172_13

def innerTail172_11 : List Nat := 5233 :: innerTail172_12
def innerCandidates172_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates172_11_checked : candidates edge 4 1039 5233 innerTail172_12 best.bound = innerCandidates172_11 := by
  unfold candidates
  have row : edge 5233 = SearchData.row5233.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan172_11 : scan edge (search edge 3) 4 [1039] 1039 innerTail172_11 best = best := by
  rw [innerTail172_11, scan, if_neg (by decide), innerCandidates172_11_checked]
  change scan edge (search edge 3) 4 [1039] 1039 innerTail172_12
    (if 3 ≤ innerCandidates172_11.length then search edge 3 [1039, 5233] innerCandidates172_11 6272 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan172_12

def innerTail172_10 : List Nat := 4723 :: innerTail172_11
def innerCandidates172_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates172_10_checked : candidates edge 4 1039 4723 innerTail172_11 best.bound = innerCandidates172_10 := by
  unfold candidates
  have row : edge 4723 = SearchData.row4723.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan172_10 : scan edge (search edge 3) 4 [1039] 1039 innerTail172_10 best = best := by
  rw [innerTail172_10, scan, if_neg (by decide), innerCandidates172_10_checked]
  change scan edge (search edge 3) 4 [1039] 1039 innerTail172_11
    (if 3 ≤ innerCandidates172_10.length then search edge 3 [1039, 4723] innerCandidates172_10 5762 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan172_11

def innerTail172_9 : List Nat := 4297 :: innerTail172_10
def innerCandidates172_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates172_9_checked : candidates edge 4 1039 4297 innerTail172_10 best.bound = innerCandidates172_9 := by
  unfold candidates
  have row : edge 4297 = SearchData.row4297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan172_9 : scan edge (search edge 3) 4 [1039] 1039 innerTail172_9 best = best := by
  rw [innerTail172_9, scan, if_neg (by decide), innerCandidates172_9_checked]
  change scan edge (search edge 3) 4 [1039] 1039 innerTail172_10
    (if 3 ≤ innerCandidates172_9.length then search edge 3 [1039, 4297] innerCandidates172_9 5336 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan172_10

def innerTail172_8 : List Nat := 3967 :: innerTail172_9
def innerCandidates172_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates172_8_checked : candidates edge 4 1039 3967 innerTail172_9 best.bound = innerCandidates172_8 := by
  unfold candidates
  have row : edge 3967 = SearchData.row3967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan172_8 : scan edge (search edge 3) 4 [1039] 1039 innerTail172_8 best = best := by
  rw [innerTail172_8, scan, if_neg (by decide), innerCandidates172_8_checked]
  change scan edge (search edge 3) 4 [1039] 1039 innerTail172_9
    (if 3 ≤ innerCandidates172_8.length then search edge 3 [1039, 3967] innerCandidates172_8 5006 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan172_9

def innerTail172_7 : List Nat := 3637 :: innerTail172_8
def innerCandidates172_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates172_7_checked : candidates edge 4 1039 3637 innerTail172_8 best.bound = innerCandidates172_7 := by
  unfold candidates
  have row : edge 3637 = SearchData.row3637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan172_7 : scan edge (search edge 3) 4 [1039] 1039 innerTail172_7 best = best := by
  rw [innerTail172_7, scan, if_neg (by decide), innerCandidates172_7_checked]
  change scan edge (search edge 3) 4 [1039] 1039 innerTail172_8
    (if 3 ≤ innerCandidates172_7.length then search edge 3 [1039, 3637] innerCandidates172_7 4676 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan172_8

def innerTail172_6 : List Nat := 3457 :: innerTail172_7
def innerCandidates172_6 : List Nat := [4297]
set_option maxRecDepth 4096 in
theorem innerCandidates172_6_checked : candidates edge 4 1039 3457 innerTail172_7 best.bound = innerCandidates172_6 := by
  unfold candidates
  have row : edge 3457 = SearchData.row3457.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan172_6 : scan edge (search edge 3) 4 [1039] 1039 innerTail172_6 best = best := by
  rw [innerTail172_6, scan, if_neg (by decide), innerCandidates172_6_checked]
  change scan edge (search edge 3) 4 [1039] 1039 innerTail172_7
    (if 3 ≤ innerCandidates172_6.length then search edge 3 [1039, 3457] innerCandidates172_6 4496 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan172_7

def innerTail172_5 : List Nat := 2917 :: innerTail172_6
def innerCandidates172_5 : List Nat := [5851]
set_option maxRecDepth 4096 in
theorem innerCandidates172_5_checked : candidates edge 4 1039 2917 innerTail172_6 best.bound = innerCandidates172_5 := by
  unfold candidates
  have row : edge 2917 = SearchData.row2917.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan172_5 : scan edge (search edge 3) 4 [1039] 1039 innerTail172_5 best = best := by
  rw [innerTail172_5, scan, if_neg (by decide), innerCandidates172_5_checked]
  change scan edge (search edge 3) 4 [1039] 1039 innerTail172_6
    (if 3 ≤ innerCandidates172_5.length then search edge 3 [1039, 2917] innerCandidates172_5 3956 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan172_6

def innerTail172_4 : List Nat := 2659 :: innerTail172_5
def innerCandidates172_4 : List Nat := [4723]
set_option maxRecDepth 4096 in
theorem innerCandidates172_4_checked : candidates edge 4 1039 2659 innerTail172_5 best.bound = innerCandidates172_4 := by
  unfold candidates
  have row : edge 2659 = PrimePairCache.smallRow2659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan172_4 : scan edge (search edge 3) 4 [1039] 1039 innerTail172_4 best = best := by
  rw [innerTail172_4, scan, if_neg (by decide), innerCandidates172_4_checked]
  change scan edge (search edge 3) 4 [1039] 1039 innerTail172_5
    (if 3 ≤ innerCandidates172_4.length then search edge 3 [1039, 2659] innerCandidates172_4 3698 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan172_5

def innerTail172_3 : List Nat := 2467 :: innerTail172_4
def innerCandidates172_3 : List Nat := [4723]
set_option maxRecDepth 4096 in
theorem innerCandidates172_3_checked : candidates edge 4 1039 2467 innerTail172_4 best.bound = innerCandidates172_3 := by
  unfold candidates
  have row : edge 2467 = PrimePairCache.smallRow2467.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan172_3 : scan edge (search edge 3) 4 [1039] 1039 innerTail172_3 best = best := by
  rw [innerTail172_3, scan, if_neg (by decide), innerCandidates172_3_checked]
  change scan edge (search edge 3) 4 [1039] 1039 innerTail172_4
    (if 3 ≤ innerCandidates172_3.length then search edge 3 [1039, 2467] innerCandidates172_3 3506 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan172_4

def innerTail172_2 : List Nat := 1999 :: innerTail172_3
def innerCandidates172_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates172_2_checked : candidates edge 4 1039 1999 innerTail172_3 best.bound = innerCandidates172_2 := by
  unfold candidates
  have row : edge 1999 = PrimePairCache.smallRow1999.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan172_2 : scan edge (search edge 3) 4 [1039] 1039 innerTail172_2 best = best := by
  rw [innerTail172_2, scan, if_neg (by decide), innerCandidates172_2_checked]
  change scan edge (search edge 3) 4 [1039] 1039 innerTail172_3
    (if 3 ≤ innerCandidates172_2.length then search edge 3 [1039, 1999] innerCandidates172_2 3038 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan172_3

def innerTail172_1 : List Nat := 1663 :: innerTail172_2
def innerCandidates172_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates172_1_checked : candidates edge 4 1039 1663 innerTail172_2 best.bound = innerCandidates172_1 := by
  unfold candidates
  have row : edge 1663 = PrimePairCache.smallRow1663.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan172_1 : scan edge (search edge 3) 4 [1039] 1039 innerTail172_1 best = best := by
  rw [innerTail172_1, scan, if_neg (by decide), innerCandidates172_1_checked]
  change scan edge (search edge 3) 4 [1039] 1039 innerTail172_2
    (if 3 ≤ innerCandidates172_1.length then search edge 3 [1039, 1663] innerCandidates172_1 2702 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan172_2

def innerTail172_0 : List Nat := 1279 :: innerTail172_1
def innerCandidates172_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates172_0_checked : candidates edge 4 1039 1279 innerTail172_1 best.bound = innerCandidates172_0 := by
  unfold candidates
  have row : edge 1279 = PrimePairCache.smallRow1279.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan172_0 : scan edge (search edge 3) 4 [1039] 1039 innerTail172_0 best = best := by
  rw [innerTail172_0, scan, if_neg (by decide), innerCandidates172_0_checked]
  change scan edge (search edge 3) 4 [1039] 1039 innerTail172_1
    (if 3 ≤ innerCandidates172_0.length then search edge 3 [1039, 1279] innerCandidates172_0 2318 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan172_1

theorem rootBranch172 : search edge 4 [1039] rootCandidates172 1039 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1039] 1039 innerTail172_0 best = best
  exact innerScan172_0

def rootCandidates171 : List Nat := [1249, 1543, 1663, 1759, 2131, 3067, 3529, 4231, 5233, 5869, 6091, 6301, 6421, 6451, 7417, 8089, 8179, 8269, 8353]
set_option maxRecDepth 4096 in
theorem rootCandidates171_checked : candidates edge 5 0 1033 tail172 best.bound = rootCandidates171 := by
  unfold candidates
  have row : edge 1033 = PrimePairCache.smallRow1033.contains := by funext q; rfl
  rw [row]
  decide

def innerTail171_19 : List Nat := []
theorem innerScan171_19 : scan edge (search edge 3) 4 [1033] 1033 innerTail171_19 best = best := rfl

def innerTail171_18 : List Nat := 8353 :: innerTail171_19
def innerCandidates171_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates171_18_checked : candidates edge 4 1033 8353 innerTail171_19 best.bound = innerCandidates171_18 := by
  unfold candidates
  have row : edge 8353 = SearchData.row8353.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan171_18 : scan edge (search edge 3) 4 [1033] 1033 innerTail171_18 best = best := by
  rw [innerTail171_18, scan, if_neg (by decide), innerCandidates171_18_checked]
  change scan edge (search edge 3) 4 [1033] 1033 innerTail171_19
    (if 3 ≤ innerCandidates171_18.length then search edge 3 [1033, 8353] innerCandidates171_18 9386 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan171_19

def innerTail171_17 : List Nat := 8269 :: innerTail171_18
def innerCandidates171_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates171_17_checked : candidates edge 4 1033 8269 innerTail171_18 best.bound = innerCandidates171_17 := by
  unfold candidates
  have row : edge 8269 = SearchData.row8269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan171_17 : scan edge (search edge 3) 4 [1033] 1033 innerTail171_17 best = best := by
  rw [innerTail171_17, scan, if_neg (by decide), innerCandidates171_17_checked]
  change scan edge (search edge 3) 4 [1033] 1033 innerTail171_18
    (if 3 ≤ innerCandidates171_17.length then search edge 3 [1033, 8269] innerCandidates171_17 9302 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan171_18

def innerTail171_16 : List Nat := 8179 :: innerTail171_17
def innerCandidates171_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates171_16_checked : candidates edge 4 1033 8179 innerTail171_17 best.bound = innerCandidates171_16 := by
  unfold candidates
  have row : edge 8179 = SearchData.row8179.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan171_16 : scan edge (search edge 3) 4 [1033] 1033 innerTail171_16 best = best := by
  rw [innerTail171_16, scan, if_neg (by decide), innerCandidates171_16_checked]
  change scan edge (search edge 3) 4 [1033] 1033 innerTail171_17
    (if 3 ≤ innerCandidates171_16.length then search edge 3 [1033, 8179] innerCandidates171_16 9212 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan171_17

def innerTail171_15 : List Nat := 8089 :: innerTail171_16
def innerCandidates171_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates171_15_checked : candidates edge 4 1033 8089 innerTail171_16 best.bound = innerCandidates171_15 := by
  unfold candidates
  have row : edge 8089 = SearchData.row8089.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan171_15 : scan edge (search edge 3) 4 [1033] 1033 innerTail171_15 best = best := by
  rw [innerTail171_15, scan, if_neg (by decide), innerCandidates171_15_checked]
  change scan edge (search edge 3) 4 [1033] 1033 innerTail171_16
    (if 3 ≤ innerCandidates171_15.length then search edge 3 [1033, 8089] innerCandidates171_15 9122 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan171_16

def innerTail171_14 : List Nat := 7417 :: innerTail171_15
def innerCandidates171_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates171_14_checked : candidates edge 4 1033 7417 innerTail171_15 best.bound = innerCandidates171_14 := by
  unfold candidates
  have row : edge 7417 = SearchData.row7417.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan171_14 : scan edge (search edge 3) 4 [1033] 1033 innerTail171_14 best = best := by
  rw [innerTail171_14, scan, if_neg (by decide), innerCandidates171_14_checked]
  change scan edge (search edge 3) 4 [1033] 1033 innerTail171_15
    (if 3 ≤ innerCandidates171_14.length then search edge 3 [1033, 7417] innerCandidates171_14 8450 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan171_15

def innerTail171_13 : List Nat := 6451 :: innerTail171_14
def innerCandidates171_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates171_13_checked : candidates edge 4 1033 6451 innerTail171_14 best.bound = innerCandidates171_13 := by
  unfold candidates
  have row : edge 6451 = SearchData.row6451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan171_13 : scan edge (search edge 3) 4 [1033] 1033 innerTail171_13 best = best := by
  rw [innerTail171_13, scan, if_neg (by decide), innerCandidates171_13_checked]
  change scan edge (search edge 3) 4 [1033] 1033 innerTail171_14
    (if 3 ≤ innerCandidates171_13.length then search edge 3 [1033, 6451] innerCandidates171_13 7484 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan171_14

def innerTail171_12 : List Nat := 6421 :: innerTail171_13
def innerCandidates171_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates171_12_checked : candidates edge 4 1033 6421 innerTail171_13 best.bound = innerCandidates171_12 := by
  unfold candidates
  have row : edge 6421 = SearchData.row6421.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan171_12 : scan edge (search edge 3) 4 [1033] 1033 innerTail171_12 best = best := by
  rw [innerTail171_12, scan, if_neg (by decide), innerCandidates171_12_checked]
  change scan edge (search edge 3) 4 [1033] 1033 innerTail171_13
    (if 3 ≤ innerCandidates171_12.length then search edge 3 [1033, 6421] innerCandidates171_12 7454 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan171_13

def innerTail171_11 : List Nat := 6301 :: innerTail171_12
def innerCandidates171_11 : List Nat := [7417]
set_option maxRecDepth 4096 in
theorem innerCandidates171_11_checked : candidates edge 4 1033 6301 innerTail171_12 best.bound = innerCandidates171_11 := by
  unfold candidates
  have row : edge 6301 = SearchData.row6301.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan171_11 : scan edge (search edge 3) 4 [1033] 1033 innerTail171_11 best = best := by
  rw [innerTail171_11, scan, if_neg (by decide), innerCandidates171_11_checked]
  change scan edge (search edge 3) 4 [1033] 1033 innerTail171_12
    (if 3 ≤ innerCandidates171_11.length then search edge 3 [1033, 6301] innerCandidates171_11 7334 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan171_12

def innerTail171_10 : List Nat := 6091 :: innerTail171_11
def innerCandidates171_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates171_10_checked : candidates edge 4 1033 6091 innerTail171_11 best.bound = innerCandidates171_10 := by
  unfold candidates
  have row : edge 6091 = SearchData.row6091.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan171_10 : scan edge (search edge 3) 4 [1033] 1033 innerTail171_10 best = best := by
  rw [innerTail171_10, scan, if_neg (by decide), innerCandidates171_10_checked]
  change scan edge (search edge 3) 4 [1033] 1033 innerTail171_11
    (if 3 ≤ innerCandidates171_10.length then search edge 3 [1033, 6091] innerCandidates171_10 7124 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan171_11

def innerTail171_9 : List Nat := 5869 :: innerTail171_10
def innerCandidates171_9 : List Nat := [8089]
set_option maxRecDepth 4096 in
theorem innerCandidates171_9_checked : candidates edge 4 1033 5869 innerTail171_10 best.bound = innerCandidates171_9 := by
  unfold candidates
  have row : edge 5869 = SearchData.row5869.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan171_9 : scan edge (search edge 3) 4 [1033] 1033 innerTail171_9 best = best := by
  rw [innerTail171_9, scan, if_neg (by decide), innerCandidates171_9_checked]
  change scan edge (search edge 3) 4 [1033] 1033 innerTail171_10
    (if 3 ≤ innerCandidates171_9.length then search edge 3 [1033, 5869] innerCandidates171_9 6902 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan171_10

def innerTail171_8 : List Nat := 5233 :: innerTail171_9
def innerCandidates171_8 : List Nat := [6451, 8089]
set_option maxRecDepth 4096 in
theorem innerCandidates171_8_checked : candidates edge 4 1033 5233 innerTail171_9 best.bound = innerCandidates171_8 := by
  unfold candidates
  have row : edge 5233 = SearchData.row5233.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan171_8 : scan edge (search edge 3) 4 [1033] 1033 innerTail171_8 best = best := by
  rw [innerTail171_8, scan, if_neg (by decide), innerCandidates171_8_checked]
  change scan edge (search edge 3) 4 [1033] 1033 innerTail171_9
    (if 3 ≤ innerCandidates171_8.length then search edge 3 [1033, 5233] innerCandidates171_8 6266 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan171_9

def innerTail171_7 : List Nat := 4231 :: innerTail171_8
def innerCandidates171_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates171_7_checked : candidates edge 4 1033 4231 innerTail171_8 best.bound = innerCandidates171_7 := by
  unfold candidates
  have row : edge 4231 = SearchData.row4231.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan171_7 : scan edge (search edge 3) 4 [1033] 1033 innerTail171_7 best = best := by
  rw [innerTail171_7, scan, if_neg (by decide), innerCandidates171_7_checked]
  change scan edge (search edge 3) 4 [1033] 1033 innerTail171_8
    (if 3 ≤ innerCandidates171_7.length then search edge 3 [1033, 4231] innerCandidates171_7 5264 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan171_8

def innerTail171_6 : List Nat := 3529 :: innerTail171_7
def innerCandidates171_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates171_6_checked : candidates edge 4 1033 3529 innerTail171_7 best.bound = innerCandidates171_6 := by
  unfold candidates
  have row : edge 3529 = SearchData.row3529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan171_6 : scan edge (search edge 3) 4 [1033] 1033 innerTail171_6 best = best := by
  rw [innerTail171_6, scan, if_neg (by decide), innerCandidates171_6_checked]
  change scan edge (search edge 3) 4 [1033] 1033 innerTail171_7
    (if 3 ≤ innerCandidates171_6.length then search edge 3 [1033, 3529] innerCandidates171_6 4562 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan171_7

def innerTail171_5 : List Nat := 3067 :: innerTail171_6
def innerCandidates171_5 : List Nat := [8269]
set_option maxRecDepth 4096 in
theorem innerCandidates171_5_checked : candidates edge 4 1033 3067 innerTail171_6 best.bound = innerCandidates171_5 := by
  unfold candidates
  have row : edge 3067 = SearchData.row3067.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan171_5 : scan edge (search edge 3) 4 [1033] 1033 innerTail171_5 best = best := by
  rw [innerTail171_5, scan, if_neg (by decide), innerCandidates171_5_checked]
  change scan edge (search edge 3) 4 [1033] 1033 innerTail171_6
    (if 3 ≤ innerCandidates171_5.length then search edge 3 [1033, 3067] innerCandidates171_5 4100 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan171_6

def innerTail171_4 : List Nat := 2131 :: innerTail171_5
def innerCandidates171_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates171_4_checked : candidates edge 4 1033 2131 innerTail171_5 best.bound = innerCandidates171_4 := by
  unfold candidates
  have row : edge 2131 = PrimePairCache.smallRow2131.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan171_4 : scan edge (search edge 3) 4 [1033] 1033 innerTail171_4 best = best := by
  rw [innerTail171_4, scan, if_neg (by decide), innerCandidates171_4_checked]
  change scan edge (search edge 3) 4 [1033] 1033 innerTail171_5
    (if 3 ≤ innerCandidates171_4.length then search edge 3 [1033, 2131] innerCandidates171_4 3164 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan171_5

def innerTail171_3 : List Nat := 1759 :: innerTail171_4
def innerCandidates171_3 : List Nat := [5233, 7417]
set_option maxRecDepth 4096 in
theorem innerCandidates171_3_checked : candidates edge 4 1033 1759 innerTail171_4 best.bound = innerCandidates171_3 := by
  unfold candidates
  have row : edge 1759 = PrimePairCache.smallRow1759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan171_3 : scan edge (search edge 3) 4 [1033] 1033 innerTail171_3 best = best := by
  rw [innerTail171_3, scan, if_neg (by decide), innerCandidates171_3_checked]
  change scan edge (search edge 3) 4 [1033] 1033 innerTail171_4
    (if 3 ≤ innerCandidates171_3.length then search edge 3 [1033, 1759] innerCandidates171_3 2792 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan171_4

def innerTail171_2 : List Nat := 1663 :: innerTail171_3
def innerCandidates171_2 : List Nat := [4231]
set_option maxRecDepth 4096 in
theorem innerCandidates171_2_checked : candidates edge 4 1033 1663 innerTail171_3 best.bound = innerCandidates171_2 := by
  unfold candidates
  have row : edge 1663 = PrimePairCache.smallRow1663.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan171_2 : scan edge (search edge 3) 4 [1033] 1033 innerTail171_2 best = best := by
  rw [innerTail171_2, scan, if_neg (by decide), innerCandidates171_2_checked]
  change scan edge (search edge 3) 4 [1033] 1033 innerTail171_3
    (if 3 ≤ innerCandidates171_2.length then search edge 3 [1033, 1663] innerCandidates171_2 2696 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan171_3

def innerTail171_1 : List Nat := 1543 :: innerTail171_2
def innerCandidates171_1 : List Nat := [6451]
set_option maxRecDepth 4096 in
theorem innerCandidates171_1_checked : candidates edge 4 1033 1543 innerTail171_2 best.bound = innerCandidates171_1 := by
  unfold candidates
  have row : edge 1543 = PrimePairCache.smallRow1543.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan171_1 : scan edge (search edge 3) 4 [1033] 1033 innerTail171_1 best = best := by
  rw [innerTail171_1, scan, if_neg (by decide), innerCandidates171_1_checked]
  change scan edge (search edge 3) 4 [1033] 1033 innerTail171_2
    (if 3 ≤ innerCandidates171_1.length then search edge 3 [1033, 1543] innerCandidates171_1 2576 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan171_2

def innerTail171_0 : List Nat := 1249 :: innerTail171_1
def innerCandidates171_0 : List Nat := [2131]
set_option maxRecDepth 4096 in
theorem innerCandidates171_0_checked : candidates edge 4 1033 1249 innerTail171_1 best.bound = innerCandidates171_0 := by
  unfold candidates
  have row : edge 1249 = PrimePairCache.smallRow1249.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan171_0 : scan edge (search edge 3) 4 [1033] 1033 innerTail171_0 best = best := by
  rw [innerTail171_0, scan, if_neg (by decide), innerCandidates171_0_checked]
  change scan edge (search edge 3) 4 [1033] 1033 innerTail171_1
    (if 3 ≤ innerCandidates171_0.length then search edge 3 [1033, 1249] innerCandidates171_0 2282 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan171_1

theorem rootBranch171 : search edge 4 [1033] rootCandidates171 1033 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1033] 1033 innerTail171_0 best = best
  exact innerScan171_0

def rootCandidates170 : List Nat := [1481, 1601, 1709, 1811, 1913, 1997, 2207, 2531, 2927, 3533, 4253, 4643, 5003, 5237, 6899, 7193, 7253, 7529]
set_option maxRecDepth 4096 in
theorem rootCandidates170_checked : candidates edge 5 0 1031 tail171 best.bound = rootCandidates170 := by
  unfold candidates
  have row : edge 1031 = PrimePairCache.smallRow1031.contains := by funext q; rfl
  rw [row]
  decide

def innerTail170_18 : List Nat := []
theorem innerScan170_18 : scan edge (search edge 3) 4 [1031] 1031 innerTail170_18 best = best := rfl

def innerTail170_17 : List Nat := 7529 :: innerTail170_18
def innerCandidates170_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates170_17_checked : candidates edge 4 1031 7529 innerTail170_18 best.bound = innerCandidates170_17 := by
  unfold candidates
  have row : edge 7529 = SearchData.row7529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan170_17 : scan edge (search edge 3) 4 [1031] 1031 innerTail170_17 best = best := by
  rw [innerTail170_17, scan, if_neg (by decide), innerCandidates170_17_checked]
  change scan edge (search edge 3) 4 [1031] 1031 innerTail170_18
    (if 3 ≤ innerCandidates170_17.length then search edge 3 [1031, 7529] innerCandidates170_17 8560 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan170_18

def innerTail170_16 : List Nat := 7253 :: innerTail170_17
def innerCandidates170_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates170_16_checked : candidates edge 4 1031 7253 innerTail170_17 best.bound = innerCandidates170_16 := by
  unfold candidates
  have row : edge 7253 = SearchData.row7253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan170_16 : scan edge (search edge 3) 4 [1031] 1031 innerTail170_16 best = best := by
  rw [innerTail170_16, scan, if_neg (by decide), innerCandidates170_16_checked]
  change scan edge (search edge 3) 4 [1031] 1031 innerTail170_17
    (if 3 ≤ innerCandidates170_16.length then search edge 3 [1031, 7253] innerCandidates170_16 8284 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan170_17

def innerTail170_15 : List Nat := 7193 :: innerTail170_16
def innerCandidates170_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates170_15_checked : candidates edge 4 1031 7193 innerTail170_16 best.bound = innerCandidates170_15 := by
  unfold candidates
  have row : edge 7193 = SearchData.row7193.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan170_15 : scan edge (search edge 3) 4 [1031] 1031 innerTail170_15 best = best := by
  rw [innerTail170_15, scan, if_neg (by decide), innerCandidates170_15_checked]
  change scan edge (search edge 3) 4 [1031] 1031 innerTail170_16
    (if 3 ≤ innerCandidates170_15.length then search edge 3 [1031, 7193] innerCandidates170_15 8224 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan170_16

def innerTail170_14 : List Nat := 6899 :: innerTail170_15
def innerCandidates170_14 : List Nat := [7529]
set_option maxRecDepth 4096 in
theorem innerCandidates170_14_checked : candidates edge 4 1031 6899 innerTail170_15 best.bound = innerCandidates170_14 := by
  unfold candidates
  have row : edge 6899 = SearchData.row6899.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan170_14 : scan edge (search edge 3) 4 [1031] 1031 innerTail170_14 best = best := by
  rw [innerTail170_14, scan, if_neg (by decide), innerCandidates170_14_checked]
  change scan edge (search edge 3) 4 [1031] 1031 innerTail170_15
    (if 3 ≤ innerCandidates170_14.length then search edge 3 [1031, 6899] innerCandidates170_14 7930 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan170_15

def innerTail170_13 : List Nat := 5237 :: innerTail170_14
def innerCandidates170_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates170_13_checked : candidates edge 4 1031 5237 innerTail170_14 best.bound = innerCandidates170_13 := by
  unfold candidates
  have row : edge 5237 = SearchData.row5237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan170_13 : scan edge (search edge 3) 4 [1031] 1031 innerTail170_13 best = best := by
  rw [innerTail170_13, scan, if_neg (by decide), innerCandidates170_13_checked]
  change scan edge (search edge 3) 4 [1031] 1031 innerTail170_14
    (if 3 ≤ innerCandidates170_13.length then search edge 3 [1031, 5237] innerCandidates170_13 6268 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan170_14

def innerTail170_12 : List Nat := 5003 :: innerTail170_13
def innerCandidates170_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates170_12_checked : candidates edge 4 1031 5003 innerTail170_13 best.bound = innerCandidates170_12 := by
  unfold candidates
  have row : edge 5003 = SearchData.row5003.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan170_12 : scan edge (search edge 3) 4 [1031] 1031 innerTail170_12 best = best := by
  rw [innerTail170_12, scan, if_neg (by decide), innerCandidates170_12_checked]
  change scan edge (search edge 3) 4 [1031] 1031 innerTail170_13
    (if 3 ≤ innerCandidates170_12.length then search edge 3 [1031, 5003] innerCandidates170_12 6034 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan170_13

def innerTail170_11 : List Nat := 4643 :: innerTail170_12
def innerCandidates170_11 : List Nat := [5237]
set_option maxRecDepth 4096 in
theorem innerCandidates170_11_checked : candidates edge 4 1031 4643 innerTail170_12 best.bound = innerCandidates170_11 := by
  unfold candidates
  have row : edge 4643 = SearchData.row4643.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan170_11 : scan edge (search edge 3) 4 [1031] 1031 innerTail170_11 best = best := by
  rw [innerTail170_11, scan, if_neg (by decide), innerCandidates170_11_checked]
  change scan edge (search edge 3) 4 [1031] 1031 innerTail170_12
    (if 3 ≤ innerCandidates170_11.length then search edge 3 [1031, 4643] innerCandidates170_11 5674 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan170_12

def innerTail170_10 : List Nat := 4253 :: innerTail170_11
def innerCandidates170_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates170_10_checked : candidates edge 4 1031 4253 innerTail170_11 best.bound = innerCandidates170_10 := by
  unfold candidates
  have row : edge 4253 = SearchData.row4253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan170_10 : scan edge (search edge 3) 4 [1031] 1031 innerTail170_10 best = best := by
  rw [innerTail170_10, scan, if_neg (by decide), innerCandidates170_10_checked]
  change scan edge (search edge 3) 4 [1031] 1031 innerTail170_11
    (if 3 ≤ innerCandidates170_10.length then search edge 3 [1031, 4253] innerCandidates170_10 5284 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan170_11

def innerTail170_9 : List Nat := 3533 :: innerTail170_10
def innerCandidates170_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates170_9_checked : candidates edge 4 1031 3533 innerTail170_10 best.bound = innerCandidates170_9 := by
  unfold candidates
  have row : edge 3533 = SearchData.row3533.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan170_9 : scan edge (search edge 3) 4 [1031] 1031 innerTail170_9 best = best := by
  rw [innerTail170_9, scan, if_neg (by decide), innerCandidates170_9_checked]
  change scan edge (search edge 3) 4 [1031] 1031 innerTail170_10
    (if 3 ≤ innerCandidates170_9.length then search edge 3 [1031, 3533] innerCandidates170_9 4564 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan170_10

def innerTail170_8 : List Nat := 2927 :: innerTail170_9
def innerCandidates170_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates170_8_checked : candidates edge 4 1031 2927 innerTail170_9 best.bound = innerCandidates170_8 := by
  unfold candidates
  have row : edge 2927 = SearchData.row2927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan170_8 : scan edge (search edge 3) 4 [1031] 1031 innerTail170_8 best = best := by
  rw [innerTail170_8, scan, if_neg (by decide), innerCandidates170_8_checked]
  change scan edge (search edge 3) 4 [1031] 1031 innerTail170_9
    (if 3 ≤ innerCandidates170_8.length then search edge 3 [1031, 2927] innerCandidates170_8 3958 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan170_9

def innerTail170_7 : List Nat := 2531 :: innerTail170_8
def innerCandidates170_7 : List Nat := [4643]
set_option maxRecDepth 4096 in
theorem innerCandidates170_7_checked : candidates edge 4 1031 2531 innerTail170_8 best.bound = innerCandidates170_7 := by
  unfold candidates
  have row : edge 2531 = PrimePairCache.smallRow2531.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan170_7 : scan edge (search edge 3) 4 [1031] 1031 innerTail170_7 best = best := by
  rw [innerTail170_7, scan, if_neg (by decide), innerCandidates170_7_checked]
  change scan edge (search edge 3) 4 [1031] 1031 innerTail170_8
    (if 3 ≤ innerCandidates170_7.length then search edge 3 [1031, 2531] innerCandidates170_7 3562 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan170_8

def innerTail170_6 : List Nat := 2207 :: innerTail170_7
def innerCandidates170_6 : List Nat := [7193]
set_option maxRecDepth 4096 in
theorem innerCandidates170_6_checked : candidates edge 4 1031 2207 innerTail170_7 best.bound = innerCandidates170_6 := by
  unfold candidates
  have row : edge 2207 = PrimePairCache.smallRow2207.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan170_6 : scan edge (search edge 3) 4 [1031] 1031 innerTail170_6 best = best := by
  rw [innerTail170_6, scan, if_neg (by decide), innerCandidates170_6_checked]
  change scan edge (search edge 3) 4 [1031] 1031 innerTail170_7
    (if 3 ≤ innerCandidates170_6.length then search edge 3 [1031, 2207] innerCandidates170_6 3238 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan170_7

def innerTail170_5 : List Nat := 1997 :: innerTail170_6
def innerCandidates170_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates170_5_checked : candidates edge 4 1031 1997 innerTail170_6 best.bound = innerCandidates170_5 := by
  unfold candidates
  have row : edge 1997 = PrimePairCache.smallRow1997.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan170_5 : scan edge (search edge 3) 4 [1031] 1031 innerTail170_5 best = best := by
  rw [innerTail170_5, scan, if_neg (by decide), innerCandidates170_5_checked]
  change scan edge (search edge 3) 4 [1031] 1031 innerTail170_6
    (if 3 ≤ innerCandidates170_5.length then search edge 3 [1031, 1997] innerCandidates170_5 3028 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan170_6

def innerTail170_4 : List Nat := 1913 :: innerTail170_5
def innerCandidates170_4 : List Nat := [7253]
set_option maxRecDepth 4096 in
theorem innerCandidates170_4_checked : candidates edge 4 1031 1913 innerTail170_5 best.bound = innerCandidates170_4 := by
  unfold candidates
  have row : edge 1913 = PrimePairCache.smallRow1913.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan170_4 : scan edge (search edge 3) 4 [1031] 1031 innerTail170_4 best = best := by
  rw [innerTail170_4, scan, if_neg (by decide), innerCandidates170_4_checked]
  change scan edge (search edge 3) 4 [1031] 1031 innerTail170_5
    (if 3 ≤ innerCandidates170_4.length then search edge 3 [1031, 1913] innerCandidates170_4 2944 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan170_5

def innerTail170_3 : List Nat := 1811 :: innerTail170_4
def innerCandidates170_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates170_3_checked : candidates edge 4 1031 1811 innerTail170_4 best.bound = innerCandidates170_3 := by
  unfold candidates
  have row : edge 1811 = PrimePairCache.smallRow1811.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan170_3 : scan edge (search edge 3) 4 [1031] 1031 innerTail170_3 best = best := by
  rw [innerTail170_3, scan, if_neg (by decide), innerCandidates170_3_checked]
  change scan edge (search edge 3) 4 [1031] 1031 innerTail170_4
    (if 3 ≤ innerCandidates170_3.length then search edge 3 [1031, 1811] innerCandidates170_3 2842 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan170_4

def innerTail170_2 : List Nat := 1709 :: innerTail170_3
def innerCandidates170_2 : List Nat := [1913]
set_option maxRecDepth 4096 in
theorem innerCandidates170_2_checked : candidates edge 4 1031 1709 innerTail170_3 best.bound = innerCandidates170_2 := by
  unfold candidates
  have row : edge 1709 = PrimePairCache.smallRow1709.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan170_2 : scan edge (search edge 3) 4 [1031] 1031 innerTail170_2 best = best := by
  rw [innerTail170_2, scan, if_neg (by decide), innerCandidates170_2_checked]
  change scan edge (search edge 3) 4 [1031] 1031 innerTail170_3
    (if 3 ≤ innerCandidates170_2.length then search edge 3 [1031, 1709] innerCandidates170_2 2740 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan170_3

def innerTail170_1 : List Nat := 1601 :: innerTail170_2
def innerCandidates170_1 : List Nat := [1811, 2207, 3533]
set_option maxRecDepth 4096 in
theorem innerCandidates170_1_checked : candidates edge 4 1031 1601 innerTail170_2 best.bound = innerCandidates170_1 := by
  unfold candidates
  have row : edge 1601 = PrimePairCache.smallRow1601.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild170_1 : search edge 3 [1031, 1601] innerCandidates170_1 2632 best = best := by decide

theorem innerScan170_1 : scan edge (search edge 3) 4 [1031] 1031 innerTail170_1 best = best := by
  rw [innerTail170_1, scan, if_neg (by decide), innerCandidates170_1_checked]
  change scan edge (search edge 3) 4 [1031] 1031 innerTail170_2
    (if 3 ≤ innerCandidates170_1.length then search edge 3 [1031, 1601] innerCandidates170_1 2632 best else best) = best
  rw [if_pos (by decide), innerChild170_1]
  exact innerScan170_2

def innerTail170_0 : List Nat := 1481 :: innerTail170_1
def innerCandidates170_0 : List Nat := [7529]
set_option maxRecDepth 4096 in
theorem innerCandidates170_0_checked : candidates edge 4 1031 1481 innerTail170_1 best.bound = innerCandidates170_0 := by
  unfold candidates
  have row : edge 1481 = PrimePairCache.smallRow1481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan170_0 : scan edge (search edge 3) 4 [1031] 1031 innerTail170_0 best = best := by
  rw [innerTail170_0, scan, if_neg (by decide), innerCandidates170_0_checked]
  change scan edge (search edge 3) 4 [1031] 1031 innerTail170_1
    (if 3 ≤ innerCandidates170_0.length then search edge 3 [1031, 1481] innerCandidates170_0 2512 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan170_1

theorem rootBranch170 : search edge 4 [1031] rootCandidates170 1031 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1031] 1031 innerTail170_0 best = best
  exact innerScan170_0

end ProjectEulerSolutions.PrimePairUpper.Search
