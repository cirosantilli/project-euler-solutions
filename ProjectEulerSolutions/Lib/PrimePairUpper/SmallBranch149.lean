import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates229 : List Nat := [1567, 1657, 1861, 1873, 2089, 3511, 3673, 4813, 5011, 5233, 5437, 6079, 6271, 6991, 7027, 7717, 7963]
set_option maxRecDepth 4096 in
theorem rootCandidates229_checked : candidates edge 5 0 1459 tail230 best.bound = rootCandidates229 := by
  unfold candidates
  have row : edge 1459 = PrimePairCache.smallRow1459.contains := by funext q; rfl
  rw [row]
  decide

def innerTail229_17 : List Nat := []
theorem innerScan229_17 : scan edge (search edge 3) 4 [1459] 1459 innerTail229_17 best = best := rfl

def innerTail229_16 : List Nat := 7963 :: innerTail229_17
def innerCandidates229_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates229_16_checked : candidates edge 4 1459 7963 innerTail229_17 best.bound = innerCandidates229_16 := by
  unfold candidates
  have row : edge 7963 = SearchData.row7963.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan229_16 : scan edge (search edge 3) 4 [1459] 1459 innerTail229_16 best = best := by
  rw [innerTail229_16, scan, if_neg (by decide), innerCandidates229_16_checked]
  change scan edge (search edge 3) 4 [1459] 1459 innerTail229_17
    (if 3 ≤ innerCandidates229_16.length then search edge 3 [1459, 7963] innerCandidates229_16 9422 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan229_17

def innerTail229_15 : List Nat := 7717 :: innerTail229_16
def innerCandidates229_15 : List Nat := [7963]
set_option maxRecDepth 4096 in
theorem innerCandidates229_15_checked : candidates edge 4 1459 7717 innerTail229_16 best.bound = innerCandidates229_15 := by
  unfold candidates
  have row : edge 7717 = SearchData.row7717.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan229_15 : scan edge (search edge 3) 4 [1459] 1459 innerTail229_15 best = best := by
  rw [innerTail229_15, scan, if_neg (by decide), innerCandidates229_15_checked]
  change scan edge (search edge 3) 4 [1459] 1459 innerTail229_16
    (if 3 ≤ innerCandidates229_15.length then search edge 3 [1459, 7717] innerCandidates229_15 9176 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan229_16

def innerTail229_14 : List Nat := 7027 :: innerTail229_15
def innerCandidates229_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates229_14_checked : candidates edge 4 1459 7027 innerTail229_15 best.bound = innerCandidates229_14 := by
  unfold candidates
  have row : edge 7027 = SearchData.row7027.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan229_14 : scan edge (search edge 3) 4 [1459] 1459 innerTail229_14 best = best := by
  rw [innerTail229_14, scan, if_neg (by decide), innerCandidates229_14_checked]
  change scan edge (search edge 3) 4 [1459] 1459 innerTail229_15
    (if 3 ≤ innerCandidates229_14.length then search edge 3 [1459, 7027] innerCandidates229_14 8486 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan229_15

def innerTail229_13 : List Nat := 6991 :: innerTail229_14
def innerCandidates229_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates229_13_checked : candidates edge 4 1459 6991 innerTail229_14 best.bound = innerCandidates229_13 := by
  unfold candidates
  have row : edge 6991 = SearchData.row6991.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan229_13 : scan edge (search edge 3) 4 [1459] 1459 innerTail229_13 best = best := by
  rw [innerTail229_13, scan, if_neg (by decide), innerCandidates229_13_checked]
  change scan edge (search edge 3) 4 [1459] 1459 innerTail229_14
    (if 3 ≤ innerCandidates229_13.length then search edge 3 [1459, 6991] innerCandidates229_13 8450 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan229_14

def innerTail229_12 : List Nat := 6271 :: innerTail229_13
def innerCandidates229_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates229_12_checked : candidates edge 4 1459 6271 innerTail229_13 best.bound = innerCandidates229_12 := by
  unfold candidates
  have row : edge 6271 = SearchData.row6271.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan229_12 : scan edge (search edge 3) 4 [1459] 1459 innerTail229_12 best = best := by
  rw [innerTail229_12, scan, if_neg (by decide), innerCandidates229_12_checked]
  change scan edge (search edge 3) 4 [1459] 1459 innerTail229_13
    (if 3 ≤ innerCandidates229_12.length then search edge 3 [1459, 6271] innerCandidates229_12 7730 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan229_13

def innerTail229_11 : List Nat := 6079 :: innerTail229_12
def innerCandidates229_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates229_11_checked : candidates edge 4 1459 6079 innerTail229_12 best.bound = innerCandidates229_11 := by
  unfold candidates
  have row : edge 6079 = SearchData.row6079.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan229_11 : scan edge (search edge 3) 4 [1459] 1459 innerTail229_11 best = best := by
  rw [innerTail229_11, scan, if_neg (by decide), innerCandidates229_11_checked]
  change scan edge (search edge 3) 4 [1459] 1459 innerTail229_12
    (if 3 ≤ innerCandidates229_11.length then search edge 3 [1459, 6079] innerCandidates229_11 7538 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan229_12

def innerTail229_10 : List Nat := 5437 :: innerTail229_11
def innerCandidates229_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates229_10_checked : candidates edge 4 1459 5437 innerTail229_11 best.bound = innerCandidates229_10 := by
  unfold candidates
  have row : edge 5437 = SearchData.row5437.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan229_10 : scan edge (search edge 3) 4 [1459] 1459 innerTail229_10 best = best := by
  rw [innerTail229_10, scan, if_neg (by decide), innerCandidates229_10_checked]
  change scan edge (search edge 3) 4 [1459] 1459 innerTail229_11
    (if 3 ≤ innerCandidates229_10.length then search edge 3 [1459, 5437] innerCandidates229_10 6896 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan229_11

def innerTail229_9 : List Nat := 5233 :: innerTail229_10
def innerCandidates229_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates229_9_checked : candidates edge 4 1459 5233 innerTail229_10 best.bound = innerCandidates229_9 := by
  unfold candidates
  have row : edge 5233 = SearchData.row5233.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan229_9 : scan edge (search edge 3) 4 [1459] 1459 innerTail229_9 best = best := by
  rw [innerTail229_9, scan, if_neg (by decide), innerCandidates229_9_checked]
  change scan edge (search edge 3) 4 [1459] 1459 innerTail229_10
    (if 3 ≤ innerCandidates229_9.length then search edge 3 [1459, 5233] innerCandidates229_9 6692 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan229_10

def innerTail229_8 : List Nat := 5011 :: innerTail229_9
def innerCandidates229_8 : List Nat := [6271]
set_option maxRecDepth 4096 in
theorem innerCandidates229_8_checked : candidates edge 4 1459 5011 innerTail229_9 best.bound = innerCandidates229_8 := by
  unfold candidates
  have row : edge 5011 = SearchData.row5011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan229_8 : scan edge (search edge 3) 4 [1459] 1459 innerTail229_8 best = best := by
  rw [innerTail229_8, scan, if_neg (by decide), innerCandidates229_8_checked]
  change scan edge (search edge 3) 4 [1459] 1459 innerTail229_9
    (if 3 ≤ innerCandidates229_8.length then search edge 3 [1459, 5011] innerCandidates229_8 6470 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan229_9

def innerTail229_7 : List Nat := 4813 :: innerTail229_8
def innerCandidates229_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates229_7_checked : candidates edge 4 1459 4813 innerTail229_8 best.bound = innerCandidates229_7 := by
  unfold candidates
  have row : edge 4813 = SearchData.row4813.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan229_7 : scan edge (search edge 3) 4 [1459] 1459 innerTail229_7 best = best := by
  rw [innerTail229_7, scan, if_neg (by decide), innerCandidates229_7_checked]
  change scan edge (search edge 3) 4 [1459] 1459 innerTail229_8
    (if 3 ≤ innerCandidates229_7.length then search edge 3 [1459, 4813] innerCandidates229_7 6272 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan229_8

def innerTail229_6 : List Nat := 3673 :: innerTail229_7
def innerCandidates229_6 : List Nat := [7717]
set_option maxRecDepth 4096 in
theorem innerCandidates229_6_checked : candidates edge 4 1459 3673 innerTail229_7 best.bound = innerCandidates229_6 := by
  unfold candidates
  have row : edge 3673 = SearchData.row3673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan229_6 : scan edge (search edge 3) 4 [1459] 1459 innerTail229_6 best = best := by
  rw [innerTail229_6, scan, if_neg (by decide), innerCandidates229_6_checked]
  change scan edge (search edge 3) 4 [1459] 1459 innerTail229_7
    (if 3 ≤ innerCandidates229_6.length then search edge 3 [1459, 3673] innerCandidates229_6 5132 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan229_7

def innerTail229_5 : List Nat := 3511 :: innerTail229_6
def innerCandidates229_5 : List Nat := [5233, 7963]
set_option maxRecDepth 4096 in
theorem innerCandidates229_5_checked : candidates edge 4 1459 3511 innerTail229_6 best.bound = innerCandidates229_5 := by
  unfold candidates
  have row : edge 3511 = SearchData.row3511.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan229_5 : scan edge (search edge 3) 4 [1459] 1459 innerTail229_5 best = best := by
  rw [innerTail229_5, scan, if_neg (by decide), innerCandidates229_5_checked]
  change scan edge (search edge 3) 4 [1459] 1459 innerTail229_6
    (if 3 ≤ innerCandidates229_5.length then search edge 3 [1459, 3511] innerCandidates229_5 4970 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan229_6

def innerTail229_4 : List Nat := 2089 :: innerTail229_5
def innerCandidates229_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates229_4_checked : candidates edge 4 1459 2089 innerTail229_5 best.bound = innerCandidates229_4 := by
  unfold candidates
  have row : edge 2089 = PrimePairCache.smallRow2089.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan229_4 : scan edge (search edge 3) 4 [1459] 1459 innerTail229_4 best = best := by
  rw [innerTail229_4, scan, if_neg (by decide), innerCandidates229_4_checked]
  change scan edge (search edge 3) 4 [1459] 1459 innerTail229_5
    (if 3 ≤ innerCandidates229_4.length then search edge 3 [1459, 2089] innerCandidates229_4 3548 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan229_5

def innerTail229_3 : List Nat := 1873 :: innerTail229_4
def innerCandidates229_3 : List Nat := [3511, 7963]
set_option maxRecDepth 4096 in
theorem innerCandidates229_3_checked : candidates edge 4 1459 1873 innerTail229_4 best.bound = innerCandidates229_3 := by
  unfold candidates
  have row : edge 1873 = PrimePairCache.smallRow1873.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan229_3 : scan edge (search edge 3) 4 [1459] 1459 innerTail229_3 best = best := by
  rw [innerTail229_3, scan, if_neg (by decide), innerCandidates229_3_checked]
  change scan edge (search edge 3) 4 [1459] 1459 innerTail229_4
    (if 3 ≤ innerCandidates229_3.length then search edge 3 [1459, 1873] innerCandidates229_3 3332 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan229_4

def innerTail229_2 : List Nat := 1861 :: innerTail229_3
def innerCandidates229_2 : List Nat := [5011]
set_option maxRecDepth 4096 in
theorem innerCandidates229_2_checked : candidates edge 4 1459 1861 innerTail229_3 best.bound = innerCandidates229_2 := by
  unfold candidates
  have row : edge 1861 = PrimePairCache.smallRow1861.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan229_2 : scan edge (search edge 3) 4 [1459] 1459 innerTail229_2 best = best := by
  rw [innerTail229_2, scan, if_neg (by decide), innerCandidates229_2_checked]
  change scan edge (search edge 3) 4 [1459] 1459 innerTail229_3
    (if 3 ≤ innerCandidates229_2.length then search edge 3 [1459, 1861] innerCandidates229_2 3320 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan229_3

def innerTail229_1 : List Nat := 1657 :: innerTail229_2
def innerCandidates229_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates229_1_checked : candidates edge 4 1459 1657 innerTail229_2 best.bound = innerCandidates229_1 := by
  unfold candidates
  have row : edge 1657 = PrimePairCache.smallRow1657.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan229_1 : scan edge (search edge 3) 4 [1459] 1459 innerTail229_1 best = best := by
  rw [innerTail229_1, scan, if_neg (by decide), innerCandidates229_1_checked]
  change scan edge (search edge 3) 4 [1459] 1459 innerTail229_2
    (if 3 ≤ innerCandidates229_1.length then search edge 3 [1459, 1657] innerCandidates229_1 3116 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan229_2

def innerTail229_0 : List Nat := 1567 :: innerTail229_1
def innerCandidates229_0 : List Nat := [7027]
set_option maxRecDepth 4096 in
theorem innerCandidates229_0_checked : candidates edge 4 1459 1567 innerTail229_1 best.bound = innerCandidates229_0 := by
  unfold candidates
  have row : edge 1567 = PrimePairCache.smallRow1567.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan229_0 : scan edge (search edge 3) 4 [1459] 1459 innerTail229_0 best = best := by
  rw [innerTail229_0, scan, if_neg (by decide), innerCandidates229_0_checked]
  change scan edge (search edge 3) 4 [1459] 1459 innerTail229_1
    (if 3 ≤ innerCandidates229_0.length then search edge 3 [1459, 1567] innerCandidates229_0 3026 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan229_1

theorem rootBranch229 : search edge 4 [1459] rootCandidates229 1459 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1459] 1459 innerTail229_0 best = best
  exact innerScan229_0

def rootCandidates228 : List Nat := [1669, 1879, 2437, 2593, 2797, 3049, 3361, 4483, 4651, 4783, 5113, 5281, 5581, 5743, 5923, 6763, 7129, 7417, 7927, 7933, 8089]
set_option maxRecDepth 4096 in
theorem rootCandidates228_checked : candidates edge 5 0 1453 tail229 best.bound = rootCandidates228 := by
  unfold candidates
  have row : edge 1453 = PrimePairCache.smallRow1453.contains := by funext q; rfl
  rw [row]
  decide

def innerTail228_21 : List Nat := []
theorem innerScan228_21 : scan edge (search edge 3) 4 [1453] 1453 innerTail228_21 best = best := rfl

def innerTail228_20 : List Nat := 8089 :: innerTail228_21
def innerCandidates228_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates228_20_checked : candidates edge 4 1453 8089 innerTail228_21 best.bound = innerCandidates228_20 := by
  unfold candidates
  have row : edge 8089 = SearchData.row8089.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan228_20 : scan edge (search edge 3) 4 [1453] 1453 innerTail228_20 best = best := by
  rw [innerTail228_20, scan, if_neg (by decide), innerCandidates228_20_checked]
  change scan edge (search edge 3) 4 [1453] 1453 innerTail228_21
    (if 3 ≤ innerCandidates228_20.length then search edge 3 [1453, 8089] innerCandidates228_20 9542 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan228_21

def innerTail228_19 : List Nat := 7933 :: innerTail228_20
def innerCandidates228_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates228_19_checked : candidates edge 4 1453 7933 innerTail228_20 best.bound = innerCandidates228_19 := by
  unfold candidates
  have row : edge 7933 = SearchData.row7933.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan228_19 : scan edge (search edge 3) 4 [1453] 1453 innerTail228_19 best = best := by
  rw [innerTail228_19, scan, if_neg (by decide), innerCandidates228_19_checked]
  change scan edge (search edge 3) 4 [1453] 1453 innerTail228_20
    (if 3 ≤ innerCandidates228_19.length then search edge 3 [1453, 7933] innerCandidates228_19 9386 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan228_20

def innerTail228_18 : List Nat := 7927 :: innerTail228_19
def innerCandidates228_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates228_18_checked : candidates edge 4 1453 7927 innerTail228_19 best.bound = innerCandidates228_18 := by
  unfold candidates
  have row : edge 7927 = SearchData.row7927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan228_18 : scan edge (search edge 3) 4 [1453] 1453 innerTail228_18 best = best := by
  rw [innerTail228_18, scan, if_neg (by decide), innerCandidates228_18_checked]
  change scan edge (search edge 3) 4 [1453] 1453 innerTail228_19
    (if 3 ≤ innerCandidates228_18.length then search edge 3 [1453, 7927] innerCandidates228_18 9380 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan228_19

def innerTail228_17 : List Nat := 7417 :: innerTail228_18
def innerCandidates228_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates228_17_checked : candidates edge 4 1453 7417 innerTail228_18 best.bound = innerCandidates228_17 := by
  unfold candidates
  have row : edge 7417 = SearchData.row7417.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan228_17 : scan edge (search edge 3) 4 [1453] 1453 innerTail228_17 best = best := by
  rw [innerTail228_17, scan, if_neg (by decide), innerCandidates228_17_checked]
  change scan edge (search edge 3) 4 [1453] 1453 innerTail228_18
    (if 3 ≤ innerCandidates228_17.length then search edge 3 [1453, 7417] innerCandidates228_17 8870 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan228_18

def innerTail228_16 : List Nat := 7129 :: innerTail228_17
def innerCandidates228_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates228_16_checked : candidates edge 4 1453 7129 innerTail228_17 best.bound = innerCandidates228_16 := by
  unfold candidates
  have row : edge 7129 = SearchData.row7129.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan228_16 : scan edge (search edge 3) 4 [1453] 1453 innerTail228_16 best = best := by
  rw [innerTail228_16, scan, if_neg (by decide), innerCandidates228_16_checked]
  change scan edge (search edge 3) 4 [1453] 1453 innerTail228_17
    (if 3 ≤ innerCandidates228_16.length then search edge 3 [1453, 7129] innerCandidates228_16 8582 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan228_17

def innerTail228_15 : List Nat := 6763 :: innerTail228_16
def innerCandidates228_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates228_15_checked : candidates edge 4 1453 6763 innerTail228_16 best.bound = innerCandidates228_15 := by
  unfold candidates
  have row : edge 6763 = SearchData.row6763.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan228_15 : scan edge (search edge 3) 4 [1453] 1453 innerTail228_15 best = best := by
  rw [innerTail228_15, scan, if_neg (by decide), innerCandidates228_15_checked]
  change scan edge (search edge 3) 4 [1453] 1453 innerTail228_16
    (if 3 ≤ innerCandidates228_15.length then search edge 3 [1453, 6763] innerCandidates228_15 8216 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan228_16

def innerTail228_14 : List Nat := 5923 :: innerTail228_15
def innerCandidates228_14 : List Nat := [7417]
set_option maxRecDepth 4096 in
theorem innerCandidates228_14_checked : candidates edge 4 1453 5923 innerTail228_15 best.bound = innerCandidates228_14 := by
  unfold candidates
  have row : edge 5923 = SearchData.row5923.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan228_14 : scan edge (search edge 3) 4 [1453] 1453 innerTail228_14 best = best := by
  rw [innerTail228_14, scan, if_neg (by decide), innerCandidates228_14_checked]
  change scan edge (search edge 3) 4 [1453] 1453 innerTail228_15
    (if 3 ≤ innerCandidates228_14.length then search edge 3 [1453, 5923] innerCandidates228_14 7376 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan228_15

def innerTail228_13 : List Nat := 5743 :: innerTail228_14
def innerCandidates228_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates228_13_checked : candidates edge 4 1453 5743 innerTail228_14 best.bound = innerCandidates228_13 := by
  unfold candidates
  have row : edge 5743 = SearchData.row5743.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan228_13 : scan edge (search edge 3) 4 [1453] 1453 innerTail228_13 best = best := by
  rw [innerTail228_13, scan, if_neg (by decide), innerCandidates228_13_checked]
  change scan edge (search edge 3) 4 [1453] 1453 innerTail228_14
    (if 3 ≤ innerCandidates228_13.length then search edge 3 [1453, 5743] innerCandidates228_13 7196 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan228_14

def innerTail228_12 : List Nat := 5581 :: innerTail228_13
def innerCandidates228_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates228_12_checked : candidates edge 4 1453 5581 innerTail228_13 best.bound = innerCandidates228_12 := by
  unfold candidates
  have row : edge 5581 = SearchData.row5581.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan228_12 : scan edge (search edge 3) 4 [1453] 1453 innerTail228_12 best = best := by
  rw [innerTail228_12, scan, if_neg (by decide), innerCandidates228_12_checked]
  change scan edge (search edge 3) 4 [1453] 1453 innerTail228_13
    (if 3 ≤ innerCandidates228_12.length then search edge 3 [1453, 5581] innerCandidates228_12 7034 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan228_13

def innerTail228_11 : List Nat := 5281 :: innerTail228_12
def innerCandidates228_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates228_11_checked : candidates edge 4 1453 5281 innerTail228_12 best.bound = innerCandidates228_11 := by
  unfold candidates
  have row : edge 5281 = SearchData.row5281.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan228_11 : scan edge (search edge 3) 4 [1453] 1453 innerTail228_11 best = best := by
  rw [innerTail228_11, scan, if_neg (by decide), innerCandidates228_11_checked]
  change scan edge (search edge 3) 4 [1453] 1453 innerTail228_12
    (if 3 ≤ innerCandidates228_11.length then search edge 3 [1453, 5281] innerCandidates228_11 6734 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan228_12

def innerTail228_10 : List Nat := 5113 :: innerTail228_11
def innerCandidates228_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates228_10_checked : candidates edge 4 1453 5113 innerTail228_11 best.bound = innerCandidates228_10 := by
  unfold candidates
  have row : edge 5113 = SearchData.row5113.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan228_10 : scan edge (search edge 3) 4 [1453] 1453 innerTail228_10 best = best := by
  rw [innerTail228_10, scan, if_neg (by decide), innerCandidates228_10_checked]
  change scan edge (search edge 3) 4 [1453] 1453 innerTail228_11
    (if 3 ≤ innerCandidates228_10.length then search edge 3 [1453, 5113] innerCandidates228_10 6566 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan228_11

def innerTail228_9 : List Nat := 4783 :: innerTail228_10
def innerCandidates228_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates228_9_checked : candidates edge 4 1453 4783 innerTail228_10 best.bound = innerCandidates228_9 := by
  unfold candidates
  have row : edge 4783 = SearchData.row4783.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan228_9 : scan edge (search edge 3) 4 [1453] 1453 innerTail228_9 best = best := by
  rw [innerTail228_9, scan, if_neg (by decide), innerCandidates228_9_checked]
  change scan edge (search edge 3) 4 [1453] 1453 innerTail228_10
    (if 3 ≤ innerCandidates228_9.length then search edge 3 [1453, 4783] innerCandidates228_9 6236 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan228_10

def innerTail228_8 : List Nat := 4651 :: innerTail228_9
def innerCandidates228_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates228_8_checked : candidates edge 4 1453 4651 innerTail228_9 best.bound = innerCandidates228_8 := by
  unfold candidates
  have row : edge 4651 = SearchData.row4651.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan228_8 : scan edge (search edge 3) 4 [1453] 1453 innerTail228_8 best = best := by
  rw [innerTail228_8, scan, if_neg (by decide), innerCandidates228_8_checked]
  change scan edge (search edge 3) 4 [1453] 1453 innerTail228_9
    (if 3 ≤ innerCandidates228_8.length then search edge 3 [1453, 4651] innerCandidates228_8 6104 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan228_9

def innerTail228_7 : List Nat := 4483 :: innerTail228_8
def innerCandidates228_7 : List Nat := [5113]
set_option maxRecDepth 4096 in
theorem innerCandidates228_7_checked : candidates edge 4 1453 4483 innerTail228_8 best.bound = innerCandidates228_7 := by
  unfold candidates
  have row : edge 4483 = SearchData.row4483.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan228_7 : scan edge (search edge 3) 4 [1453] 1453 innerTail228_7 best = best := by
  rw [innerTail228_7, scan, if_neg (by decide), innerCandidates228_7_checked]
  change scan edge (search edge 3) 4 [1453] 1453 innerTail228_8
    (if 3 ≤ innerCandidates228_7.length then search edge 3 [1453, 4483] innerCandidates228_7 5936 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan228_8

def innerTail228_6 : List Nat := 3361 :: innerTail228_7
def innerCandidates228_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates228_6_checked : candidates edge 4 1453 3361 innerTail228_7 best.bound = innerCandidates228_6 := by
  unfold candidates
  have row : edge 3361 = SearchData.row3361.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan228_6 : scan edge (search edge 3) 4 [1453] 1453 innerTail228_6 best = best := by
  rw [innerTail228_6, scan, if_neg (by decide), innerCandidates228_6_checked]
  change scan edge (search edge 3) 4 [1453] 1453 innerTail228_7
    (if 3 ≤ innerCandidates228_6.length then search edge 3 [1453, 3361] innerCandidates228_6 4814 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan228_7

def innerTail228_5 : List Nat := 3049 :: innerTail228_6
def innerCandidates228_5 : List Nat := [5581, 8089]
set_option maxRecDepth 4096 in
theorem innerCandidates228_5_checked : candidates edge 4 1453 3049 innerTail228_6 best.bound = innerCandidates228_5 := by
  unfold candidates
  have row : edge 3049 = SearchData.row3049.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan228_5 : scan edge (search edge 3) 4 [1453] 1453 innerTail228_5 best = best := by
  rw [innerTail228_5, scan, if_neg (by decide), innerCandidates228_5_checked]
  change scan edge (search edge 3) 4 [1453] 1453 innerTail228_6
    (if 3 ≤ innerCandidates228_5.length then search edge 3 [1453, 3049] innerCandidates228_5 4502 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan228_6

def innerTail228_4 : List Nat := 2797 :: innerTail228_5
def innerCandidates228_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates228_4_checked : candidates edge 4 1453 2797 innerTail228_5 best.bound = innerCandidates228_4 := by
  unfold candidates
  have row : edge 2797 = SearchData.row2797.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan228_4 : scan edge (search edge 3) 4 [1453] 1453 innerTail228_4 best = best := by
  rw [innerTail228_4, scan, if_neg (by decide), innerCandidates228_4_checked]
  change scan edge (search edge 3) 4 [1453] 1453 innerTail228_5
    (if 3 ≤ innerCandidates228_4.length then search edge 3 [1453, 2797] innerCandidates228_4 4250 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan228_5

def innerTail228_3 : List Nat := 2593 :: innerTail228_4
def innerCandidates228_3 : List Nat := [5281, 7927]
set_option maxRecDepth 4096 in
theorem innerCandidates228_3_checked : candidates edge 4 1453 2593 innerTail228_4 best.bound = innerCandidates228_3 := by
  unfold candidates
  have row : edge 2593 = PrimePairCache.smallRow2593.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan228_3 : scan edge (search edge 3) 4 [1453] 1453 innerTail228_3 best = best := by
  rw [innerTail228_3, scan, if_neg (by decide), innerCandidates228_3_checked]
  change scan edge (search edge 3) 4 [1453] 1453 innerTail228_4
    (if 3 ≤ innerCandidates228_3.length then search edge 3 [1453, 2593] innerCandidates228_3 4046 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan228_4

def innerTail228_2 : List Nat := 2437 :: innerTail228_3
def innerCandidates228_2 : List Nat := [3361]
set_option maxRecDepth 4096 in
theorem innerCandidates228_2_checked : candidates edge 4 1453 2437 innerTail228_3 best.bound = innerCandidates228_2 := by
  unfold candidates
  have row : edge 2437 = PrimePairCache.smallRow2437.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan228_2 : scan edge (search edge 3) 4 [1453] 1453 innerTail228_2 best = best := by
  rw [innerTail228_2, scan, if_neg (by decide), innerCandidates228_2_checked]
  change scan edge (search edge 3) 4 [1453] 1453 innerTail228_3
    (if 3 ≤ innerCandidates228_2.length then search edge 3 [1453, 2437] innerCandidates228_2 3890 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan228_3

def innerTail228_1 : List Nat := 1879 :: innerTail228_2
def innerCandidates228_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates228_1_checked : candidates edge 4 1453 1879 innerTail228_2 best.bound = innerCandidates228_1 := by
  unfold candidates
  have row : edge 1879 = PrimePairCache.smallRow1879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan228_1 : scan edge (search edge 3) 4 [1453] 1453 innerTail228_1 best = best := by
  rw [innerTail228_1, scan, if_neg (by decide), innerCandidates228_1_checked]
  change scan edge (search edge 3) 4 [1453] 1453 innerTail228_2
    (if 3 ≤ innerCandidates228_1.length then search edge 3 [1453, 1879] innerCandidates228_1 3332 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan228_2

def innerTail228_0 : List Nat := 1669 :: innerTail228_1
def innerCandidates228_0 : List Nat := [2593, 4651, 5743]
set_option maxRecDepth 4096 in
theorem innerCandidates228_0_checked : candidates edge 4 1453 1669 innerTail228_1 best.bound = innerCandidates228_0 := by
  unfold candidates
  have row : edge 1669 = PrimePairCache.smallRow1669.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild228_0 : search edge 3 [1453, 1669] innerCandidates228_0 3122 best = best := by decide

theorem innerScan228_0 : scan edge (search edge 3) 4 [1453] 1453 innerTail228_0 best = best := by
  rw [innerTail228_0, scan, if_neg (by decide), innerCandidates228_0_checked]
  change scan edge (search edge 3) 4 [1453] 1453 innerTail228_1
    (if 3 ≤ innerCandidates228_0.length then search edge 3 [1453, 1669] innerCandidates228_0 3122 best else best) = best
  rw [if_pos (by decide), innerChild228_0]
  exact innerScan228_1

theorem rootBranch228 : search edge 4 [1453] rootCandidates228 1453 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1453] 1453 innerTail228_0 best = best
  exact innerScan228_0

def rootCandidates227 : List Nat := [1481, 2039, 2699, 3413, 3557, 3761, 4373, 4457, 5147, 6581, 6659, 6869, 7001, 7193, 7499, 7823, 7883, 7907]
set_option maxRecDepth 4096 in
theorem rootCandidates227_checked : candidates edge 5 0 1451 tail228 best.bound = rootCandidates227 := by
  unfold candidates
  have row : edge 1451 = PrimePairCache.smallRow1451.contains := by funext q; rfl
  rw [row]
  decide

def innerTail227_18 : List Nat := []
theorem innerScan227_18 : scan edge (search edge 3) 4 [1451] 1451 innerTail227_18 best = best := rfl

def innerTail227_17 : List Nat := 7907 :: innerTail227_18
def innerCandidates227_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates227_17_checked : candidates edge 4 1451 7907 innerTail227_18 best.bound = innerCandidates227_17 := by
  unfold candidates
  have row : edge 7907 = SearchData.row7907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan227_17 : scan edge (search edge 3) 4 [1451] 1451 innerTail227_17 best = best := by
  rw [innerTail227_17, scan, if_neg (by decide), innerCandidates227_17_checked]
  change scan edge (search edge 3) 4 [1451] 1451 innerTail227_18
    (if 3 ≤ innerCandidates227_17.length then search edge 3 [1451, 7907] innerCandidates227_17 9358 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan227_18

def innerTail227_16 : List Nat := 7883 :: innerTail227_17
def innerCandidates227_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates227_16_checked : candidates edge 4 1451 7883 innerTail227_17 best.bound = innerCandidates227_16 := by
  unfold candidates
  have row : edge 7883 = SearchData.row7883.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan227_16 : scan edge (search edge 3) 4 [1451] 1451 innerTail227_16 best = best := by
  rw [innerTail227_16, scan, if_neg (by decide), innerCandidates227_16_checked]
  change scan edge (search edge 3) 4 [1451] 1451 innerTail227_17
    (if 3 ≤ innerCandidates227_16.length then search edge 3 [1451, 7883] innerCandidates227_16 9334 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan227_17

def innerTail227_15 : List Nat := 7823 :: innerTail227_16
def innerCandidates227_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates227_15_checked : candidates edge 4 1451 7823 innerTail227_16 best.bound = innerCandidates227_15 := by
  unfold candidates
  have row : edge 7823 = SearchData.row7823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan227_15 : scan edge (search edge 3) 4 [1451] 1451 innerTail227_15 best = best := by
  rw [innerTail227_15, scan, if_neg (by decide), innerCandidates227_15_checked]
  change scan edge (search edge 3) 4 [1451] 1451 innerTail227_16
    (if 3 ≤ innerCandidates227_15.length then search edge 3 [1451, 7823] innerCandidates227_15 9274 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan227_16

def innerTail227_14 : List Nat := 7499 :: innerTail227_15
def innerCandidates227_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates227_14_checked : candidates edge 4 1451 7499 innerTail227_15 best.bound = innerCandidates227_14 := by
  unfold candidates
  have row : edge 7499 = SearchData.row7499.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan227_14 : scan edge (search edge 3) 4 [1451] 1451 innerTail227_14 best = best := by
  rw [innerTail227_14, scan, if_neg (by decide), innerCandidates227_14_checked]
  change scan edge (search edge 3) 4 [1451] 1451 innerTail227_15
    (if 3 ≤ innerCandidates227_14.length then search edge 3 [1451, 7499] innerCandidates227_14 8950 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan227_15

def innerTail227_13 : List Nat := 7193 :: innerTail227_14
def innerCandidates227_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates227_13_checked : candidates edge 4 1451 7193 innerTail227_14 best.bound = innerCandidates227_13 := by
  unfold candidates
  have row : edge 7193 = SearchData.row7193.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan227_13 : scan edge (search edge 3) 4 [1451] 1451 innerTail227_13 best = best := by
  rw [innerTail227_13, scan, if_neg (by decide), innerCandidates227_13_checked]
  change scan edge (search edge 3) 4 [1451] 1451 innerTail227_14
    (if 3 ≤ innerCandidates227_13.length then search edge 3 [1451, 7193] innerCandidates227_13 8644 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan227_14

def innerTail227_12 : List Nat := 7001 :: innerTail227_13
def innerCandidates227_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates227_12_checked : candidates edge 4 1451 7001 innerTail227_13 best.bound = innerCandidates227_12 := by
  unfold candidates
  have row : edge 7001 = SearchData.row7001.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan227_12 : scan edge (search edge 3) 4 [1451] 1451 innerTail227_12 best = best := by
  rw [innerTail227_12, scan, if_neg (by decide), innerCandidates227_12_checked]
  change scan edge (search edge 3) 4 [1451] 1451 innerTail227_13
    (if 3 ≤ innerCandidates227_12.length then search edge 3 [1451, 7001] innerCandidates227_12 8452 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan227_13

def innerTail227_11 : List Nat := 6869 :: innerTail227_12
def innerCandidates227_11 : List Nat := [7907]
set_option maxRecDepth 4096 in
theorem innerCandidates227_11_checked : candidates edge 4 1451 6869 innerTail227_12 best.bound = innerCandidates227_11 := by
  unfold candidates
  have row : edge 6869 = SearchData.row6869.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan227_11 : scan edge (search edge 3) 4 [1451] 1451 innerTail227_11 best = best := by
  rw [innerTail227_11, scan, if_neg (by decide), innerCandidates227_11_checked]
  change scan edge (search edge 3) 4 [1451] 1451 innerTail227_12
    (if 3 ≤ innerCandidates227_11.length then search edge 3 [1451, 6869] innerCandidates227_11 8320 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan227_12

def innerTail227_10 : List Nat := 6659 :: innerTail227_11
def innerCandidates227_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates227_10_checked : candidates edge 4 1451 6659 innerTail227_11 best.bound = innerCandidates227_10 := by
  unfold candidates
  have row : edge 6659 = SearchData.row6659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan227_10 : scan edge (search edge 3) 4 [1451] 1451 innerTail227_10 best = best := by
  rw [innerTail227_10, scan, if_neg (by decide), innerCandidates227_10_checked]
  change scan edge (search edge 3) 4 [1451] 1451 innerTail227_11
    (if 3 ≤ innerCandidates227_10.length then search edge 3 [1451, 6659] innerCandidates227_10 8110 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan227_11

def innerTail227_9 : List Nat := 6581 :: innerTail227_10
def innerCandidates227_9 : List Nat := [7883]
set_option maxRecDepth 4096 in
theorem innerCandidates227_9_checked : candidates edge 4 1451 6581 innerTail227_10 best.bound = innerCandidates227_9 := by
  unfold candidates
  have row : edge 6581 = SearchData.row6581.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan227_9 : scan edge (search edge 3) 4 [1451] 1451 innerTail227_9 best = best := by
  rw [innerTail227_9, scan, if_neg (by decide), innerCandidates227_9_checked]
  change scan edge (search edge 3) 4 [1451] 1451 innerTail227_10
    (if 3 ≤ innerCandidates227_9.length then search edge 3 [1451, 6581] innerCandidates227_9 8032 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan227_10

def innerTail227_8 : List Nat := 5147 :: innerTail227_9
def innerCandidates227_8 : List Nat := [7823]
set_option maxRecDepth 4096 in
theorem innerCandidates227_8_checked : candidates edge 4 1451 5147 innerTail227_9 best.bound = innerCandidates227_8 := by
  unfold candidates
  have row : edge 5147 = SearchData.row5147.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan227_8 : scan edge (search edge 3) 4 [1451] 1451 innerTail227_8 best = best := by
  rw [innerTail227_8, scan, if_neg (by decide), innerCandidates227_8_checked]
  change scan edge (search edge 3) 4 [1451] 1451 innerTail227_9
    (if 3 ≤ innerCandidates227_8.length then search edge 3 [1451, 5147] innerCandidates227_8 6598 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan227_9

def innerTail227_7 : List Nat := 4457 :: innerTail227_8
def innerCandidates227_7 : List Nat := [7499]
set_option maxRecDepth 4096 in
theorem innerCandidates227_7_checked : candidates edge 4 1451 4457 innerTail227_8 best.bound = innerCandidates227_7 := by
  unfold candidates
  have row : edge 4457 = SearchData.row4457.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan227_7 : scan edge (search edge 3) 4 [1451] 1451 innerTail227_7 best = best := by
  rw [innerTail227_7, scan, if_neg (by decide), innerCandidates227_7_checked]
  change scan edge (search edge 3) 4 [1451] 1451 innerTail227_8
    (if 3 ≤ innerCandidates227_7.length then search edge 3 [1451, 4457] innerCandidates227_7 5908 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan227_8

def innerTail227_6 : List Nat := 4373 :: innerTail227_7
def innerCandidates227_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates227_6_checked : candidates edge 4 1451 4373 innerTail227_7 best.bound = innerCandidates227_6 := by
  unfold candidates
  have row : edge 4373 = SearchData.row4373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan227_6 : scan edge (search edge 3) 4 [1451] 1451 innerTail227_6 best = best := by
  rw [innerTail227_6, scan, if_neg (by decide), innerCandidates227_6_checked]
  change scan edge (search edge 3) 4 [1451] 1451 innerTail227_7
    (if 3 ≤ innerCandidates227_6.length then search edge 3 [1451, 4373] innerCandidates227_6 5824 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan227_7

def innerTail227_5 : List Nat := 3761 :: innerTail227_6
def innerCandidates227_5 : List Nat := [7499]
set_option maxRecDepth 4096 in
theorem innerCandidates227_5_checked : candidates edge 4 1451 3761 innerTail227_6 best.bound = innerCandidates227_5 := by
  unfold candidates
  have row : edge 3761 = SearchData.row3761.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan227_5 : scan edge (search edge 3) 4 [1451] 1451 innerTail227_5 best = best := by
  rw [innerTail227_5, scan, if_neg (by decide), innerCandidates227_5_checked]
  change scan edge (search edge 3) 4 [1451] 1451 innerTail227_6
    (if 3 ≤ innerCandidates227_5.length then search edge 3 [1451, 3761] innerCandidates227_5 5212 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan227_6

def innerTail227_4 : List Nat := 3557 :: innerTail227_5
def innerCandidates227_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates227_4_checked : candidates edge 4 1451 3557 innerTail227_5 best.bound = innerCandidates227_4 := by
  unfold candidates
  have row : edge 3557 = SearchData.row3557.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan227_4 : scan edge (search edge 3) 4 [1451] 1451 innerTail227_4 best = best := by
  rw [innerTail227_4, scan, if_neg (by decide), innerCandidates227_4_checked]
  change scan edge (search edge 3) 4 [1451] 1451 innerTail227_5
    (if 3 ≤ innerCandidates227_4.length then search edge 3 [1451, 3557] innerCandidates227_4 5008 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan227_5

def innerTail227_3 : List Nat := 3413 :: innerTail227_4
def innerCandidates227_3 : List Nat := [3761, 7001]
set_option maxRecDepth 4096 in
theorem innerCandidates227_3_checked : candidates edge 4 1451 3413 innerTail227_4 best.bound = innerCandidates227_3 := by
  unfold candidates
  have row : edge 3413 = SearchData.row3413.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan227_3 : scan edge (search edge 3) 4 [1451] 1451 innerTail227_3 best = best := by
  rw [innerTail227_3, scan, if_neg (by decide), innerCandidates227_3_checked]
  change scan edge (search edge 3) 4 [1451] 1451 innerTail227_4
    (if 3 ≤ innerCandidates227_3.length then search edge 3 [1451, 3413] innerCandidates227_3 4864 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan227_4

def innerTail227_2 : List Nat := 2699 :: innerTail227_3
def innerCandidates227_2 : List Nat := [3413, 3761]
set_option maxRecDepth 4096 in
theorem innerCandidates227_2_checked : candidates edge 4 1451 2699 innerTail227_3 best.bound = innerCandidates227_2 := by
  unfold candidates
  have row : edge 2699 = SearchData.row2699.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan227_2 : scan edge (search edge 3) 4 [1451] 1451 innerTail227_2 best = best := by
  rw [innerTail227_2, scan, if_neg (by decide), innerCandidates227_2_checked]
  change scan edge (search edge 3) 4 [1451] 1451 innerTail227_3
    (if 3 ≤ innerCandidates227_2.length then search edge 3 [1451, 2699] innerCandidates227_2 4150 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan227_3

def innerTail227_1 : List Nat := 2039 :: innerTail227_2
def innerCandidates227_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates227_1_checked : candidates edge 4 1451 2039 innerTail227_2 best.bound = innerCandidates227_1 := by
  unfold candidates
  have row : edge 2039 = PrimePairCache.smallRow2039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan227_1 : scan edge (search edge 3) 4 [1451] 1451 innerTail227_1 best = best := by
  rw [innerTail227_1, scan, if_neg (by decide), innerCandidates227_1_checked]
  change scan edge (search edge 3) 4 [1451] 1451 innerTail227_2
    (if 3 ≤ innerCandidates227_1.length then search edge 3 [1451, 2039] innerCandidates227_1 3490 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan227_2

def innerTail227_0 : List Nat := 1481 :: innerTail227_1
def innerCandidates227_0 : List Nat := [6869, 7823]
set_option maxRecDepth 4096 in
theorem innerCandidates227_0_checked : candidates edge 4 1451 1481 innerTail227_1 best.bound = innerCandidates227_0 := by
  unfold candidates
  have row : edge 1481 = PrimePairCache.smallRow1481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan227_0 : scan edge (search edge 3) 4 [1451] 1451 innerTail227_0 best = best := by
  rw [innerTail227_0, scan, if_neg (by decide), innerCandidates227_0_checked]
  change scan edge (search edge 3) 4 [1451] 1451 innerTail227_1
    (if 3 ≤ innerCandidates227_0.length then search edge 3 [1451, 1481] innerCandidates227_0 2932 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan227_1

theorem rootBranch227 : search edge 4 [1451] rootCandidates227 1451 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1451] 1451 innerTail227_0 best = best
  exact innerScan227_0

end ProjectEulerSolutions.PrimePairUpper.Search
