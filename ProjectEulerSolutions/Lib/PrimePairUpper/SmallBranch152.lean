import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates221 : List Nat := [1699, 1999, 2017, 2137, 2287, 2389, 2797, 3187, 3253, 4027, 4057, 5521, 5791, 5821, 6337, 6367, 6427, 8017]
set_option maxRecDepth 4096 in
theorem rootCandidates221_checked : candidates edge 5 0 1423 tail222 best.bound = rootCandidates221 := by
  unfold candidates
  have row : edge 1423 = PrimePairCache.smallRow1423.contains := by funext q; rfl
  rw [row]
  decide

def innerTail221_18 : List Nat := []
theorem innerScan221_18 : scan edge (search edge 3) 4 [1423] 1423 innerTail221_18 best = best := rfl

def innerTail221_17 : List Nat := 8017 :: innerTail221_18
def innerCandidates221_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates221_17_checked : candidates edge 4 1423 8017 innerTail221_18 best.bound = innerCandidates221_17 := by
  unfold candidates
  have row : edge 8017 = SearchData.row8017.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan221_17 : scan edge (search edge 3) 4 [1423] 1423 innerTail221_17 best = best := by
  rw [innerTail221_17, scan, if_neg (by decide), innerCandidates221_17_checked]
  change scan edge (search edge 3) 4 [1423] 1423 innerTail221_18
    (if 3 ≤ innerCandidates221_17.length then search edge 3 [1423, 8017] innerCandidates221_17 9440 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan221_18

def innerTail221_16 : List Nat := 6427 :: innerTail221_17
def innerCandidates221_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates221_16_checked : candidates edge 4 1423 6427 innerTail221_17 best.bound = innerCandidates221_16 := by
  unfold candidates
  have row : edge 6427 = SearchData.row6427.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan221_16 : scan edge (search edge 3) 4 [1423] 1423 innerTail221_16 best = best := by
  rw [innerTail221_16, scan, if_neg (by decide), innerCandidates221_16_checked]
  change scan edge (search edge 3) 4 [1423] 1423 innerTail221_17
    (if 3 ≤ innerCandidates221_16.length then search edge 3 [1423, 6427] innerCandidates221_16 7850 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan221_17

def innerTail221_15 : List Nat := 6367 :: innerTail221_16
def innerCandidates221_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates221_15_checked : candidates edge 4 1423 6367 innerTail221_16 best.bound = innerCandidates221_15 := by
  unfold candidates
  have row : edge 6367 = SearchData.row6367.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan221_15 : scan edge (search edge 3) 4 [1423] 1423 innerTail221_15 best = best := by
  rw [innerTail221_15, scan, if_neg (by decide), innerCandidates221_15_checked]
  change scan edge (search edge 3) 4 [1423] 1423 innerTail221_16
    (if 3 ≤ innerCandidates221_15.length then search edge 3 [1423, 6367] innerCandidates221_15 7790 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan221_16

def innerTail221_14 : List Nat := 6337 :: innerTail221_15
def innerCandidates221_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates221_14_checked : candidates edge 4 1423 6337 innerTail221_15 best.bound = innerCandidates221_14 := by
  unfold candidates
  have row : edge 6337 = SearchData.row6337.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan221_14 : scan edge (search edge 3) 4 [1423] 1423 innerTail221_14 best = best := by
  rw [innerTail221_14, scan, if_neg (by decide), innerCandidates221_14_checked]
  change scan edge (search edge 3) 4 [1423] 1423 innerTail221_15
    (if 3 ≤ innerCandidates221_14.length then search edge 3 [1423, 6337] innerCandidates221_14 7760 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan221_15

def innerTail221_13 : List Nat := 5821 :: innerTail221_14
def innerCandidates221_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates221_13_checked : candidates edge 4 1423 5821 innerTail221_14 best.bound = innerCandidates221_13 := by
  unfold candidates
  have row : edge 5821 = SearchData.row5821.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan221_13 : scan edge (search edge 3) 4 [1423] 1423 innerTail221_13 best = best := by
  rw [innerTail221_13, scan, if_neg (by decide), innerCandidates221_13_checked]
  change scan edge (search edge 3) 4 [1423] 1423 innerTail221_14
    (if 3 ≤ innerCandidates221_13.length then search edge 3 [1423, 5821] innerCandidates221_13 7244 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan221_14

def innerTail221_12 : List Nat := 5791 :: innerTail221_13
def innerCandidates221_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates221_12_checked : candidates edge 4 1423 5791 innerTail221_13 best.bound = innerCandidates221_12 := by
  unfold candidates
  have row : edge 5791 = SearchData.row5791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan221_12 : scan edge (search edge 3) 4 [1423] 1423 innerTail221_12 best = best := by
  rw [innerTail221_12, scan, if_neg (by decide), innerCandidates221_12_checked]
  change scan edge (search edge 3) 4 [1423] 1423 innerTail221_13
    (if 3 ≤ innerCandidates221_12.length then search edge 3 [1423, 5791] innerCandidates221_12 7214 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan221_13

def innerTail221_11 : List Nat := 5521 :: innerTail221_12
def innerCandidates221_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates221_11_checked : candidates edge 4 1423 5521 innerTail221_12 best.bound = innerCandidates221_11 := by
  unfold candidates
  have row : edge 5521 = SearchData.row5521.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan221_11 : scan edge (search edge 3) 4 [1423] 1423 innerTail221_11 best = best := by
  rw [innerTail221_11, scan, if_neg (by decide), innerCandidates221_11_checked]
  change scan edge (search edge 3) 4 [1423] 1423 innerTail221_12
    (if 3 ≤ innerCandidates221_11.length then search edge 3 [1423, 5521] innerCandidates221_11 6944 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan221_12

def innerTail221_10 : List Nat := 4057 :: innerTail221_11
def innerCandidates221_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates221_10_checked : candidates edge 4 1423 4057 innerTail221_11 best.bound = innerCandidates221_10 := by
  unfold candidates
  have row : edge 4057 = SearchData.row4057.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan221_10 : scan edge (search edge 3) 4 [1423] 1423 innerTail221_10 best = best := by
  rw [innerTail221_10, scan, if_neg (by decide), innerCandidates221_10_checked]
  change scan edge (search edge 3) 4 [1423] 1423 innerTail221_11
    (if 3 ≤ innerCandidates221_10.length then search edge 3 [1423, 4057] innerCandidates221_10 5480 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan221_11

def innerTail221_9 : List Nat := 4027 :: innerTail221_10
def innerCandidates221_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates221_9_checked : candidates edge 4 1423 4027 innerTail221_10 best.bound = innerCandidates221_9 := by
  unfold candidates
  have row : edge 4027 = SearchData.row4027.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan221_9 : scan edge (search edge 3) 4 [1423] 1423 innerTail221_9 best = best := by
  rw [innerTail221_9, scan, if_neg (by decide), innerCandidates221_9_checked]
  change scan edge (search edge 3) 4 [1423] 1423 innerTail221_10
    (if 3 ≤ innerCandidates221_9.length then search edge 3 [1423, 4027] innerCandidates221_9 5450 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan221_10

def innerTail221_8 : List Nat := 3253 :: innerTail221_9
def innerCandidates221_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates221_8_checked : candidates edge 4 1423 3253 innerTail221_9 best.bound = innerCandidates221_8 := by
  unfold candidates
  have row : edge 3253 = SearchData.row3253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan221_8 : scan edge (search edge 3) 4 [1423] 1423 innerTail221_8 best = best := by
  rw [innerTail221_8, scan, if_neg (by decide), innerCandidates221_8_checked]
  change scan edge (search edge 3) 4 [1423] 1423 innerTail221_9
    (if 3 ≤ innerCandidates221_8.length then search edge 3 [1423, 3253] innerCandidates221_8 4676 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan221_9

def innerTail221_7 : List Nat := 3187 :: innerTail221_8
def innerCandidates221_7 : List Nat := [8017]
set_option maxRecDepth 4096 in
theorem innerCandidates221_7_checked : candidates edge 4 1423 3187 innerTail221_8 best.bound = innerCandidates221_7 := by
  unfold candidates
  have row : edge 3187 = SearchData.row3187.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan221_7 : scan edge (search edge 3) 4 [1423] 1423 innerTail221_7 best = best := by
  rw [innerTail221_7, scan, if_neg (by decide), innerCandidates221_7_checked]
  change scan edge (search edge 3) 4 [1423] 1423 innerTail221_8
    (if 3 ≤ innerCandidates221_7.length then search edge 3 [1423, 3187] innerCandidates221_7 4610 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan221_8

def innerTail221_6 : List Nat := 2797 :: innerTail221_7
def innerCandidates221_6 : List Nat := [6367, 8017]
set_option maxRecDepth 4096 in
theorem innerCandidates221_6_checked : candidates edge 4 1423 2797 innerTail221_7 best.bound = innerCandidates221_6 := by
  unfold candidates
  have row : edge 2797 = SearchData.row2797.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan221_6 : scan edge (search edge 3) 4 [1423] 1423 innerTail221_6 best = best := by
  rw [innerTail221_6, scan, if_neg (by decide), innerCandidates221_6_checked]
  change scan edge (search edge 3) 4 [1423] 1423 innerTail221_7
    (if 3 ≤ innerCandidates221_6.length then search edge 3 [1423, 2797] innerCandidates221_6 4220 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan221_7

def innerTail221_5 : List Nat := 2389 :: innerTail221_6
def innerCandidates221_5 : List Nat := [6427]
set_option maxRecDepth 4096 in
theorem innerCandidates221_5_checked : candidates edge 4 1423 2389 innerTail221_6 best.bound = innerCandidates221_5 := by
  unfold candidates
  have row : edge 2389 = PrimePairCache.smallRow2389.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan221_5 : scan edge (search edge 3) 4 [1423] 1423 innerTail221_5 best = best := by
  rw [innerTail221_5, scan, if_neg (by decide), innerCandidates221_5_checked]
  change scan edge (search edge 3) 4 [1423] 1423 innerTail221_6
    (if 3 ≤ innerCandidates221_5.length then search edge 3 [1423, 2389] innerCandidates221_5 3812 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan221_6

def innerTail221_4 : List Nat := 2287 :: innerTail221_5
def innerCandidates221_4 : List Nat := [3187]
set_option maxRecDepth 4096 in
theorem innerCandidates221_4_checked : candidates edge 4 1423 2287 innerTail221_5 best.bound = innerCandidates221_4 := by
  unfold candidates
  have row : edge 2287 = PrimePairCache.smallRow2287.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan221_4 : scan edge (search edge 3) 4 [1423] 1423 innerTail221_4 best = best := by
  rw [innerTail221_4, scan, if_neg (by decide), innerCandidates221_4_checked]
  change scan edge (search edge 3) 4 [1423] 1423 innerTail221_5
    (if 3 ≤ innerCandidates221_4.length then search edge 3 [1423, 2287] innerCandidates221_4 3710 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan221_5

def innerTail221_3 : List Nat := 2137 :: innerTail221_4
def innerCandidates221_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates221_3_checked : candidates edge 4 1423 2137 innerTail221_4 best.bound = innerCandidates221_3 := by
  unfold candidates
  have row : edge 2137 = PrimePairCache.smallRow2137.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan221_3 : scan edge (search edge 3) 4 [1423] 1423 innerTail221_3 best = best := by
  rw [innerTail221_3, scan, if_neg (by decide), innerCandidates221_3_checked]
  change scan edge (search edge 3) 4 [1423] 1423 innerTail221_4
    (if 3 ≤ innerCandidates221_3.length then search edge 3 [1423, 2137] innerCandidates221_3 3560 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan221_4

def innerTail221_2 : List Nat := 2017 :: innerTail221_3
def innerCandidates221_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates221_2_checked : candidates edge 4 1423 2017 innerTail221_3 best.bound = innerCandidates221_2 := by
  unfold candidates
  have row : edge 2017 = PrimePairCache.smallRow2017.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan221_2 : scan edge (search edge 3) 4 [1423] 1423 innerTail221_2 best = best := by
  rw [innerTail221_2, scan, if_neg (by decide), innerCandidates221_2_checked]
  change scan edge (search edge 3) 4 [1423] 1423 innerTail221_3
    (if 3 ≤ innerCandidates221_2.length then search edge 3 [1423, 2017] innerCandidates221_2 3440 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan221_3

def innerTail221_1 : List Nat := 1999 :: innerTail221_2
def innerCandidates221_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates221_1_checked : candidates edge 4 1423 1999 innerTail221_2 best.bound = innerCandidates221_1 := by
  unfold candidates
  have row : edge 1999 = PrimePairCache.smallRow1999.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan221_1 : scan edge (search edge 3) 4 [1423] 1423 innerTail221_1 best = best := by
  rw [innerTail221_1, scan, if_neg (by decide), innerCandidates221_1_checked]
  change scan edge (search edge 3) 4 [1423] 1423 innerTail221_2
    (if 3 ≤ innerCandidates221_1.length then search edge 3 [1423, 1999] innerCandidates221_1 3422 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan221_2

def innerTail221_0 : List Nat := 1699 :: innerTail221_1
def innerCandidates221_0 : List Nat := [5791]
set_option maxRecDepth 4096 in
theorem innerCandidates221_0_checked : candidates edge 4 1423 1699 innerTail221_1 best.bound = innerCandidates221_0 := by
  unfold candidates
  have row : edge 1699 = PrimePairCache.smallRow1699.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan221_0 : scan edge (search edge 3) 4 [1423] 1423 innerTail221_0 best = best := by
  rw [innerTail221_0, scan, if_neg (by decide), innerCandidates221_0_checked]
  change scan edge (search edge 3) 4 [1423] 1423 innerTail221_1
    (if 3 ≤ innerCandidates221_0.length then search edge 3 [1423, 1699] innerCandidates221_0 3122 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan221_1

theorem rootBranch221 : search edge 4 [1423] rootCandidates221 1423 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1423] 1423 innerTail221_0 best = best
  exact innerScan221_0

def rootCandidates220 : List Nat := [1499, 3329, 4013, 4517, 4643, 5231, 5387, 6029, 6521, 6857, 6983, 7247, 7817, 7907]
set_option maxRecDepth 4096 in
theorem rootCandidates220_checked : candidates edge 5 0 1409 tail221 best.bound = rootCandidates220 := by
  unfold candidates
  have row : edge 1409 = PrimePairCache.smallRow1409.contains := by funext q; rfl
  rw [row]
  decide

def innerTail220_14 : List Nat := []
theorem innerScan220_14 : scan edge (search edge 3) 4 [1409] 1409 innerTail220_14 best = best := rfl

def innerTail220_13 : List Nat := 7907 :: innerTail220_14
def innerCandidates220_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates220_13_checked : candidates edge 4 1409 7907 innerTail220_14 best.bound = innerCandidates220_13 := by
  unfold candidates
  have row : edge 7907 = SearchData.row7907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan220_13 : scan edge (search edge 3) 4 [1409] 1409 innerTail220_13 best = best := by
  rw [innerTail220_13, scan, if_neg (by decide), innerCandidates220_13_checked]
  change scan edge (search edge 3) 4 [1409] 1409 innerTail220_14
    (if 3 ≤ innerCandidates220_13.length then search edge 3 [1409, 7907] innerCandidates220_13 9316 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan220_14

def innerTail220_12 : List Nat := 7817 :: innerTail220_13
def innerCandidates220_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates220_12_checked : candidates edge 4 1409 7817 innerTail220_13 best.bound = innerCandidates220_12 := by
  unfold candidates
  have row : edge 7817 = SearchData.row7817.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan220_12 : scan edge (search edge 3) 4 [1409] 1409 innerTail220_12 best = best := by
  rw [innerTail220_12, scan, if_neg (by decide), innerCandidates220_12_checked]
  change scan edge (search edge 3) 4 [1409] 1409 innerTail220_13
    (if 3 ≤ innerCandidates220_12.length then search edge 3 [1409, 7817] innerCandidates220_12 9226 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan220_13

def innerTail220_11 : List Nat := 7247 :: innerTail220_12
def innerCandidates220_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates220_11_checked : candidates edge 4 1409 7247 innerTail220_12 best.bound = innerCandidates220_11 := by
  unfold candidates
  have row : edge 7247 = SearchData.row7247.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan220_11 : scan edge (search edge 3) 4 [1409] 1409 innerTail220_11 best = best := by
  rw [innerTail220_11, scan, if_neg (by decide), innerCandidates220_11_checked]
  change scan edge (search edge 3) 4 [1409] 1409 innerTail220_12
    (if 3 ≤ innerCandidates220_11.length then search edge 3 [1409, 7247] innerCandidates220_11 8656 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan220_12

def innerTail220_10 : List Nat := 6983 :: innerTail220_11
def innerCandidates220_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates220_10_checked : candidates edge 4 1409 6983 innerTail220_11 best.bound = innerCandidates220_10 := by
  unfold candidates
  have row : edge 6983 = SearchData.row6983.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan220_10 : scan edge (search edge 3) 4 [1409] 1409 innerTail220_10 best = best := by
  rw [innerTail220_10, scan, if_neg (by decide), innerCandidates220_10_checked]
  change scan edge (search edge 3) 4 [1409] 1409 innerTail220_11
    (if 3 ≤ innerCandidates220_10.length then search edge 3 [1409, 6983] innerCandidates220_10 8392 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan220_11

def innerTail220_9 : List Nat := 6857 :: innerTail220_10
def innerCandidates220_9 : List Nat := [7907]
set_option maxRecDepth 4096 in
theorem innerCandidates220_9_checked : candidates edge 4 1409 6857 innerTail220_10 best.bound = innerCandidates220_9 := by
  unfold candidates
  have row : edge 6857 = SearchData.row6857.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan220_9 : scan edge (search edge 3) 4 [1409] 1409 innerTail220_9 best = best := by
  rw [innerTail220_9, scan, if_neg (by decide), innerCandidates220_9_checked]
  change scan edge (search edge 3) 4 [1409] 1409 innerTail220_10
    (if 3 ≤ innerCandidates220_9.length then search edge 3 [1409, 6857] innerCandidates220_9 8266 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan220_10

def innerTail220_8 : List Nat := 6521 :: innerTail220_9
def innerCandidates220_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates220_8_checked : candidates edge 4 1409 6521 innerTail220_9 best.bound = innerCandidates220_8 := by
  unfold candidates
  have row : edge 6521 = SearchData.row6521.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan220_8 : scan edge (search edge 3) 4 [1409] 1409 innerTail220_8 best = best := by
  rw [innerTail220_8, scan, if_neg (by decide), innerCandidates220_8_checked]
  change scan edge (search edge 3) 4 [1409] 1409 innerTail220_9
    (if 3 ≤ innerCandidates220_8.length then search edge 3 [1409, 6521] innerCandidates220_8 7930 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan220_9

def innerTail220_7 : List Nat := 6029 :: innerTail220_8
def innerCandidates220_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates220_7_checked : candidates edge 4 1409 6029 innerTail220_8 best.bound = innerCandidates220_7 := by
  unfold candidates
  have row : edge 6029 = SearchData.row6029.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan220_7 : scan edge (search edge 3) 4 [1409] 1409 innerTail220_7 best = best := by
  rw [innerTail220_7, scan, if_neg (by decide), innerCandidates220_7_checked]
  change scan edge (search edge 3) 4 [1409] 1409 innerTail220_8
    (if 3 ≤ innerCandidates220_7.length then search edge 3 [1409, 6029] innerCandidates220_7 7438 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan220_8

def innerTail220_6 : List Nat := 5387 :: innerTail220_7
def innerCandidates220_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates220_6_checked : candidates edge 4 1409 5387 innerTail220_7 best.bound = innerCandidates220_6 := by
  unfold candidates
  have row : edge 5387 = SearchData.row5387.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan220_6 : scan edge (search edge 3) 4 [1409] 1409 innerTail220_6 best = best := by
  rw [innerTail220_6, scan, if_neg (by decide), innerCandidates220_6_checked]
  change scan edge (search edge 3) 4 [1409] 1409 innerTail220_7
    (if 3 ≤ innerCandidates220_6.length then search edge 3 [1409, 5387] innerCandidates220_6 6796 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan220_7

def innerTail220_5 : List Nat := 5231 :: innerTail220_6
def innerCandidates220_5 : List Nat := [5387]
set_option maxRecDepth 4096 in
theorem innerCandidates220_5_checked : candidates edge 4 1409 5231 innerTail220_6 best.bound = innerCandidates220_5 := by
  unfold candidates
  have row : edge 5231 = SearchData.row5231.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan220_5 : scan edge (search edge 3) 4 [1409] 1409 innerTail220_5 best = best := by
  rw [innerTail220_5, scan, if_neg (by decide), innerCandidates220_5_checked]
  change scan edge (search edge 3) 4 [1409] 1409 innerTail220_6
    (if 3 ≤ innerCandidates220_5.length then search edge 3 [1409, 5231] innerCandidates220_5 6640 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan220_6

def innerTail220_4 : List Nat := 4643 :: innerTail220_5
def innerCandidates220_4 : List Nat := [5231]
set_option maxRecDepth 4096 in
theorem innerCandidates220_4_checked : candidates edge 4 1409 4643 innerTail220_5 best.bound = innerCandidates220_4 := by
  unfold candidates
  have row : edge 4643 = SearchData.row4643.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan220_4 : scan edge (search edge 3) 4 [1409] 1409 innerTail220_4 best = best := by
  rw [innerTail220_4, scan, if_neg (by decide), innerCandidates220_4_checked]
  change scan edge (search edge 3) 4 [1409] 1409 innerTail220_5
    (if 3 ≤ innerCandidates220_4.length then search edge 3 [1409, 4643] innerCandidates220_4 6052 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan220_5

def innerTail220_3 : List Nat := 4517 :: innerTail220_4
def innerCandidates220_3 : List Nat := [5387]
set_option maxRecDepth 4096 in
theorem innerCandidates220_3_checked : candidates edge 4 1409 4517 innerTail220_4 best.bound = innerCandidates220_3 := by
  unfold candidates
  have row : edge 4517 = SearchData.row4517.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan220_3 : scan edge (search edge 3) 4 [1409] 1409 innerTail220_3 best = best := by
  rw [innerTail220_3, scan, if_neg (by decide), innerCandidates220_3_checked]
  change scan edge (search edge 3) 4 [1409] 1409 innerTail220_4
    (if 3 ≤ innerCandidates220_3.length then search edge 3 [1409, 4517] innerCandidates220_3 5926 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan220_4

def innerTail220_2 : List Nat := 4013 :: innerTail220_3
def innerCandidates220_2 : List Nat := [4643]
set_option maxRecDepth 4096 in
theorem innerCandidates220_2_checked : candidates edge 4 1409 4013 innerTail220_3 best.bound = innerCandidates220_2 := by
  unfold candidates
  have row : edge 4013 = SearchData.row4013.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan220_2 : scan edge (search edge 3) 4 [1409] 1409 innerTail220_2 best = best := by
  rw [innerTail220_2, scan, if_neg (by decide), innerCandidates220_2_checked]
  change scan edge (search edge 3) 4 [1409] 1409 innerTail220_3
    (if 3 ≤ innerCandidates220_2.length then search edge 3 [1409, 4013] innerCandidates220_2 5422 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan220_3

def innerTail220_1 : List Nat := 3329 :: innerTail220_2
def innerCandidates220_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates220_1_checked : candidates edge 4 1409 3329 innerTail220_2 best.bound = innerCandidates220_1 := by
  unfold candidates
  have row : edge 3329 = SearchData.row3329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan220_1 : scan edge (search edge 3) 4 [1409] 1409 innerTail220_1 best = best := by
  rw [innerTail220_1, scan, if_neg (by decide), innerCandidates220_1_checked]
  change scan edge (search edge 3) 4 [1409] 1409 innerTail220_2
    (if 3 ≤ innerCandidates220_1.length then search edge 3 [1409, 3329] innerCandidates220_1 4738 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan220_2

def innerTail220_0 : List Nat := 1499 :: innerTail220_1
def innerCandidates220_0 : List Nat := [4013, 6521]
set_option maxRecDepth 4096 in
theorem innerCandidates220_0_checked : candidates edge 4 1409 1499 innerTail220_1 best.bound = innerCandidates220_0 := by
  unfold candidates
  have row : edge 1499 = PrimePairCache.smallRow1499.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan220_0 : scan edge (search edge 3) 4 [1409] 1409 innerTail220_0 best = best := by
  rw [innerTail220_0, scan, if_neg (by decide), innerCandidates220_0_checked]
  change scan edge (search edge 3) 4 [1409] 1409 innerTail220_1
    (if 3 ≤ innerCandidates220_0.length then search edge 3 [1409, 1499] innerCandidates220_0 2908 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan220_1

theorem rootBranch220 : search edge 4 [1409] rootCandidates220 1409 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1409] 1409 innerTail220_0 best = best
  exact innerScan220_0

def rootCandidates219 : List Nat := [1489, 1741, 2131, 2551, 3433, 4261, 4339, 4801, 4987, 5881, 6133, 6481, 6529, 6793, 7759]
set_option maxRecDepth 4096 in
theorem rootCandidates219_checked : candidates edge 5 0 1399 tail220 best.bound = rootCandidates219 := by
  unfold candidates
  have row : edge 1399 = PrimePairCache.smallRow1399.contains := by funext q; rfl
  rw [row]
  decide

def innerTail219_15 : List Nat := []
theorem innerScan219_15 : scan edge (search edge 3) 4 [1399] 1399 innerTail219_15 best = best := rfl

def innerTail219_14 : List Nat := 7759 :: innerTail219_15
def innerCandidates219_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates219_14_checked : candidates edge 4 1399 7759 innerTail219_15 best.bound = innerCandidates219_14 := by
  unfold candidates
  have row : edge 7759 = SearchData.row7759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan219_14 : scan edge (search edge 3) 4 [1399] 1399 innerTail219_14 best = best := by
  rw [innerTail219_14, scan, if_neg (by decide), innerCandidates219_14_checked]
  change scan edge (search edge 3) 4 [1399] 1399 innerTail219_15
    (if 3 ≤ innerCandidates219_14.length then search edge 3 [1399, 7759] innerCandidates219_14 9158 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan219_15

def innerTail219_13 : List Nat := 6793 :: innerTail219_14
def innerCandidates219_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates219_13_checked : candidates edge 4 1399 6793 innerTail219_14 best.bound = innerCandidates219_13 := by
  unfold candidates
  have row : edge 6793 = SearchData.row6793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan219_13 : scan edge (search edge 3) 4 [1399] 1399 innerTail219_13 best = best := by
  rw [innerTail219_13, scan, if_neg (by decide), innerCandidates219_13_checked]
  change scan edge (search edge 3) 4 [1399] 1399 innerTail219_14
    (if 3 ≤ innerCandidates219_13.length then search edge 3 [1399, 6793] innerCandidates219_13 8192 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan219_14

def innerTail219_12 : List Nat := 6529 :: innerTail219_13
def innerCandidates219_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates219_12_checked : candidates edge 4 1399 6529 innerTail219_13 best.bound = innerCandidates219_12 := by
  unfold candidates
  have row : edge 6529 = SearchData.row6529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan219_12 : scan edge (search edge 3) 4 [1399] 1399 innerTail219_12 best = best := by
  rw [innerTail219_12, scan, if_neg (by decide), innerCandidates219_12_checked]
  change scan edge (search edge 3) 4 [1399] 1399 innerTail219_13
    (if 3 ≤ innerCandidates219_12.length then search edge 3 [1399, 6529] innerCandidates219_12 7928 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan219_13

def innerTail219_11 : List Nat := 6481 :: innerTail219_12
def innerCandidates219_11 : List Nat := [6793]
set_option maxRecDepth 4096 in
theorem innerCandidates219_11_checked : candidates edge 4 1399 6481 innerTail219_12 best.bound = innerCandidates219_11 := by
  unfold candidates
  have row : edge 6481 = SearchData.row6481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan219_11 : scan edge (search edge 3) 4 [1399] 1399 innerTail219_11 best = best := by
  rw [innerTail219_11, scan, if_neg (by decide), innerCandidates219_11_checked]
  change scan edge (search edge 3) 4 [1399] 1399 innerTail219_12
    (if 3 ≤ innerCandidates219_11.length then search edge 3 [1399, 6481] innerCandidates219_11 7880 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan219_12

def innerTail219_10 : List Nat := 6133 :: innerTail219_11
def innerCandidates219_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates219_10_checked : candidates edge 4 1399 6133 innerTail219_11 best.bound = innerCandidates219_10 := by
  unfold candidates
  have row : edge 6133 = SearchData.row6133.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan219_10 : scan edge (search edge 3) 4 [1399] 1399 innerTail219_10 best = best := by
  rw [innerTail219_10, scan, if_neg (by decide), innerCandidates219_10_checked]
  change scan edge (search edge 3) 4 [1399] 1399 innerTail219_11
    (if 3 ≤ innerCandidates219_10.length then search edge 3 [1399, 6133] innerCandidates219_10 7532 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan219_11

def innerTail219_9 : List Nat := 5881 :: innerTail219_10
def innerCandidates219_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates219_9_checked : candidates edge 4 1399 5881 innerTail219_10 best.bound = innerCandidates219_9 := by
  unfold candidates
  have row : edge 5881 = SearchData.row5881.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan219_9 : scan edge (search edge 3) 4 [1399] 1399 innerTail219_9 best = best := by
  rw [innerTail219_9, scan, if_neg (by decide), innerCandidates219_9_checked]
  change scan edge (search edge 3) 4 [1399] 1399 innerTail219_10
    (if 3 ≤ innerCandidates219_9.length then search edge 3 [1399, 5881] innerCandidates219_9 7280 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan219_10

def innerTail219_8 : List Nat := 4987 :: innerTail219_9
def innerCandidates219_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates219_8_checked : candidates edge 4 1399 4987 innerTail219_9 best.bound = innerCandidates219_8 := by
  unfold candidates
  have row : edge 4987 = SearchData.row4987.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan219_8 : scan edge (search edge 3) 4 [1399] 1399 innerTail219_8 best = best := by
  rw [innerTail219_8, scan, if_neg (by decide), innerCandidates219_8_checked]
  change scan edge (search edge 3) 4 [1399] 1399 innerTail219_9
    (if 3 ≤ innerCandidates219_8.length then search edge 3 [1399, 4987] innerCandidates219_8 6386 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan219_9

def innerTail219_7 : List Nat := 4801 :: innerTail219_8
def innerCandidates219_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates219_7_checked : candidates edge 4 1399 4801 innerTail219_8 best.bound = innerCandidates219_7 := by
  unfold candidates
  have row : edge 4801 = SearchData.row4801.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan219_7 : scan edge (search edge 3) 4 [1399] 1399 innerTail219_7 best = best := by
  rw [innerTail219_7, scan, if_neg (by decide), innerCandidates219_7_checked]
  change scan edge (search edge 3) 4 [1399] 1399 innerTail219_8
    (if 3 ≤ innerCandidates219_7.length then search edge 3 [1399, 4801] innerCandidates219_7 6200 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan219_8

def innerTail219_6 : List Nat := 4339 :: innerTail219_7
def innerCandidates219_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates219_6_checked : candidates edge 4 1399 4339 innerTail219_7 best.bound = innerCandidates219_6 := by
  unfold candidates
  have row : edge 4339 = SearchData.row4339.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan219_6 : scan edge (search edge 3) 4 [1399] 1399 innerTail219_6 best = best := by
  rw [innerTail219_6, scan, if_neg (by decide), innerCandidates219_6_checked]
  change scan edge (search edge 3) 4 [1399] 1399 innerTail219_7
    (if 3 ≤ innerCandidates219_6.length then search edge 3 [1399, 4339] innerCandidates219_6 5738 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan219_7

def innerTail219_5 : List Nat := 4261 :: innerTail219_6
def innerCandidates219_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates219_5_checked : candidates edge 4 1399 4261 innerTail219_6 best.bound = innerCandidates219_5 := by
  unfold candidates
  have row : edge 4261 = SearchData.row4261.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan219_5 : scan edge (search edge 3) 4 [1399] 1399 innerTail219_5 best = best := by
  rw [innerTail219_5, scan, if_neg (by decide), innerCandidates219_5_checked]
  change scan edge (search edge 3) 4 [1399] 1399 innerTail219_6
    (if 3 ≤ innerCandidates219_5.length then search edge 3 [1399, 4261] innerCandidates219_5 5660 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan219_6

def innerTail219_4 : List Nat := 3433 :: innerTail219_5
def innerCandidates219_4 : List Nat := [4987]
set_option maxRecDepth 4096 in
theorem innerCandidates219_4_checked : candidates edge 4 1399 3433 innerTail219_5 best.bound = innerCandidates219_4 := by
  unfold candidates
  have row : edge 3433 = SearchData.row3433.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan219_4 : scan edge (search edge 3) 4 [1399] 1399 innerTail219_4 best = best := by
  rw [innerTail219_4, scan, if_neg (by decide), innerCandidates219_4_checked]
  change scan edge (search edge 3) 4 [1399] 1399 innerTail219_5
    (if 3 ≤ innerCandidates219_4.length then search edge 3 [1399, 3433] innerCandidates219_4 4832 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan219_5

def innerTail219_3 : List Nat := 2551 :: innerTail219_4
def innerCandidates219_3 : List Nat := [6793]
set_option maxRecDepth 4096 in
theorem innerCandidates219_3_checked : candidates edge 4 1399 2551 innerTail219_4 best.bound = innerCandidates219_3 := by
  unfold candidates
  have row : edge 2551 = PrimePairCache.smallRow2551.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan219_3 : scan edge (search edge 3) 4 [1399] 1399 innerTail219_3 best = best := by
  rw [innerTail219_3, scan, if_neg (by decide), innerCandidates219_3_checked]
  change scan edge (search edge 3) 4 [1399] 1399 innerTail219_4
    (if 3 ≤ innerCandidates219_3.length then search edge 3 [1399, 2551] innerCandidates219_3 3950 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan219_4

def innerTail219_2 : List Nat := 2131 :: innerTail219_3
def innerCandidates219_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates219_2_checked : candidates edge 4 1399 2131 innerTail219_3 best.bound = innerCandidates219_2 := by
  unfold candidates
  have row : edge 2131 = PrimePairCache.smallRow2131.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan219_2 : scan edge (search edge 3) 4 [1399] 1399 innerTail219_2 best = best := by
  rw [innerTail219_2, scan, if_neg (by decide), innerCandidates219_2_checked]
  change scan edge (search edge 3) 4 [1399] 1399 innerTail219_3
    (if 3 ≤ innerCandidates219_2.length then search edge 3 [1399, 2131] innerCandidates219_2 3530 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan219_3

def innerTail219_1 : List Nat := 1741 :: innerTail219_2
def innerCandidates219_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates219_1_checked : candidates edge 4 1399 1741 innerTail219_2 best.bound = innerCandidates219_1 := by
  unfold candidates
  have row : edge 1741 = PrimePairCache.smallRow1741.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan219_1 : scan edge (search edge 3) 4 [1399] 1399 innerTail219_1 best = best := by
  rw [innerTail219_1, scan, if_neg (by decide), innerCandidates219_1_checked]
  change scan edge (search edge 3) 4 [1399] 1399 innerTail219_2
    (if 3 ≤ innerCandidates219_1.length then search edge 3 [1399, 1741] innerCandidates219_1 3140 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan219_2

def innerTail219_0 : List Nat := 1489 :: innerTail219_1
def innerCandidates219_0 : List Nat := [4801]
set_option maxRecDepth 4096 in
theorem innerCandidates219_0_checked : candidates edge 4 1399 1489 innerTail219_1 best.bound = innerCandidates219_0 := by
  unfold candidates
  have row : edge 1489 = PrimePairCache.smallRow1489.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan219_0 : scan edge (search edge 3) 4 [1399] 1399 innerTail219_0 best = best := by
  rw [innerTail219_0, scan, if_neg (by decide), innerCandidates219_0_checked]
  change scan edge (search edge 3) 4 [1399] 1399 innerTail219_1
    (if 3 ≤ innerCandidates219_0.length then search edge 3 [1399, 1489] innerCandidates219_0 2888 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan219_1

theorem rootBranch219 : search edge 4 [1399] rootCandidates219 1399 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1399] 1399 innerTail219_0 best = best
  exact innerScan219_0

def rootCandidates218 : List Nat := [1657, 1789, 2683, 2767, 3169, 4027, 4111, 5233, 6571, 6823, 7933, 8017]
set_option maxRecDepth 4096 in
theorem rootCandidates218_checked : candidates edge 5 0 1381 tail219 best.bound = rootCandidates218 := by
  unfold candidates
  have row : edge 1381 = PrimePairCache.smallRow1381.contains := by funext q; rfl
  rw [row]
  decide

def innerTail218_12 : List Nat := []
theorem innerScan218_12 : scan edge (search edge 3) 4 [1381] 1381 innerTail218_12 best = best := rfl

def innerTail218_11 : List Nat := 8017 :: innerTail218_12
def innerCandidates218_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates218_11_checked : candidates edge 4 1381 8017 innerTail218_12 best.bound = innerCandidates218_11 := by
  unfold candidates
  have row : edge 8017 = SearchData.row8017.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan218_11 : scan edge (search edge 3) 4 [1381] 1381 innerTail218_11 best = best := by
  rw [innerTail218_11, scan, if_neg (by decide), innerCandidates218_11_checked]
  change scan edge (search edge 3) 4 [1381] 1381 innerTail218_12
    (if 3 ≤ innerCandidates218_11.length then search edge 3 [1381, 8017] innerCandidates218_11 9398 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan218_12

def innerTail218_10 : List Nat := 7933 :: innerTail218_11
def innerCandidates218_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates218_10_checked : candidates edge 4 1381 7933 innerTail218_11 best.bound = innerCandidates218_10 := by
  unfold candidates
  have row : edge 7933 = SearchData.row7933.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan218_10 : scan edge (search edge 3) 4 [1381] 1381 innerTail218_10 best = best := by
  rw [innerTail218_10, scan, if_neg (by decide), innerCandidates218_10_checked]
  change scan edge (search edge 3) 4 [1381] 1381 innerTail218_11
    (if 3 ≤ innerCandidates218_10.length then search edge 3 [1381, 7933] innerCandidates218_10 9314 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan218_11

def innerTail218_9 : List Nat := 6823 :: innerTail218_10
def innerCandidates218_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates218_9_checked : candidates edge 4 1381 6823 innerTail218_10 best.bound = innerCandidates218_9 := by
  unfold candidates
  have row : edge 6823 = SearchData.row6823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan218_9 : scan edge (search edge 3) 4 [1381] 1381 innerTail218_9 best = best := by
  rw [innerTail218_9, scan, if_neg (by decide), innerCandidates218_9_checked]
  change scan edge (search edge 3) 4 [1381] 1381 innerTail218_10
    (if 3 ≤ innerCandidates218_9.length then search edge 3 [1381, 6823] innerCandidates218_9 8204 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan218_10

def innerTail218_8 : List Nat := 6571 :: innerTail218_9
def innerCandidates218_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates218_8_checked : candidates edge 4 1381 6571 innerTail218_9 best.bound = innerCandidates218_8 := by
  unfold candidates
  have row : edge 6571 = SearchData.row6571.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan218_8 : scan edge (search edge 3) 4 [1381] 1381 innerTail218_8 best = best := by
  rw [innerTail218_8, scan, if_neg (by decide), innerCandidates218_8_checked]
  change scan edge (search edge 3) 4 [1381] 1381 innerTail218_9
    (if 3 ≤ innerCandidates218_8.length then search edge 3 [1381, 6571] innerCandidates218_8 7952 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan218_9

def innerTail218_7 : List Nat := 5233 :: innerTail218_8
def innerCandidates218_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates218_7_checked : candidates edge 4 1381 5233 innerTail218_8 best.bound = innerCandidates218_7 := by
  unfold candidates
  have row : edge 5233 = SearchData.row5233.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan218_7 : scan edge (search edge 3) 4 [1381] 1381 innerTail218_7 best = best := by
  rw [innerTail218_7, scan, if_neg (by decide), innerCandidates218_7_checked]
  change scan edge (search edge 3) 4 [1381] 1381 innerTail218_8
    (if 3 ≤ innerCandidates218_7.length then search edge 3 [1381, 5233] innerCandidates218_7 6614 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan218_8

def innerTail218_6 : List Nat := 4111 :: innerTail218_7
def innerCandidates218_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates218_6_checked : candidates edge 4 1381 4111 innerTail218_7 best.bound = innerCandidates218_6 := by
  unfold candidates
  have row : edge 4111 = SearchData.row4111.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan218_6 : scan edge (search edge 3) 4 [1381] 1381 innerTail218_6 best = best := by
  rw [innerTail218_6, scan, if_neg (by decide), innerCandidates218_6_checked]
  change scan edge (search edge 3) 4 [1381] 1381 innerTail218_7
    (if 3 ≤ innerCandidates218_6.length then search edge 3 [1381, 4111] innerCandidates218_6 5492 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan218_7

def innerTail218_5 : List Nat := 4027 :: innerTail218_6
def innerCandidates218_5 : List Nat := [5233]
set_option maxRecDepth 4096 in
theorem innerCandidates218_5_checked : candidates edge 4 1381 4027 innerTail218_6 best.bound = innerCandidates218_5 := by
  unfold candidates
  have row : edge 4027 = SearchData.row4027.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan218_5 : scan edge (search edge 3) 4 [1381] 1381 innerTail218_5 best = best := by
  rw [innerTail218_5, scan, if_neg (by decide), innerCandidates218_5_checked]
  change scan edge (search edge 3) 4 [1381] 1381 innerTail218_6
    (if 3 ≤ innerCandidates218_5.length then search edge 3 [1381, 4027] innerCandidates218_5 5408 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan218_6

def innerTail218_4 : List Nat := 3169 :: innerTail218_5
def innerCandidates218_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates218_4_checked : candidates edge 4 1381 3169 innerTail218_5 best.bound = innerCandidates218_4 := by
  unfold candidates
  have row : edge 3169 = SearchData.row3169.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan218_4 : scan edge (search edge 3) 4 [1381] 1381 innerTail218_4 best = best := by
  rw [innerTail218_4, scan, if_neg (by decide), innerCandidates218_4_checked]
  change scan edge (search edge 3) 4 [1381] 1381 innerTail218_5
    (if 3 ≤ innerCandidates218_4.length then search edge 3 [1381, 3169] innerCandidates218_4 4550 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan218_5

def innerTail218_3 : List Nat := 2767 :: innerTail218_4
def innerCandidates218_3 : List Nat := [7933]
set_option maxRecDepth 4096 in
theorem innerCandidates218_3_checked : candidates edge 4 1381 2767 innerTail218_4 best.bound = innerCandidates218_3 := by
  unfold candidates
  have row : edge 2767 = SearchData.row2767.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan218_3 : scan edge (search edge 3) 4 [1381] 1381 innerTail218_3 best = best := by
  rw [innerTail218_3, scan, if_neg (by decide), innerCandidates218_3_checked]
  change scan edge (search edge 3) 4 [1381] 1381 innerTail218_4
    (if 3 ≤ innerCandidates218_3.length then search edge 3 [1381, 2767] innerCandidates218_3 4148 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan218_4

def innerTail218_2 : List Nat := 2683 :: innerTail218_3
def innerCandidates218_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates218_2_checked : candidates edge 4 1381 2683 innerTail218_3 best.bound = innerCandidates218_2 := by
  unfold candidates
  have row : edge 2683 = SearchData.row2683.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan218_2 : scan edge (search edge 3) 4 [1381] 1381 innerTail218_2 best = best := by
  rw [innerTail218_2, scan, if_neg (by decide), innerCandidates218_2_checked]
  change scan edge (search edge 3) 4 [1381] 1381 innerTail218_3
    (if 3 ≤ innerCandidates218_2.length then search edge 3 [1381, 2683] innerCandidates218_2 4064 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan218_3

def innerTail218_1 : List Nat := 1789 :: innerTail218_2
def innerCandidates218_1 : List Nat := [2683, 4111]
set_option maxRecDepth 4096 in
theorem innerCandidates218_1_checked : candidates edge 4 1381 1789 innerTail218_2 best.bound = innerCandidates218_1 := by
  unfold candidates
  have row : edge 1789 = PrimePairCache.smallRow1789.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan218_1 : scan edge (search edge 3) 4 [1381] 1381 innerTail218_1 best = best := by
  rw [innerTail218_1, scan, if_neg (by decide), innerCandidates218_1_checked]
  change scan edge (search edge 3) 4 [1381] 1381 innerTail218_2
    (if 3 ≤ innerCandidates218_1.length then search edge 3 [1381, 1789] innerCandidates218_1 3170 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan218_2

def innerTail218_0 : List Nat := 1657 :: innerTail218_1
def innerCandidates218_0 : List Nat := [2683, 4111, 8017]
set_option maxRecDepth 4096 in
theorem innerCandidates218_0_checked : candidates edge 4 1381 1657 innerTail218_1 best.bound = innerCandidates218_0 := by
  unfold candidates
  have row : edge 1657 = PrimePairCache.smallRow1657.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild218_0 : search edge 3 [1381, 1657] innerCandidates218_0 3038 best = best := by decide

theorem innerScan218_0 : scan edge (search edge 3) 4 [1381] 1381 innerTail218_0 best = best := by
  rw [innerTail218_0, scan, if_neg (by decide), innerCandidates218_0_checked]
  change scan edge (search edge 3) 4 [1381] 1381 innerTail218_1
    (if 3 ≤ innerCandidates218_0.length then search edge 3 [1381, 1657] innerCandidates218_0 3038 best else best) = best
  rw [if_pos (by decide), innerChild218_0]
  exact innerScan218_1

theorem rootBranch218 : search edge 4 [1381] rootCandidates218 1381 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1381] 1381 innerTail218_0 best = best
  exact innerScan218_0

end ProjectEulerSolutions.PrimePairUpper.Search
