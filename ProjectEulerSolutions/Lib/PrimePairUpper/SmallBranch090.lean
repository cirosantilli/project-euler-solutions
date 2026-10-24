import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates433 : List Nat := [3209, 3491, 3533, 4133, 4337, 4409, 5897, 6317, 6719, 7481, 7529, 8273]
set_option maxRecDepth 4096 in
theorem rootCandidates433_checked : candidates edge 5 0 3041 tail434 best.bound = rootCandidates433 := by
  unfold candidates
  have row : edge 3041 = SearchData.row3041.contains := by funext q; rfl
  rw [row]
  decide

def innerTail433_12 : List Nat := []
theorem innerScan433_12 : scan edge (search edge 3) 4 [3041] 3041 innerTail433_12 best = best := rfl

def innerTail433_11 : List Nat := 8273 :: innerTail433_12
def innerCandidates433_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates433_11_checked : candidates edge 4 3041 8273 innerTail433_12 best.bound = innerCandidates433_11 := by
  unfold candidates
  have row : edge 8273 = SearchData.row8273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan433_11 : scan edge (search edge 3) 4 [3041] 3041 innerTail433_11 best = best := by
  rw [innerTail433_11, scan, if_neg (by decide), innerCandidates433_11_checked]
  change scan edge (search edge 3) 4 [3041] 3041 innerTail433_12
    (if 3 ≤ innerCandidates433_11.length then search edge 3 [3041, 8273] innerCandidates433_11 11314 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan433_12

def innerTail433_10 : List Nat := 7529 :: innerTail433_11
def innerCandidates433_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates433_10_checked : candidates edge 4 3041 7529 innerTail433_11 best.bound = innerCandidates433_10 := by
  unfold candidates
  have row : edge 7529 = SearchData.row7529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan433_10 : scan edge (search edge 3) 4 [3041] 3041 innerTail433_10 best = best := by
  rw [innerTail433_10, scan, if_neg (by decide), innerCandidates433_10_checked]
  change scan edge (search edge 3) 4 [3041] 3041 innerTail433_11
    (if 3 ≤ innerCandidates433_10.length then search edge 3 [3041, 7529] innerCandidates433_10 10570 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan433_11

def innerTail433_9 : List Nat := 7481 :: innerTail433_10
def innerCandidates433_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates433_9_checked : candidates edge 4 3041 7481 innerTail433_10 best.bound = innerCandidates433_9 := by
  unfold candidates
  have row : edge 7481 = SearchData.row7481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan433_9 : scan edge (search edge 3) 4 [3041] 3041 innerTail433_9 best = best := by
  rw [innerTail433_9, scan, if_neg (by decide), innerCandidates433_9_checked]
  change scan edge (search edge 3) 4 [3041] 3041 innerTail433_10
    (if 3 ≤ innerCandidates433_9.length then search edge 3 [3041, 7481] innerCandidates433_9 10522 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan433_10

def innerTail433_8 : List Nat := 6719 :: innerTail433_9
def innerCandidates433_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates433_8_checked : candidates edge 4 3041 6719 innerTail433_9 best.bound = innerCandidates433_8 := by
  unfold candidates
  have row : edge 6719 = SearchData.row6719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan433_8 : scan edge (search edge 3) 4 [3041] 3041 innerTail433_8 best = best := by
  rw [innerTail433_8, scan, if_neg (by decide), innerCandidates433_8_checked]
  change scan edge (search edge 3) 4 [3041] 3041 innerTail433_9
    (if 3 ≤ innerCandidates433_8.length then search edge 3 [3041, 6719] innerCandidates433_8 9760 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan433_9

def innerTail433_7 : List Nat := 6317 :: innerTail433_8
def innerCandidates433_7 : List Nat := [7529]
set_option maxRecDepth 4096 in
theorem innerCandidates433_7_checked : candidates edge 4 3041 6317 innerTail433_8 best.bound = innerCandidates433_7 := by
  unfold candidates
  have row : edge 6317 = SearchData.row6317.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan433_7 : scan edge (search edge 3) 4 [3041] 3041 innerTail433_7 best = best := by
  rw [innerTail433_7, scan, if_neg (by decide), innerCandidates433_7_checked]
  change scan edge (search edge 3) 4 [3041] 3041 innerTail433_8
    (if 3 ≤ innerCandidates433_7.length then search edge 3 [3041, 6317] innerCandidates433_7 9358 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan433_8

def innerTail433_6 : List Nat := 5897 :: innerTail433_7
def innerCandidates433_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates433_6_checked : candidates edge 4 3041 5897 innerTail433_7 best.bound = innerCandidates433_6 := by
  unfold candidates
  have row : edge 5897 = SearchData.row5897.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan433_6 : scan edge (search edge 3) 4 [3041] 3041 innerTail433_6 best = best := by
  rw [innerTail433_6, scan, if_neg (by decide), innerCandidates433_6_checked]
  change scan edge (search edge 3) 4 [3041] 3041 innerTail433_7
    (if 3 ≤ innerCandidates433_6.length then search edge 3 [3041, 5897] innerCandidates433_6 8938 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan433_7

def innerTail433_5 : List Nat := 4409 :: innerTail433_6
def innerCandidates433_5 : List Nat := [6719]
set_option maxRecDepth 4096 in
theorem innerCandidates433_5_checked : candidates edge 4 3041 4409 innerTail433_6 best.bound = innerCandidates433_5 := by
  unfold candidates
  have row : edge 4409 = SearchData.row4409.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan433_5 : scan edge (search edge 3) 4 [3041] 3041 innerTail433_5 best = best := by
  rw [innerTail433_5, scan, if_neg (by decide), innerCandidates433_5_checked]
  change scan edge (search edge 3) 4 [3041] 3041 innerTail433_6
    (if 3 ≤ innerCandidates433_5.length then search edge 3 [3041, 4409] innerCandidates433_5 7450 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan433_6

def innerTail433_4 : List Nat := 4337 :: innerTail433_5
def innerCandidates433_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates433_4_checked : candidates edge 4 3041 4337 innerTail433_5 best.bound = innerCandidates433_4 := by
  unfold candidates
  have row : edge 4337 = SearchData.row4337.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan433_4 : scan edge (search edge 3) 4 [3041] 3041 innerTail433_4 best = best := by
  rw [innerTail433_4, scan, if_neg (by decide), innerCandidates433_4_checked]
  change scan edge (search edge 3) 4 [3041] 3041 innerTail433_5
    (if 3 ≤ innerCandidates433_4.length then search edge 3 [3041, 4337] innerCandidates433_4 7378 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan433_5

def innerTail433_3 : List Nat := 4133 :: innerTail433_4
def innerCandidates433_3 : List Nat := [4409]
set_option maxRecDepth 4096 in
theorem innerCandidates433_3_checked : candidates edge 4 3041 4133 innerTail433_4 best.bound = innerCandidates433_3 := by
  unfold candidates
  have row : edge 4133 = SearchData.row4133.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan433_3 : scan edge (search edge 3) 4 [3041] 3041 innerTail433_3 best = best := by
  rw [innerTail433_3, scan, if_neg (by decide), innerCandidates433_3_checked]
  change scan edge (search edge 3) 4 [3041] 3041 innerTail433_4
    (if 3 ≤ innerCandidates433_3.length then search edge 3 [3041, 4133] innerCandidates433_3 7174 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan433_4

def innerTail433_2 : List Nat := 3533 :: innerTail433_3
def innerCandidates433_2 : List Nat := [4133, 8273]
set_option maxRecDepth 4096 in
theorem innerCandidates433_2_checked : candidates edge 4 3041 3533 innerTail433_3 best.bound = innerCandidates433_2 := by
  unfold candidates
  have row : edge 3533 = SearchData.row3533.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan433_2 : scan edge (search edge 3) 4 [3041] 3041 innerTail433_2 best = best := by
  rw [innerTail433_2, scan, if_neg (by decide), innerCandidates433_2_checked]
  change scan edge (search edge 3) 4 [3041] 3041 innerTail433_3
    (if 3 ≤ innerCandidates433_2.length then search edge 3 [3041, 3533] innerCandidates433_2 6574 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan433_3

def innerTail433_1 : List Nat := 3491 :: innerTail433_2
def innerCandidates433_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates433_1_checked : candidates edge 4 3041 3491 innerTail433_2 best.bound = innerCandidates433_1 := by
  unfold candidates
  have row : edge 3491 = SearchData.row3491.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan433_1 : scan edge (search edge 3) 4 [3041] 3041 innerTail433_1 best = best := by
  rw [innerTail433_1, scan, if_neg (by decide), innerCandidates433_1_checked]
  change scan edge (search edge 3) 4 [3041] 3041 innerTail433_2
    (if 3 ≤ innerCandidates433_1.length then search edge 3 [3041, 3491] innerCandidates433_1 6532 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan433_2

def innerTail433_0 : List Nat := 3209 :: innerTail433_1
def innerCandidates433_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates433_0_checked : candidates edge 4 3041 3209 innerTail433_1 best.bound = innerCandidates433_0 := by
  unfold candidates
  have row : edge 3209 = SearchData.row3209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan433_0 : scan edge (search edge 3) 4 [3041] 3041 innerTail433_0 best = best := by
  rw [innerTail433_0, scan, if_neg (by decide), innerCandidates433_0_checked]
  change scan edge (search edge 3) 4 [3041] 3041 innerTail433_1
    (if 3 ≤ innerCandidates433_0.length then search edge 3 [3041, 3209] innerCandidates433_0 6250 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan433_1

theorem rootBranch433 : search edge 4 [3041] rootCandidates433 3041 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3041] 3041 innerTail433_0 best = best
  exact innerScan433_0

def rootCandidates432 : List Nat := [3253, 3331, 3433, 3517, 3709, 4423, 4507, 4987, 5689, 5701, 6007, 6301, 6607, 6823, 7129, 8329]
set_option maxRecDepth 4096 in
theorem rootCandidates432_checked : candidates edge 5 0 3037 tail433 best.bound = rootCandidates432 := by
  unfold candidates
  have row : edge 3037 = SearchData.row3037.contains := by funext q; rfl
  rw [row]
  decide

def innerTail432_16 : List Nat := []
theorem innerScan432_16 : scan edge (search edge 3) 4 [3037] 3037 innerTail432_16 best = best := rfl

def innerTail432_15 : List Nat := 8329 :: innerTail432_16
def innerCandidates432_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates432_15_checked : candidates edge 4 3037 8329 innerTail432_16 best.bound = innerCandidates432_15 := by
  unfold candidates
  have row : edge 8329 = SearchData.row8329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan432_15 : scan edge (search edge 3) 4 [3037] 3037 innerTail432_15 best = best := by
  rw [innerTail432_15, scan, if_neg (by decide), innerCandidates432_15_checked]
  change scan edge (search edge 3) 4 [3037] 3037 innerTail432_16
    (if 3 ≤ innerCandidates432_15.length then search edge 3 [3037, 8329] innerCandidates432_15 11366 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan432_16

def innerTail432_14 : List Nat := 7129 :: innerTail432_15
def innerCandidates432_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates432_14_checked : candidates edge 4 3037 7129 innerTail432_15 best.bound = innerCandidates432_14 := by
  unfold candidates
  have row : edge 7129 = SearchData.row7129.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan432_14 : scan edge (search edge 3) 4 [3037] 3037 innerTail432_14 best = best := by
  rw [innerTail432_14, scan, if_neg (by decide), innerCandidates432_14_checked]
  change scan edge (search edge 3) 4 [3037] 3037 innerTail432_15
    (if 3 ≤ innerCandidates432_14.length then search edge 3 [3037, 7129] innerCandidates432_14 10166 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan432_15

def innerTail432_13 : List Nat := 6823 :: innerTail432_14
def innerCandidates432_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates432_13_checked : candidates edge 4 3037 6823 innerTail432_14 best.bound = innerCandidates432_13 := by
  unfold candidates
  have row : edge 6823 = SearchData.row6823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan432_13 : scan edge (search edge 3) 4 [3037] 3037 innerTail432_13 best = best := by
  rw [innerTail432_13, scan, if_neg (by decide), innerCandidates432_13_checked]
  change scan edge (search edge 3) 4 [3037] 3037 innerTail432_14
    (if 3 ≤ innerCandidates432_13.length then search edge 3 [3037, 6823] innerCandidates432_13 9860 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan432_14

def innerTail432_12 : List Nat := 6607 :: innerTail432_13
def innerCandidates432_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates432_12_checked : candidates edge 4 3037 6607 innerTail432_13 best.bound = innerCandidates432_12 := by
  unfold candidates
  have row : edge 6607 = SearchData.row6607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan432_12 : scan edge (search edge 3) 4 [3037] 3037 innerTail432_12 best = best := by
  rw [innerTail432_12, scan, if_neg (by decide), innerCandidates432_12_checked]
  change scan edge (search edge 3) 4 [3037] 3037 innerTail432_13
    (if 3 ≤ innerCandidates432_12.length then search edge 3 [3037, 6607] innerCandidates432_12 9644 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan432_13

def innerTail432_11 : List Nat := 6301 :: innerTail432_12
def innerCandidates432_11 : List Nat := [6607]
set_option maxRecDepth 4096 in
theorem innerCandidates432_11_checked : candidates edge 4 3037 6301 innerTail432_12 best.bound = innerCandidates432_11 := by
  unfold candidates
  have row : edge 6301 = SearchData.row6301.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan432_11 : scan edge (search edge 3) 4 [3037] 3037 innerTail432_11 best = best := by
  rw [innerTail432_11, scan, if_neg (by decide), innerCandidates432_11_checked]
  change scan edge (search edge 3) 4 [3037] 3037 innerTail432_12
    (if 3 ≤ innerCandidates432_11.length then search edge 3 [3037, 6301] innerCandidates432_11 9338 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan432_12

def innerTail432_10 : List Nat := 6007 :: innerTail432_11
def innerCandidates432_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates432_10_checked : candidates edge 4 3037 6007 innerTail432_11 best.bound = innerCandidates432_10 := by
  unfold candidates
  have row : edge 6007 = SearchData.row6007.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan432_10 : scan edge (search edge 3) 4 [3037] 3037 innerTail432_10 best = best := by
  rw [innerTail432_10, scan, if_neg (by decide), innerCandidates432_10_checked]
  change scan edge (search edge 3) 4 [3037] 3037 innerTail432_11
    (if 3 ≤ innerCandidates432_10.length then search edge 3 [3037, 6007] innerCandidates432_10 9044 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan432_11

def innerTail432_9 : List Nat := 5701 :: innerTail432_10
def innerCandidates432_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates432_9_checked : candidates edge 4 3037 5701 innerTail432_10 best.bound = innerCandidates432_9 := by
  unfold candidates
  have row : edge 5701 = SearchData.row5701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan432_9 : scan edge (search edge 3) 4 [3037] 3037 innerTail432_9 best = best := by
  rw [innerTail432_9, scan, if_neg (by decide), innerCandidates432_9_checked]
  change scan edge (search edge 3) 4 [3037] 3037 innerTail432_10
    (if 3 ≤ innerCandidates432_9.length then search edge 3 [3037, 5701] innerCandidates432_9 8738 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan432_10

def innerTail432_8 : List Nat := 5689 :: innerTail432_9
def innerCandidates432_8 : List Nat := [6823, 8329]
set_option maxRecDepth 4096 in
theorem innerCandidates432_8_checked : candidates edge 4 3037 5689 innerTail432_9 best.bound = innerCandidates432_8 := by
  unfold candidates
  have row : edge 5689 = SearchData.row5689.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan432_8 : scan edge (search edge 3) 4 [3037] 3037 innerTail432_8 best = best := by
  rw [innerTail432_8, scan, if_neg (by decide), innerCandidates432_8_checked]
  change scan edge (search edge 3) 4 [3037] 3037 innerTail432_9
    (if 3 ≤ innerCandidates432_8.length then search edge 3 [3037, 5689] innerCandidates432_8 8726 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan432_9

def innerTail432_7 : List Nat := 4987 :: innerTail432_8
def innerCandidates432_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates432_7_checked : candidates edge 4 3037 4987 innerTail432_8 best.bound = innerCandidates432_7 := by
  unfold candidates
  have row : edge 4987 = SearchData.row4987.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan432_7 : scan edge (search edge 3) 4 [3037] 3037 innerTail432_7 best = best := by
  rw [innerTail432_7, scan, if_neg (by decide), innerCandidates432_7_checked]
  change scan edge (search edge 3) 4 [3037] 3037 innerTail432_8
    (if 3 ≤ innerCandidates432_7.length then search edge 3 [3037, 4987] innerCandidates432_7 8024 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan432_8

def innerTail432_6 : List Nat := 4507 :: innerTail432_7
def innerCandidates432_6 : List Nat := [7129]
set_option maxRecDepth 4096 in
theorem innerCandidates432_6_checked : candidates edge 4 3037 4507 innerTail432_7 best.bound = innerCandidates432_6 := by
  unfold candidates
  have row : edge 4507 = SearchData.row4507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan432_6 : scan edge (search edge 3) 4 [3037] 3037 innerTail432_6 best = best := by
  rw [innerTail432_6, scan, if_neg (by decide), innerCandidates432_6_checked]
  change scan edge (search edge 3) 4 [3037] 3037 innerTail432_7
    (if 3 ≤ innerCandidates432_6.length then search edge 3 [3037, 4507] innerCandidates432_6 7544 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan432_7

def innerTail432_5 : List Nat := 4423 :: innerTail432_6
def innerCandidates432_5 : List Nat := [5689]
set_option maxRecDepth 4096 in
theorem innerCandidates432_5_checked : candidates edge 4 3037 4423 innerTail432_6 best.bound = innerCandidates432_5 := by
  unfold candidates
  have row : edge 4423 = SearchData.row4423.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan432_5 : scan edge (search edge 3) 4 [3037] 3037 innerTail432_5 best = best := by
  rw [innerTail432_5, scan, if_neg (by decide), innerCandidates432_5_checked]
  change scan edge (search edge 3) 4 [3037] 3037 innerTail432_6
    (if 3 ≤ innerCandidates432_5.length then search edge 3 [3037, 4423] innerCandidates432_5 7460 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan432_6

def innerTail432_4 : List Nat := 3709 :: innerTail432_5
def innerCandidates432_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates432_4_checked : candidates edge 4 3037 3709 innerTail432_5 best.bound = innerCandidates432_4 := by
  unfold candidates
  have row : edge 3709 = SearchData.row3709.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan432_4 : scan edge (search edge 3) 4 [3037] 3037 innerTail432_4 best = best := by
  rw [innerTail432_4, scan, if_neg (by decide), innerCandidates432_4_checked]
  change scan edge (search edge 3) 4 [3037] 3037 innerTail432_5
    (if 3 ≤ innerCandidates432_4.length then search edge 3 [3037, 3709] innerCandidates432_4 6746 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan432_5

def innerTail432_3 : List Nat := 3517 :: innerTail432_4
def innerCandidates432_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates432_3_checked : candidates edge 4 3037 3517 innerTail432_4 best.bound = innerCandidates432_3 := by
  unfold candidates
  have row : edge 3517 = SearchData.row3517.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan432_3 : scan edge (search edge 3) 4 [3037] 3037 innerTail432_3 best = best := by
  rw [innerTail432_3, scan, if_neg (by decide), innerCandidates432_3_checked]
  change scan edge (search edge 3) 4 [3037] 3037 innerTail432_4
    (if 3 ≤ innerCandidates432_3.length then search edge 3 [3037, 3517] innerCandidates432_3 6554 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan432_4

def innerTail432_2 : List Nat := 3433 :: innerTail432_3
def innerCandidates432_2 : List Nat := [4987, 6607, 7129]
set_option maxRecDepth 4096 in
theorem innerCandidates432_2_checked : candidates edge 4 3037 3433 innerTail432_3 best.bound = innerCandidates432_2 := by
  unfold candidates
  have row : edge 3433 = SearchData.row3433.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild432_2 : search edge 3 [3037, 3433] innerCandidates432_2 6470 best = best := by decide

theorem innerScan432_2 : scan edge (search edge 3) 4 [3037] 3037 innerTail432_2 best = best := by
  rw [innerTail432_2, scan, if_neg (by decide), innerCandidates432_2_checked]
  change scan edge (search edge 3) 4 [3037] 3037 innerTail432_3
    (if 3 ≤ innerCandidates432_2.length then search edge 3 [3037, 3433] innerCandidates432_2 6470 best else best) = best
  rw [if_pos (by decide), innerChild432_2]
  exact innerScan432_3

def innerTail432_1 : List Nat := 3331 :: innerTail432_2
def innerCandidates432_1 : List Nat := [5701]
set_option maxRecDepth 4096 in
theorem innerCandidates432_1_checked : candidates edge 4 3037 3331 innerTail432_2 best.bound = innerCandidates432_1 := by
  unfold candidates
  have row : edge 3331 = SearchData.row3331.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan432_1 : scan edge (search edge 3) 4 [3037] 3037 innerTail432_1 best = best := by
  rw [innerTail432_1, scan, if_neg (by decide), innerCandidates432_1_checked]
  change scan edge (search edge 3) 4 [3037] 3037 innerTail432_2
    (if 3 ≤ innerCandidates432_1.length then search edge 3 [3037, 3331] innerCandidates432_1 6368 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan432_2

def innerTail432_0 : List Nat := 3253 :: innerTail432_1
def innerCandidates432_0 : List Nat := [3517]
set_option maxRecDepth 4096 in
theorem innerCandidates432_0_checked : candidates edge 4 3037 3253 innerTail432_1 best.bound = innerCandidates432_0 := by
  unfold candidates
  have row : edge 3253 = SearchData.row3253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan432_0 : scan edge (search edge 3) 4 [3037] 3037 innerTail432_0 best = best := by
  rw [innerTail432_0, scan, if_neg (by decide), innerCandidates432_0_checked]
  change scan edge (search edge 3) 4 [3037] 3037 innerTail432_1
    (if 3 ≤ innerCandidates432_0.length then search edge 3 [3037, 3253] innerCandidates432_0 6290 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan432_1

theorem rootBranch432 : search edge 4 [3037] rootCandidates432 3037 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3037] 3037 innerTail432_0 best = best
  exact innerScan432_0

def rootCandidates431 : List Nat := [3527, 3767, 3989, 4229, 4637, 4649, 5879, 5981, 6551, 6917, 6959, 7589, 8069]
set_option maxRecDepth 4096 in
theorem rootCandidates431_checked : candidates edge 5 0 3023 tail432 best.bound = rootCandidates431 := by
  unfold candidates
  have row : edge 3023 = SearchData.row3023.contains := by funext q; rfl
  rw [row]
  decide

def innerTail431_13 : List Nat := []
theorem innerScan431_13 : scan edge (search edge 3) 4 [3023] 3023 innerTail431_13 best = best := rfl

def innerTail431_12 : List Nat := 8069 :: innerTail431_13
def innerCandidates431_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates431_12_checked : candidates edge 4 3023 8069 innerTail431_13 best.bound = innerCandidates431_12 := by
  unfold candidates
  have row : edge 8069 = SearchData.row8069.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan431_12 : scan edge (search edge 3) 4 [3023] 3023 innerTail431_12 best = best := by
  rw [innerTail431_12, scan, if_neg (by decide), innerCandidates431_12_checked]
  change scan edge (search edge 3) 4 [3023] 3023 innerTail431_13
    (if 3 ≤ innerCandidates431_12.length then search edge 3 [3023, 8069] innerCandidates431_12 11092 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan431_13

def innerTail431_11 : List Nat := 7589 :: innerTail431_12
def innerCandidates431_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates431_11_checked : candidates edge 4 3023 7589 innerTail431_12 best.bound = innerCandidates431_11 := by
  unfold candidates
  have row : edge 7589 = SearchData.row7589.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan431_11 : scan edge (search edge 3) 4 [3023] 3023 innerTail431_11 best = best := by
  rw [innerTail431_11, scan, if_neg (by decide), innerCandidates431_11_checked]
  change scan edge (search edge 3) 4 [3023] 3023 innerTail431_12
    (if 3 ≤ innerCandidates431_11.length then search edge 3 [3023, 7589] innerCandidates431_11 10612 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan431_12

def innerTail431_10 : List Nat := 6959 :: innerTail431_11
def innerCandidates431_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates431_10_checked : candidates edge 4 3023 6959 innerTail431_11 best.bound = innerCandidates431_10 := by
  unfold candidates
  have row : edge 6959 = SearchData.row6959.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan431_10 : scan edge (search edge 3) 4 [3023] 3023 innerTail431_10 best = best := by
  rw [innerTail431_10, scan, if_neg (by decide), innerCandidates431_10_checked]
  change scan edge (search edge 3) 4 [3023] 3023 innerTail431_11
    (if 3 ≤ innerCandidates431_10.length then search edge 3 [3023, 6959] innerCandidates431_10 9982 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan431_11

def innerTail431_9 : List Nat := 6917 :: innerTail431_10
def innerCandidates431_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates431_9_checked : candidates edge 4 3023 6917 innerTail431_10 best.bound = innerCandidates431_9 := by
  unfold candidates
  have row : edge 6917 = SearchData.row6917.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan431_9 : scan edge (search edge 3) 4 [3023] 3023 innerTail431_9 best = best := by
  rw [innerTail431_9, scan, if_neg (by decide), innerCandidates431_9_checked]
  change scan edge (search edge 3) 4 [3023] 3023 innerTail431_10
    (if 3 ≤ innerCandidates431_9.length then search edge 3 [3023, 6917] innerCandidates431_9 9940 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan431_10

def innerTail431_8 : List Nat := 6551 :: innerTail431_9
def innerCandidates431_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates431_8_checked : candidates edge 4 3023 6551 innerTail431_9 best.bound = innerCandidates431_8 := by
  unfold candidates
  have row : edge 6551 = SearchData.row6551.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan431_8 : scan edge (search edge 3) 4 [3023] 3023 innerTail431_8 best = best := by
  rw [innerTail431_8, scan, if_neg (by decide), innerCandidates431_8_checked]
  change scan edge (search edge 3) 4 [3023] 3023 innerTail431_9
    (if 3 ≤ innerCandidates431_8.length then search edge 3 [3023, 6551] innerCandidates431_8 9574 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan431_9

def innerTail431_7 : List Nat := 5981 :: innerTail431_8
def innerCandidates431_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates431_7_checked : candidates edge 4 3023 5981 innerTail431_8 best.bound = innerCandidates431_7 := by
  unfold candidates
  have row : edge 5981 = SearchData.row5981.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan431_7 : scan edge (search edge 3) 4 [3023] 3023 innerTail431_7 best = best := by
  rw [innerTail431_7, scan, if_neg (by decide), innerCandidates431_7_checked]
  change scan edge (search edge 3) 4 [3023] 3023 innerTail431_8
    (if 3 ≤ innerCandidates431_7.length then search edge 3 [3023, 5981] innerCandidates431_7 9004 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan431_8

def innerTail431_6 : List Nat := 5879 :: innerTail431_7
def innerCandidates431_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates431_6_checked : candidates edge 4 3023 5879 innerTail431_7 best.bound = innerCandidates431_6 := by
  unfold candidates
  have row : edge 5879 = SearchData.row5879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan431_6 : scan edge (search edge 3) 4 [3023] 3023 innerTail431_6 best = best := by
  rw [innerTail431_6, scan, if_neg (by decide), innerCandidates431_6_checked]
  change scan edge (search edge 3) 4 [3023] 3023 innerTail431_7
    (if 3 ≤ innerCandidates431_6.length then search edge 3 [3023, 5879] innerCandidates431_6 8902 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan431_7

def innerTail431_5 : List Nat := 4649 :: innerTail431_6
def innerCandidates431_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates431_5_checked : candidates edge 4 3023 4649 innerTail431_6 best.bound = innerCandidates431_5 := by
  unfold candidates
  have row : edge 4649 = SearchData.row4649.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan431_5 : scan edge (search edge 3) 4 [3023] 3023 innerTail431_5 best = best := by
  rw [innerTail431_5, scan, if_neg (by decide), innerCandidates431_5_checked]
  change scan edge (search edge 3) 4 [3023] 3023 innerTail431_6
    (if 3 ≤ innerCandidates431_5.length then search edge 3 [3023, 4649] innerCandidates431_5 7672 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan431_6

def innerTail431_4 : List Nat := 4637 :: innerTail431_5
def innerCandidates431_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates431_4_checked : candidates edge 4 3023 4637 innerTail431_5 best.bound = innerCandidates431_4 := by
  unfold candidates
  have row : edge 4637 = SearchData.row4637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan431_4 : scan edge (search edge 3) 4 [3023] 3023 innerTail431_4 best = best := by
  rw [innerTail431_4, scan, if_neg (by decide), innerCandidates431_4_checked]
  change scan edge (search edge 3) 4 [3023] 3023 innerTail431_5
    (if 3 ≤ innerCandidates431_4.length then search edge 3 [3023, 4637] innerCandidates431_4 7660 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan431_5

def innerTail431_3 : List Nat := 4229 :: innerTail431_4
def innerCandidates431_3 : List Nat := [7589]
set_option maxRecDepth 4096 in
theorem innerCandidates431_3_checked : candidates edge 4 3023 4229 innerTail431_4 best.bound = innerCandidates431_3 := by
  unfold candidates
  have row : edge 4229 = SearchData.row4229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan431_3 : scan edge (search edge 3) 4 [3023] 3023 innerTail431_3 best = best := by
  rw [innerTail431_3, scan, if_neg (by decide), innerCandidates431_3_checked]
  change scan edge (search edge 3) 4 [3023] 3023 innerTail431_4
    (if 3 ≤ innerCandidates431_3.length then search edge 3 [3023, 4229] innerCandidates431_3 7252 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan431_4

def innerTail431_2 : List Nat := 3989 :: innerTail431_3
def innerCandidates431_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates431_2_checked : candidates edge 4 3023 3989 innerTail431_3 best.bound = innerCandidates431_2 := by
  unfold candidates
  have row : edge 3989 = SearchData.row3989.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan431_2 : scan edge (search edge 3) 4 [3023] 3023 innerTail431_2 best = best := by
  rw [innerTail431_2, scan, if_neg (by decide), innerCandidates431_2_checked]
  change scan edge (search edge 3) 4 [3023] 3023 innerTail431_3
    (if 3 ≤ innerCandidates431_2.length then search edge 3 [3023, 3989] innerCandidates431_2 7012 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan431_3

def innerTail431_1 : List Nat := 3767 :: innerTail431_2
def innerCandidates431_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates431_1_checked : candidates edge 4 3023 3767 innerTail431_2 best.bound = innerCandidates431_1 := by
  unfold candidates
  have row : edge 3767 = SearchData.row3767.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan431_1 : scan edge (search edge 3) 4 [3023] 3023 innerTail431_1 best = best := by
  rw [innerTail431_1, scan, if_neg (by decide), innerCandidates431_1_checked]
  change scan edge (search edge 3) 4 [3023] 3023 innerTail431_2
    (if 3 ≤ innerCandidates431_1.length then search edge 3 [3023, 3767] innerCandidates431_1 6790 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan431_2

def innerTail431_0 : List Nat := 3527 :: innerTail431_1
def innerCandidates431_0 : List Nat := [4649, 6551]
set_option maxRecDepth 4096 in
theorem innerCandidates431_0_checked : candidates edge 4 3023 3527 innerTail431_1 best.bound = innerCandidates431_0 := by
  unfold candidates
  have row : edge 3527 = SearchData.row3527.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan431_0 : scan edge (search edge 3) 4 [3023] 3023 innerTail431_0 best = best := by
  rw [innerTail431_0, scan, if_neg (by decide), innerCandidates431_0_checked]
  change scan edge (search edge 3) 4 [3023] 3023 innerTail431_1
    (if 3 ≤ innerCandidates431_0.length then search edge 3 [3023, 3527] innerCandidates431_0 6550 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan431_1

theorem rootBranch431 : search edge 4 [3023] rootCandidates431 3023 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3023] 3023 innerTail431_0 best = best
  exact innerScan431_0

def rootCandidates430 : List Nat := [3217, 3571, 4519, 4603, 4621, 5413, 5419, 5779, 5839, 6553, 6571, 6841, 7351, 7417, 7603, 8269, 8317]
set_option maxRecDepth 4096 in
theorem rootCandidates430_checked : candidates edge 5 0 3019 tail431 best.bound = rootCandidates430 := by
  unfold candidates
  have row : edge 3019 = SearchData.row3019.contains := by funext q; rfl
  rw [row]
  decide

def innerTail430_17 : List Nat := []
theorem innerScan430_17 : scan edge (search edge 3) 4 [3019] 3019 innerTail430_17 best = best := rfl

def innerTail430_16 : List Nat := 8317 :: innerTail430_17
def innerCandidates430_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates430_16_checked : candidates edge 4 3019 8317 innerTail430_17 best.bound = innerCandidates430_16 := by
  unfold candidates
  have row : edge 8317 = SearchData.row8317.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan430_16 : scan edge (search edge 3) 4 [3019] 3019 innerTail430_16 best = best := by
  rw [innerTail430_16, scan, if_neg (by decide), innerCandidates430_16_checked]
  change scan edge (search edge 3) 4 [3019] 3019 innerTail430_17
    (if 3 ≤ innerCandidates430_16.length then search edge 3 [3019, 8317] innerCandidates430_16 11336 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan430_17

def innerTail430_15 : List Nat := 8269 :: innerTail430_16
def innerCandidates430_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates430_15_checked : candidates edge 4 3019 8269 innerTail430_16 best.bound = innerCandidates430_15 := by
  unfold candidates
  have row : edge 8269 = SearchData.row8269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan430_15 : scan edge (search edge 3) 4 [3019] 3019 innerTail430_15 best = best := by
  rw [innerTail430_15, scan, if_neg (by decide), innerCandidates430_15_checked]
  change scan edge (search edge 3) 4 [3019] 3019 innerTail430_16
    (if 3 ≤ innerCandidates430_15.length then search edge 3 [3019, 8269] innerCandidates430_15 11288 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan430_16

def innerTail430_14 : List Nat := 7603 :: innerTail430_15
def innerCandidates430_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates430_14_checked : candidates edge 4 3019 7603 innerTail430_15 best.bound = innerCandidates430_14 := by
  unfold candidates
  have row : edge 7603 = SearchData.row7603.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan430_14 : scan edge (search edge 3) 4 [3019] 3019 innerTail430_14 best = best := by
  rw [innerTail430_14, scan, if_neg (by decide), innerCandidates430_14_checked]
  change scan edge (search edge 3) 4 [3019] 3019 innerTail430_15
    (if 3 ≤ innerCandidates430_14.length then search edge 3 [3019, 7603] innerCandidates430_14 10622 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan430_15

def innerTail430_13 : List Nat := 7417 :: innerTail430_14
def innerCandidates430_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates430_13_checked : candidates edge 4 3019 7417 innerTail430_14 best.bound = innerCandidates430_13 := by
  unfold candidates
  have row : edge 7417 = SearchData.row7417.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan430_13 : scan edge (search edge 3) 4 [3019] 3019 innerTail430_13 best = best := by
  rw [innerTail430_13, scan, if_neg (by decide), innerCandidates430_13_checked]
  change scan edge (search edge 3) 4 [3019] 3019 innerTail430_14
    (if 3 ≤ innerCandidates430_13.length then search edge 3 [3019, 7417] innerCandidates430_13 10436 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan430_14

def innerTail430_12 : List Nat := 7351 :: innerTail430_13
def innerCandidates430_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates430_12_checked : candidates edge 4 3019 7351 innerTail430_13 best.bound = innerCandidates430_12 := by
  unfold candidates
  have row : edge 7351 = SearchData.row7351.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan430_12 : scan edge (search edge 3) 4 [3019] 3019 innerTail430_12 best = best := by
  rw [innerTail430_12, scan, if_neg (by decide), innerCandidates430_12_checked]
  change scan edge (search edge 3) 4 [3019] 3019 innerTail430_13
    (if 3 ≤ innerCandidates430_12.length then search edge 3 [3019, 7351] innerCandidates430_12 10370 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan430_13

def innerTail430_11 : List Nat := 6841 :: innerTail430_12
def innerCandidates430_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates430_11_checked : candidates edge 4 3019 6841 innerTail430_12 best.bound = innerCandidates430_11 := by
  unfold candidates
  have row : edge 6841 = SearchData.row6841.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan430_11 : scan edge (search edge 3) 4 [3019] 3019 innerTail430_11 best = best := by
  rw [innerTail430_11, scan, if_neg (by decide), innerCandidates430_11_checked]
  change scan edge (search edge 3) 4 [3019] 3019 innerTail430_12
    (if 3 ≤ innerCandidates430_11.length then search edge 3 [3019, 6841] innerCandidates430_11 9860 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan430_12

def innerTail430_10 : List Nat := 6571 :: innerTail430_11
def innerCandidates430_10 : List Nat := [8269]
set_option maxRecDepth 4096 in
theorem innerCandidates430_10_checked : candidates edge 4 3019 6571 innerTail430_11 best.bound = innerCandidates430_10 := by
  unfold candidates
  have row : edge 6571 = SearchData.row6571.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan430_10 : scan edge (search edge 3) 4 [3019] 3019 innerTail430_10 best = best := by
  rw [innerTail430_10, scan, if_neg (by decide), innerCandidates430_10_checked]
  change scan edge (search edge 3) 4 [3019] 3019 innerTail430_11
    (if 3 ≤ innerCandidates430_10.length then search edge 3 [3019, 6571] innerCandidates430_10 9590 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan430_11

def innerTail430_9 : List Nat := 6553 :: innerTail430_10
def innerCandidates430_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates430_9_checked : candidates edge 4 3019 6553 innerTail430_10 best.bound = innerCandidates430_9 := by
  unfold candidates
  have row : edge 6553 = SearchData.row6553.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan430_9 : scan edge (search edge 3) 4 [3019] 3019 innerTail430_9 best = best := by
  rw [innerTail430_9, scan, if_neg (by decide), innerCandidates430_9_checked]
  change scan edge (search edge 3) 4 [3019] 3019 innerTail430_10
    (if 3 ≤ innerCandidates430_9.length then search edge 3 [3019, 6553] innerCandidates430_9 9572 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan430_10

def innerTail430_8 : List Nat := 5839 :: innerTail430_9
def innerCandidates430_8 : List Nat := [8269]
set_option maxRecDepth 4096 in
theorem innerCandidates430_8_checked : candidates edge 4 3019 5839 innerTail430_9 best.bound = innerCandidates430_8 := by
  unfold candidates
  have row : edge 5839 = SearchData.row5839.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan430_8 : scan edge (search edge 3) 4 [3019] 3019 innerTail430_8 best = best := by
  rw [innerTail430_8, scan, if_neg (by decide), innerCandidates430_8_checked]
  change scan edge (search edge 3) 4 [3019] 3019 innerTail430_9
    (if 3 ≤ innerCandidates430_8.length then search edge 3 [3019, 5839] innerCandidates430_8 8858 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan430_9

def innerTail430_7 : List Nat := 5779 :: innerTail430_8
def innerCandidates430_7 : List Nat := [8269]
set_option maxRecDepth 4096 in
theorem innerCandidates430_7_checked : candidates edge 4 3019 5779 innerTail430_8 best.bound = innerCandidates430_7 := by
  unfold candidates
  have row : edge 5779 = SearchData.row5779.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan430_7 : scan edge (search edge 3) 4 [3019] 3019 innerTail430_7 best = best := by
  rw [innerTail430_7, scan, if_neg (by decide), innerCandidates430_7_checked]
  change scan edge (search edge 3) 4 [3019] 3019 innerTail430_8
    (if 3 ≤ innerCandidates430_7.length then search edge 3 [3019, 5779] innerCandidates430_7 8798 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan430_8

def innerTail430_6 : List Nat := 5419 :: innerTail430_7
def innerCandidates430_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates430_6_checked : candidates edge 4 3019 5419 innerTail430_7 best.bound = innerCandidates430_6 := by
  unfold candidates
  have row : edge 5419 = SearchData.row5419.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan430_6 : scan edge (search edge 3) 4 [3019] 3019 innerTail430_6 best = best := by
  rw [innerTail430_6, scan, if_neg (by decide), innerCandidates430_6_checked]
  change scan edge (search edge 3) 4 [3019] 3019 innerTail430_7
    (if 3 ≤ innerCandidates430_6.length then search edge 3 [3019, 5419] innerCandidates430_6 8438 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan430_7

def innerTail430_5 : List Nat := 5413 :: innerTail430_6
def innerCandidates430_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates430_5_checked : candidates edge 4 3019 5413 innerTail430_6 best.bound = innerCandidates430_5 := by
  unfold candidates
  have row : edge 5413 = SearchData.row5413.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan430_5 : scan edge (search edge 3) 4 [3019] 3019 innerTail430_5 best = best := by
  rw [innerTail430_5, scan, if_neg (by decide), innerCandidates430_5_checked]
  change scan edge (search edge 3) 4 [3019] 3019 innerTail430_6
    (if 3 ≤ innerCandidates430_5.length then search edge 3 [3019, 5413] innerCandidates430_5 8432 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan430_6

def innerTail430_4 : List Nat := 4621 :: innerTail430_5
def innerCandidates430_4 : List Nat := [5779]
set_option maxRecDepth 4096 in
theorem innerCandidates430_4_checked : candidates edge 4 3019 4621 innerTail430_5 best.bound = innerCandidates430_4 := by
  unfold candidates
  have row : edge 4621 = SearchData.row4621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan430_4 : scan edge (search edge 3) 4 [3019] 3019 innerTail430_4 best = best := by
  rw [innerTail430_4, scan, if_neg (by decide), innerCandidates430_4_checked]
  change scan edge (search edge 3) 4 [3019] 3019 innerTail430_5
    (if 3 ≤ innerCandidates430_4.length then search edge 3 [3019, 4621] innerCandidates430_4 7640 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan430_5

def innerTail430_3 : List Nat := 4603 :: innerTail430_4
def innerCandidates430_3 : List Nat := [7603]
set_option maxRecDepth 4096 in
theorem innerCandidates430_3_checked : candidates edge 4 3019 4603 innerTail430_4 best.bound = innerCandidates430_3 := by
  unfold candidates
  have row : edge 4603 = SearchData.row4603.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan430_3 : scan edge (search edge 3) 4 [3019] 3019 innerTail430_3 best = best := by
  rw [innerTail430_3, scan, if_neg (by decide), innerCandidates430_3_checked]
  change scan edge (search edge 3) 4 [3019] 3019 innerTail430_4
    (if 3 ≤ innerCandidates430_3.length then search edge 3 [3019, 4603] innerCandidates430_3 7622 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan430_4

def innerTail430_2 : List Nat := 4519 :: innerTail430_3
def innerCandidates430_2 : List Nat := [5419]
set_option maxRecDepth 4096 in
theorem innerCandidates430_2_checked : candidates edge 4 3019 4519 innerTail430_3 best.bound = innerCandidates430_2 := by
  unfold candidates
  have row : edge 4519 = SearchData.row4519.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan430_2 : scan edge (search edge 3) 4 [3019] 3019 innerTail430_2 best = best := by
  rw [innerTail430_2, scan, if_neg (by decide), innerCandidates430_2_checked]
  change scan edge (search edge 3) 4 [3019] 3019 innerTail430_3
    (if 3 ≤ innerCandidates430_2.length then search edge 3 [3019, 4519] innerCandidates430_2 7538 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan430_3

def innerTail430_1 : List Nat := 3571 :: innerTail430_2
def innerCandidates430_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates430_1_checked : candidates edge 4 3019 3571 innerTail430_2 best.bound = innerCandidates430_1 := by
  unfold candidates
  have row : edge 3571 = SearchData.row3571.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan430_1 : scan edge (search edge 3) 4 [3019] 3019 innerTail430_1 best = best := by
  rw [innerTail430_1, scan, if_neg (by decide), innerCandidates430_1_checked]
  change scan edge (search edge 3) 4 [3019] 3019 innerTail430_2
    (if 3 ≤ innerCandidates430_1.length then search edge 3 [3019, 3571] innerCandidates430_1 6590 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan430_2

def innerTail430_0 : List Nat := 3217 :: innerTail430_1
def innerCandidates430_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates430_0_checked : candidates edge 4 3019 3217 innerTail430_1 best.bound = innerCandidates430_0 := by
  unfold candidates
  have row : edge 3217 = SearchData.row3217.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan430_0 : scan edge (search edge 3) 4 [3019] 3019 innerTail430_0 best = best := by
  rw [innerTail430_0, scan, if_neg (by decide), innerCandidates430_0_checked]
  change scan edge (search edge 3) 4 [3019] 3019 innerTail430_1
    (if 3 ≤ innerCandidates430_0.length then search edge 3 [3019, 3217] innerCandidates430_0 6236 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan430_1

theorem rootBranch430 : search edge 4 [3019] rootCandidates430 3019 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3019] 3019 innerTail430_0 best = best
  exact innerScan430_0

end ProjectEulerSolutions.PrimePairUpper.Search
