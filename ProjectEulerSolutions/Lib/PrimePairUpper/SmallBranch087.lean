import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates447 : List Nat := [3343, 4099, 4219, 4273, 4507, 4903, 5011, 5107, 5527, 6073, 6481, 6709, 7507, 8377]
set_option maxRecDepth 4096 in
theorem rootCandidates447_checked : candidates edge 5 0 3181 tail448 best.bound = rootCandidates447 := by
  unfold candidates
  have row : edge 3181 = SearchData.row3181.contains := by funext q; rfl
  rw [row]
  decide

def innerTail447_14 : List Nat := []
theorem innerScan447_14 : scan edge (search edge 3) 4 [3181] 3181 innerTail447_14 best = best := rfl

def innerTail447_13 : List Nat := 8377 :: innerTail447_14
def innerCandidates447_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates447_13_checked : candidates edge 4 3181 8377 innerTail447_14 best.bound = innerCandidates447_13 := by
  unfold candidates
  have row : edge 8377 = SearchData.row8377.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan447_13 : scan edge (search edge 3) 4 [3181] 3181 innerTail447_13 best = best := by
  rw [innerTail447_13, scan, if_neg (by decide), innerCandidates447_13_checked]
  change scan edge (search edge 3) 4 [3181] 3181 innerTail447_14
    (if 3 ≤ innerCandidates447_13.length then search edge 3 [3181, 8377] innerCandidates447_13 11558 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan447_14

def innerTail447_12 : List Nat := 7507 :: innerTail447_13
def innerCandidates447_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates447_12_checked : candidates edge 4 3181 7507 innerTail447_13 best.bound = innerCandidates447_12 := by
  unfold candidates
  have row : edge 7507 = SearchData.row7507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan447_12 : scan edge (search edge 3) 4 [3181] 3181 innerTail447_12 best = best := by
  rw [innerTail447_12, scan, if_neg (by decide), innerCandidates447_12_checked]
  change scan edge (search edge 3) 4 [3181] 3181 innerTail447_13
    (if 3 ≤ innerCandidates447_12.length then search edge 3 [3181, 7507] innerCandidates447_12 10688 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan447_13

def innerTail447_11 : List Nat := 6709 :: innerTail447_12
def innerCandidates447_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates447_11_checked : candidates edge 4 3181 6709 innerTail447_12 best.bound = innerCandidates447_11 := by
  unfold candidates
  have row : edge 6709 = SearchData.row6709.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan447_11 : scan edge (search edge 3) 4 [3181] 3181 innerTail447_11 best = best := by
  rw [innerTail447_11, scan, if_neg (by decide), innerCandidates447_11_checked]
  change scan edge (search edge 3) 4 [3181] 3181 innerTail447_12
    (if 3 ≤ innerCandidates447_11.length then search edge 3 [3181, 6709] innerCandidates447_11 9890 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan447_12

def innerTail447_10 : List Nat := 6481 :: innerTail447_11
def innerCandidates447_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates447_10_checked : candidates edge 4 3181 6481 innerTail447_11 best.bound = innerCandidates447_10 := by
  unfold candidates
  have row : edge 6481 = SearchData.row6481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan447_10 : scan edge (search edge 3) 4 [3181] 3181 innerTail447_10 best = best := by
  rw [innerTail447_10, scan, if_neg (by decide), innerCandidates447_10_checked]
  change scan edge (search edge 3) 4 [3181] 3181 innerTail447_11
    (if 3 ≤ innerCandidates447_10.length then search edge 3 [3181, 6481] innerCandidates447_10 9662 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan447_11

def innerTail447_9 : List Nat := 6073 :: innerTail447_10
def innerCandidates447_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates447_9_checked : candidates edge 4 3181 6073 innerTail447_10 best.bound = innerCandidates447_9 := by
  unfold candidates
  have row : edge 6073 = SearchData.row6073.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan447_9 : scan edge (search edge 3) 4 [3181] 3181 innerTail447_9 best = best := by
  rw [innerTail447_9, scan, if_neg (by decide), innerCandidates447_9_checked]
  change scan edge (search edge 3) 4 [3181] 3181 innerTail447_10
    (if 3 ≤ innerCandidates447_9.length then search edge 3 [3181, 6073] innerCandidates447_9 9254 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan447_10

def innerTail447_8 : List Nat := 5527 :: innerTail447_9
def innerCandidates447_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates447_8_checked : candidates edge 4 3181 5527 innerTail447_9 best.bound = innerCandidates447_8 := by
  unfold candidates
  have row : edge 5527 = SearchData.row5527.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan447_8 : scan edge (search edge 3) 4 [3181] 3181 innerTail447_8 best = best := by
  rw [innerTail447_8, scan, if_neg (by decide), innerCandidates447_8_checked]
  change scan edge (search edge 3) 4 [3181] 3181 innerTail447_9
    (if 3 ≤ innerCandidates447_8.length then search edge 3 [3181, 5527] innerCandidates447_8 8708 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan447_9

def innerTail447_7 : List Nat := 5107 :: innerTail447_8
def innerCandidates447_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates447_7_checked : candidates edge 4 3181 5107 innerTail447_8 best.bound = innerCandidates447_7 := by
  unfold candidates
  have row : edge 5107 = SearchData.row5107.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan447_7 : scan edge (search edge 3) 4 [3181] 3181 innerTail447_7 best = best := by
  rw [innerTail447_7, scan, if_neg (by decide), innerCandidates447_7_checked]
  change scan edge (search edge 3) 4 [3181] 3181 innerTail447_8
    (if 3 ≤ innerCandidates447_7.length then search edge 3 [3181, 5107] innerCandidates447_7 8288 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan447_8

def innerTail447_6 : List Nat := 5011 :: innerTail447_7
def innerCandidates447_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates447_6_checked : candidates edge 4 3181 5011 innerTail447_7 best.bound = innerCandidates447_6 := by
  unfold candidates
  have row : edge 5011 = SearchData.row5011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan447_6 : scan edge (search edge 3) 4 [3181] 3181 innerTail447_6 best = best := by
  rw [innerTail447_6, scan, if_neg (by decide), innerCandidates447_6_checked]
  change scan edge (search edge 3) 4 [3181] 3181 innerTail447_7
    (if 3 ≤ innerCandidates447_6.length then search edge 3 [3181, 5011] innerCandidates447_6 8192 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan447_7

def innerTail447_5 : List Nat := 4903 :: innerTail447_6
def innerCandidates447_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates447_5_checked : candidates edge 4 3181 4903 innerTail447_6 best.bound = innerCandidates447_5 := by
  unfold candidates
  have row : edge 4903 = SearchData.row4903.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan447_5 : scan edge (search edge 3) 4 [3181] 3181 innerTail447_5 best = best := by
  rw [innerTail447_5, scan, if_neg (by decide), innerCandidates447_5_checked]
  change scan edge (search edge 3) 4 [3181] 3181 innerTail447_6
    (if 3 ≤ innerCandidates447_5.length then search edge 3 [3181, 4903] innerCandidates447_5 8084 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan447_6

def innerTail447_4 : List Nat := 4507 :: innerTail447_5
def innerCandidates447_4 : List Nat := [6709, 8377]
set_option maxRecDepth 4096 in
theorem innerCandidates447_4_checked : candidates edge 4 3181 4507 innerTail447_5 best.bound = innerCandidates447_4 := by
  unfold candidates
  have row : edge 4507 = SearchData.row4507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan447_4 : scan edge (search edge 3) 4 [3181] 3181 innerTail447_4 best = best := by
  rw [innerTail447_4, scan, if_neg (by decide), innerCandidates447_4_checked]
  change scan edge (search edge 3) 4 [3181] 3181 innerTail447_5
    (if 3 ≤ innerCandidates447_4.length then search edge 3 [3181, 4507] innerCandidates447_4 7688 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan447_5

def innerTail447_3 : List Nat := 4273 :: innerTail447_4
def innerCandidates447_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates447_3_checked : candidates edge 4 3181 4273 innerTail447_4 best.bound = innerCandidates447_3 := by
  unfold candidates
  have row : edge 4273 = SearchData.row4273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan447_3 : scan edge (search edge 3) 4 [3181] 3181 innerTail447_3 best = best := by
  rw [innerTail447_3, scan, if_neg (by decide), innerCandidates447_3_checked]
  change scan edge (search edge 3) 4 [3181] 3181 innerTail447_4
    (if 3 ≤ innerCandidates447_3.length then search edge 3 [3181, 4273] innerCandidates447_3 7454 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan447_4

def innerTail447_2 : List Nat := 4219 :: innerTail447_3
def innerCandidates447_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates447_2_checked : candidates edge 4 3181 4219 innerTail447_3 best.bound = innerCandidates447_2 := by
  unfold candidates
  have row : edge 4219 = SearchData.row4219.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan447_2 : scan edge (search edge 3) 4 [3181] 3181 innerTail447_2 best = best := by
  rw [innerTail447_2, scan, if_neg (by decide), innerCandidates447_2_checked]
  change scan edge (search edge 3) 4 [3181] 3181 innerTail447_3
    (if 3 ≤ innerCandidates447_2.length then search edge 3 [3181, 4219] innerCandidates447_2 7400 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan447_3

def innerTail447_1 : List Nat := 4099 :: innerTail447_2
def innerCandidates447_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates447_1_checked : candidates edge 4 3181 4099 innerTail447_2 best.bound = innerCandidates447_1 := by
  unfold candidates
  have row : edge 4099 = SearchData.row4099.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan447_1 : scan edge (search edge 3) 4 [3181] 3181 innerTail447_1 best = best := by
  rw [innerTail447_1, scan, if_neg (by decide), innerCandidates447_1_checked]
  change scan edge (search edge 3) 4 [3181] 3181 innerTail447_2
    (if 3 ≤ innerCandidates447_1.length then search edge 3 [3181, 4099] innerCandidates447_1 7280 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan447_2

def innerTail447_0 : List Nat := 3343 :: innerTail447_1
def innerCandidates447_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates447_0_checked : candidates edge 4 3181 3343 innerTail447_1 best.bound = innerCandidates447_0 := by
  unfold candidates
  have row : edge 3343 = SearchData.row3343.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan447_0 : scan edge (search edge 3) 4 [3181] 3181 innerTail447_0 best = best := by
  rw [innerTail447_0, scan, if_neg (by decide), innerCandidates447_0_checked]
  change scan edge (search edge 3) 4 [3181] 3181 innerTail447_1
    (if 3 ≤ innerCandidates447_0.length then search edge 3 [3181, 3343] innerCandidates447_0 6524 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan447_1

theorem rootBranch447 : search edge 4 [3181] rootCandidates447 3181 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3181] 3181 innerTail447_0 best = best
  exact innerScan447_0

def rootCandidates446 : List Nat := [3229, 3391, 3733, 5101, 5119, 5953, 6037, 6361, 7213, 8179, 8377]
set_option maxRecDepth 4096 in
theorem rootCandidates446_checked : candidates edge 5 0 3169 tail447 best.bound = rootCandidates446 := by
  unfold candidates
  have row : edge 3169 = SearchData.row3169.contains := by funext q; rfl
  rw [row]
  decide

def innerTail446_11 : List Nat := []
theorem innerScan446_11 : scan edge (search edge 3) 4 [3169] 3169 innerTail446_11 best = best := rfl

def innerTail446_10 : List Nat := 8377 :: innerTail446_11
def innerCandidates446_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates446_10_checked : candidates edge 4 3169 8377 innerTail446_11 best.bound = innerCandidates446_10 := by
  unfold candidates
  have row : edge 8377 = SearchData.row8377.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan446_10 : scan edge (search edge 3) 4 [3169] 3169 innerTail446_10 best = best := by
  rw [innerTail446_10, scan, if_neg (by decide), innerCandidates446_10_checked]
  change scan edge (search edge 3) 4 [3169] 3169 innerTail446_11
    (if 3 ≤ innerCandidates446_10.length then search edge 3 [3169, 8377] innerCandidates446_10 11546 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan446_11

def innerTail446_9 : List Nat := 8179 :: innerTail446_10
def innerCandidates446_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates446_9_checked : candidates edge 4 3169 8179 innerTail446_10 best.bound = innerCandidates446_9 := by
  unfold candidates
  have row : edge 8179 = SearchData.row8179.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan446_9 : scan edge (search edge 3) 4 [3169] 3169 innerTail446_9 best = best := by
  rw [innerTail446_9, scan, if_neg (by decide), innerCandidates446_9_checked]
  change scan edge (search edge 3) 4 [3169] 3169 innerTail446_10
    (if 3 ≤ innerCandidates446_9.length then search edge 3 [3169, 8179] innerCandidates446_9 11348 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan446_10

def innerTail446_8 : List Nat := 7213 :: innerTail446_9
def innerCandidates446_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates446_8_checked : candidates edge 4 3169 7213 innerTail446_9 best.bound = innerCandidates446_8 := by
  unfold candidates
  have row : edge 7213 = SearchData.row7213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan446_8 : scan edge (search edge 3) 4 [3169] 3169 innerTail446_8 best = best := by
  rw [innerTail446_8, scan, if_neg (by decide), innerCandidates446_8_checked]
  change scan edge (search edge 3) 4 [3169] 3169 innerTail446_9
    (if 3 ≤ innerCandidates446_8.length then search edge 3 [3169, 7213] innerCandidates446_8 10382 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan446_9

def innerTail446_7 : List Nat := 6361 :: innerTail446_8
def innerCandidates446_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates446_7_checked : candidates edge 4 3169 6361 innerTail446_8 best.bound = innerCandidates446_7 := by
  unfold candidates
  have row : edge 6361 = SearchData.row6361.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan446_7 : scan edge (search edge 3) 4 [3169] 3169 innerTail446_7 best = best := by
  rw [innerTail446_7, scan, if_neg (by decide), innerCandidates446_7_checked]
  change scan edge (search edge 3) 4 [3169] 3169 innerTail446_8
    (if 3 ≤ innerCandidates446_7.length then search edge 3 [3169, 6361] innerCandidates446_7 9530 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan446_8

def innerTail446_6 : List Nat := 6037 :: innerTail446_7
def innerCandidates446_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates446_6_checked : candidates edge 4 3169 6037 innerTail446_7 best.bound = innerCandidates446_6 := by
  unfold candidates
  have row : edge 6037 = SearchData.row6037.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan446_6 : scan edge (search edge 3) 4 [3169] 3169 innerTail446_6 best = best := by
  rw [innerTail446_6, scan, if_neg (by decide), innerCandidates446_6_checked]
  change scan edge (search edge 3) 4 [3169] 3169 innerTail446_7
    (if 3 ≤ innerCandidates446_6.length then search edge 3 [3169, 6037] innerCandidates446_6 9206 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan446_7

def innerTail446_5 : List Nat := 5953 :: innerTail446_6
def innerCandidates446_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates446_5_checked : candidates edge 4 3169 5953 innerTail446_6 best.bound = innerCandidates446_5 := by
  unfold candidates
  have row : edge 5953 = SearchData.row5953.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan446_5 : scan edge (search edge 3) 4 [3169] 3169 innerTail446_5 best = best := by
  rw [innerTail446_5, scan, if_neg (by decide), innerCandidates446_5_checked]
  change scan edge (search edge 3) 4 [3169] 3169 innerTail446_6
    (if 3 ≤ innerCandidates446_5.length then search edge 3 [3169, 5953] innerCandidates446_5 9122 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan446_6

def innerTail446_4 : List Nat := 5119 :: innerTail446_5
def innerCandidates446_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates446_4_checked : candidates edge 4 3169 5119 innerTail446_5 best.bound = innerCandidates446_4 := by
  unfold candidates
  have row : edge 5119 = SearchData.row5119.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan446_4 : scan edge (search edge 3) 4 [3169] 3169 innerTail446_4 best = best := by
  rw [innerTail446_4, scan, if_neg (by decide), innerCandidates446_4_checked]
  change scan edge (search edge 3) 4 [3169] 3169 innerTail446_5
    (if 3 ≤ innerCandidates446_4.length then search edge 3 [3169, 5119] innerCandidates446_4 8288 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan446_5

def innerTail446_3 : List Nat := 5101 :: innerTail446_4
def innerCandidates446_3 : List Nat := [5953]
set_option maxRecDepth 4096 in
theorem innerCandidates446_3_checked : candidates edge 4 3169 5101 innerTail446_4 best.bound = innerCandidates446_3 := by
  unfold candidates
  have row : edge 5101 = SearchData.row5101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan446_3 : scan edge (search edge 3) 4 [3169] 3169 innerTail446_3 best = best := by
  rw [innerTail446_3, scan, if_neg (by decide), innerCandidates446_3_checked]
  change scan edge (search edge 3) 4 [3169] 3169 innerTail446_4
    (if 3 ≤ innerCandidates446_3.length then search edge 3 [3169, 5101] innerCandidates446_3 8270 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan446_4

def innerTail446_2 : List Nat := 3733 :: innerTail446_3
def innerCandidates446_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates446_2_checked : candidates edge 4 3169 3733 innerTail446_3 best.bound = innerCandidates446_2 := by
  unfold candidates
  have row : edge 3733 = SearchData.row3733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan446_2 : scan edge (search edge 3) 4 [3169] 3169 innerTail446_2 best = best := by
  rw [innerTail446_2, scan, if_neg (by decide), innerCandidates446_2_checked]
  change scan edge (search edge 3) 4 [3169] 3169 innerTail446_3
    (if 3 ≤ innerCandidates446_2.length then search edge 3 [3169, 3733] innerCandidates446_2 6902 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan446_3

def innerTail446_1 : List Nat := 3391 :: innerTail446_2
def innerCandidates446_1 : List Nat := [6361]
set_option maxRecDepth 4096 in
theorem innerCandidates446_1_checked : candidates edge 4 3169 3391 innerTail446_2 best.bound = innerCandidates446_1 := by
  unfold candidates
  have row : edge 3391 = SearchData.row3391.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan446_1 : scan edge (search edge 3) 4 [3169] 3169 innerTail446_1 best = best := by
  rw [innerTail446_1, scan, if_neg (by decide), innerCandidates446_1_checked]
  change scan edge (search edge 3) 4 [3169] 3169 innerTail446_2
    (if 3 ≤ innerCandidates446_1.length then search edge 3 [3169, 3391] innerCandidates446_1 6560 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan446_2

def innerTail446_0 : List Nat := 3229 :: innerTail446_1
def innerCandidates446_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates446_0_checked : candidates edge 4 3169 3229 innerTail446_1 best.bound = innerCandidates446_0 := by
  unfold candidates
  have row : edge 3229 = SearchData.row3229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan446_0 : scan edge (search edge 3) 4 [3169] 3169 innerTail446_0 best = best := by
  rw [innerTail446_0, scan, if_neg (by decide), innerCandidates446_0_checked]
  change scan edge (search edge 3) 4 [3169] 3169 innerTail446_1
    (if 3 ≤ innerCandidates446_0.length then search edge 3 [3169, 3229] innerCandidates446_0 6398 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan446_1

theorem rootBranch446 : search edge 4 [3169] rootCandidates446 3169 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3169] 3169 innerTail446_0 best = best
  exact innerScan446_0

def rootCandidates445 : List Nat := [3527, 4001, 4217, 4421, 5309, 5711, 6269, 6299, 6521, 8039]
set_option maxRecDepth 4096 in
theorem rootCandidates445_checked : candidates edge 5 0 3167 tail446 best.bound = rootCandidates445 := by
  unfold candidates
  have row : edge 3167 = SearchData.row3167.contains := by funext q; rfl
  rw [row]
  decide

def innerTail445_10 : List Nat := []
theorem innerScan445_10 : scan edge (search edge 3) 4 [3167] 3167 innerTail445_10 best = best := rfl

def innerTail445_9 : List Nat := 8039 :: innerTail445_10
def innerCandidates445_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates445_9_checked : candidates edge 4 3167 8039 innerTail445_10 best.bound = innerCandidates445_9 := by
  unfold candidates
  have row : edge 8039 = SearchData.row8039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan445_9 : scan edge (search edge 3) 4 [3167] 3167 innerTail445_9 best = best := by
  rw [innerTail445_9, scan, if_neg (by decide), innerCandidates445_9_checked]
  change scan edge (search edge 3) 4 [3167] 3167 innerTail445_10
    (if 3 ≤ innerCandidates445_9.length then search edge 3 [3167, 8039] innerCandidates445_9 11206 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan445_10

def innerTail445_8 : List Nat := 6521 :: innerTail445_9
def innerCandidates445_8 : List Nat := [8039]
set_option maxRecDepth 4096 in
theorem innerCandidates445_8_checked : candidates edge 4 3167 6521 innerTail445_9 best.bound = innerCandidates445_8 := by
  unfold candidates
  have row : edge 6521 = SearchData.row6521.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan445_8 : scan edge (search edge 3) 4 [3167] 3167 innerTail445_8 best = best := by
  rw [innerTail445_8, scan, if_neg (by decide), innerCandidates445_8_checked]
  change scan edge (search edge 3) 4 [3167] 3167 innerTail445_9
    (if 3 ≤ innerCandidates445_8.length then search edge 3 [3167, 6521] innerCandidates445_8 9688 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan445_9

def innerTail445_7 : List Nat := 6299 :: innerTail445_8
def innerCandidates445_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates445_7_checked : candidates edge 4 3167 6299 innerTail445_8 best.bound = innerCandidates445_7 := by
  unfold candidates
  have row : edge 6299 = SearchData.row6299.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan445_7 : scan edge (search edge 3) 4 [3167] 3167 innerTail445_7 best = best := by
  rw [innerTail445_7, scan, if_neg (by decide), innerCandidates445_7_checked]
  change scan edge (search edge 3) 4 [3167] 3167 innerTail445_8
    (if 3 ≤ innerCandidates445_7.length then search edge 3 [3167, 6299] innerCandidates445_7 9466 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan445_8

def innerTail445_6 : List Nat := 6269 :: innerTail445_7
def innerCandidates445_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates445_6_checked : candidates edge 4 3167 6269 innerTail445_7 best.bound = innerCandidates445_6 := by
  unfold candidates
  have row : edge 6269 = SearchData.row6269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan445_6 : scan edge (search edge 3) 4 [3167] 3167 innerTail445_6 best = best := by
  rw [innerTail445_6, scan, if_neg (by decide), innerCandidates445_6_checked]
  change scan edge (search edge 3) 4 [3167] 3167 innerTail445_7
    (if 3 ≤ innerCandidates445_6.length then search edge 3 [3167, 6269] innerCandidates445_6 9436 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan445_7

def innerTail445_5 : List Nat := 5711 :: innerTail445_6
def innerCandidates445_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates445_5_checked : candidates edge 4 3167 5711 innerTail445_6 best.bound = innerCandidates445_5 := by
  unfold candidates
  have row : edge 5711 = SearchData.row5711.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan445_5 : scan edge (search edge 3) 4 [3167] 3167 innerTail445_5 best = best := by
  rw [innerTail445_5, scan, if_neg (by decide), innerCandidates445_5_checked]
  change scan edge (search edge 3) 4 [3167] 3167 innerTail445_6
    (if 3 ≤ innerCandidates445_5.length then search edge 3 [3167, 5711] innerCandidates445_5 8878 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan445_6

def innerTail445_4 : List Nat := 5309 :: innerTail445_5
def innerCandidates445_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates445_4_checked : candidates edge 4 3167 5309 innerTail445_5 best.bound = innerCandidates445_4 := by
  unfold candidates
  have row : edge 5309 = SearchData.row5309.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan445_4 : scan edge (search edge 3) 4 [3167] 3167 innerTail445_4 best = best := by
  rw [innerTail445_4, scan, if_neg (by decide), innerCandidates445_4_checked]
  change scan edge (search edge 3) 4 [3167] 3167 innerTail445_5
    (if 3 ≤ innerCandidates445_4.length then search edge 3 [3167, 5309] innerCandidates445_4 8476 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan445_5

def innerTail445_3 : List Nat := 4421 :: innerTail445_4
def innerCandidates445_3 : List Nat := [8039]
set_option maxRecDepth 4096 in
theorem innerCandidates445_3_checked : candidates edge 4 3167 4421 innerTail445_4 best.bound = innerCandidates445_3 := by
  unfold candidates
  have row : edge 4421 = SearchData.row4421.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan445_3 : scan edge (search edge 3) 4 [3167] 3167 innerTail445_3 best = best := by
  rw [innerTail445_3, scan, if_neg (by decide), innerCandidates445_3_checked]
  change scan edge (search edge 3) 4 [3167] 3167 innerTail445_4
    (if 3 ≤ innerCandidates445_3.length then search edge 3 [3167, 4421] innerCandidates445_3 7588 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan445_4

def innerTail445_2 : List Nat := 4217 :: innerTail445_3
def innerCandidates445_2 : List Nat := [4421]
set_option maxRecDepth 4096 in
theorem innerCandidates445_2_checked : candidates edge 4 3167 4217 innerTail445_3 best.bound = innerCandidates445_2 := by
  unfold candidates
  have row : edge 4217 = SearchData.row4217.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan445_2 : scan edge (search edge 3) 4 [3167] 3167 innerTail445_2 best = best := by
  rw [innerTail445_2, scan, if_neg (by decide), innerCandidates445_2_checked]
  change scan edge (search edge 3) 4 [3167] 3167 innerTail445_3
    (if 3 ≤ innerCandidates445_2.length then search edge 3 [3167, 4217] innerCandidates445_2 7384 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan445_3

def innerTail445_1 : List Nat := 4001 :: innerTail445_2
def innerCandidates445_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates445_1_checked : candidates edge 4 3167 4001 innerTail445_2 best.bound = innerCandidates445_1 := by
  unfold candidates
  have row : edge 4001 = SearchData.row4001.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan445_1 : scan edge (search edge 3) 4 [3167] 3167 innerTail445_1 best = best := by
  rw [innerTail445_1, scan, if_neg (by decide), innerCandidates445_1_checked]
  change scan edge (search edge 3) 4 [3167] 3167 innerTail445_2
    (if 3 ≤ innerCandidates445_1.length then search edge 3 [3167, 4001] innerCandidates445_1 7168 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan445_2

def innerTail445_0 : List Nat := 3527 :: innerTail445_1
def innerCandidates445_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates445_0_checked : candidates edge 4 3167 3527 innerTail445_1 best.bound = innerCandidates445_0 := by
  unfold candidates
  have row : edge 3527 = SearchData.row3527.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan445_0 : scan edge (search edge 3) 4 [3167] 3167 innerTail445_0 best = best := by
  rw [innerTail445_0, scan, if_neg (by decide), innerCandidates445_0_checked]
  change scan edge (search edge 3) 4 [3167] 3167 innerTail445_1
    (if 3 ≤ innerCandidates445_0.length then search edge 3 [3167, 3527] innerCandidates445_0 6694 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan445_1

theorem rootBranch445 : search edge 4 [3167] rootCandidates445 3167 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3167] 3167 innerTail445_0 best = best
  exact innerScan445_0

def rootCandidates444 : List Nat := [3529, 3571, 3769, 3967, 4129, 5101, 5431, 5743, 6037, 6301, 6733, 7477, 7951]
set_option maxRecDepth 4096 in
theorem rootCandidates444_checked : candidates edge 5 0 3163 tail445 best.bound = rootCandidates444 := by
  unfold candidates
  have row : edge 3163 = SearchData.row3163.contains := by funext q; rfl
  rw [row]
  decide

def innerTail444_13 : List Nat := []
theorem innerScan444_13 : scan edge (search edge 3) 4 [3163] 3163 innerTail444_13 best = best := rfl

def innerTail444_12 : List Nat := 7951 :: innerTail444_13
def innerCandidates444_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates444_12_checked : candidates edge 4 3163 7951 innerTail444_13 best.bound = innerCandidates444_12 := by
  unfold candidates
  have row : edge 7951 = SearchData.row7951.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan444_12 : scan edge (search edge 3) 4 [3163] 3163 innerTail444_12 best = best := by
  rw [innerTail444_12, scan, if_neg (by decide), innerCandidates444_12_checked]
  change scan edge (search edge 3) 4 [3163] 3163 innerTail444_13
    (if 3 ≤ innerCandidates444_12.length then search edge 3 [3163, 7951] innerCandidates444_12 11114 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan444_13

def innerTail444_11 : List Nat := 7477 :: innerTail444_12
def innerCandidates444_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates444_11_checked : candidates edge 4 3163 7477 innerTail444_12 best.bound = innerCandidates444_11 := by
  unfold candidates
  have row : edge 7477 = SearchData.row7477.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan444_11 : scan edge (search edge 3) 4 [3163] 3163 innerTail444_11 best = best := by
  rw [innerTail444_11, scan, if_neg (by decide), innerCandidates444_11_checked]
  change scan edge (search edge 3) 4 [3163] 3163 innerTail444_12
    (if 3 ≤ innerCandidates444_11.length then search edge 3 [3163, 7477] innerCandidates444_11 10640 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan444_12

def innerTail444_10 : List Nat := 6733 :: innerTail444_11
def innerCandidates444_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates444_10_checked : candidates edge 4 3163 6733 innerTail444_11 best.bound = innerCandidates444_10 := by
  unfold candidates
  have row : edge 6733 = SearchData.row6733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan444_10 : scan edge (search edge 3) 4 [3163] 3163 innerTail444_10 best = best := by
  rw [innerTail444_10, scan, if_neg (by decide), innerCandidates444_10_checked]
  change scan edge (search edge 3) 4 [3163] 3163 innerTail444_11
    (if 3 ≤ innerCandidates444_10.length then search edge 3 [3163, 6733] innerCandidates444_10 9896 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan444_11

def innerTail444_9 : List Nat := 6301 :: innerTail444_10
def innerCandidates444_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates444_9_checked : candidates edge 4 3163 6301 innerTail444_10 best.bound = innerCandidates444_9 := by
  unfold candidates
  have row : edge 6301 = SearchData.row6301.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan444_9 : scan edge (search edge 3) 4 [3163] 3163 innerTail444_9 best = best := by
  rw [innerTail444_9, scan, if_neg (by decide), innerCandidates444_9_checked]
  change scan edge (search edge 3) 4 [3163] 3163 innerTail444_10
    (if 3 ≤ innerCandidates444_9.length then search edge 3 [3163, 6301] innerCandidates444_9 9464 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan444_10

def innerTail444_8 : List Nat := 6037 :: innerTail444_9
def innerCandidates444_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates444_8_checked : candidates edge 4 3163 6037 innerTail444_9 best.bound = innerCandidates444_8 := by
  unfold candidates
  have row : edge 6037 = SearchData.row6037.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan444_8 : scan edge (search edge 3) 4 [3163] 3163 innerTail444_8 best = best := by
  rw [innerTail444_8, scan, if_neg (by decide), innerCandidates444_8_checked]
  change scan edge (search edge 3) 4 [3163] 3163 innerTail444_9
    (if 3 ≤ innerCandidates444_8.length then search edge 3 [3163, 6037] innerCandidates444_8 9200 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan444_9

def innerTail444_7 : List Nat := 5743 :: innerTail444_8
def innerCandidates444_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates444_7_checked : candidates edge 4 3163 5743 innerTail444_8 best.bound = innerCandidates444_7 := by
  unfold candidates
  have row : edge 5743 = SearchData.row5743.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan444_7 : scan edge (search edge 3) 4 [3163] 3163 innerTail444_7 best = best := by
  rw [innerTail444_7, scan, if_neg (by decide), innerCandidates444_7_checked]
  change scan edge (search edge 3) 4 [3163] 3163 innerTail444_8
    (if 3 ≤ innerCandidates444_7.length then search edge 3 [3163, 5743] innerCandidates444_7 8906 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan444_8

def innerTail444_6 : List Nat := 5431 :: innerTail444_7
def innerCandidates444_6 : List Nat := [5743]
set_option maxRecDepth 4096 in
theorem innerCandidates444_6_checked : candidates edge 4 3163 5431 innerTail444_7 best.bound = innerCandidates444_6 := by
  unfold candidates
  have row : edge 5431 = SearchData.row5431.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan444_6 : scan edge (search edge 3) 4 [3163] 3163 innerTail444_6 best = best := by
  rw [innerTail444_6, scan, if_neg (by decide), innerCandidates444_6_checked]
  change scan edge (search edge 3) 4 [3163] 3163 innerTail444_7
    (if 3 ≤ innerCandidates444_6.length then search edge 3 [3163, 5431] innerCandidates444_6 8594 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan444_7

def innerTail444_5 : List Nat := 5101 :: innerTail444_6
def innerCandidates444_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates444_5_checked : candidates edge 4 3163 5101 innerTail444_6 best.bound = innerCandidates444_5 := by
  unfold candidates
  have row : edge 5101 = SearchData.row5101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan444_5 : scan edge (search edge 3) 4 [3163] 3163 innerTail444_5 best = best := by
  rw [innerTail444_5, scan, if_neg (by decide), innerCandidates444_5_checked]
  change scan edge (search edge 3) 4 [3163] 3163 innerTail444_6
    (if 3 ≤ innerCandidates444_5.length then search edge 3 [3163, 5101] innerCandidates444_5 8264 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan444_6

def innerTail444_4 : List Nat := 4129 :: innerTail444_5
def innerCandidates444_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates444_4_checked : candidates edge 4 3163 4129 innerTail444_5 best.bound = innerCandidates444_4 := by
  unfold candidates
  have row : edge 4129 = SearchData.row4129.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan444_4 : scan edge (search edge 3) 4 [3163] 3163 innerTail444_4 best = best := by
  rw [innerTail444_4, scan, if_neg (by decide), innerCandidates444_4_checked]
  change scan edge (search edge 3) 4 [3163] 3163 innerTail444_5
    (if 3 ≤ innerCandidates444_4.length then search edge 3 [3163, 4129] innerCandidates444_4 7292 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan444_5

def innerTail444_3 : List Nat := 3967 :: innerTail444_4
def innerCandidates444_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates444_3_checked : candidates edge 4 3163 3967 innerTail444_4 best.bound = innerCandidates444_3 := by
  unfold candidates
  have row : edge 3967 = SearchData.row3967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan444_3 : scan edge (search edge 3) 4 [3163] 3163 innerTail444_3 best = best := by
  rw [innerTail444_3, scan, if_neg (by decide), innerCandidates444_3_checked]
  change scan edge (search edge 3) 4 [3163] 3163 innerTail444_4
    (if 3 ≤ innerCandidates444_3.length then search edge 3 [3163, 3967] innerCandidates444_3 7130 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan444_4

def innerTail444_2 : List Nat := 3769 :: innerTail444_3
def innerCandidates444_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates444_2_checked : candidates edge 4 3163 3769 innerTail444_3 best.bound = innerCandidates444_2 := by
  unfold candidates
  have row : edge 3769 = SearchData.row3769.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan444_2 : scan edge (search edge 3) 4 [3163] 3163 innerTail444_2 best = best := by
  rw [innerTail444_2, scan, if_neg (by decide), innerCandidates444_2_checked]
  change scan edge (search edge 3) 4 [3163] 3163 innerTail444_3
    (if 3 ≤ innerCandidates444_2.length then search edge 3 [3163, 3769] innerCandidates444_2 6932 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan444_3

def innerTail444_1 : List Nat := 3571 :: innerTail444_2
def innerCandidates444_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates444_1_checked : candidates edge 4 3163 3571 innerTail444_2 best.bound = innerCandidates444_1 := by
  unfold candidates
  have row : edge 3571 = SearchData.row3571.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan444_1 : scan edge (search edge 3) 4 [3163] 3163 innerTail444_1 best = best := by
  rw [innerTail444_1, scan, if_neg (by decide), innerCandidates444_1_checked]
  change scan edge (search edge 3) 4 [3163] 3163 innerTail444_2
    (if 3 ≤ innerCandidates444_1.length then search edge 3 [3163, 3571] innerCandidates444_1 6734 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan444_2

def innerTail444_0 : List Nat := 3529 :: innerTail444_1
def innerCandidates444_0 : List Nat := [6733]
set_option maxRecDepth 4096 in
theorem innerCandidates444_0_checked : candidates edge 4 3163 3529 innerTail444_1 best.bound = innerCandidates444_0 := by
  unfold candidates
  have row : edge 3529 = SearchData.row3529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan444_0 : scan edge (search edge 3) 4 [3163] 3163 innerTail444_0 best = best := by
  rw [innerTail444_0, scan, if_neg (by decide), innerCandidates444_0_checked]
  change scan edge (search edge 3) 4 [3163] 3163 innerTail444_1
    (if 3 ≤ innerCandidates444_0.length then search edge 3 [3163, 3529] innerCandidates444_0 6692 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan444_1

theorem rootBranch444 : search edge 4 [3163] rootCandidates444 3163 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3163] 3163 innerTail444_0 best = best
  exact innerScan444_0

def rootCandidates443 : List Nat := [3371, 3593, 3821, 4349, 5009, 5189, 5501, 6311, 6803, 7331, 7919, 8387]
set_option maxRecDepth 4096 in
theorem rootCandidates443_checked : candidates edge 5 0 3137 tail444 best.bound = rootCandidates443 := by
  unfold candidates
  have row : edge 3137 = SearchData.row3137.contains := by funext q; rfl
  rw [row]
  decide

def innerTail443_12 : List Nat := []
theorem innerScan443_12 : scan edge (search edge 3) 4 [3137] 3137 innerTail443_12 best = best := rfl

def innerTail443_11 : List Nat := 8387 :: innerTail443_12
def innerCandidates443_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates443_11_checked : candidates edge 4 3137 8387 innerTail443_12 best.bound = innerCandidates443_11 := by
  unfold candidates
  have row : edge 8387 = SearchData.row8387.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan443_11 : scan edge (search edge 3) 4 [3137] 3137 innerTail443_11 best = best := by
  rw [innerTail443_11, scan, if_neg (by decide), innerCandidates443_11_checked]
  change scan edge (search edge 3) 4 [3137] 3137 innerTail443_12
    (if 3 ≤ innerCandidates443_11.length then search edge 3 [3137, 8387] innerCandidates443_11 11524 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan443_12

def innerTail443_10 : List Nat := 7919 :: innerTail443_11
def innerCandidates443_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates443_10_checked : candidates edge 4 3137 7919 innerTail443_11 best.bound = innerCandidates443_10 := by
  unfold candidates
  have row : edge 7919 = SearchData.row7919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan443_10 : scan edge (search edge 3) 4 [3137] 3137 innerTail443_10 best = best := by
  rw [innerTail443_10, scan, if_neg (by decide), innerCandidates443_10_checked]
  change scan edge (search edge 3) 4 [3137] 3137 innerTail443_11
    (if 3 ≤ innerCandidates443_10.length then search edge 3 [3137, 7919] innerCandidates443_10 11056 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan443_11

def innerTail443_9 : List Nat := 7331 :: innerTail443_10
def innerCandidates443_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates443_9_checked : candidates edge 4 3137 7331 innerTail443_10 best.bound = innerCandidates443_9 := by
  unfold candidates
  have row : edge 7331 = SearchData.row7331.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan443_9 : scan edge (search edge 3) 4 [3137] 3137 innerTail443_9 best = best := by
  rw [innerTail443_9, scan, if_neg (by decide), innerCandidates443_9_checked]
  change scan edge (search edge 3) 4 [3137] 3137 innerTail443_10
    (if 3 ≤ innerCandidates443_9.length then search edge 3 [3137, 7331] innerCandidates443_9 10468 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan443_10

def innerTail443_8 : List Nat := 6803 :: innerTail443_9
def innerCandidates443_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates443_8_checked : candidates edge 4 3137 6803 innerTail443_9 best.bound = innerCandidates443_8 := by
  unfold candidates
  have row : edge 6803 = SearchData.row6803.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan443_8 : scan edge (search edge 3) 4 [3137] 3137 innerTail443_8 best = best := by
  rw [innerTail443_8, scan, if_neg (by decide), innerCandidates443_8_checked]
  change scan edge (search edge 3) 4 [3137] 3137 innerTail443_9
    (if 3 ≤ innerCandidates443_8.length then search edge 3 [3137, 6803] innerCandidates443_8 9940 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan443_9

def innerTail443_7 : List Nat := 6311 :: innerTail443_8
def innerCandidates443_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates443_7_checked : candidates edge 4 3137 6311 innerTail443_8 best.bound = innerCandidates443_7 := by
  unfold candidates
  have row : edge 6311 = SearchData.row6311.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan443_7 : scan edge (search edge 3) 4 [3137] 3137 innerTail443_7 best = best := by
  rw [innerTail443_7, scan, if_neg (by decide), innerCandidates443_7_checked]
  change scan edge (search edge 3) 4 [3137] 3137 innerTail443_8
    (if 3 ≤ innerCandidates443_7.length then search edge 3 [3137, 6311] innerCandidates443_7 9448 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan443_8

def innerTail443_6 : List Nat := 5501 :: innerTail443_7
def innerCandidates443_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates443_6_checked : candidates edge 4 3137 5501 innerTail443_7 best.bound = innerCandidates443_6 := by
  unfold candidates
  have row : edge 5501 = SearchData.row5501.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan443_6 : scan edge (search edge 3) 4 [3137] 3137 innerTail443_6 best = best := by
  rw [innerTail443_6, scan, if_neg (by decide), innerCandidates443_6_checked]
  change scan edge (search edge 3) 4 [3137] 3137 innerTail443_7
    (if 3 ≤ innerCandidates443_6.length then search edge 3 [3137, 5501] innerCandidates443_6 8638 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan443_7

def innerTail443_5 : List Nat := 5189 :: innerTail443_6
def innerCandidates443_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates443_5_checked : candidates edge 4 3137 5189 innerTail443_6 best.bound = innerCandidates443_5 := by
  unfold candidates
  have row : edge 5189 = SearchData.row5189.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan443_5 : scan edge (search edge 3) 4 [3137] 3137 innerTail443_5 best = best := by
  rw [innerTail443_5, scan, if_neg (by decide), innerCandidates443_5_checked]
  change scan edge (search edge 3) 4 [3137] 3137 innerTail443_6
    (if 3 ≤ innerCandidates443_5.length then search edge 3 [3137, 5189] innerCandidates443_5 8326 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan443_6

def innerTail443_4 : List Nat := 5009 :: innerTail443_5
def innerCandidates443_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates443_4_checked : candidates edge 4 3137 5009 innerTail443_5 best.bound = innerCandidates443_4 := by
  unfold candidates
  have row : edge 5009 = SearchData.row5009.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan443_4 : scan edge (search edge 3) 4 [3137] 3137 innerTail443_4 best = best := by
  rw [innerTail443_4, scan, if_neg (by decide), innerCandidates443_4_checked]
  change scan edge (search edge 3) 4 [3137] 3137 innerTail443_5
    (if 3 ≤ innerCandidates443_4.length then search edge 3 [3137, 5009] innerCandidates443_4 8146 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan443_5

def innerTail443_3 : List Nat := 4349 :: innerTail443_4
def innerCandidates443_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates443_3_checked : candidates edge 4 3137 4349 innerTail443_4 best.bound = innerCandidates443_3 := by
  unfold candidates
  have row : edge 4349 = SearchData.row4349.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan443_3 : scan edge (search edge 3) 4 [3137] 3137 innerTail443_3 best = best := by
  rw [innerTail443_3, scan, if_neg (by decide), innerCandidates443_3_checked]
  change scan edge (search edge 3) 4 [3137] 3137 innerTail443_4
    (if 3 ≤ innerCandidates443_3.length then search edge 3 [3137, 4349] innerCandidates443_3 7486 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan443_4

def innerTail443_2 : List Nat := 3821 :: innerTail443_3
def innerCandidates443_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates443_2_checked : candidates edge 4 3137 3821 innerTail443_3 best.bound = innerCandidates443_2 := by
  unfold candidates
  have row : edge 3821 = SearchData.row3821.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan443_2 : scan edge (search edge 3) 4 [3137] 3137 innerTail443_2 best = best := by
  rw [innerTail443_2, scan, if_neg (by decide), innerCandidates443_2_checked]
  change scan edge (search edge 3) 4 [3137] 3137 innerTail443_3
    (if 3 ≤ innerCandidates443_2.length then search edge 3 [3137, 3821] innerCandidates443_2 6958 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan443_3

def innerTail443_1 : List Nat := 3593 :: innerTail443_2
def innerCandidates443_1 : List Nat := [6311]
set_option maxRecDepth 4096 in
theorem innerCandidates443_1_checked : candidates edge 4 3137 3593 innerTail443_2 best.bound = innerCandidates443_1 := by
  unfold candidates
  have row : edge 3593 = SearchData.row3593.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan443_1 : scan edge (search edge 3) 4 [3137] 3137 innerTail443_1 best = best := by
  rw [innerTail443_1, scan, if_neg (by decide), innerCandidates443_1_checked]
  change scan edge (search edge 3) 4 [3137] 3137 innerTail443_2
    (if 3 ≤ innerCandidates443_1.length then search edge 3 [3137, 3593] innerCandidates443_1 6730 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan443_2

def innerTail443_0 : List Nat := 3371 :: innerTail443_1
def innerCandidates443_0 : List Nat := [5009]
set_option maxRecDepth 4096 in
theorem innerCandidates443_0_checked : candidates edge 4 3137 3371 innerTail443_1 best.bound = innerCandidates443_0 := by
  unfold candidates
  have row : edge 3371 = SearchData.row3371.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan443_0 : scan edge (search edge 3) 4 [3137] 3137 innerTail443_0 best = best := by
  rw [innerTail443_0, scan, if_neg (by decide), innerCandidates443_0_checked]
  change scan edge (search edge 3) 4 [3137] 3137 innerTail443_1
    (if 3 ≤ innerCandidates443_0.length then search edge 3 [3137, 3371] innerCandidates443_0 6508 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan443_1

theorem rootBranch443 : search edge 4 [3137] rootCandidates443 3137 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3137] 3137 innerTail443_0 best = best
  exact innerScan443_0

end ProjectEulerSolutions.PrimePairUpper.Search
