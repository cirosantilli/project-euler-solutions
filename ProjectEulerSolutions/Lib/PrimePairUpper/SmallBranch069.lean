import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates535 : List Nat := [4421, 4793, 5261, 5519, 6269, 6857, 8363, 8369]
set_option maxRecDepth 4096 in
theorem rootCandidates535_checked : candidates edge 5 0 3881 tail536 best.bound = rootCandidates535 := by
  unfold candidates
  have row : edge 3881 = SearchData.row3881.contains := by funext q; rfl
  rw [row]
  decide

def innerTail535_8 : List Nat := []
theorem innerScan535_8 : scan edge (search edge 3) 4 [3881] 3881 innerTail535_8 best = best := rfl

def innerTail535_7 : List Nat := 8369 :: innerTail535_8
def innerCandidates535_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates535_7_checked : candidates edge 4 3881 8369 innerTail535_8 best.bound = innerCandidates535_7 := by
  unfold candidates
  have row : edge 8369 = SearchData.row8369.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan535_7 : scan edge (search edge 3) 4 [3881] 3881 innerTail535_7 best = best := by
  rw [innerTail535_7, scan, if_neg (by decide), innerCandidates535_7_checked]
  change scan edge (search edge 3) 4 [3881] 3881 innerTail535_8
    (if 3 ≤ innerCandidates535_7.length then search edge 3 [3881, 8369] innerCandidates535_7 12250 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan535_8

def innerTail535_6 : List Nat := 8363 :: innerTail535_7
def innerCandidates535_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates535_6_checked : candidates edge 4 3881 8363 innerTail535_7 best.bound = innerCandidates535_6 := by
  unfold candidates
  have row : edge 8363 = SearchData.row8363.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan535_6 : scan edge (search edge 3) 4 [3881] 3881 innerTail535_6 best = best := by
  rw [innerTail535_6, scan, if_neg (by decide), innerCandidates535_6_checked]
  change scan edge (search edge 3) 4 [3881] 3881 innerTail535_7
    (if 3 ≤ innerCandidates535_6.length then search edge 3 [3881, 8363] innerCandidates535_6 12244 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan535_7

def innerTail535_5 : List Nat := 6857 :: innerTail535_6
def innerCandidates535_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates535_5_checked : candidates edge 4 3881 6857 innerTail535_6 best.bound = innerCandidates535_5 := by
  unfold candidates
  have row : edge 6857 = SearchData.row6857.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan535_5 : scan edge (search edge 3) 4 [3881] 3881 innerTail535_5 best = best := by
  rw [innerTail535_5, scan, if_neg (by decide), innerCandidates535_5_checked]
  change scan edge (search edge 3) 4 [3881] 3881 innerTail535_6
    (if 3 ≤ innerCandidates535_5.length then search edge 3 [3881, 6857] innerCandidates535_5 10738 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan535_6

def innerTail535_4 : List Nat := 6269 :: innerTail535_5
def innerCandidates535_4 : List Nat := [8369]
set_option maxRecDepth 4096 in
theorem innerCandidates535_4_checked : candidates edge 4 3881 6269 innerTail535_5 best.bound = innerCandidates535_4 := by
  unfold candidates
  have row : edge 6269 = SearchData.row6269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan535_4 : scan edge (search edge 3) 4 [3881] 3881 innerTail535_4 best = best := by
  rw [innerTail535_4, scan, if_neg (by decide), innerCandidates535_4_checked]
  change scan edge (search edge 3) 4 [3881] 3881 innerTail535_5
    (if 3 ≤ innerCandidates535_4.length then search edge 3 [3881, 6269] innerCandidates535_4 10150 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan535_5

def innerTail535_3 : List Nat := 5519 :: innerTail535_4
def innerCandidates535_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates535_3_checked : candidates edge 4 3881 5519 innerTail535_4 best.bound = innerCandidates535_3 := by
  unfold candidates
  have row : edge 5519 = SearchData.row5519.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan535_3 : scan edge (search edge 3) 4 [3881] 3881 innerTail535_3 best = best := by
  rw [innerTail535_3, scan, if_neg (by decide), innerCandidates535_3_checked]
  change scan edge (search edge 3) 4 [3881] 3881 innerTail535_4
    (if 3 ≤ innerCandidates535_3.length then search edge 3 [3881, 5519] innerCandidates535_3 9400 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan535_4

def innerTail535_2 : List Nat := 5261 :: innerTail535_3
def innerCandidates535_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates535_2_checked : candidates edge 4 3881 5261 innerTail535_3 best.bound = innerCandidates535_2 := by
  unfold candidates
  have row : edge 5261 = SearchData.row5261.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan535_2 : scan edge (search edge 3) 4 [3881] 3881 innerTail535_2 best = best := by
  rw [innerTail535_2, scan, if_neg (by decide), innerCandidates535_2_checked]
  change scan edge (search edge 3) 4 [3881] 3881 innerTail535_3
    (if 3 ≤ innerCandidates535_2.length then search edge 3 [3881, 5261] innerCandidates535_2 9142 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan535_3

def innerTail535_1 : List Nat := 4793 :: innerTail535_2
def innerCandidates535_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates535_1_checked : candidates edge 4 3881 4793 innerTail535_2 best.bound = innerCandidates535_1 := by
  unfold candidates
  have row : edge 4793 = SearchData.row4793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan535_1 : scan edge (search edge 3) 4 [3881] 3881 innerTail535_1 best = best := by
  rw [innerTail535_1, scan, if_neg (by decide), innerCandidates535_1_checked]
  change scan edge (search edge 3) 4 [3881] 3881 innerTail535_2
    (if 3 ≤ innerCandidates535_1.length then search edge 3 [3881, 4793] innerCandidates535_1 8674 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan535_2

def innerTail535_0 : List Nat := 4421 :: innerTail535_1
def innerCandidates535_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates535_0_checked : candidates edge 4 3881 4421 innerTail535_1 best.bound = innerCandidates535_0 := by
  unfold candidates
  have row : edge 4421 = SearchData.row4421.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan535_0 : scan edge (search edge 3) 4 [3881] 3881 innerTail535_0 best = best := by
  rw [innerTail535_0, scan, if_neg (by decide), innerCandidates535_0_checked]
  change scan edge (search edge 3) 4 [3881] 3881 innerTail535_1
    (if 3 ≤ innerCandidates535_0.length then search edge 3 [3881, 4421] innerCandidates535_0 8302 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan535_1

theorem rootBranch535 : search edge 4 [3881] rootCandidates535 3881 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3881] 3881 innerTail535_0 best = best
  exact innerScan535_0

def rootCandidates534 : List Nat := [4507, 5167, 5227, 5479, 5683, 6151, 6163, 6373, 6469, 6607, 7159, 8263]
set_option maxRecDepth 4096 in
theorem rootCandidates534_checked : candidates edge 5 0 3877 tail535 best.bound = rootCandidates534 := by
  unfold candidates
  have row : edge 3877 = SearchData.row3877.contains := by funext q; rfl
  rw [row]
  decide

def innerTail534_12 : List Nat := []
theorem innerScan534_12 : scan edge (search edge 3) 4 [3877] 3877 innerTail534_12 best = best := rfl

def innerTail534_11 : List Nat := 8263 :: innerTail534_12
def innerCandidates534_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates534_11_checked : candidates edge 4 3877 8263 innerTail534_12 best.bound = innerCandidates534_11 := by
  unfold candidates
  have row : edge 8263 = SearchData.row8263.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan534_11 : scan edge (search edge 3) 4 [3877] 3877 innerTail534_11 best = best := by
  rw [innerTail534_11, scan, if_neg (by decide), innerCandidates534_11_checked]
  change scan edge (search edge 3) 4 [3877] 3877 innerTail534_12
    (if 3 ≤ innerCandidates534_11.length then search edge 3 [3877, 8263] innerCandidates534_11 12140 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan534_12

def innerTail534_10 : List Nat := 7159 :: innerTail534_11
def innerCandidates534_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates534_10_checked : candidates edge 4 3877 7159 innerTail534_11 best.bound = innerCandidates534_10 := by
  unfold candidates
  have row : edge 7159 = SearchData.row7159.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan534_10 : scan edge (search edge 3) 4 [3877] 3877 innerTail534_10 best = best := by
  rw [innerTail534_10, scan, if_neg (by decide), innerCandidates534_10_checked]
  change scan edge (search edge 3) 4 [3877] 3877 innerTail534_11
    (if 3 ≤ innerCandidates534_10.length then search edge 3 [3877, 7159] innerCandidates534_10 11036 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan534_11

def innerTail534_9 : List Nat := 6607 :: innerTail534_10
def innerCandidates534_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates534_9_checked : candidates edge 4 3877 6607 innerTail534_10 best.bound = innerCandidates534_9 := by
  unfold candidates
  have row : edge 6607 = SearchData.row6607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan534_9 : scan edge (search edge 3) 4 [3877] 3877 innerTail534_9 best = best := by
  rw [innerTail534_9, scan, if_neg (by decide), innerCandidates534_9_checked]
  change scan edge (search edge 3) 4 [3877] 3877 innerTail534_10
    (if 3 ≤ innerCandidates534_9.length then search edge 3 [3877, 6607] innerCandidates534_9 10484 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan534_10

def innerTail534_8 : List Nat := 6469 :: innerTail534_9
def innerCandidates534_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates534_8_checked : candidates edge 4 3877 6469 innerTail534_9 best.bound = innerCandidates534_8 := by
  unfold candidates
  have row : edge 6469 = SearchData.row6469.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan534_8 : scan edge (search edge 3) 4 [3877] 3877 innerTail534_8 best = best := by
  rw [innerTail534_8, scan, if_neg (by decide), innerCandidates534_8_checked]
  change scan edge (search edge 3) 4 [3877] 3877 innerTail534_9
    (if 3 ≤ innerCandidates534_8.length then search edge 3 [3877, 6469] innerCandidates534_8 10346 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan534_9

def innerTail534_7 : List Nat := 6373 :: innerTail534_8
def innerCandidates534_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates534_7_checked : candidates edge 4 3877 6373 innerTail534_8 best.bound = innerCandidates534_7 := by
  unfold candidates
  have row : edge 6373 = SearchData.row6373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan534_7 : scan edge (search edge 3) 4 [3877] 3877 innerTail534_7 best = best := by
  rw [innerTail534_7, scan, if_neg (by decide), innerCandidates534_7_checked]
  change scan edge (search edge 3) 4 [3877] 3877 innerTail534_8
    (if 3 ≤ innerCandidates534_7.length then search edge 3 [3877, 6373] innerCandidates534_7 10250 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan534_8

def innerTail534_6 : List Nat := 6163 :: innerTail534_7
def innerCandidates534_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates534_6_checked : candidates edge 4 3877 6163 innerTail534_7 best.bound = innerCandidates534_6 := by
  unfold candidates
  have row : edge 6163 = SearchData.row6163.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan534_6 : scan edge (search edge 3) 4 [3877] 3877 innerTail534_6 best = best := by
  rw [innerTail534_6, scan, if_neg (by decide), innerCandidates534_6_checked]
  change scan edge (search edge 3) 4 [3877] 3877 innerTail534_7
    (if 3 ≤ innerCandidates534_6.length then search edge 3 [3877, 6163] innerCandidates534_6 10040 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan534_7

def innerTail534_5 : List Nat := 6151 :: innerTail534_6
def innerCandidates534_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates534_5_checked : candidates edge 4 3877 6151 innerTail534_6 best.bound = innerCandidates534_5 := by
  unfold candidates
  have row : edge 6151 = SearchData.row6151.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan534_5 : scan edge (search edge 3) 4 [3877] 3877 innerTail534_5 best = best := by
  rw [innerTail534_5, scan, if_neg (by decide), innerCandidates534_5_checked]
  change scan edge (search edge 3) 4 [3877] 3877 innerTail534_6
    (if 3 ≤ innerCandidates534_5.length then search edge 3 [3877, 6151] innerCandidates534_5 10028 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan534_6

def innerTail534_4 : List Nat := 5683 :: innerTail534_5
def innerCandidates534_4 : List Nat := [7159]
set_option maxRecDepth 4096 in
theorem innerCandidates534_4_checked : candidates edge 4 3877 5683 innerTail534_5 best.bound = innerCandidates534_4 := by
  unfold candidates
  have row : edge 5683 = SearchData.row5683.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan534_4 : scan edge (search edge 3) 4 [3877] 3877 innerTail534_4 best = best := by
  rw [innerTail534_4, scan, if_neg (by decide), innerCandidates534_4_checked]
  change scan edge (search edge 3) 4 [3877] 3877 innerTail534_5
    (if 3 ≤ innerCandidates534_4.length then search edge 3 [3877, 5683] innerCandidates534_4 9560 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan534_5

def innerTail534_3 : List Nat := 5479 :: innerTail534_4
def innerCandidates534_3 : List Nat := [6607]
set_option maxRecDepth 4096 in
theorem innerCandidates534_3_checked : candidates edge 4 3877 5479 innerTail534_4 best.bound = innerCandidates534_3 := by
  unfold candidates
  have row : edge 5479 = SearchData.row5479.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan534_3 : scan edge (search edge 3) 4 [3877] 3877 innerTail534_3 best = best := by
  rw [innerTail534_3, scan, if_neg (by decide), innerCandidates534_3_checked]
  change scan edge (search edge 3) 4 [3877] 3877 innerTail534_4
    (if 3 ≤ innerCandidates534_3.length then search edge 3 [3877, 5479] innerCandidates534_3 9356 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan534_4

def innerTail534_2 : List Nat := 5227 :: innerTail534_3
def innerCandidates534_2 : List Nat := [5683]
set_option maxRecDepth 4096 in
theorem innerCandidates534_2_checked : candidates edge 4 3877 5227 innerTail534_3 best.bound = innerCandidates534_2 := by
  unfold candidates
  have row : edge 5227 = SearchData.row5227.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan534_2 : scan edge (search edge 3) 4 [3877] 3877 innerTail534_2 best = best := by
  rw [innerTail534_2, scan, if_neg (by decide), innerCandidates534_2_checked]
  change scan edge (search edge 3) 4 [3877] 3877 innerTail534_3
    (if 3 ≤ innerCandidates534_2.length then search edge 3 [3877, 5227] innerCandidates534_2 9104 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan534_3

def innerTail534_1 : List Nat := 5167 :: innerTail534_2
def innerCandidates534_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates534_1_checked : candidates edge 4 3877 5167 innerTail534_2 best.bound = innerCandidates534_1 := by
  unfold candidates
  have row : edge 5167 = SearchData.row5167.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan534_1 : scan edge (search edge 3) 4 [3877] 3877 innerTail534_1 best = best := by
  rw [innerTail534_1, scan, if_neg (by decide), innerCandidates534_1_checked]
  change scan edge (search edge 3) 4 [3877] 3877 innerTail534_2
    (if 3 ≤ innerCandidates534_1.length then search edge 3 [3877, 5167] innerCandidates534_1 9044 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan534_2

def innerTail534_0 : List Nat := 4507 :: innerTail534_1
def innerCandidates534_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates534_0_checked : candidates edge 4 3877 4507 innerTail534_1 best.bound = innerCandidates534_0 := by
  unfold candidates
  have row : edge 4507 = SearchData.row4507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan534_0 : scan edge (search edge 3) 4 [3877] 3877 innerTail534_0 best = best := by
  rw [innerTail534_0, scan, if_neg (by decide), innerCandidates534_0_checked]
  change scan edge (search edge 3) 4 [3877] 3877 innerTail534_1
    (if 3 ≤ innerCandidates534_0.length then search edge 3 [3877, 4507] innerCandidates534_0 8384 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan534_1

theorem rootBranch534 : search edge 4 [3877] rootCandidates534 3877 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3877] 3877 innerTail534_0 best = best
  exact innerScan534_0

def rootCandidates533 : List Nat := [4007, 4451, 5297, 6569, 6779, 7187, 7673, 7757]
set_option maxRecDepth 4096 in
theorem rootCandidates533_checked : candidates edge 5 0 3863 tail534 best.bound = rootCandidates533 := by
  unfold candidates
  have row : edge 3863 = SearchData.row3863.contains := by funext q; rfl
  rw [row]
  decide

def innerTail533_8 : List Nat := []
theorem innerScan533_8 : scan edge (search edge 3) 4 [3863] 3863 innerTail533_8 best = best := rfl

def innerTail533_7 : List Nat := 7757 :: innerTail533_8
def innerCandidates533_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates533_7_checked : candidates edge 4 3863 7757 innerTail533_8 best.bound = innerCandidates533_7 := by
  unfold candidates
  have row : edge 7757 = SearchData.row7757.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan533_7 : scan edge (search edge 3) 4 [3863] 3863 innerTail533_7 best = best := by
  rw [innerTail533_7, scan, if_neg (by decide), innerCandidates533_7_checked]
  change scan edge (search edge 3) 4 [3863] 3863 innerTail533_8
    (if 3 ≤ innerCandidates533_7.length then search edge 3 [3863, 7757] innerCandidates533_7 11620 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan533_8

def innerTail533_6 : List Nat := 7673 :: innerTail533_7
def innerCandidates533_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates533_6_checked : candidates edge 4 3863 7673 innerTail533_7 best.bound = innerCandidates533_6 := by
  unfold candidates
  have row : edge 7673 = SearchData.row7673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan533_6 : scan edge (search edge 3) 4 [3863] 3863 innerTail533_6 best = best := by
  rw [innerTail533_6, scan, if_neg (by decide), innerCandidates533_6_checked]
  change scan edge (search edge 3) 4 [3863] 3863 innerTail533_7
    (if 3 ≤ innerCandidates533_6.length then search edge 3 [3863, 7673] innerCandidates533_6 11536 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan533_7

def innerTail533_5 : List Nat := 7187 :: innerTail533_6
def innerCandidates533_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates533_5_checked : candidates edge 4 3863 7187 innerTail533_6 best.bound = innerCandidates533_5 := by
  unfold candidates
  have row : edge 7187 = SearchData.row7187.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan533_5 : scan edge (search edge 3) 4 [3863] 3863 innerTail533_5 best = best := by
  rw [innerTail533_5, scan, if_neg (by decide), innerCandidates533_5_checked]
  change scan edge (search edge 3) 4 [3863] 3863 innerTail533_6
    (if 3 ≤ innerCandidates533_5.length then search edge 3 [3863, 7187] innerCandidates533_5 11050 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan533_6

def innerTail533_4 : List Nat := 6779 :: innerTail533_5
def innerCandidates533_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates533_4_checked : candidates edge 4 3863 6779 innerTail533_5 best.bound = innerCandidates533_4 := by
  unfold candidates
  have row : edge 6779 = SearchData.row6779.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan533_4 : scan edge (search edge 3) 4 [3863] 3863 innerTail533_4 best = best := by
  rw [innerTail533_4, scan, if_neg (by decide), innerCandidates533_4_checked]
  change scan edge (search edge 3) 4 [3863] 3863 innerTail533_5
    (if 3 ≤ innerCandidates533_4.length then search edge 3 [3863, 6779] innerCandidates533_4 10642 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan533_5

def innerTail533_3 : List Nat := 6569 :: innerTail533_4
def innerCandidates533_3 : List Nat := [6779]
set_option maxRecDepth 4096 in
theorem innerCandidates533_3_checked : candidates edge 4 3863 6569 innerTail533_4 best.bound = innerCandidates533_3 := by
  unfold candidates
  have row : edge 6569 = SearchData.row6569.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan533_3 : scan edge (search edge 3) 4 [3863] 3863 innerTail533_3 best = best := by
  rw [innerTail533_3, scan, if_neg (by decide), innerCandidates533_3_checked]
  change scan edge (search edge 3) 4 [3863] 3863 innerTail533_4
    (if 3 ≤ innerCandidates533_3.length then search edge 3 [3863, 6569] innerCandidates533_3 10432 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan533_4

def innerTail533_2 : List Nat := 5297 :: innerTail533_3
def innerCandidates533_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates533_2_checked : candidates edge 4 3863 5297 innerTail533_3 best.bound = innerCandidates533_2 := by
  unfold candidates
  have row : edge 5297 = SearchData.row5297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan533_2 : scan edge (search edge 3) 4 [3863] 3863 innerTail533_2 best = best := by
  rw [innerTail533_2, scan, if_neg (by decide), innerCandidates533_2_checked]
  change scan edge (search edge 3) 4 [3863] 3863 innerTail533_3
    (if 3 ≤ innerCandidates533_2.length then search edge 3 [3863, 5297] innerCandidates533_2 9160 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan533_3

def innerTail533_1 : List Nat := 4451 :: innerTail533_2
def innerCandidates533_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates533_1_checked : candidates edge 4 3863 4451 innerTail533_2 best.bound = innerCandidates533_1 := by
  unfold candidates
  have row : edge 4451 = SearchData.row4451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan533_1 : scan edge (search edge 3) 4 [3863] 3863 innerTail533_1 best = best := by
  rw [innerTail533_1, scan, if_neg (by decide), innerCandidates533_1_checked]
  change scan edge (search edge 3) 4 [3863] 3863 innerTail533_2
    (if 3 ≤ innerCandidates533_1.length then search edge 3 [3863, 4451] innerCandidates533_1 8314 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan533_2

def innerTail533_0 : List Nat := 4007 :: innerTail533_1
def innerCandidates533_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates533_0_checked : candidates edge 4 3863 4007 innerTail533_1 best.bound = innerCandidates533_0 := by
  unfold candidates
  have row : edge 4007 = SearchData.row4007.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan533_0 : scan edge (search edge 3) 4 [3863] 3863 innerTail533_0 best = best := by
  rw [innerTail533_0, scan, if_neg (by decide), innerCandidates533_0_checked]
  change scan edge (search edge 3) 4 [3863] 3863 innerTail533_1
    (if 3 ≤ innerCandidates533_0.length then search edge 3 [3863, 4007] innerCandidates533_0 7870 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan533_1

theorem rootBranch533 : search edge 4 [3863] rootCandidates533 3863 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3863] 3863 innerTail533_0 best = best
  exact innerScan533_0

def rootCandidates532 : List Nat := [4099, 4759, 5521, 5683, 6361, 6367, 6607, 6709, 6991, 7237, 7243, 7297, 7591, 7699, 8293]
set_option maxRecDepth 4096 in
theorem rootCandidates532_checked : candidates edge 5 0 3853 tail533 best.bound = rootCandidates532 := by
  unfold candidates
  have row : edge 3853 = SearchData.row3853.contains := by funext q; rfl
  rw [row]
  decide

def innerTail532_15 : List Nat := []
theorem innerScan532_15 : scan edge (search edge 3) 4 [3853] 3853 innerTail532_15 best = best := rfl

def innerTail532_14 : List Nat := 8293 :: innerTail532_15
def innerCandidates532_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates532_14_checked : candidates edge 4 3853 8293 innerTail532_15 best.bound = innerCandidates532_14 := by
  unfold candidates
  have row : edge 8293 = SearchData.row8293.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan532_14 : scan edge (search edge 3) 4 [3853] 3853 innerTail532_14 best = best := by
  rw [innerTail532_14, scan, if_neg (by decide), innerCandidates532_14_checked]
  change scan edge (search edge 3) 4 [3853] 3853 innerTail532_15
    (if 3 ≤ innerCandidates532_14.length then search edge 3 [3853, 8293] innerCandidates532_14 12146 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan532_15

def innerTail532_13 : List Nat := 7699 :: innerTail532_14
def innerCandidates532_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates532_13_checked : candidates edge 4 3853 7699 innerTail532_14 best.bound = innerCandidates532_13 := by
  unfold candidates
  have row : edge 7699 = SearchData.row7699.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan532_13 : scan edge (search edge 3) 4 [3853] 3853 innerTail532_13 best = best := by
  rw [innerTail532_13, scan, if_neg (by decide), innerCandidates532_13_checked]
  change scan edge (search edge 3) 4 [3853] 3853 innerTail532_14
    (if 3 ≤ innerCandidates532_13.length then search edge 3 [3853, 7699] innerCandidates532_13 11552 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan532_14

def innerTail532_12 : List Nat := 7591 :: innerTail532_13
def innerCandidates532_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates532_12_checked : candidates edge 4 3853 7591 innerTail532_13 best.bound = innerCandidates532_12 := by
  unfold candidates
  have row : edge 7591 = SearchData.row7591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan532_12 : scan edge (search edge 3) 4 [3853] 3853 innerTail532_12 best = best := by
  rw [innerTail532_12, scan, if_neg (by decide), innerCandidates532_12_checked]
  change scan edge (search edge 3) 4 [3853] 3853 innerTail532_13
    (if 3 ≤ innerCandidates532_12.length then search edge 3 [3853, 7591] innerCandidates532_12 11444 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan532_13

def innerTail532_11 : List Nat := 7297 :: innerTail532_12
def innerCandidates532_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates532_11_checked : candidates edge 4 3853 7297 innerTail532_12 best.bound = innerCandidates532_11 := by
  unfold candidates
  have row : edge 7297 = SearchData.row7297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan532_11 : scan edge (search edge 3) 4 [3853] 3853 innerTail532_11 best = best := by
  rw [innerTail532_11, scan, if_neg (by decide), innerCandidates532_11_checked]
  change scan edge (search edge 3) 4 [3853] 3853 innerTail532_12
    (if 3 ≤ innerCandidates532_11.length then search edge 3 [3853, 7297] innerCandidates532_11 11150 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan532_12

def innerTail532_10 : List Nat := 7243 :: innerTail532_11
def innerCandidates532_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates532_10_checked : candidates edge 4 3853 7243 innerTail532_11 best.bound = innerCandidates532_10 := by
  unfold candidates
  have row : edge 7243 = SearchData.row7243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan532_10 : scan edge (search edge 3) 4 [3853] 3853 innerTail532_10 best = best := by
  rw [innerTail532_10, scan, if_neg (by decide), innerCandidates532_10_checked]
  change scan edge (search edge 3) 4 [3853] 3853 innerTail532_11
    (if 3 ≤ innerCandidates532_10.length then search edge 3 [3853, 7243] innerCandidates532_10 11096 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan532_11

def innerTail532_9 : List Nat := 7237 :: innerTail532_10
def innerCandidates532_9 : List Nat := [7699]
set_option maxRecDepth 4096 in
theorem innerCandidates532_9_checked : candidates edge 4 3853 7237 innerTail532_10 best.bound = innerCandidates532_9 := by
  unfold candidates
  have row : edge 7237 = SearchData.row7237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan532_9 : scan edge (search edge 3) 4 [3853] 3853 innerTail532_9 best = best := by
  rw [innerTail532_9, scan, if_neg (by decide), innerCandidates532_9_checked]
  change scan edge (search edge 3) 4 [3853] 3853 innerTail532_10
    (if 3 ≤ innerCandidates532_9.length then search edge 3 [3853, 7237] innerCandidates532_9 11090 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan532_10

def innerTail532_8 : List Nat := 6991 :: innerTail532_9
def innerCandidates532_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates532_8_checked : candidates edge 4 3853 6991 innerTail532_9 best.bound = innerCandidates532_8 := by
  unfold candidates
  have row : edge 6991 = SearchData.row6991.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan532_8 : scan edge (search edge 3) 4 [3853] 3853 innerTail532_8 best = best := by
  rw [innerTail532_8, scan, if_neg (by decide), innerCandidates532_8_checked]
  change scan edge (search edge 3) 4 [3853] 3853 innerTail532_9
    (if 3 ≤ innerCandidates532_8.length then search edge 3 [3853, 6991] innerCandidates532_8 10844 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan532_9

def innerTail532_7 : List Nat := 6709 :: innerTail532_8
def innerCandidates532_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates532_7_checked : candidates edge 4 3853 6709 innerTail532_8 best.bound = innerCandidates532_7 := by
  unfold candidates
  have row : edge 6709 = SearchData.row6709.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan532_7 : scan edge (search edge 3) 4 [3853] 3853 innerTail532_7 best = best := by
  rw [innerTail532_7, scan, if_neg (by decide), innerCandidates532_7_checked]
  change scan edge (search edge 3) 4 [3853] 3853 innerTail532_8
    (if 3 ≤ innerCandidates532_7.length then search edge 3 [3853, 6709] innerCandidates532_7 10562 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan532_8

def innerTail532_6 : List Nat := 6607 :: innerTail532_7
def innerCandidates532_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates532_6_checked : candidates edge 4 3853 6607 innerTail532_7 best.bound = innerCandidates532_6 := by
  unfold candidates
  have row : edge 6607 = SearchData.row6607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan532_6 : scan edge (search edge 3) 4 [3853] 3853 innerTail532_6 best = best := by
  rw [innerTail532_6, scan, if_neg (by decide), innerCandidates532_6_checked]
  change scan edge (search edge 3) 4 [3853] 3853 innerTail532_7
    (if 3 ≤ innerCandidates532_6.length then search edge 3 [3853, 6607] innerCandidates532_6 10460 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan532_7

def innerTail532_5 : List Nat := 6367 :: innerTail532_6
def innerCandidates532_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates532_5_checked : candidates edge 4 3853 6367 innerTail532_6 best.bound = innerCandidates532_5 := by
  unfold candidates
  have row : edge 6367 = SearchData.row6367.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan532_5 : scan edge (search edge 3) 4 [3853] 3853 innerTail532_5 best = best := by
  rw [innerTail532_5, scan, if_neg (by decide), innerCandidates532_5_checked]
  change scan edge (search edge 3) 4 [3853] 3853 innerTail532_6
    (if 3 ≤ innerCandidates532_5.length then search edge 3 [3853, 6367] innerCandidates532_5 10220 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan532_6

def innerTail532_4 : List Nat := 6361 :: innerTail532_5
def innerCandidates532_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates532_4_checked : candidates edge 4 3853 6361 innerTail532_5 best.bound = innerCandidates532_4 := by
  unfold candidates
  have row : edge 6361 = SearchData.row6361.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan532_4 : scan edge (search edge 3) 4 [3853] 3853 innerTail532_4 best = best := by
  rw [innerTail532_4, scan, if_neg (by decide), innerCandidates532_4_checked]
  change scan edge (search edge 3) 4 [3853] 3853 innerTail532_5
    (if 3 ≤ innerCandidates532_4.length then search edge 3 [3853, 6361] innerCandidates532_4 10214 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan532_5

def innerTail532_3 : List Nat := 5683 :: innerTail532_4
def innerCandidates532_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates532_3_checked : candidates edge 4 3853 5683 innerTail532_4 best.bound = innerCandidates532_3 := by
  unfold candidates
  have row : edge 5683 = SearchData.row5683.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan532_3 : scan edge (search edge 3) 4 [3853] 3853 innerTail532_3 best = best := by
  rw [innerTail532_3, scan, if_neg (by decide), innerCandidates532_3_checked]
  change scan edge (search edge 3) 4 [3853] 3853 innerTail532_4
    (if 3 ≤ innerCandidates532_3.length then search edge 3 [3853, 5683] innerCandidates532_3 9536 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan532_4

def innerTail532_2 : List Nat := 5521 :: innerTail532_3
def innerCandidates532_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates532_2_checked : candidates edge 4 3853 5521 innerTail532_3 best.bound = innerCandidates532_2 := by
  unfold candidates
  have row : edge 5521 = SearchData.row5521.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan532_2 : scan edge (search edge 3) 4 [3853] 3853 innerTail532_2 best = best := by
  rw [innerTail532_2, scan, if_neg (by decide), innerCandidates532_2_checked]
  change scan edge (search edge 3) 4 [3853] 3853 innerTail532_3
    (if 3 ≤ innerCandidates532_2.length then search edge 3 [3853, 5521] innerCandidates532_2 9374 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan532_3

def innerTail532_1 : List Nat := 4759 :: innerTail532_2
def innerCandidates532_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates532_1_checked : candidates edge 4 3853 4759 innerTail532_2 best.bound = innerCandidates532_1 := by
  unfold candidates
  have row : edge 4759 = SearchData.row4759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan532_1 : scan edge (search edge 3) 4 [3853] 3853 innerTail532_1 best = best := by
  rw [innerTail532_1, scan, if_neg (by decide), innerCandidates532_1_checked]
  change scan edge (search edge 3) 4 [3853] 3853 innerTail532_2
    (if 3 ≤ innerCandidates532_1.length then search edge 3 [3853, 4759] innerCandidates532_1 8612 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan532_2

def innerTail532_0 : List Nat := 4099 :: innerTail532_1
def innerCandidates532_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates532_0_checked : candidates edge 4 3853 4099 innerTail532_1 best.bound = innerCandidates532_0 := by
  unfold candidates
  have row : edge 4099 = SearchData.row4099.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan532_0 : scan edge (search edge 3) 4 [3853] 3853 innerTail532_0 best = best := by
  rw [innerTail532_0, scan, if_neg (by decide), innerCandidates532_0_checked]
  change scan edge (search edge 3) 4 [3853] 3853 innerTail532_1
    (if 3 ≤ innerCandidates532_0.length then search edge 3 [3853, 4099] innerCandidates532_0 7952 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan532_1

theorem rootBranch532 : search edge 4 [3853] rootCandidates532 3853 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3853] 3853 innerTail532_0 best = best
  exact innerScan532_0

def rootCandidates531 : List Nat := [4013, 4139, 4271, 4649, 5651, 5903, 6521, 6659, 6899, 6983, 7043, 7151, 7547, 7673]
set_option maxRecDepth 4096 in
theorem rootCandidates531_checked : candidates edge 5 0 3851 tail532 best.bound = rootCandidates531 := by
  unfold candidates
  have row : edge 3851 = SearchData.row3851.contains := by funext q; rfl
  rw [row]
  decide

def innerTail531_14 : List Nat := []
theorem innerScan531_14 : scan edge (search edge 3) 4 [3851] 3851 innerTail531_14 best = best := rfl

def innerTail531_13 : List Nat := 7673 :: innerTail531_14
def innerCandidates531_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates531_13_checked : candidates edge 4 3851 7673 innerTail531_14 best.bound = innerCandidates531_13 := by
  unfold candidates
  have row : edge 7673 = SearchData.row7673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan531_13 : scan edge (search edge 3) 4 [3851] 3851 innerTail531_13 best = best := by
  rw [innerTail531_13, scan, if_neg (by decide), innerCandidates531_13_checked]
  change scan edge (search edge 3) 4 [3851] 3851 innerTail531_14
    (if 3 ≤ innerCandidates531_13.length then search edge 3 [3851, 7673] innerCandidates531_13 11524 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan531_14

def innerTail531_12 : List Nat := 7547 :: innerTail531_13
def innerCandidates531_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates531_12_checked : candidates edge 4 3851 7547 innerTail531_13 best.bound = innerCandidates531_12 := by
  unfold candidates
  have row : edge 7547 = SearchData.row7547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan531_12 : scan edge (search edge 3) 4 [3851] 3851 innerTail531_12 best = best := by
  rw [innerTail531_12, scan, if_neg (by decide), innerCandidates531_12_checked]
  change scan edge (search edge 3) 4 [3851] 3851 innerTail531_13
    (if 3 ≤ innerCandidates531_12.length then search edge 3 [3851, 7547] innerCandidates531_12 11398 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan531_13

def innerTail531_11 : List Nat := 7151 :: innerTail531_12
def innerCandidates531_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates531_11_checked : candidates edge 4 3851 7151 innerTail531_12 best.bound = innerCandidates531_11 := by
  unfold candidates
  have row : edge 7151 = SearchData.row7151.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan531_11 : scan edge (search edge 3) 4 [3851] 3851 innerTail531_11 best = best := by
  rw [innerTail531_11, scan, if_neg (by decide), innerCandidates531_11_checked]
  change scan edge (search edge 3) 4 [3851] 3851 innerTail531_12
    (if 3 ≤ innerCandidates531_11.length then search edge 3 [3851, 7151] innerCandidates531_11 11002 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan531_12

def innerTail531_10 : List Nat := 7043 :: innerTail531_11
def innerCandidates531_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates531_10_checked : candidates edge 4 3851 7043 innerTail531_11 best.bound = innerCandidates531_10 := by
  unfold candidates
  have row : edge 7043 = SearchData.row7043.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan531_10 : scan edge (search edge 3) 4 [3851] 3851 innerTail531_10 best = best := by
  rw [innerTail531_10, scan, if_neg (by decide), innerCandidates531_10_checked]
  change scan edge (search edge 3) 4 [3851] 3851 innerTail531_11
    (if 3 ≤ innerCandidates531_10.length then search edge 3 [3851, 7043] innerCandidates531_10 10894 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan531_11

def innerTail531_9 : List Nat := 6983 :: innerTail531_10
def innerCandidates531_9 : List Nat := [7043]
set_option maxRecDepth 4096 in
theorem innerCandidates531_9_checked : candidates edge 4 3851 6983 innerTail531_10 best.bound = innerCandidates531_9 := by
  unfold candidates
  have row : edge 6983 = SearchData.row6983.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan531_9 : scan edge (search edge 3) 4 [3851] 3851 innerTail531_9 best = best := by
  rw [innerTail531_9, scan, if_neg (by decide), innerCandidates531_9_checked]
  change scan edge (search edge 3) 4 [3851] 3851 innerTail531_10
    (if 3 ≤ innerCandidates531_9.length then search edge 3 [3851, 6983] innerCandidates531_9 10834 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan531_10

def innerTail531_8 : List Nat := 6899 :: innerTail531_9
def innerCandidates531_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates531_8_checked : candidates edge 4 3851 6899 innerTail531_9 best.bound = innerCandidates531_8 := by
  unfold candidates
  have row : edge 6899 = SearchData.row6899.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan531_8 : scan edge (search edge 3) 4 [3851] 3851 innerTail531_8 best = best := by
  rw [innerTail531_8, scan, if_neg (by decide), innerCandidates531_8_checked]
  change scan edge (search edge 3) 4 [3851] 3851 innerTail531_9
    (if 3 ≤ innerCandidates531_8.length then search edge 3 [3851, 6899] innerCandidates531_8 10750 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan531_9

def innerTail531_7 : List Nat := 6659 :: innerTail531_8
def innerCandidates531_7 : List Nat := [7043]
set_option maxRecDepth 4096 in
theorem innerCandidates531_7_checked : candidates edge 4 3851 6659 innerTail531_8 best.bound = innerCandidates531_7 := by
  unfold candidates
  have row : edge 6659 = SearchData.row6659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan531_7 : scan edge (search edge 3) 4 [3851] 3851 innerTail531_7 best = best := by
  rw [innerTail531_7, scan, if_neg (by decide), innerCandidates531_7_checked]
  change scan edge (search edge 3) 4 [3851] 3851 innerTail531_8
    (if 3 ≤ innerCandidates531_7.length then search edge 3 [3851, 6659] innerCandidates531_7 10510 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan531_8

def innerTail531_6 : List Nat := 6521 :: innerTail531_7
def innerCandidates531_6 : List Nat := [7043]
set_option maxRecDepth 4096 in
theorem innerCandidates531_6_checked : candidates edge 4 3851 6521 innerTail531_7 best.bound = innerCandidates531_6 := by
  unfold candidates
  have row : edge 6521 = SearchData.row6521.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan531_6 : scan edge (search edge 3) 4 [3851] 3851 innerTail531_6 best = best := by
  rw [innerTail531_6, scan, if_neg (by decide), innerCandidates531_6_checked]
  change scan edge (search edge 3) 4 [3851] 3851 innerTail531_7
    (if 3 ≤ innerCandidates531_6.length then search edge 3 [3851, 6521] innerCandidates531_6 10372 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan531_7

def innerTail531_5 : List Nat := 5903 :: innerTail531_6
def innerCandidates531_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates531_5_checked : candidates edge 4 3851 5903 innerTail531_6 best.bound = innerCandidates531_5 := by
  unfold candidates
  have row : edge 5903 = SearchData.row5903.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan531_5 : scan edge (search edge 3) 4 [3851] 3851 innerTail531_5 best = best := by
  rw [innerTail531_5, scan, if_neg (by decide), innerCandidates531_5_checked]
  change scan edge (search edge 3) 4 [3851] 3851 innerTail531_6
    (if 3 ≤ innerCandidates531_5.length then search edge 3 [3851, 5903] innerCandidates531_5 9754 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan531_6

def innerTail531_4 : List Nat := 5651 :: innerTail531_5
def innerCandidates531_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates531_4_checked : candidates edge 4 3851 5651 innerTail531_5 best.bound = innerCandidates531_4 := by
  unfold candidates
  have row : edge 5651 = SearchData.row5651.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan531_4 : scan edge (search edge 3) 4 [3851] 3851 innerTail531_4 best = best := by
  rw [innerTail531_4, scan, if_neg (by decide), innerCandidates531_4_checked]
  change scan edge (search edge 3) 4 [3851] 3851 innerTail531_5
    (if 3 ≤ innerCandidates531_4.length then search edge 3 [3851, 5651] innerCandidates531_4 9502 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan531_5

def innerTail531_3 : List Nat := 4649 :: innerTail531_4
def innerCandidates531_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates531_3_checked : candidates edge 4 3851 4649 innerTail531_4 best.bound = innerCandidates531_3 := by
  unfold candidates
  have row : edge 4649 = SearchData.row4649.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan531_3 : scan edge (search edge 3) 4 [3851] 3851 innerTail531_3 best = best := by
  rw [innerTail531_3, scan, if_neg (by decide), innerCandidates531_3_checked]
  change scan edge (search edge 3) 4 [3851] 3851 innerTail531_4
    (if 3 ≤ innerCandidates531_3.length then search edge 3 [3851, 4649] innerCandidates531_3 8500 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan531_4

def innerTail531_2 : List Nat := 4271 :: innerTail531_3
def innerCandidates531_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates531_2_checked : candidates edge 4 3851 4271 innerTail531_3 best.bound = innerCandidates531_2 := by
  unfold candidates
  have row : edge 4271 = SearchData.row4271.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan531_2 : scan edge (search edge 3) 4 [3851] 3851 innerTail531_2 best = best := by
  rw [innerTail531_2, scan, if_neg (by decide), innerCandidates531_2_checked]
  change scan edge (search edge 3) 4 [3851] 3851 innerTail531_3
    (if 3 ≤ innerCandidates531_2.length then search edge 3 [3851, 4271] innerCandidates531_2 8122 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan531_3

def innerTail531_1 : List Nat := 4139 :: innerTail531_2
def innerCandidates531_1 : List Nat := [7673]
set_option maxRecDepth 4096 in
theorem innerCandidates531_1_checked : candidates edge 4 3851 4139 innerTail531_2 best.bound = innerCandidates531_1 := by
  unfold candidates
  have row : edge 4139 = SearchData.row4139.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan531_1 : scan edge (search edge 3) 4 [3851] 3851 innerTail531_1 best = best := by
  rw [innerTail531_1, scan, if_neg (by decide), innerCandidates531_1_checked]
  change scan edge (search edge 3) 4 [3851] 3851 innerTail531_2
    (if 3 ≤ innerCandidates531_1.length then search edge 3 [3851, 4139] innerCandidates531_1 7990 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan531_2

def innerTail531_0 : List Nat := 4013 :: innerTail531_1
def innerCandidates531_0 : List Nat := [6659]
set_option maxRecDepth 4096 in
theorem innerCandidates531_0_checked : candidates edge 4 3851 4013 innerTail531_1 best.bound = innerCandidates531_0 := by
  unfold candidates
  have row : edge 4013 = SearchData.row4013.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan531_0 : scan edge (search edge 3) 4 [3851] 3851 innerTail531_0 best = best := by
  rw [innerTail531_0, scan, if_neg (by decide), innerCandidates531_0_checked]
  change scan edge (search edge 3) 4 [3851] 3851 innerTail531_1
    (if 3 ≤ innerCandidates531_0.length then search edge 3 [3851, 4013] innerCandidates531_0 7864 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan531_1

theorem rootBranch531 : search edge 4 [3851] rootCandidates531 3851 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3851] 3851 innerTail531_0 best = best
  exact innerScan531_0

end ProjectEulerSolutions.PrimePairUpper.Search
