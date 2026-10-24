import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates567 : List Nat := [4937, 5657, 6263, 6911, 6959, 7253, 7457, 7541, 7607, 7673, 8369, 8387]
set_option maxRecDepth 4096 in
theorem rootCandidates567_checked : candidates edge 5 0 4139 tail568 best.bound = rootCandidates567 := by
  unfold candidates
  have row : edge 4139 = SearchData.row4139.contains := by funext q; rfl
  rw [row]
  decide

def innerTail567_12 : List Nat := []
theorem innerScan567_12 : scan edge (search edge 3) 4 [4139] 4139 innerTail567_12 best = best := rfl

def innerTail567_11 : List Nat := 8387 :: innerTail567_12
def innerCandidates567_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates567_11_checked : candidates edge 4 4139 8387 innerTail567_12 best.bound = innerCandidates567_11 := by
  unfold candidates
  have row : edge 8387 = SearchData.row8387.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan567_11 : scan edge (search edge 3) 4 [4139] 4139 innerTail567_11 best = best := by
  rw [innerTail567_11, scan, if_neg (by decide), innerCandidates567_11_checked]
  change scan edge (search edge 3) 4 [4139] 4139 innerTail567_12
    (if 3 ≤ innerCandidates567_11.length then search edge 3 [4139, 8387] innerCandidates567_11 12526 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan567_12

def innerTail567_10 : List Nat := 8369 :: innerTail567_11
def innerCandidates567_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates567_10_checked : candidates edge 4 4139 8369 innerTail567_11 best.bound = innerCandidates567_10 := by
  unfold candidates
  have row : edge 8369 = SearchData.row8369.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan567_10 : scan edge (search edge 3) 4 [4139] 4139 innerTail567_10 best = best := by
  rw [innerTail567_10, scan, if_neg (by decide), innerCandidates567_10_checked]
  change scan edge (search edge 3) 4 [4139] 4139 innerTail567_11
    (if 3 ≤ innerCandidates567_10.length then search edge 3 [4139, 8369] innerCandidates567_10 12508 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan567_11

def innerTail567_9 : List Nat := 7673 :: innerTail567_10
def innerCandidates567_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates567_9_checked : candidates edge 4 4139 7673 innerTail567_10 best.bound = innerCandidates567_9 := by
  unfold candidates
  have row : edge 7673 = SearchData.row7673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan567_9 : scan edge (search edge 3) 4 [4139] 4139 innerTail567_9 best = best := by
  rw [innerTail567_9, scan, if_neg (by decide), innerCandidates567_9_checked]
  change scan edge (search edge 3) 4 [4139] 4139 innerTail567_10
    (if 3 ≤ innerCandidates567_9.length then search edge 3 [4139, 7673] innerCandidates567_9 11812 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan567_10

def innerTail567_8 : List Nat := 7607 :: innerTail567_9
def innerCandidates567_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates567_8_checked : candidates edge 4 4139 7607 innerTail567_9 best.bound = innerCandidates567_8 := by
  unfold candidates
  have row : edge 7607 = SearchData.row7607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan567_8 : scan edge (search edge 3) 4 [4139] 4139 innerTail567_8 best = best := by
  rw [innerTail567_8, scan, if_neg (by decide), innerCandidates567_8_checked]
  change scan edge (search edge 3) 4 [4139] 4139 innerTail567_9
    (if 3 ≤ innerCandidates567_8.length then search edge 3 [4139, 7607] innerCandidates567_8 11746 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan567_9

def innerTail567_7 : List Nat := 7541 :: innerTail567_8
def innerCandidates567_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates567_7_checked : candidates edge 4 4139 7541 innerTail567_8 best.bound = innerCandidates567_7 := by
  unfold candidates
  have row : edge 7541 = SearchData.row7541.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan567_7 : scan edge (search edge 3) 4 [4139] 4139 innerTail567_7 best = best := by
  rw [innerTail567_7, scan, if_neg (by decide), innerCandidates567_7_checked]
  change scan edge (search edge 3) 4 [4139] 4139 innerTail567_8
    (if 3 ≤ innerCandidates567_7.length then search edge 3 [4139, 7541] innerCandidates567_7 11680 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan567_8

def innerTail567_6 : List Nat := 7457 :: innerTail567_7
def innerCandidates567_6 : List Nat := [8369]
set_option maxRecDepth 4096 in
theorem innerCandidates567_6_checked : candidates edge 4 4139 7457 innerTail567_7 best.bound = innerCandidates567_6 := by
  unfold candidates
  have row : edge 7457 = SearchData.row7457.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan567_6 : scan edge (search edge 3) 4 [4139] 4139 innerTail567_6 best = best := by
  rw [innerTail567_6, scan, if_neg (by decide), innerCandidates567_6_checked]
  change scan edge (search edge 3) 4 [4139] 4139 innerTail567_7
    (if 3 ≤ innerCandidates567_6.length then search edge 3 [4139, 7457] innerCandidates567_6 11596 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan567_7

def innerTail567_5 : List Nat := 7253 :: innerTail567_6
def innerCandidates567_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates567_5_checked : candidates edge 4 4139 7253 innerTail567_6 best.bound = innerCandidates567_5 := by
  unfold candidates
  have row : edge 7253 = SearchData.row7253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan567_5 : scan edge (search edge 3) 4 [4139] 4139 innerTail567_5 best = best := by
  rw [innerTail567_5, scan, if_neg (by decide), innerCandidates567_5_checked]
  change scan edge (search edge 3) 4 [4139] 4139 innerTail567_6
    (if 3 ≤ innerCandidates567_5.length then search edge 3 [4139, 7253] innerCandidates567_5 11392 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan567_6

def innerTail567_4 : List Nat := 6959 :: innerTail567_5
def innerCandidates567_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates567_4_checked : candidates edge 4 4139 6959 innerTail567_5 best.bound = innerCandidates567_4 := by
  unfold candidates
  have row : edge 6959 = SearchData.row6959.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan567_4 : scan edge (search edge 3) 4 [4139] 4139 innerTail567_4 best = best := by
  rw [innerTail567_4, scan, if_neg (by decide), innerCandidates567_4_checked]
  change scan edge (search edge 3) 4 [4139] 4139 innerTail567_5
    (if 3 ≤ innerCandidates567_4.length then search edge 3 [4139, 6959] innerCandidates567_4 11098 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan567_5

def innerTail567_3 : List Nat := 6911 :: innerTail567_4
def innerCandidates567_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates567_3_checked : candidates edge 4 4139 6911 innerTail567_4 best.bound = innerCandidates567_3 := by
  unfold candidates
  have row : edge 6911 = SearchData.row6911.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan567_3 : scan edge (search edge 3) 4 [4139] 4139 innerTail567_3 best = best := by
  rw [innerTail567_3, scan, if_neg (by decide), innerCandidates567_3_checked]
  change scan edge (search edge 3) 4 [4139] 4139 innerTail567_4
    (if 3 ≤ innerCandidates567_3.length then search edge 3 [4139, 6911] innerCandidates567_3 11050 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan567_4

def innerTail567_2 : List Nat := 6263 :: innerTail567_3
def innerCandidates567_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates567_2_checked : candidates edge 4 4139 6263 innerTail567_3 best.bound = innerCandidates567_2 := by
  unfold candidates
  have row : edge 6263 = SearchData.row6263.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan567_2 : scan edge (search edge 3) 4 [4139] 4139 innerTail567_2 best = best := by
  rw [innerTail567_2, scan, if_neg (by decide), innerCandidates567_2_checked]
  change scan edge (search edge 3) 4 [4139] 4139 innerTail567_3
    (if 3 ≤ innerCandidates567_2.length then search edge 3 [4139, 6263] innerCandidates567_2 10402 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan567_3

def innerTail567_1 : List Nat := 5657 :: innerTail567_2
def innerCandidates567_1 : List Nat := [7673, 8369, 8387]
set_option maxRecDepth 4096 in
theorem innerCandidates567_1_checked : candidates edge 4 4139 5657 innerTail567_2 best.bound = innerCandidates567_1 := by
  unfold candidates
  have row : edge 5657 = SearchData.row5657.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild567_1 : search edge 3 [4139, 5657] innerCandidates567_1 9796 best = best := by decide

theorem innerScan567_1 : scan edge (search edge 3) 4 [4139] 4139 innerTail567_1 best = best := by
  rw [innerTail567_1, scan, if_neg (by decide), innerCandidates567_1_checked]
  change scan edge (search edge 3) 4 [4139] 4139 innerTail567_2
    (if 3 ≤ innerCandidates567_1.length then search edge 3 [4139, 5657] innerCandidates567_1 9796 best else best) = best
  rw [if_pos (by decide), innerChild567_1]
  exact innerScan567_2

def innerTail567_0 : List Nat := 4937 :: innerTail567_1
def innerCandidates567_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates567_0_checked : candidates edge 4 4139 4937 innerTail567_1 best.bound = innerCandidates567_0 := by
  unfold candidates
  have row : edge 4937 = SearchData.row4937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan567_0 : scan edge (search edge 3) 4 [4139] 4139 innerTail567_0 best = best := by
  rw [innerTail567_0, scan, if_neg (by decide), innerCandidates567_0_checked]
  change scan edge (search edge 3) 4 [4139] 4139 innerTail567_1
    (if 3 ≤ innerCandidates567_0.length then search edge 3 [4139, 4937] innerCandidates567_0 9076 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan567_1

theorem rootBranch567 : search edge 4 [4139] rootCandidates567 4139 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4139] 4139 innerTail567_0 best = best
  exact innerScan567_0

def rootCandidates566 : List Nat := [4409, 4493, 4721, 5081, 5171, 5351, 5387, 6173, 6947, 7433, 7451, 8111]
set_option maxRecDepth 4096 in
theorem rootCandidates566_checked : candidates edge 5 0 4133 tail567 best.bound = rootCandidates566 := by
  unfold candidates
  have row : edge 4133 = SearchData.row4133.contains := by funext q; rfl
  rw [row]
  decide

def innerTail566_12 : List Nat := []
theorem innerScan566_12 : scan edge (search edge 3) 4 [4133] 4133 innerTail566_12 best = best := rfl

def innerTail566_11 : List Nat := 8111 :: innerTail566_12
def innerCandidates566_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates566_11_checked : candidates edge 4 4133 8111 innerTail566_12 best.bound = innerCandidates566_11 := by
  unfold candidates
  have row : edge 8111 = SearchData.row8111.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan566_11 : scan edge (search edge 3) 4 [4133] 4133 innerTail566_11 best = best := by
  rw [innerTail566_11, scan, if_neg (by decide), innerCandidates566_11_checked]
  change scan edge (search edge 3) 4 [4133] 4133 innerTail566_12
    (if 3 ≤ innerCandidates566_11.length then search edge 3 [4133, 8111] innerCandidates566_11 12244 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan566_12

def innerTail566_10 : List Nat := 7451 :: innerTail566_11
def innerCandidates566_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates566_10_checked : candidates edge 4 4133 7451 innerTail566_11 best.bound = innerCandidates566_10 := by
  unfold candidates
  have row : edge 7451 = SearchData.row7451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan566_10 : scan edge (search edge 3) 4 [4133] 4133 innerTail566_10 best = best := by
  rw [innerTail566_10, scan, if_neg (by decide), innerCandidates566_10_checked]
  change scan edge (search edge 3) 4 [4133] 4133 innerTail566_11
    (if 3 ≤ innerCandidates566_10.length then search edge 3 [4133, 7451] innerCandidates566_10 11584 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan566_11

def innerTail566_9 : List Nat := 7433 :: innerTail566_10
def innerCandidates566_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates566_9_checked : candidates edge 4 4133 7433 innerTail566_10 best.bound = innerCandidates566_9 := by
  unfold candidates
  have row : edge 7433 = SearchData.row7433.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan566_9 : scan edge (search edge 3) 4 [4133] 4133 innerTail566_9 best = best := by
  rw [innerTail566_9, scan, if_neg (by decide), innerCandidates566_9_checked]
  change scan edge (search edge 3) 4 [4133] 4133 innerTail566_10
    (if 3 ≤ innerCandidates566_9.length then search edge 3 [4133, 7433] innerCandidates566_9 11566 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan566_10

def innerTail566_8 : List Nat := 6947 :: innerTail566_9
def innerCandidates566_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates566_8_checked : candidates edge 4 4133 6947 innerTail566_9 best.bound = innerCandidates566_8 := by
  unfold candidates
  have row : edge 6947 = SearchData.row6947.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan566_8 : scan edge (search edge 3) 4 [4133] 4133 innerTail566_8 best = best := by
  rw [innerTail566_8, scan, if_neg (by decide), innerCandidates566_8_checked]
  change scan edge (search edge 3) 4 [4133] 4133 innerTail566_9
    (if 3 ≤ innerCandidates566_8.length then search edge 3 [4133, 6947] innerCandidates566_8 11080 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan566_9

def innerTail566_7 : List Nat := 6173 :: innerTail566_8
def innerCandidates566_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates566_7_checked : candidates edge 4 4133 6173 innerTail566_8 best.bound = innerCandidates566_7 := by
  unfold candidates
  have row : edge 6173 = SearchData.row6173.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan566_7 : scan edge (search edge 3) 4 [4133] 4133 innerTail566_7 best = best := by
  rw [innerTail566_7, scan, if_neg (by decide), innerCandidates566_7_checked]
  change scan edge (search edge 3) 4 [4133] 4133 innerTail566_8
    (if 3 ≤ innerCandidates566_7.length then search edge 3 [4133, 6173] innerCandidates566_7 10306 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan566_8

def innerTail566_6 : List Nat := 5387 :: innerTail566_7
def innerCandidates566_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates566_6_checked : candidates edge 4 4133 5387 innerTail566_7 best.bound = innerCandidates566_6 := by
  unfold candidates
  have row : edge 5387 = SearchData.row5387.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan566_6 : scan edge (search edge 3) 4 [4133] 4133 innerTail566_6 best = best := by
  rw [innerTail566_6, scan, if_neg (by decide), innerCandidates566_6_checked]
  change scan edge (search edge 3) 4 [4133] 4133 innerTail566_7
    (if 3 ≤ innerCandidates566_6.length then search edge 3 [4133, 5387] innerCandidates566_6 9520 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan566_7

def innerTail566_5 : List Nat := 5351 :: innerTail566_6
def innerCandidates566_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates566_5_checked : candidates edge 4 4133 5351 innerTail566_6 best.bound = innerCandidates566_5 := by
  unfold candidates
  have row : edge 5351 = SearchData.row5351.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan566_5 : scan edge (search edge 3) 4 [4133] 4133 innerTail566_5 best = best := by
  rw [innerTail566_5, scan, if_neg (by decide), innerCandidates566_5_checked]
  change scan edge (search edge 3) 4 [4133] 4133 innerTail566_6
    (if 3 ≤ innerCandidates566_5.length then search edge 3 [4133, 5351] innerCandidates566_5 9484 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan566_6

def innerTail566_4 : List Nat := 5171 :: innerTail566_5
def innerCandidates566_4 : List Nat := [8111]
set_option maxRecDepth 4096 in
theorem innerCandidates566_4_checked : candidates edge 4 4133 5171 innerTail566_5 best.bound = innerCandidates566_4 := by
  unfold candidates
  have row : edge 5171 = SearchData.row5171.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan566_4 : scan edge (search edge 3) 4 [4133] 4133 innerTail566_4 best = best := by
  rw [innerTail566_4, scan, if_neg (by decide), innerCandidates566_4_checked]
  change scan edge (search edge 3) 4 [4133] 4133 innerTail566_5
    (if 3 ≤ innerCandidates566_4.length then search edge 3 [4133, 5171] innerCandidates566_4 9304 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan566_5

def innerTail566_3 : List Nat := 5081 :: innerTail566_4
def innerCandidates566_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates566_3_checked : candidates edge 4 4133 5081 innerTail566_4 best.bound = innerCandidates566_3 := by
  unfold candidates
  have row : edge 5081 = SearchData.row5081.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan566_3 : scan edge (search edge 3) 4 [4133] 4133 innerTail566_3 best = best := by
  rw [innerTail566_3, scan, if_neg (by decide), innerCandidates566_3_checked]
  change scan edge (search edge 3) 4 [4133] 4133 innerTail566_4
    (if 3 ≤ innerCandidates566_3.length then search edge 3 [4133, 5081] innerCandidates566_3 9214 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan566_4

def innerTail566_2 : List Nat := 4721 :: innerTail566_3
def innerCandidates566_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates566_2_checked : candidates edge 4 4133 4721 innerTail566_3 best.bound = innerCandidates566_2 := by
  unfold candidates
  have row : edge 4721 = SearchData.row4721.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan566_2 : scan edge (search edge 3) 4 [4133] 4133 innerTail566_2 best = best := by
  rw [innerTail566_2, scan, if_neg (by decide), innerCandidates566_2_checked]
  change scan edge (search edge 3) 4 [4133] 4133 innerTail566_3
    (if 3 ≤ innerCandidates566_2.length then search edge 3 [4133, 4721] innerCandidates566_2 8854 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan566_3

def innerTail566_1 : List Nat := 4493 :: innerTail566_2
def innerCandidates566_1 : List Nat := [6173]
set_option maxRecDepth 4096 in
theorem innerCandidates566_1_checked : candidates edge 4 4133 4493 innerTail566_2 best.bound = innerCandidates566_1 := by
  unfold candidates
  have row : edge 4493 = SearchData.row4493.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan566_1 : scan edge (search edge 3) 4 [4133] 4133 innerTail566_1 best = best := by
  rw [innerTail566_1, scan, if_neg (by decide), innerCandidates566_1_checked]
  change scan edge (search edge 3) 4 [4133] 4133 innerTail566_2
    (if 3 ≤ innerCandidates566_1.length then search edge 3 [4133, 4493] innerCandidates566_1 8626 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan566_2

def innerTail566_0 : List Nat := 4409 :: innerTail566_1
def innerCandidates566_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates566_0_checked : candidates edge 4 4133 4409 innerTail566_1 best.bound = innerCandidates566_0 := by
  unfold candidates
  have row : edge 4409 = SearchData.row4409.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan566_0 : scan edge (search edge 3) 4 [4133] 4133 innerTail566_0 best = best := by
  rw [innerTail566_0, scan, if_neg (by decide), innerCandidates566_0_checked]
  change scan edge (search edge 3) 4 [4133] 4133 innerTail566_1
    (if 3 ≤ innerCandidates566_0.length then search edge 3 [4133, 4409] innerCandidates566_0 8542 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan566_1

theorem rootBranch566 : search edge 4 [4133] rootCandidates566 4133 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4133] 4133 innerTail566_0 best = best
  exact innerScan566_0

def rootCandidates565 : List Nat := [4651, 4861, 5521, 6271, 6361, 6781, 6949, 7159, 7717, 8167, 8221, 8377]
set_option maxRecDepth 4096 in
theorem rootCandidates565_checked : candidates edge 5 0 4129 tail566 best.bound = rootCandidates565 := by
  unfold candidates
  have row : edge 4129 = SearchData.row4129.contains := by funext q; rfl
  rw [row]
  decide

def innerTail565_12 : List Nat := []
theorem innerScan565_12 : scan edge (search edge 3) 4 [4129] 4129 innerTail565_12 best = best := rfl

def innerTail565_11 : List Nat := 8377 :: innerTail565_12
def innerCandidates565_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates565_11_checked : candidates edge 4 4129 8377 innerTail565_12 best.bound = innerCandidates565_11 := by
  unfold candidates
  have row : edge 8377 = SearchData.row8377.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan565_11 : scan edge (search edge 3) 4 [4129] 4129 innerTail565_11 best = best := by
  rw [innerTail565_11, scan, if_neg (by decide), innerCandidates565_11_checked]
  change scan edge (search edge 3) 4 [4129] 4129 innerTail565_12
    (if 3 ≤ innerCandidates565_11.length then search edge 3 [4129, 8377] innerCandidates565_11 12506 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan565_12

def innerTail565_10 : List Nat := 8221 :: innerTail565_11
def innerCandidates565_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates565_10_checked : candidates edge 4 4129 8221 innerTail565_11 best.bound = innerCandidates565_10 := by
  unfold candidates
  have row : edge 8221 = SearchData.row8221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan565_10 : scan edge (search edge 3) 4 [4129] 4129 innerTail565_10 best = best := by
  rw [innerTail565_10, scan, if_neg (by decide), innerCandidates565_10_checked]
  change scan edge (search edge 3) 4 [4129] 4129 innerTail565_11
    (if 3 ≤ innerCandidates565_10.length then search edge 3 [4129, 8221] innerCandidates565_10 12350 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan565_11

def innerTail565_9 : List Nat := 8167 :: innerTail565_10
def innerCandidates565_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates565_9_checked : candidates edge 4 4129 8167 innerTail565_10 best.bound = innerCandidates565_9 := by
  unfold candidates
  have row : edge 8167 = SearchData.row8167.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan565_9 : scan edge (search edge 3) 4 [4129] 4129 innerTail565_9 best = best := by
  rw [innerTail565_9, scan, if_neg (by decide), innerCandidates565_9_checked]
  change scan edge (search edge 3) 4 [4129] 4129 innerTail565_10
    (if 3 ≤ innerCandidates565_9.length then search edge 3 [4129, 8167] innerCandidates565_9 12296 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan565_10

def innerTail565_8 : List Nat := 7717 :: innerTail565_9
def innerCandidates565_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates565_8_checked : candidates edge 4 4129 7717 innerTail565_9 best.bound = innerCandidates565_8 := by
  unfold candidates
  have row : edge 7717 = SearchData.row7717.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan565_8 : scan edge (search edge 3) 4 [4129] 4129 innerTail565_8 best = best := by
  rw [innerTail565_8, scan, if_neg (by decide), innerCandidates565_8_checked]
  change scan edge (search edge 3) 4 [4129] 4129 innerTail565_9
    (if 3 ≤ innerCandidates565_8.length then search edge 3 [4129, 7717] innerCandidates565_8 11846 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan565_9

def innerTail565_7 : List Nat := 7159 :: innerTail565_8
def innerCandidates565_7 : List Nat := [7717]
set_option maxRecDepth 4096 in
theorem innerCandidates565_7_checked : candidates edge 4 4129 7159 innerTail565_8 best.bound = innerCandidates565_7 := by
  unfold candidates
  have row : edge 7159 = SearchData.row7159.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan565_7 : scan edge (search edge 3) 4 [4129] 4129 innerTail565_7 best = best := by
  rw [innerTail565_7, scan, if_neg (by decide), innerCandidates565_7_checked]
  change scan edge (search edge 3) 4 [4129] 4129 innerTail565_8
    (if 3 ≤ innerCandidates565_7.length then search edge 3 [4129, 7159] innerCandidates565_7 11288 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan565_8

def innerTail565_6 : List Nat := 6949 :: innerTail565_7
def innerCandidates565_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates565_6_checked : candidates edge 4 4129 6949 innerTail565_7 best.bound = innerCandidates565_6 := by
  unfold candidates
  have row : edge 6949 = SearchData.row6949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan565_6 : scan edge (search edge 3) 4 [4129] 4129 innerTail565_6 best = best := by
  rw [innerTail565_6, scan, if_neg (by decide), innerCandidates565_6_checked]
  change scan edge (search edge 3) 4 [4129] 4129 innerTail565_7
    (if 3 ≤ innerCandidates565_6.length then search edge 3 [4129, 6949] innerCandidates565_6 11078 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan565_7

def innerTail565_5 : List Nat := 6781 :: innerTail565_6
def innerCandidates565_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates565_5_checked : candidates edge 4 4129 6781 innerTail565_6 best.bound = innerCandidates565_5 := by
  unfold candidates
  have row : edge 6781 = SearchData.row6781.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan565_5 : scan edge (search edge 3) 4 [4129] 4129 innerTail565_5 best = best := by
  rw [innerTail565_5, scan, if_neg (by decide), innerCandidates565_5_checked]
  change scan edge (search edge 3) 4 [4129] 4129 innerTail565_6
    (if 3 ≤ innerCandidates565_5.length then search edge 3 [4129, 6781] innerCandidates565_5 10910 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan565_6

def innerTail565_4 : List Nat := 6361 :: innerTail565_5
def innerCandidates565_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates565_4_checked : candidates edge 4 4129 6361 innerTail565_5 best.bound = innerCandidates565_4 := by
  unfold candidates
  have row : edge 6361 = SearchData.row6361.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan565_4 : scan edge (search edge 3) 4 [4129] 4129 innerTail565_4 best = best := by
  rw [innerTail565_4, scan, if_neg (by decide), innerCandidates565_4_checked]
  change scan edge (search edge 3) 4 [4129] 4129 innerTail565_5
    (if 3 ≤ innerCandidates565_4.length then search edge 3 [4129, 6361] innerCandidates565_4 10490 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan565_5

def innerTail565_3 : List Nat := 6271 :: innerTail565_4
def innerCandidates565_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates565_3_checked : candidates edge 4 4129 6271 innerTail565_4 best.bound = innerCandidates565_3 := by
  unfold candidates
  have row : edge 6271 = SearchData.row6271.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan565_3 : scan edge (search edge 3) 4 [4129] 4129 innerTail565_3 best = best := by
  rw [innerTail565_3, scan, if_neg (by decide), innerCandidates565_3_checked]
  change scan edge (search edge 3) 4 [4129] 4129 innerTail565_4
    (if 3 ≤ innerCandidates565_3.length then search edge 3 [4129, 6271] innerCandidates565_3 10400 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan565_4

def innerTail565_2 : List Nat := 5521 :: innerTail565_3
def innerCandidates565_2 : List Nat := [7159]
set_option maxRecDepth 4096 in
theorem innerCandidates565_2_checked : candidates edge 4 4129 5521 innerTail565_3 best.bound = innerCandidates565_2 := by
  unfold candidates
  have row : edge 5521 = SearchData.row5521.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan565_2 : scan edge (search edge 3) 4 [4129] 4129 innerTail565_2 best = best := by
  rw [innerTail565_2, scan, if_neg (by decide), innerCandidates565_2_checked]
  change scan edge (search edge 3) 4 [4129] 4129 innerTail565_3
    (if 3 ≤ innerCandidates565_2.length then search edge 3 [4129, 5521] innerCandidates565_2 9650 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan565_3

def innerTail565_1 : List Nat := 4861 :: innerTail565_2
def innerCandidates565_1 : List Nat := [6361]
set_option maxRecDepth 4096 in
theorem innerCandidates565_1_checked : candidates edge 4 4129 4861 innerTail565_2 best.bound = innerCandidates565_1 := by
  unfold candidates
  have row : edge 4861 = SearchData.row4861.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan565_1 : scan edge (search edge 3) 4 [4129] 4129 innerTail565_1 best = best := by
  rw [innerTail565_1, scan, if_neg (by decide), innerCandidates565_1_checked]
  change scan edge (search edge 3) 4 [4129] 4129 innerTail565_2
    (if 3 ≤ innerCandidates565_1.length then search edge 3 [4129, 4861] innerCandidates565_1 8990 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan565_2

def innerTail565_0 : List Nat := 4651 :: innerTail565_1
def innerCandidates565_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates565_0_checked : candidates edge 4 4129 4651 innerTail565_1 best.bound = innerCandidates565_0 := by
  unfold candidates
  have row : edge 4651 = SearchData.row4651.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan565_0 : scan edge (search edge 3) 4 [4129] 4129 innerTail565_0 best = best := by
  rw [innerTail565_0, scan, if_neg (by decide), innerCandidates565_0_checked]
  change scan edge (search edge 3) 4 [4129] 4129 innerTail565_1
    (if 3 ≤ innerCandidates565_0.length then search edge 3 [4129, 4651] innerCandidates565_0 8780 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan565_1

theorem rootBranch565 : search edge 4 [4129] rootCandidates565 4129 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4129] 4129 innerTail565_0 best = best
  exact innerScan565_0

def rootCandidates564 : List Nat := [4967, 5309, 5519, 6581, 6737, 6959, 7079, 7949, 8243, 8291]
set_option maxRecDepth 4096 in
theorem rootCandidates564_checked : candidates edge 5 0 4127 tail565 best.bound = rootCandidates564 := by
  unfold candidates
  have row : edge 4127 = SearchData.row4127.contains := by funext q; rfl
  rw [row]
  decide

def innerTail564_10 : List Nat := []
theorem innerScan564_10 : scan edge (search edge 3) 4 [4127] 4127 innerTail564_10 best = best := rfl

def innerTail564_9 : List Nat := 8291 :: innerTail564_10
def innerCandidates564_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates564_9_checked : candidates edge 4 4127 8291 innerTail564_10 best.bound = innerCandidates564_9 := by
  unfold candidates
  have row : edge 8291 = SearchData.row8291.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan564_9 : scan edge (search edge 3) 4 [4127] 4127 innerTail564_9 best = best := by
  rw [innerTail564_9, scan, if_neg (by decide), innerCandidates564_9_checked]
  change scan edge (search edge 3) 4 [4127] 4127 innerTail564_10
    (if 3 ≤ innerCandidates564_9.length then search edge 3 [4127, 8291] innerCandidates564_9 12418 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan564_10

def innerTail564_8 : List Nat := 8243 :: innerTail564_9
def innerCandidates564_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates564_8_checked : candidates edge 4 4127 8243 innerTail564_9 best.bound = innerCandidates564_8 := by
  unfold candidates
  have row : edge 8243 = SearchData.row8243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan564_8 : scan edge (search edge 3) 4 [4127] 4127 innerTail564_8 best = best := by
  rw [innerTail564_8, scan, if_neg (by decide), innerCandidates564_8_checked]
  change scan edge (search edge 3) 4 [4127] 4127 innerTail564_9
    (if 3 ≤ innerCandidates564_8.length then search edge 3 [4127, 8243] innerCandidates564_8 12370 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan564_9

def innerTail564_7 : List Nat := 7949 :: innerTail564_8
def innerCandidates564_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates564_7_checked : candidates edge 4 4127 7949 innerTail564_8 best.bound = innerCandidates564_7 := by
  unfold candidates
  have row : edge 7949 = SearchData.row7949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan564_7 : scan edge (search edge 3) 4 [4127] 4127 innerTail564_7 best = best := by
  rw [innerTail564_7, scan, if_neg (by decide), innerCandidates564_7_checked]
  change scan edge (search edge 3) 4 [4127] 4127 innerTail564_8
    (if 3 ≤ innerCandidates564_7.length then search edge 3 [4127, 7949] innerCandidates564_7 12076 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan564_8

def innerTail564_6 : List Nat := 7079 :: innerTail564_7
def innerCandidates564_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates564_6_checked : candidates edge 4 4127 7079 innerTail564_7 best.bound = innerCandidates564_6 := by
  unfold candidates
  have row : edge 7079 = SearchData.row7079.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan564_6 : scan edge (search edge 3) 4 [4127] 4127 innerTail564_6 best = best := by
  rw [innerTail564_6, scan, if_neg (by decide), innerCandidates564_6_checked]
  change scan edge (search edge 3) 4 [4127] 4127 innerTail564_7
    (if 3 ≤ innerCandidates564_6.length then search edge 3 [4127, 7079] innerCandidates564_6 11206 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan564_7

def innerTail564_5 : List Nat := 6959 :: innerTail564_6
def innerCandidates564_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates564_5_checked : candidates edge 4 4127 6959 innerTail564_6 best.bound = innerCandidates564_5 := by
  unfold candidates
  have row : edge 6959 = SearchData.row6959.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan564_5 : scan edge (search edge 3) 4 [4127] 4127 innerTail564_5 best = best := by
  rw [innerTail564_5, scan, if_neg (by decide), innerCandidates564_5_checked]
  change scan edge (search edge 3) 4 [4127] 4127 innerTail564_6
    (if 3 ≤ innerCandidates564_5.length then search edge 3 [4127, 6959] innerCandidates564_5 11086 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan564_6

def innerTail564_4 : List Nat := 6737 :: innerTail564_5
def innerCandidates564_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates564_4_checked : candidates edge 4 4127 6737 innerTail564_5 best.bound = innerCandidates564_4 := by
  unfold candidates
  have row : edge 6737 = SearchData.row6737.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan564_4 : scan edge (search edge 3) 4 [4127] 4127 innerTail564_4 best = best := by
  rw [innerTail564_4, scan, if_neg (by decide), innerCandidates564_4_checked]
  change scan edge (search edge 3) 4 [4127] 4127 innerTail564_5
    (if 3 ≤ innerCandidates564_4.length then search edge 3 [4127, 6737] innerCandidates564_4 10864 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan564_5

def innerTail564_3 : List Nat := 6581 :: innerTail564_4
def innerCandidates564_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates564_3_checked : candidates edge 4 4127 6581 innerTail564_4 best.bound = innerCandidates564_3 := by
  unfold candidates
  have row : edge 6581 = SearchData.row6581.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan564_3 : scan edge (search edge 3) 4 [4127] 4127 innerTail564_3 best = best := by
  rw [innerTail564_3, scan, if_neg (by decide), innerCandidates564_3_checked]
  change scan edge (search edge 3) 4 [4127] 4127 innerTail564_4
    (if 3 ≤ innerCandidates564_3.length then search edge 3 [4127, 6581] innerCandidates564_3 10708 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan564_4

def innerTail564_2 : List Nat := 5519 :: innerTail564_3
def innerCandidates564_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates564_2_checked : candidates edge 4 4127 5519 innerTail564_3 best.bound = innerCandidates564_2 := by
  unfold candidates
  have row : edge 5519 = SearchData.row5519.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan564_2 : scan edge (search edge 3) 4 [4127] 4127 innerTail564_2 best = best := by
  rw [innerTail564_2, scan, if_neg (by decide), innerCandidates564_2_checked]
  change scan edge (search edge 3) 4 [4127] 4127 innerTail564_3
    (if 3 ≤ innerCandidates564_2.length then search edge 3 [4127, 5519] innerCandidates564_2 9646 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan564_3

def innerTail564_1 : List Nat := 5309 :: innerTail564_2
def innerCandidates564_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates564_1_checked : candidates edge 4 4127 5309 innerTail564_2 best.bound = innerCandidates564_1 := by
  unfold candidates
  have row : edge 5309 = SearchData.row5309.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan564_1 : scan edge (search edge 3) 4 [4127] 4127 innerTail564_1 best = best := by
  rw [innerTail564_1, scan, if_neg (by decide), innerCandidates564_1_checked]
  change scan edge (search edge 3) 4 [4127] 4127 innerTail564_2
    (if 3 ≤ innerCandidates564_1.length then search edge 3 [4127, 5309] innerCandidates564_1 9436 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan564_2

def innerTail564_0 : List Nat := 4967 :: innerTail564_1
def innerCandidates564_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates564_0_checked : candidates edge 4 4127 4967 innerTail564_1 best.bound = innerCandidates564_0 := by
  unfold candidates
  have row : edge 4967 = SearchData.row4967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan564_0 : scan edge (search edge 3) 4 [4127] 4127 innerTail564_0 best = best := by
  rw [innerTail564_0, scan, if_neg (by decide), innerCandidates564_0_checked]
  change scan edge (search edge 3) 4 [4127] 4127 innerTail564_1
    (if 3 ≤ innerCandidates564_0.length then search edge 3 [4127, 4967] innerCandidates564_0 9094 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan564_1

theorem rootBranch564 : search edge 4 [4127] rootCandidates564 4127 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4127] 4127 innerTail564_0 best = best
  exact innerScan564_0

def rootCandidates563 : List Nat := [4261, 5119, 5521, 5527, 5857, 6073, 6553, 6637, 6661, 6679, 6949, 7369, 7789, 8167]
set_option maxRecDepth 4096 in
theorem rootCandidates563_checked : candidates edge 5 0 4111 tail564 best.bound = rootCandidates563 := by
  unfold candidates
  have row : edge 4111 = SearchData.row4111.contains := by funext q; rfl
  rw [row]
  decide

def innerTail563_14 : List Nat := []
theorem innerScan563_14 : scan edge (search edge 3) 4 [4111] 4111 innerTail563_14 best = best := rfl

def innerTail563_13 : List Nat := 8167 :: innerTail563_14
def innerCandidates563_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates563_13_checked : candidates edge 4 4111 8167 innerTail563_14 best.bound = innerCandidates563_13 := by
  unfold candidates
  have row : edge 8167 = SearchData.row8167.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan563_13 : scan edge (search edge 3) 4 [4111] 4111 innerTail563_13 best = best := by
  rw [innerTail563_13, scan, if_neg (by decide), innerCandidates563_13_checked]
  change scan edge (search edge 3) 4 [4111] 4111 innerTail563_14
    (if 3 ≤ innerCandidates563_13.length then search edge 3 [4111, 8167] innerCandidates563_13 12278 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan563_14

def innerTail563_12 : List Nat := 7789 :: innerTail563_13
def innerCandidates563_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates563_12_checked : candidates edge 4 4111 7789 innerTail563_13 best.bound = innerCandidates563_12 := by
  unfold candidates
  have row : edge 7789 = SearchData.row7789.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan563_12 : scan edge (search edge 3) 4 [4111] 4111 innerTail563_12 best = best := by
  rw [innerTail563_12, scan, if_neg (by decide), innerCandidates563_12_checked]
  change scan edge (search edge 3) 4 [4111] 4111 innerTail563_13
    (if 3 ≤ innerCandidates563_12.length then search edge 3 [4111, 7789] innerCandidates563_12 11900 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan563_13

def innerTail563_11 : List Nat := 7369 :: innerTail563_12
def innerCandidates563_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates563_11_checked : candidates edge 4 4111 7369 innerTail563_12 best.bound = innerCandidates563_11 := by
  unfold candidates
  have row : edge 7369 = SearchData.row7369.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan563_11 : scan edge (search edge 3) 4 [4111] 4111 innerTail563_11 best = best := by
  rw [innerTail563_11, scan, if_neg (by decide), innerCandidates563_11_checked]
  change scan edge (search edge 3) 4 [4111] 4111 innerTail563_12
    (if 3 ≤ innerCandidates563_11.length then search edge 3 [4111, 7369] innerCandidates563_11 11480 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan563_12

def innerTail563_10 : List Nat := 6949 :: innerTail563_11
def innerCandidates563_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates563_10_checked : candidates edge 4 4111 6949 innerTail563_11 best.bound = innerCandidates563_10 := by
  unfold candidates
  have row : edge 6949 = SearchData.row6949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan563_10 : scan edge (search edge 3) 4 [4111] 4111 innerTail563_10 best = best := by
  rw [innerTail563_10, scan, if_neg (by decide), innerCandidates563_10_checked]
  change scan edge (search edge 3) 4 [4111] 4111 innerTail563_11
    (if 3 ≤ innerCandidates563_10.length then search edge 3 [4111, 6949] innerCandidates563_10 11060 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan563_11

def innerTail563_9 : List Nat := 6679 :: innerTail563_10
def innerCandidates563_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates563_9_checked : candidates edge 4 4111 6679 innerTail563_10 best.bound = innerCandidates563_9 := by
  unfold candidates
  have row : edge 6679 = SearchData.row6679.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan563_9 : scan edge (search edge 3) 4 [4111] 4111 innerTail563_9 best = best := by
  rw [innerTail563_9, scan, if_neg (by decide), innerCandidates563_9_checked]
  change scan edge (search edge 3) 4 [4111] 4111 innerTail563_10
    (if 3 ≤ innerCandidates563_9.length then search edge 3 [4111, 6679] innerCandidates563_9 10790 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan563_10

def innerTail563_8 : List Nat := 6661 :: innerTail563_9
def innerCandidates563_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates563_8_checked : candidates edge 4 4111 6661 innerTail563_9 best.bound = innerCandidates563_8 := by
  unfold candidates
  have row : edge 6661 = SearchData.row6661.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan563_8 : scan edge (search edge 3) 4 [4111] 4111 innerTail563_8 best = best := by
  rw [innerTail563_8, scan, if_neg (by decide), innerCandidates563_8_checked]
  change scan edge (search edge 3) 4 [4111] 4111 innerTail563_9
    (if 3 ≤ innerCandidates563_8.length then search edge 3 [4111, 6661] innerCandidates563_8 10772 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan563_9

def innerTail563_7 : List Nat := 6637 :: innerTail563_8
def innerCandidates563_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates563_7_checked : candidates edge 4 4111 6637 innerTail563_8 best.bound = innerCandidates563_7 := by
  unfold candidates
  have row : edge 6637 = SearchData.row6637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan563_7 : scan edge (search edge 3) 4 [4111] 4111 innerTail563_7 best = best := by
  rw [innerTail563_7, scan, if_neg (by decide), innerCandidates563_7_checked]
  change scan edge (search edge 3) 4 [4111] 4111 innerTail563_8
    (if 3 ≤ innerCandidates563_7.length then search edge 3 [4111, 6637] innerCandidates563_7 10748 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan563_8

def innerTail563_6 : List Nat := 6553 :: innerTail563_7
def innerCandidates563_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates563_6_checked : candidates edge 4 4111 6553 innerTail563_7 best.bound = innerCandidates563_6 := by
  unfold candidates
  have row : edge 6553 = SearchData.row6553.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan563_6 : scan edge (search edge 3) 4 [4111] 4111 innerTail563_6 best = best := by
  rw [innerTail563_6, scan, if_neg (by decide), innerCandidates563_6_checked]
  change scan edge (search edge 3) 4 [4111] 4111 innerTail563_7
    (if 3 ≤ innerCandidates563_6.length then search edge 3 [4111, 6553] innerCandidates563_6 10664 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan563_7

def innerTail563_5 : List Nat := 6073 :: innerTail563_6
def innerCandidates563_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates563_5_checked : candidates edge 4 4111 6073 innerTail563_6 best.bound = innerCandidates563_5 := by
  unfold candidates
  have row : edge 6073 = SearchData.row6073.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan563_5 : scan edge (search edge 3) 4 [4111] 4111 innerTail563_5 best = best := by
  rw [innerTail563_5, scan, if_neg (by decide), innerCandidates563_5_checked]
  change scan edge (search edge 3) 4 [4111] 4111 innerTail563_6
    (if 3 ≤ innerCandidates563_5.length then search edge 3 [4111, 6073] innerCandidates563_5 10184 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan563_6

def innerTail563_4 : List Nat := 5857 :: innerTail563_5
def innerCandidates563_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates563_4_checked : candidates edge 4 4111 5857 innerTail563_5 best.bound = innerCandidates563_4 := by
  unfold candidates
  have row : edge 5857 = SearchData.row5857.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan563_4 : scan edge (search edge 3) 4 [4111] 4111 innerTail563_4 best = best := by
  rw [innerTail563_4, scan, if_neg (by decide), innerCandidates563_4_checked]
  change scan edge (search edge 3) 4 [4111] 4111 innerTail563_5
    (if 3 ≤ innerCandidates563_4.length then search edge 3 [4111, 5857] innerCandidates563_4 9968 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan563_5

def innerTail563_3 : List Nat := 5527 :: innerTail563_4
def innerCandidates563_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates563_3_checked : candidates edge 4 4111 5527 innerTail563_4 best.bound = innerCandidates563_3 := by
  unfold candidates
  have row : edge 5527 = SearchData.row5527.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan563_3 : scan edge (search edge 3) 4 [4111] 4111 innerTail563_3 best = best := by
  rw [innerTail563_3, scan, if_neg (by decide), innerCandidates563_3_checked]
  change scan edge (search edge 3) 4 [4111] 4111 innerTail563_4
    (if 3 ≤ innerCandidates563_3.length then search edge 3 [4111, 5527] innerCandidates563_3 9638 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan563_4

def innerTail563_2 : List Nat := 5521 :: innerTail563_3
def innerCandidates563_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates563_2_checked : candidates edge 4 4111 5521 innerTail563_3 best.bound = innerCandidates563_2 := by
  unfold candidates
  have row : edge 5521 = SearchData.row5521.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan563_2 : scan edge (search edge 3) 4 [4111] 4111 innerTail563_2 best = best := by
  rw [innerTail563_2, scan, if_neg (by decide), innerCandidates563_2_checked]
  change scan edge (search edge 3) 4 [4111] 4111 innerTail563_3
    (if 3 ≤ innerCandidates563_2.length then search edge 3 [4111, 5521] innerCandidates563_2 9632 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan563_3

def innerTail563_1 : List Nat := 5119 :: innerTail563_2
def innerCandidates563_1 : List Nat := [6073, 6637]
set_option maxRecDepth 4096 in
theorem innerCandidates563_1_checked : candidates edge 4 4111 5119 innerTail563_2 best.bound = innerCandidates563_1 := by
  unfold candidates
  have row : edge 5119 = SearchData.row5119.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan563_1 : scan edge (search edge 3) 4 [4111] 4111 innerTail563_1 best = best := by
  rw [innerTail563_1, scan, if_neg (by decide), innerCandidates563_1_checked]
  change scan edge (search edge 3) 4 [4111] 4111 innerTail563_2
    (if 3 ≤ innerCandidates563_1.length then search edge 3 [4111, 5119] innerCandidates563_1 9230 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan563_2

def innerTail563_0 : List Nat := 4261 :: innerTail563_1
def innerCandidates563_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates563_0_checked : candidates edge 4 4111 4261 innerTail563_1 best.bound = innerCandidates563_0 := by
  unfold candidates
  have row : edge 4261 = SearchData.row4261.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan563_0 : scan edge (search edge 3) 4 [4111] 4111 innerTail563_0 best = best := by
  rw [innerTail563_0, scan, if_neg (by decide), innerCandidates563_0_checked]
  change scan edge (search edge 3) 4 [4111] 4111 innerTail563_1
    (if 3 ≤ innerCandidates563_0.length then search edge 3 [4111, 4261] innerCandidates563_0 8372 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan563_1

theorem rootBranch563 : search edge 4 [4111] rootCandidates563 4111 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4111] 4111 innerTail563_0 best = best
  exact innerScan563_0

end ProjectEulerSolutions.PrimePairUpper.Search
