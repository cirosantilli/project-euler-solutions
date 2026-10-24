import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates667 : List Nat := [5087, 5939, 6131, 6197, 6551, 6911, 7349, 7541, 7703, 8147]
set_option maxRecDepth 4096 in
theorem rootCandidates667_checked : candidates edge 5 0 5003 tail668 best.bound = rootCandidates667 := by
  unfold candidates
  have row : edge 5003 = SearchData.row5003.contains := by funext q; rfl
  rw [row]
  decide

def innerTail667_10 : List Nat := []
theorem innerScan667_10 : scan edge (search edge 3) 4 [5003] 5003 innerTail667_10 best = best := rfl

def innerTail667_9 : List Nat := 8147 :: innerTail667_10
def innerCandidates667_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates667_9_checked : candidates edge 4 5003 8147 innerTail667_10 best.bound = innerCandidates667_9 := by
  unfold candidates
  have row : edge 8147 = SearchData.row8147.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan667_9 : scan edge (search edge 3) 4 [5003] 5003 innerTail667_9 best = best := by
  rw [innerTail667_9, scan, if_neg (by decide), innerCandidates667_9_checked]
  change scan edge (search edge 3) 4 [5003] 5003 innerTail667_10
    (if 3 ≤ innerCandidates667_9.length then search edge 3 [5003, 8147] innerCandidates667_9 13150 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan667_10

def innerTail667_8 : List Nat := 7703 :: innerTail667_9
def innerCandidates667_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates667_8_checked : candidates edge 4 5003 7703 innerTail667_9 best.bound = innerCandidates667_8 := by
  unfold candidates
  have row : edge 7703 = SearchData.row7703.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan667_8 : scan edge (search edge 3) 4 [5003] 5003 innerTail667_8 best = best := by
  rw [innerTail667_8, scan, if_neg (by decide), innerCandidates667_8_checked]
  change scan edge (search edge 3) 4 [5003] 5003 innerTail667_9
    (if 3 ≤ innerCandidates667_8.length then search edge 3 [5003, 7703] innerCandidates667_8 12706 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan667_9

def innerTail667_7 : List Nat := 7541 :: innerTail667_8
def innerCandidates667_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates667_7_checked : candidates edge 4 5003 7541 innerTail667_8 best.bound = innerCandidates667_7 := by
  unfold candidates
  have row : edge 7541 = SearchData.row7541.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan667_7 : scan edge (search edge 3) 4 [5003] 5003 innerTail667_7 best = best := by
  rw [innerTail667_7, scan, if_neg (by decide), innerCandidates667_7_checked]
  change scan edge (search edge 3) 4 [5003] 5003 innerTail667_8
    (if 3 ≤ innerCandidates667_7.length then search edge 3 [5003, 7541] innerCandidates667_7 12544 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan667_8

def innerTail667_6 : List Nat := 7349 :: innerTail667_7
def innerCandidates667_6 : List Nat := [8147]
set_option maxRecDepth 4096 in
theorem innerCandidates667_6_checked : candidates edge 4 5003 7349 innerTail667_7 best.bound = innerCandidates667_6 := by
  unfold candidates
  have row : edge 7349 = SearchData.row7349.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan667_6 : scan edge (search edge 3) 4 [5003] 5003 innerTail667_6 best = best := by
  rw [innerTail667_6, scan, if_neg (by decide), innerCandidates667_6_checked]
  change scan edge (search edge 3) 4 [5003] 5003 innerTail667_7
    (if 3 ≤ innerCandidates667_6.length then search edge 3 [5003, 7349] innerCandidates667_6 12352 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan667_7

def innerTail667_5 : List Nat := 6911 :: innerTail667_6
def innerCandidates667_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates667_5_checked : candidates edge 4 5003 6911 innerTail667_6 best.bound = innerCandidates667_5 := by
  unfold candidates
  have row : edge 6911 = SearchData.row6911.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan667_5 : scan edge (search edge 3) 4 [5003] 5003 innerTail667_5 best = best := by
  rw [innerTail667_5, scan, if_neg (by decide), innerCandidates667_5_checked]
  change scan edge (search edge 3) 4 [5003] 5003 innerTail667_6
    (if 3 ≤ innerCandidates667_5.length then search edge 3 [5003, 6911] innerCandidates667_5 11914 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan667_6

def innerTail667_4 : List Nat := 6551 :: innerTail667_5
def innerCandidates667_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates667_4_checked : candidates edge 4 5003 6551 innerTail667_5 best.bound = innerCandidates667_4 := by
  unfold candidates
  have row : edge 6551 = SearchData.row6551.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan667_4 : scan edge (search edge 3) 4 [5003] 5003 innerTail667_4 best = best := by
  rw [innerTail667_4, scan, if_neg (by decide), innerCandidates667_4_checked]
  change scan edge (search edge 3) 4 [5003] 5003 innerTail667_5
    (if 3 ≤ innerCandidates667_4.length then search edge 3 [5003, 6551] innerCandidates667_4 11554 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan667_5

def innerTail667_3 : List Nat := 6197 :: innerTail667_4
def innerCandidates667_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates667_3_checked : candidates edge 4 5003 6197 innerTail667_4 best.bound = innerCandidates667_3 := by
  unfold candidates
  have row : edge 6197 = SearchData.row6197.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan667_3 : scan edge (search edge 3) 4 [5003] 5003 innerTail667_3 best = best := by
  rw [innerTail667_3, scan, if_neg (by decide), innerCandidates667_3_checked]
  change scan edge (search edge 3) 4 [5003] 5003 innerTail667_4
    (if 3 ≤ innerCandidates667_3.length then search edge 3 [5003, 6197] innerCandidates667_3 11200 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan667_4

def innerTail667_2 : List Nat := 6131 :: innerTail667_3
def innerCandidates667_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates667_2_checked : candidates edge 4 5003 6131 innerTail667_3 best.bound = innerCandidates667_2 := by
  unfold candidates
  have row : edge 6131 = SearchData.row6131.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan667_2 : scan edge (search edge 3) 4 [5003] 5003 innerTail667_2 best = best := by
  rw [innerTail667_2, scan, if_neg (by decide), innerCandidates667_2_checked]
  change scan edge (search edge 3) 4 [5003] 5003 innerTail667_3
    (if 3 ≤ innerCandidates667_2.length then search edge 3 [5003, 6131] innerCandidates667_2 11134 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan667_3

def innerTail667_1 : List Nat := 5939 :: innerTail667_2
def innerCandidates667_1 : List Nat := [6551]
set_option maxRecDepth 4096 in
theorem innerCandidates667_1_checked : candidates edge 4 5003 5939 innerTail667_2 best.bound = innerCandidates667_1 := by
  unfold candidates
  have row : edge 5939 = SearchData.row5939.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan667_1 : scan edge (search edge 3) 4 [5003] 5003 innerTail667_1 best = best := by
  rw [innerTail667_1, scan, if_neg (by decide), innerCandidates667_1_checked]
  change scan edge (search edge 3) 4 [5003] 5003 innerTail667_2
    (if 3 ≤ innerCandidates667_1.length then search edge 3 [5003, 5939] innerCandidates667_1 10942 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan667_2

def innerTail667_0 : List Nat := 5087 :: innerTail667_1
def innerCandidates667_0 : List Nat := [8147]
set_option maxRecDepth 4096 in
theorem innerCandidates667_0_checked : candidates edge 4 5003 5087 innerTail667_1 best.bound = innerCandidates667_0 := by
  unfold candidates
  have row : edge 5087 = SearchData.row5087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan667_0 : scan edge (search edge 3) 4 [5003] 5003 innerTail667_0 best = best := by
  rw [innerTail667_0, scan, if_neg (by decide), innerCandidates667_0_checked]
  change scan edge (search edge 3) 4 [5003] 5003 innerTail667_1
    (if 3 ≤ innerCandidates667_0.length then search edge 3 [5003, 5087] innerCandidates667_0 10090 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan667_1

theorem rootBranch667 : search edge 4 [5003] rootCandidates667 5003 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5003] 5003 innerTail667_0 best = best
  exact innerScan667_0

def rootCandidates666 : List Nat := [5107, 5119, 5791, 6367, 6763, 6829, 7069, 7309]
set_option maxRecDepth 4096 in
theorem rootCandidates666_checked : candidates edge 5 0 4999 tail667 best.bound = rootCandidates666 := by
  unfold candidates
  have row : edge 4999 = SearchData.row4999.contains := by funext q; rfl
  rw [row]
  decide

def innerTail666_8 : List Nat := []
theorem innerScan666_8 : scan edge (search edge 3) 4 [4999] 4999 innerTail666_8 best = best := rfl

def innerTail666_7 : List Nat := 7309 :: innerTail666_8
def innerCandidates666_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates666_7_checked : candidates edge 4 4999 7309 innerTail666_8 best.bound = innerCandidates666_7 := by
  unfold candidates
  have row : edge 7309 = SearchData.row7309.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan666_7 : scan edge (search edge 3) 4 [4999] 4999 innerTail666_7 best = best := by
  rw [innerTail666_7, scan, if_neg (by decide), innerCandidates666_7_checked]
  change scan edge (search edge 3) 4 [4999] 4999 innerTail666_8
    (if 3 ≤ innerCandidates666_7.length then search edge 3 [4999, 7309] innerCandidates666_7 12308 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan666_8

def innerTail666_6 : List Nat := 7069 :: innerTail666_7
def innerCandidates666_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates666_6_checked : candidates edge 4 4999 7069 innerTail666_7 best.bound = innerCandidates666_6 := by
  unfold candidates
  have row : edge 7069 = SearchData.row7069.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan666_6 : scan edge (search edge 3) 4 [4999] 4999 innerTail666_6 best = best := by
  rw [innerTail666_6, scan, if_neg (by decide), innerCandidates666_6_checked]
  change scan edge (search edge 3) 4 [4999] 4999 innerTail666_7
    (if 3 ≤ innerCandidates666_6.length then search edge 3 [4999, 7069] innerCandidates666_6 12068 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan666_7

def innerTail666_5 : List Nat := 6829 :: innerTail666_6
def innerCandidates666_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates666_5_checked : candidates edge 4 4999 6829 innerTail666_6 best.bound = innerCandidates666_5 := by
  unfold candidates
  have row : edge 6829 = SearchData.row6829.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan666_5 : scan edge (search edge 3) 4 [4999] 4999 innerTail666_5 best = best := by
  rw [innerTail666_5, scan, if_neg (by decide), innerCandidates666_5_checked]
  change scan edge (search edge 3) 4 [4999] 4999 innerTail666_6
    (if 3 ≤ innerCandidates666_5.length then search edge 3 [4999, 6829] innerCandidates666_5 11828 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan666_6

def innerTail666_4 : List Nat := 6763 :: innerTail666_5
def innerCandidates666_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates666_4_checked : candidates edge 4 4999 6763 innerTail666_5 best.bound = innerCandidates666_4 := by
  unfold candidates
  have row : edge 6763 = SearchData.row6763.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan666_4 : scan edge (search edge 3) 4 [4999] 4999 innerTail666_4 best = best := by
  rw [innerTail666_4, scan, if_neg (by decide), innerCandidates666_4_checked]
  change scan edge (search edge 3) 4 [4999] 4999 innerTail666_5
    (if 3 ≤ innerCandidates666_4.length then search edge 3 [4999, 6763] innerCandidates666_4 11762 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan666_5

def innerTail666_3 : List Nat := 6367 :: innerTail666_4
def innerCandidates666_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates666_3_checked : candidates edge 4 4999 6367 innerTail666_4 best.bound = innerCandidates666_3 := by
  unfold candidates
  have row : edge 6367 = SearchData.row6367.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan666_3 : scan edge (search edge 3) 4 [4999] 4999 innerTail666_3 best = best := by
  rw [innerTail666_3, scan, if_neg (by decide), innerCandidates666_3_checked]
  change scan edge (search edge 3) 4 [4999] 4999 innerTail666_4
    (if 3 ≤ innerCandidates666_3.length then search edge 3 [4999, 6367] innerCandidates666_3 11366 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan666_4

def innerTail666_2 : List Nat := 5791 :: innerTail666_3
def innerCandidates666_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates666_2_checked : candidates edge 4 4999 5791 innerTail666_3 best.bound = innerCandidates666_2 := by
  unfold candidates
  have row : edge 5791 = SearchData.row5791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan666_2 : scan edge (search edge 3) 4 [4999] 4999 innerTail666_2 best = best := by
  rw [innerTail666_2, scan, if_neg (by decide), innerCandidates666_2_checked]
  change scan edge (search edge 3) 4 [4999] 4999 innerTail666_3
    (if 3 ≤ innerCandidates666_2.length then search edge 3 [4999, 5791] innerCandidates666_2 10790 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan666_3

def innerTail666_1 : List Nat := 5119 :: innerTail666_2
def innerCandidates666_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates666_1_checked : candidates edge 4 4999 5119 innerTail666_2 best.bound = innerCandidates666_1 := by
  unfold candidates
  have row : edge 5119 = SearchData.row5119.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan666_1 : scan edge (search edge 3) 4 [4999] 4999 innerTail666_1 best = best := by
  rw [innerTail666_1, scan, if_neg (by decide), innerCandidates666_1_checked]
  change scan edge (search edge 3) 4 [4999] 4999 innerTail666_2
    (if 3 ≤ innerCandidates666_1.length then search edge 3 [4999, 5119] innerCandidates666_1 10118 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan666_2

def innerTail666_0 : List Nat := 5107 :: innerTail666_1
def innerCandidates666_0 : List Nat := [6763]
set_option maxRecDepth 4096 in
theorem innerCandidates666_0_checked : candidates edge 4 4999 5107 innerTail666_1 best.bound = innerCandidates666_0 := by
  unfold candidates
  have row : edge 5107 = SearchData.row5107.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan666_0 : scan edge (search edge 3) 4 [4999] 4999 innerTail666_0 best = best := by
  rw [innerTail666_0, scan, if_neg (by decide), innerCandidates666_0_checked]
  change scan edge (search edge 3) 4 [4999] 4999 innerTail666_1
    (if 3 ≤ innerCandidates666_0.length then search edge 3 [4999, 5107] innerCandidates666_0 10106 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan666_1

theorem rootBranch666 : search edge 4 [4999] rootCandidates666 4999 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4999] 4999 innerTail666_0 best = best
  exact innerScan666_0

def rootCandidates665 : List Nat := [4999, 5527, 5779, 6571, 7561, 8269]
set_option maxRecDepth 4096 in
theorem rootCandidates665_checked : candidates edge 5 0 4993 tail666 best.bound = rootCandidates665 := by
  unfold candidates
  have row : edge 4993 = SearchData.row4993.contains := by funext q; rfl
  rw [row]
  decide

def innerTail665_6 : List Nat := []
theorem innerScan665_6 : scan edge (search edge 3) 4 [4993] 4993 innerTail665_6 best = best := rfl

def innerTail665_5 : List Nat := 8269 :: innerTail665_6
def innerCandidates665_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates665_5_checked : candidates edge 4 4993 8269 innerTail665_6 best.bound = innerCandidates665_5 := by
  unfold candidates
  have row : edge 8269 = SearchData.row8269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan665_5 : scan edge (search edge 3) 4 [4993] 4993 innerTail665_5 best = best := by
  rw [innerTail665_5, scan, if_neg (by decide), innerCandidates665_5_checked]
  change scan edge (search edge 3) 4 [4993] 4993 innerTail665_6
    (if 3 ≤ innerCandidates665_5.length then search edge 3 [4993, 8269] innerCandidates665_5 13262 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan665_6

def innerTail665_4 : List Nat := 7561 :: innerTail665_5
def innerCandidates665_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates665_4_checked : candidates edge 4 4993 7561 innerTail665_5 best.bound = innerCandidates665_4 := by
  unfold candidates
  have row : edge 7561 = SearchData.row7561.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan665_4 : scan edge (search edge 3) 4 [4993] 4993 innerTail665_4 best = best := by
  rw [innerTail665_4, scan, if_neg (by decide), innerCandidates665_4_checked]
  change scan edge (search edge 3) 4 [4993] 4993 innerTail665_5
    (if 3 ≤ innerCandidates665_4.length then search edge 3 [4993, 7561] innerCandidates665_4 12554 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan665_5

def innerTail665_3 : List Nat := 6571 :: innerTail665_4
def innerCandidates665_3 : List Nat := [8269]
set_option maxRecDepth 4096 in
theorem innerCandidates665_3_checked : candidates edge 4 4993 6571 innerTail665_4 best.bound = innerCandidates665_3 := by
  unfold candidates
  have row : edge 6571 = SearchData.row6571.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan665_3 : scan edge (search edge 3) 4 [4993] 4993 innerTail665_3 best = best := by
  rw [innerTail665_3, scan, if_neg (by decide), innerCandidates665_3_checked]
  change scan edge (search edge 3) 4 [4993] 4993 innerTail665_4
    (if 3 ≤ innerCandidates665_3.length then search edge 3 [4993, 6571] innerCandidates665_3 11564 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan665_4

def innerTail665_2 : List Nat := 5779 :: innerTail665_3
def innerCandidates665_2 : List Nat := [8269]
set_option maxRecDepth 4096 in
theorem innerCandidates665_2_checked : candidates edge 4 4993 5779 innerTail665_3 best.bound = innerCandidates665_2 := by
  unfold candidates
  have row : edge 5779 = SearchData.row5779.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan665_2 : scan edge (search edge 3) 4 [4993] 4993 innerTail665_2 best = best := by
  rw [innerTail665_2, scan, if_neg (by decide), innerCandidates665_2_checked]
  change scan edge (search edge 3) 4 [4993] 4993 innerTail665_3
    (if 3 ≤ innerCandidates665_2.length then search edge 3 [4993, 5779] innerCandidates665_2 10772 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan665_3

def innerTail665_1 : List Nat := 5527 :: innerTail665_2
def innerCandidates665_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates665_1_checked : candidates edge 4 4993 5527 innerTail665_2 best.bound = innerCandidates665_1 := by
  unfold candidates
  have row : edge 5527 = SearchData.row5527.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan665_1 : scan edge (search edge 3) 4 [4993] 4993 innerTail665_1 best = best := by
  rw [innerTail665_1, scan, if_neg (by decide), innerCandidates665_1_checked]
  change scan edge (search edge 3) 4 [4993] 4993 innerTail665_2
    (if 3 ≤ innerCandidates665_1.length then search edge 3 [4993, 5527] innerCandidates665_1 10520 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan665_2

def innerTail665_0 : List Nat := 4999 :: innerTail665_1
def innerCandidates665_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates665_0_checked : candidates edge 4 4993 4999 innerTail665_1 best.bound = innerCandidates665_0 := by
  unfold candidates
  have row : edge 4999 = SearchData.row4999.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan665_0 : scan edge (search edge 3) 4 [4993] 4993 innerTail665_0 best = best := by
  rw [innerTail665_0, scan, if_neg (by decide), innerCandidates665_0_checked]
  change scan edge (search edge 3) 4 [4993] 4993 innerTail665_1
    (if 3 ≤ innerCandidates665_0.length then search edge 3 [4993, 4999] innerCandidates665_0 9992 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan665_1

theorem rootBranch665 : search edge 4 [4993] rootCandidates665 4993 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4993] 4993 innerTail665_0 best = best
  exact innerScan665_0

def rootCandidates664 : List Nat := [5197, 6991]
set_option maxRecDepth 4096 in
theorem rootCandidates664_checked : candidates edge 5 0 4987 tail665 best.bound = rootCandidates664 := by
  unfold candidates
  have row : edge 4987 = SearchData.row4987.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates663 : List Nat := [5303, 5717, 6221, 6389, 6569, 7829, 7901]
set_option maxRecDepth 4096 in
theorem rootCandidates663_checked : candidates edge 5 0 4973 tail664 best.bound = rootCandidates663 := by
  unfold candidates
  have row : edge 4973 = SearchData.row4973.contains := by funext q; rfl
  rw [row]
  decide

def innerTail663_7 : List Nat := []
theorem innerScan663_7 : scan edge (search edge 3) 4 [4973] 4973 innerTail663_7 best = best := rfl

def innerTail663_6 : List Nat := 7901 :: innerTail663_7
def innerCandidates663_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates663_6_checked : candidates edge 4 4973 7901 innerTail663_7 best.bound = innerCandidates663_6 := by
  unfold candidates
  have row : edge 7901 = SearchData.row7901.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan663_6 : scan edge (search edge 3) 4 [4973] 4973 innerTail663_6 best = best := by
  rw [innerTail663_6, scan, if_neg (by decide), innerCandidates663_6_checked]
  change scan edge (search edge 3) 4 [4973] 4973 innerTail663_7
    (if 3 ≤ innerCandidates663_6.length then search edge 3 [4973, 7901] innerCandidates663_6 12874 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan663_7

def innerTail663_5 : List Nat := 7829 :: innerTail663_6
def innerCandidates663_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates663_5_checked : candidates edge 4 4973 7829 innerTail663_6 best.bound = innerCandidates663_5 := by
  unfold candidates
  have row : edge 7829 = SearchData.row7829.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan663_5 : scan edge (search edge 3) 4 [4973] 4973 innerTail663_5 best = best := by
  rw [innerTail663_5, scan, if_neg (by decide), innerCandidates663_5_checked]
  change scan edge (search edge 3) 4 [4973] 4973 innerTail663_6
    (if 3 ≤ innerCandidates663_5.length then search edge 3 [4973, 7829] innerCandidates663_5 12802 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan663_6

def innerTail663_4 : List Nat := 6569 :: innerTail663_5
def innerCandidates663_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates663_4_checked : candidates edge 4 4973 6569 innerTail663_5 best.bound = innerCandidates663_4 := by
  unfold candidates
  have row : edge 6569 = SearchData.row6569.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan663_4 : scan edge (search edge 3) 4 [4973] 4973 innerTail663_4 best = best := by
  rw [innerTail663_4, scan, if_neg (by decide), innerCandidates663_4_checked]
  change scan edge (search edge 3) 4 [4973] 4973 innerTail663_5
    (if 3 ≤ innerCandidates663_4.length then search edge 3 [4973, 6569] innerCandidates663_4 11542 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan663_5

def innerTail663_3 : List Nat := 6389 :: innerTail663_4
def innerCandidates663_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates663_3_checked : candidates edge 4 4973 6389 innerTail663_4 best.bound = innerCandidates663_3 := by
  unfold candidates
  have row : edge 6389 = SearchData.row6389.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan663_3 : scan edge (search edge 3) 4 [4973] 4973 innerTail663_3 best = best := by
  rw [innerTail663_3, scan, if_neg (by decide), innerCandidates663_3_checked]
  change scan edge (search edge 3) 4 [4973] 4973 innerTail663_4
    (if 3 ≤ innerCandidates663_3.length then search edge 3 [4973, 6389] innerCandidates663_3 11362 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan663_4

def innerTail663_2 : List Nat := 6221 :: innerTail663_3
def innerCandidates663_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates663_2_checked : candidates edge 4 4973 6221 innerTail663_3 best.bound = innerCandidates663_2 := by
  unfold candidates
  have row : edge 6221 = SearchData.row6221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan663_2 : scan edge (search edge 3) 4 [4973] 4973 innerTail663_2 best = best := by
  rw [innerTail663_2, scan, if_neg (by decide), innerCandidates663_2_checked]
  change scan edge (search edge 3) 4 [4973] 4973 innerTail663_3
    (if 3 ≤ innerCandidates663_2.length then search edge 3 [4973, 6221] innerCandidates663_2 11194 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan663_3

def innerTail663_1 : List Nat := 5717 :: innerTail663_2
def innerCandidates663_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates663_1_checked : candidates edge 4 4973 5717 innerTail663_2 best.bound = innerCandidates663_1 := by
  unfold candidates
  have row : edge 5717 = SearchData.row5717.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan663_1 : scan edge (search edge 3) 4 [4973] 4973 innerTail663_1 best = best := by
  rw [innerTail663_1, scan, if_neg (by decide), innerCandidates663_1_checked]
  change scan edge (search edge 3) 4 [4973] 4973 innerTail663_2
    (if 3 ≤ innerCandidates663_1.length then search edge 3 [4973, 5717] innerCandidates663_1 10690 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan663_2

def innerTail663_0 : List Nat := 5303 :: innerTail663_1
def innerCandidates663_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates663_0_checked : candidates edge 4 4973 5303 innerTail663_1 best.bound = innerCandidates663_0 := by
  unfold candidates
  have row : edge 5303 = SearchData.row5303.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan663_0 : scan edge (search edge 3) 4 [4973] 4973 innerTail663_0 best = best := by
  rw [innerTail663_0, scan, if_neg (by decide), innerCandidates663_0_checked]
  change scan edge (search edge 3) 4 [4973] 4973 innerTail663_1
    (if 3 ≤ innerCandidates663_0.length then search edge 3 [4973, 5303] innerCandidates663_0 10276 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan663_1

theorem rootBranch663 : search edge 4 [4973] rootCandidates663 4973 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4973] 4973 innerTail663_0 best = best
  exact innerScan663_0

def rootCandidates662 : List Nat := [5827, 5881, 6067, 6133, 6397, 6679, 7321, 7507, 8329]
set_option maxRecDepth 4096 in
theorem rootCandidates662_checked : candidates edge 5 0 4969 tail663 best.bound = rootCandidates662 := by
  unfold candidates
  have row : edge 4969 = SearchData.row4969.contains := by funext q; rfl
  rw [row]
  decide

def innerTail662_9 : List Nat := []
theorem innerScan662_9 : scan edge (search edge 3) 4 [4969] 4969 innerTail662_9 best = best := rfl

def innerTail662_8 : List Nat := 8329 :: innerTail662_9
def innerCandidates662_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates662_8_checked : candidates edge 4 4969 8329 innerTail662_9 best.bound = innerCandidates662_8 := by
  unfold candidates
  have row : edge 8329 = SearchData.row8329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan662_8 : scan edge (search edge 3) 4 [4969] 4969 innerTail662_8 best = best := by
  rw [innerTail662_8, scan, if_neg (by decide), innerCandidates662_8_checked]
  change scan edge (search edge 3) 4 [4969] 4969 innerTail662_9
    (if 3 ≤ innerCandidates662_8.length then search edge 3 [4969, 8329] innerCandidates662_8 13298 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan662_9

def innerTail662_7 : List Nat := 7507 :: innerTail662_8
def innerCandidates662_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates662_7_checked : candidates edge 4 4969 7507 innerTail662_8 best.bound = innerCandidates662_7 := by
  unfold candidates
  have row : edge 7507 = SearchData.row7507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan662_7 : scan edge (search edge 3) 4 [4969] 4969 innerTail662_7 best = best := by
  rw [innerTail662_7, scan, if_neg (by decide), innerCandidates662_7_checked]
  change scan edge (search edge 3) 4 [4969] 4969 innerTail662_8
    (if 3 ≤ innerCandidates662_7.length then search edge 3 [4969, 7507] innerCandidates662_7 12476 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan662_8

def innerTail662_6 : List Nat := 7321 :: innerTail662_7
def innerCandidates662_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates662_6_checked : candidates edge 4 4969 7321 innerTail662_7 best.bound = innerCandidates662_6 := by
  unfold candidates
  have row : edge 7321 = SearchData.row7321.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan662_6 : scan edge (search edge 3) 4 [4969] 4969 innerTail662_6 best = best := by
  rw [innerTail662_6, scan, if_neg (by decide), innerCandidates662_6_checked]
  change scan edge (search edge 3) 4 [4969] 4969 innerTail662_7
    (if 3 ≤ innerCandidates662_6.length then search edge 3 [4969, 7321] innerCandidates662_6 12290 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan662_7

def innerTail662_5 : List Nat := 6679 :: innerTail662_6
def innerCandidates662_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates662_5_checked : candidates edge 4 4969 6679 innerTail662_6 best.bound = innerCandidates662_5 := by
  unfold candidates
  have row : edge 6679 = SearchData.row6679.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan662_5 : scan edge (search edge 3) 4 [4969] 4969 innerTail662_5 best = best := by
  rw [innerTail662_5, scan, if_neg (by decide), innerCandidates662_5_checked]
  change scan edge (search edge 3) 4 [4969] 4969 innerTail662_6
    (if 3 ≤ innerCandidates662_5.length then search edge 3 [4969, 6679] innerCandidates662_5 11648 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan662_6

def innerTail662_4 : List Nat := 6397 :: innerTail662_5
def innerCandidates662_4 : List Nat := [7321]
set_option maxRecDepth 4096 in
theorem innerCandidates662_4_checked : candidates edge 4 4969 6397 innerTail662_5 best.bound = innerCandidates662_4 := by
  unfold candidates
  have row : edge 6397 = SearchData.row6397.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan662_4 : scan edge (search edge 3) 4 [4969] 4969 innerTail662_4 best = best := by
  rw [innerTail662_4, scan, if_neg (by decide), innerCandidates662_4_checked]
  change scan edge (search edge 3) 4 [4969] 4969 innerTail662_5
    (if 3 ≤ innerCandidates662_4.length then search edge 3 [4969, 6397] innerCandidates662_4 11366 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan662_5

def innerTail662_3 : List Nat := 6133 :: innerTail662_4
def innerCandidates662_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates662_3_checked : candidates edge 4 4969 6133 innerTail662_4 best.bound = innerCandidates662_3 := by
  unfold candidates
  have row : edge 6133 = SearchData.row6133.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan662_3 : scan edge (search edge 3) 4 [4969] 4969 innerTail662_3 best = best := by
  rw [innerTail662_3, scan, if_neg (by decide), innerCandidates662_3_checked]
  change scan edge (search edge 3) 4 [4969] 4969 innerTail662_4
    (if 3 ≤ innerCandidates662_3.length then search edge 3 [4969, 6133] innerCandidates662_3 11102 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan662_4

def innerTail662_2 : List Nat := 6067 :: innerTail662_3
def innerCandidates662_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates662_2_checked : candidates edge 4 4969 6067 innerTail662_3 best.bound = innerCandidates662_2 := by
  unfold candidates
  have row : edge 6067 = SearchData.row6067.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan662_2 : scan edge (search edge 3) 4 [4969] 4969 innerTail662_2 best = best := by
  rw [innerTail662_2, scan, if_neg (by decide), innerCandidates662_2_checked]
  change scan edge (search edge 3) 4 [4969] 4969 innerTail662_3
    (if 3 ≤ innerCandidates662_2.length then search edge 3 [4969, 6067] innerCandidates662_2 11036 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan662_3

def innerTail662_1 : List Nat := 5881 :: innerTail662_2
def innerCandidates662_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates662_1_checked : candidates edge 4 4969 5881 innerTail662_2 best.bound = innerCandidates662_1 := by
  unfold candidates
  have row : edge 5881 = SearchData.row5881.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan662_1 : scan edge (search edge 3) 4 [4969] 4969 innerTail662_1 best = best := by
  rw [innerTail662_1, scan, if_neg (by decide), innerCandidates662_1_checked]
  change scan edge (search edge 3) 4 [4969] 4969 innerTail662_2
    (if 3 ≤ innerCandidates662_1.length then search edge 3 [4969, 5881] innerCandidates662_1 10850 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan662_2

def innerTail662_0 : List Nat := 5827 :: innerTail662_1
def innerCandidates662_0 : List Nat := [7507]
set_option maxRecDepth 4096 in
theorem innerCandidates662_0_checked : candidates edge 4 4969 5827 innerTail662_1 best.bound = innerCandidates662_0 := by
  unfold candidates
  have row : edge 5827 = SearchData.row5827.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan662_0 : scan edge (search edge 3) 4 [4969] 4969 innerTail662_0 best = best := by
  rw [innerTail662_0, scan, if_neg (by decide), innerCandidates662_0_checked]
  change scan edge (search edge 3) 4 [4969] 4969 innerTail662_1
    (if 3 ≤ innerCandidates662_0.length then search edge 3 [4969, 5827] innerCandidates662_0 10796 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan662_1

theorem rootBranch662 : search edge 4 [4969] rootCandidates662 4969 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4969] 4969 innerTail662_0 best = best
  exact innerScan662_0

def rootCandidates661 : List Nat := [5387, 5441, 5531, 5573, 5849, 5903, 5939, 6827, 7127, 7547, 7877, 8117, 8219]
set_option maxRecDepth 4096 in
theorem rootCandidates661_checked : candidates edge 5 0 4967 tail662 best.bound = rootCandidates661 := by
  unfold candidates
  have row : edge 4967 = SearchData.row4967.contains := by funext q; rfl
  rw [row]
  decide

def innerTail661_13 : List Nat := []
theorem innerScan661_13 : scan edge (search edge 3) 4 [4967] 4967 innerTail661_13 best = best := rfl

def innerTail661_12 : List Nat := 8219 :: innerTail661_13
def innerCandidates661_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates661_12_checked : candidates edge 4 4967 8219 innerTail661_13 best.bound = innerCandidates661_12 := by
  unfold candidates
  have row : edge 8219 = SearchData.row8219.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan661_12 : scan edge (search edge 3) 4 [4967] 4967 innerTail661_12 best = best := by
  rw [innerTail661_12, scan, if_neg (by decide), innerCandidates661_12_checked]
  change scan edge (search edge 3) 4 [4967] 4967 innerTail661_13
    (if 3 ≤ innerCandidates661_12.length then search edge 3 [4967, 8219] innerCandidates661_12 13186 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan661_13

def innerTail661_11 : List Nat := 8117 :: innerTail661_12
def innerCandidates661_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates661_11_checked : candidates edge 4 4967 8117 innerTail661_12 best.bound = innerCandidates661_11 := by
  unfold candidates
  have row : edge 8117 = SearchData.row8117.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan661_11 : scan edge (search edge 3) 4 [4967] 4967 innerTail661_11 best = best := by
  rw [innerTail661_11, scan, if_neg (by decide), innerCandidates661_11_checked]
  change scan edge (search edge 3) 4 [4967] 4967 innerTail661_12
    (if 3 ≤ innerCandidates661_11.length then search edge 3 [4967, 8117] innerCandidates661_11 13084 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan661_12

def innerTail661_10 : List Nat := 7877 :: innerTail661_11
def innerCandidates661_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates661_10_checked : candidates edge 4 4967 7877 innerTail661_11 best.bound = innerCandidates661_10 := by
  unfold candidates
  have row : edge 7877 = SearchData.row7877.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan661_10 : scan edge (search edge 3) 4 [4967] 4967 innerTail661_10 best = best := by
  rw [innerTail661_10, scan, if_neg (by decide), innerCandidates661_10_checked]
  change scan edge (search edge 3) 4 [4967] 4967 innerTail661_11
    (if 3 ≤ innerCandidates661_10.length then search edge 3 [4967, 7877] innerCandidates661_10 12844 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan661_11

def innerTail661_9 : List Nat := 7547 :: innerTail661_10
def innerCandidates661_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates661_9_checked : candidates edge 4 4967 7547 innerTail661_10 best.bound = innerCandidates661_9 := by
  unfold candidates
  have row : edge 7547 = SearchData.row7547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan661_9 : scan edge (search edge 3) 4 [4967] 4967 innerTail661_9 best = best := by
  rw [innerTail661_9, scan, if_neg (by decide), innerCandidates661_9_checked]
  change scan edge (search edge 3) 4 [4967] 4967 innerTail661_10
    (if 3 ≤ innerCandidates661_9.length then search edge 3 [4967, 7547] innerCandidates661_9 12514 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan661_10

def innerTail661_8 : List Nat := 7127 :: innerTail661_9
def innerCandidates661_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates661_8_checked : candidates edge 4 4967 7127 innerTail661_9 best.bound = innerCandidates661_8 := by
  unfold candidates
  have row : edge 7127 = SearchData.row7127.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan661_8 : scan edge (search edge 3) 4 [4967] 4967 innerTail661_8 best = best := by
  rw [innerTail661_8, scan, if_neg (by decide), innerCandidates661_8_checked]
  change scan edge (search edge 3) 4 [4967] 4967 innerTail661_9
    (if 3 ≤ innerCandidates661_8.length then search edge 3 [4967, 7127] innerCandidates661_8 12094 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan661_9

def innerTail661_7 : List Nat := 6827 :: innerTail661_8
def innerCandidates661_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates661_7_checked : candidates edge 4 4967 6827 innerTail661_8 best.bound = innerCandidates661_7 := by
  unfold candidates
  have row : edge 6827 = SearchData.row6827.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan661_7 : scan edge (search edge 3) 4 [4967] 4967 innerTail661_7 best = best := by
  rw [innerTail661_7, scan, if_neg (by decide), innerCandidates661_7_checked]
  change scan edge (search edge 3) 4 [4967] 4967 innerTail661_8
    (if 3 ≤ innerCandidates661_7.length then search edge 3 [4967, 6827] innerCandidates661_7 11794 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan661_8

def innerTail661_6 : List Nat := 5939 :: innerTail661_7
def innerCandidates661_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates661_6_checked : candidates edge 4 4967 5939 innerTail661_7 best.bound = innerCandidates661_6 := by
  unfold candidates
  have row : edge 5939 = SearchData.row5939.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan661_6 : scan edge (search edge 3) 4 [4967] 4967 innerTail661_6 best = best := by
  rw [innerTail661_6, scan, if_neg (by decide), innerCandidates661_6_checked]
  change scan edge (search edge 3) 4 [4967] 4967 innerTail661_7
    (if 3 ≤ innerCandidates661_6.length then search edge 3 [4967, 5939] innerCandidates661_6 10906 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan661_7

def innerTail661_5 : List Nat := 5903 :: innerTail661_6
def innerCandidates661_5 : List Nat := [6827, 7877]
set_option maxRecDepth 4096 in
theorem innerCandidates661_5_checked : candidates edge 4 4967 5903 innerTail661_6 best.bound = innerCandidates661_5 := by
  unfold candidates
  have row : edge 5903 = SearchData.row5903.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan661_5 : scan edge (search edge 3) 4 [4967] 4967 innerTail661_5 best = best := by
  rw [innerTail661_5, scan, if_neg (by decide), innerCandidates661_5_checked]
  change scan edge (search edge 3) 4 [4967] 4967 innerTail661_6
    (if 3 ≤ innerCandidates661_5.length then search edge 3 [4967, 5903] innerCandidates661_5 10870 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan661_6

def innerTail661_4 : List Nat := 5849 :: innerTail661_5
def innerCandidates661_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates661_4_checked : candidates edge 4 4967 5849 innerTail661_5 best.bound = innerCandidates661_4 := by
  unfold candidates
  have row : edge 5849 = SearchData.row5849.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan661_4 : scan edge (search edge 3) 4 [4967] 4967 innerTail661_4 best = best := by
  rw [innerTail661_4, scan, if_neg (by decide), innerCandidates661_4_checked]
  change scan edge (search edge 3) 4 [4967] 4967 innerTail661_5
    (if 3 ≤ innerCandidates661_4.length then search edge 3 [4967, 5849] innerCandidates661_4 10816 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan661_5

def innerTail661_3 : List Nat := 5573 :: innerTail661_4
def innerCandidates661_3 : List Nat := [8219]
set_option maxRecDepth 4096 in
theorem innerCandidates661_3_checked : candidates edge 4 4967 5573 innerTail661_4 best.bound = innerCandidates661_3 := by
  unfold candidates
  have row : edge 5573 = SearchData.row5573.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan661_3 : scan edge (search edge 3) 4 [4967] 4967 innerTail661_3 best = best := by
  rw [innerTail661_3, scan, if_neg (by decide), innerCandidates661_3_checked]
  change scan edge (search edge 3) 4 [4967] 4967 innerTail661_4
    (if 3 ≤ innerCandidates661_3.length then search edge 3 [4967, 5573] innerCandidates661_3 10540 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan661_4

def innerTail661_2 : List Nat := 5531 :: innerTail661_3
def innerCandidates661_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates661_2_checked : candidates edge 4 4967 5531 innerTail661_3 best.bound = innerCandidates661_2 := by
  unfold candidates
  have row : edge 5531 = SearchData.row5531.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan661_2 : scan edge (search edge 3) 4 [4967] 4967 innerTail661_2 best = best := by
  rw [innerTail661_2, scan, if_neg (by decide), innerCandidates661_2_checked]
  change scan edge (search edge 3) 4 [4967] 4967 innerTail661_3
    (if 3 ≤ innerCandidates661_2.length then search edge 3 [4967, 5531] innerCandidates661_2 10498 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan661_3

def innerTail661_1 : List Nat := 5441 :: innerTail661_2
def innerCandidates661_1 : List Nat := [5849, 7547]
set_option maxRecDepth 4096 in
theorem innerCandidates661_1_checked : candidates edge 4 4967 5441 innerTail661_2 best.bound = innerCandidates661_1 := by
  unfold candidates
  have row : edge 5441 = SearchData.row5441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan661_1 : scan edge (search edge 3) 4 [4967] 4967 innerTail661_1 best = best := by
  rw [innerTail661_1, scan, if_neg (by decide), innerCandidates661_1_checked]
  change scan edge (search edge 3) 4 [4967] 4967 innerTail661_2
    (if 3 ≤ innerCandidates661_1.length then search edge 3 [4967, 5441] innerCandidates661_1 10408 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan661_2

def innerTail661_0 : List Nat := 5387 :: innerTail661_1
def innerCandidates661_0 : List Nat := [7127]
set_option maxRecDepth 4096 in
theorem innerCandidates661_0_checked : candidates edge 4 4967 5387 innerTail661_1 best.bound = innerCandidates661_0 := by
  unfold candidates
  have row : edge 5387 = SearchData.row5387.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan661_0 : scan edge (search edge 3) 4 [4967] 4967 innerTail661_0 best = best := by
  rw [innerTail661_0, scan, if_neg (by decide), innerCandidates661_0_checked]
  change scan edge (search edge 3) 4 [4967] 4967 innerTail661_1
    (if 3 ≤ innerCandidates661_0.length then search edge 3 [4967, 5387] innerCandidates661_0 10354 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan661_1

theorem rootBranch661 : search edge 4 [4967] rootCandidates661 4967 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4967] 4967 innerTail661_0 best = best
  exact innerScan661_0

def rootCandidates660 : List Nat := [5023, 5569, 5623, 5839, 6577, 7603, 8017, 8059, 8233]
set_option maxRecDepth 4096 in
theorem rootCandidates660_checked : candidates edge 5 0 4957 tail661 best.bound = rootCandidates660 := by
  unfold candidates
  have row : edge 4957 = SearchData.row4957.contains := by funext q; rfl
  rw [row]
  decide

def innerTail660_9 : List Nat := []
theorem innerScan660_9 : scan edge (search edge 3) 4 [4957] 4957 innerTail660_9 best = best := rfl

def innerTail660_8 : List Nat := 8233 :: innerTail660_9
def innerCandidates660_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates660_8_checked : candidates edge 4 4957 8233 innerTail660_9 best.bound = innerCandidates660_8 := by
  unfold candidates
  have row : edge 8233 = SearchData.row8233.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan660_8 : scan edge (search edge 3) 4 [4957] 4957 innerTail660_8 best = best := by
  rw [innerTail660_8, scan, if_neg (by decide), innerCandidates660_8_checked]
  change scan edge (search edge 3) 4 [4957] 4957 innerTail660_9
    (if 3 ≤ innerCandidates660_8.length then search edge 3 [4957, 8233] innerCandidates660_8 13190 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan660_9

def innerTail660_7 : List Nat := 8059 :: innerTail660_8
def innerCandidates660_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates660_7_checked : candidates edge 4 4957 8059 innerTail660_8 best.bound = innerCandidates660_7 := by
  unfold candidates
  have row : edge 8059 = SearchData.row8059.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan660_7 : scan edge (search edge 3) 4 [4957] 4957 innerTail660_7 best = best := by
  rw [innerTail660_7, scan, if_neg (by decide), innerCandidates660_7_checked]
  change scan edge (search edge 3) 4 [4957] 4957 innerTail660_8
    (if 3 ≤ innerCandidates660_7.length then search edge 3 [4957, 8059] innerCandidates660_7 13016 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan660_8

def innerTail660_6 : List Nat := 8017 :: innerTail660_7
def innerCandidates660_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates660_6_checked : candidates edge 4 4957 8017 innerTail660_7 best.bound = innerCandidates660_6 := by
  unfold candidates
  have row : edge 8017 = SearchData.row8017.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan660_6 : scan edge (search edge 3) 4 [4957] 4957 innerTail660_6 best = best := by
  rw [innerTail660_6, scan, if_neg (by decide), innerCandidates660_6_checked]
  change scan edge (search edge 3) 4 [4957] 4957 innerTail660_7
    (if 3 ≤ innerCandidates660_6.length then search edge 3 [4957, 8017] innerCandidates660_6 12974 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan660_7

def innerTail660_5 : List Nat := 7603 :: innerTail660_6
def innerCandidates660_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates660_5_checked : candidates edge 4 4957 7603 innerTail660_6 best.bound = innerCandidates660_5 := by
  unfold candidates
  have row : edge 7603 = SearchData.row7603.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan660_5 : scan edge (search edge 3) 4 [4957] 4957 innerTail660_5 best = best := by
  rw [innerTail660_5, scan, if_neg (by decide), innerCandidates660_5_checked]
  change scan edge (search edge 3) 4 [4957] 4957 innerTail660_6
    (if 3 ≤ innerCandidates660_5.length then search edge 3 [4957, 7603] innerCandidates660_5 12560 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan660_6

def innerTail660_4 : List Nat := 6577 :: innerTail660_5
def innerCandidates660_4 : List Nat := [7603]
set_option maxRecDepth 4096 in
theorem innerCandidates660_4_checked : candidates edge 4 4957 6577 innerTail660_5 best.bound = innerCandidates660_4 := by
  unfold candidates
  have row : edge 6577 = SearchData.row6577.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan660_4 : scan edge (search edge 3) 4 [4957] 4957 innerTail660_4 best = best := by
  rw [innerTail660_4, scan, if_neg (by decide), innerCandidates660_4_checked]
  change scan edge (search edge 3) 4 [4957] 4957 innerTail660_5
    (if 3 ≤ innerCandidates660_4.length then search edge 3 [4957, 6577] innerCandidates660_4 11534 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan660_5

def innerTail660_3 : List Nat := 5839 :: innerTail660_4
def innerCandidates660_3 : List Nat := [8233]
set_option maxRecDepth 4096 in
theorem innerCandidates660_3_checked : candidates edge 4 4957 5839 innerTail660_4 best.bound = innerCandidates660_3 := by
  unfold candidates
  have row : edge 5839 = SearchData.row5839.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan660_3 : scan edge (search edge 3) 4 [4957] 4957 innerTail660_3 best = best := by
  rw [innerTail660_3, scan, if_neg (by decide), innerCandidates660_3_checked]
  change scan edge (search edge 3) 4 [4957] 4957 innerTail660_4
    (if 3 ≤ innerCandidates660_3.length then search edge 3 [4957, 5839] innerCandidates660_3 10796 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan660_4

def innerTail660_2 : List Nat := 5623 :: innerTail660_3
def innerCandidates660_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates660_2_checked : candidates edge 4 4957 5623 innerTail660_3 best.bound = innerCandidates660_2 := by
  unfold candidates
  have row : edge 5623 = SearchData.row5623.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan660_2 : scan edge (search edge 3) 4 [4957] 4957 innerTail660_2 best = best := by
  rw [innerTail660_2, scan, if_neg (by decide), innerCandidates660_2_checked]
  change scan edge (search edge 3) 4 [4957] 4957 innerTail660_3
    (if 3 ≤ innerCandidates660_2.length then search edge 3 [4957, 5623] innerCandidates660_2 10580 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan660_3

def innerTail660_1 : List Nat := 5569 :: innerTail660_2
def innerCandidates660_1 : List Nat := [6577]
set_option maxRecDepth 4096 in
theorem innerCandidates660_1_checked : candidates edge 4 4957 5569 innerTail660_2 best.bound = innerCandidates660_1 := by
  unfold candidates
  have row : edge 5569 = SearchData.row5569.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan660_1 : scan edge (search edge 3) 4 [4957] 4957 innerTail660_1 best = best := by
  rw [innerTail660_1, scan, if_neg (by decide), innerCandidates660_1_checked]
  change scan edge (search edge 3) 4 [4957] 4957 innerTail660_2
    (if 3 ≤ innerCandidates660_1.length then search edge 3 [4957, 5569] innerCandidates660_1 10526 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan660_2

def innerTail660_0 : List Nat := 5023 :: innerTail660_1
def innerCandidates660_0 : List Nat := [8059]
set_option maxRecDepth 4096 in
theorem innerCandidates660_0_checked : candidates edge 4 4957 5023 innerTail660_1 best.bound = innerCandidates660_0 := by
  unfold candidates
  have row : edge 5023 = SearchData.row5023.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan660_0 : scan edge (search edge 3) 4 [4957] 4957 innerTail660_0 best = best := by
  rw [innerTail660_0, scan, if_neg (by decide), innerCandidates660_0_checked]
  change scan edge (search edge 3) 4 [4957] 4957 innerTail660_1
    (if 3 ≤ innerCandidates660_0.length then search edge 3 [4957, 5023] innerCandidates660_0 9980 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan660_1

theorem rootBranch660 : search edge 4 [4957] rootCandidates660 4957 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4957] 4957 innerTail660_0 best = best
  exact innerScan660_0

end ProjectEulerSolutions.PrimePairUpper.Search
