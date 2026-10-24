import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates807 : List Nat := [6599, 6653, 6971, 7577, 8363]
set_option maxRecDepth 4096 in
theorem rootCandidates807_checked : candidates edge 5 0 6221 tail808 best.bound = rootCandidates807 := by
  unfold candidates
  have row : edge 6221 = SearchData.row6221.contains := by funext q; rfl
  rw [row]
  decide

def innerTail807_5 : List Nat := []
theorem innerScan807_5 : scan edge (search edge 3) 4 [6221] 6221 innerTail807_5 best = best := rfl

def innerTail807_4 : List Nat := 8363 :: innerTail807_5
def innerCandidates807_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates807_4_checked : candidates edge 4 6221 8363 innerTail807_5 best.bound = innerCandidates807_4 := by
  unfold candidates
  have row : edge 8363 = SearchData.row8363.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan807_4 : scan edge (search edge 3) 4 [6221] 6221 innerTail807_4 best = best := by
  rw [innerTail807_4, scan, if_neg (by decide), innerCandidates807_4_checked]
  change scan edge (search edge 3) 4 [6221] 6221 innerTail807_5
    (if 3 ≤ innerCandidates807_4.length then search edge 3 [6221, 8363] innerCandidates807_4 14584 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan807_5

def innerTail807_3 : List Nat := 7577 :: innerTail807_4
def innerCandidates807_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates807_3_checked : candidates edge 4 6221 7577 innerTail807_4 best.bound = innerCandidates807_3 := by
  unfold candidates
  have row : edge 7577 = SearchData.row7577.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan807_3 : scan edge (search edge 3) 4 [6221] 6221 innerTail807_3 best = best := by
  rw [innerTail807_3, scan, if_neg (by decide), innerCandidates807_3_checked]
  change scan edge (search edge 3) 4 [6221] 6221 innerTail807_4
    (if 3 ≤ innerCandidates807_3.length then search edge 3 [6221, 7577] innerCandidates807_3 13798 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan807_4

def innerTail807_2 : List Nat := 6971 :: innerTail807_3
def innerCandidates807_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates807_2_checked : candidates edge 4 6221 6971 innerTail807_3 best.bound = innerCandidates807_2 := by
  unfold candidates
  have row : edge 6971 = SearchData.row6971.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan807_2 : scan edge (search edge 3) 4 [6221] 6221 innerTail807_2 best = best := by
  rw [innerTail807_2, scan, if_neg (by decide), innerCandidates807_2_checked]
  change scan edge (search edge 3) 4 [6221] 6221 innerTail807_3
    (if 3 ≤ innerCandidates807_2.length then search edge 3 [6221, 6971] innerCandidates807_2 13192 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan807_3

def innerTail807_1 : List Nat := 6653 :: innerTail807_2
def innerCandidates807_1 : List Nat := [8363]
set_option maxRecDepth 4096 in
theorem innerCandidates807_1_checked : candidates edge 4 6221 6653 innerTail807_2 best.bound = innerCandidates807_1 := by
  unfold candidates
  have row : edge 6653 = SearchData.row6653.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan807_1 : scan edge (search edge 3) 4 [6221] 6221 innerTail807_1 best = best := by
  rw [innerTail807_1, scan, if_neg (by decide), innerCandidates807_1_checked]
  change scan edge (search edge 3) 4 [6221] 6221 innerTail807_2
    (if 3 ≤ innerCandidates807_1.length then search edge 3 [6221, 6653] innerCandidates807_1 12874 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan807_2

def innerTail807_0 : List Nat := 6599 :: innerTail807_1
def innerCandidates807_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates807_0_checked : candidates edge 4 6221 6599 innerTail807_1 best.bound = innerCandidates807_0 := by
  unfold candidates
  have row : edge 6599 = SearchData.row6599.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan807_0 : scan edge (search edge 3) 4 [6221] 6221 innerTail807_0 best = best := by
  rw [innerTail807_0, scan, if_neg (by decide), innerCandidates807_0_checked]
  change scan edge (search edge 3) 4 [6221] 6221 innerTail807_1
    (if 3 ≤ innerCandidates807_0.length then search edge 3 [6221, 6599] innerCandidates807_0 12820 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan807_1

theorem rootBranch807 : search edge 4 [6221] rootCandidates807 6221 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6221] 6221 innerTail807_0 best = best
  exact innerScan807_0

def rootCandidates806 : List Nat := [6481, 6679, 7933, 8161, 8329]
set_option maxRecDepth 4096 in
theorem rootCandidates806_checked : candidates edge 5 0 6217 tail807 best.bound = rootCandidates806 := by
  unfold candidates
  have row : edge 6217 = SearchData.row6217.contains := by funext q; rfl
  rw [row]
  decide

def innerTail806_5 : List Nat := []
theorem innerScan806_5 : scan edge (search edge 3) 4 [6217] 6217 innerTail806_5 best = best := rfl

def innerTail806_4 : List Nat := 8329 :: innerTail806_5
def innerCandidates806_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates806_4_checked : candidates edge 4 6217 8329 innerTail806_5 best.bound = innerCandidates806_4 := by
  unfold candidates
  have row : edge 8329 = SearchData.row8329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan806_4 : scan edge (search edge 3) 4 [6217] 6217 innerTail806_4 best = best := by
  rw [innerTail806_4, scan, if_neg (by decide), innerCandidates806_4_checked]
  change scan edge (search edge 3) 4 [6217] 6217 innerTail806_5
    (if 3 ≤ innerCandidates806_4.length then search edge 3 [6217, 8329] innerCandidates806_4 14546 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan806_5

def innerTail806_3 : List Nat := 8161 :: innerTail806_4
def innerCandidates806_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates806_3_checked : candidates edge 4 6217 8161 innerTail806_4 best.bound = innerCandidates806_3 := by
  unfold candidates
  have row : edge 8161 = SearchData.row8161.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan806_3 : scan edge (search edge 3) 4 [6217] 6217 innerTail806_3 best = best := by
  rw [innerTail806_3, scan, if_neg (by decide), innerCandidates806_3_checked]
  change scan edge (search edge 3) 4 [6217] 6217 innerTail806_4
    (if 3 ≤ innerCandidates806_3.length then search edge 3 [6217, 8161] innerCandidates806_3 14378 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan806_4

def innerTail806_2 : List Nat := 7933 :: innerTail806_3
def innerCandidates806_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates806_2_checked : candidates edge 4 6217 7933 innerTail806_3 best.bound = innerCandidates806_2 := by
  unfold candidates
  have row : edge 7933 = SearchData.row7933.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan806_2 : scan edge (search edge 3) 4 [6217] 6217 innerTail806_2 best = best := by
  rw [innerTail806_2, scan, if_neg (by decide), innerCandidates806_2_checked]
  change scan edge (search edge 3) 4 [6217] 6217 innerTail806_3
    (if 3 ≤ innerCandidates806_2.length then search edge 3 [6217, 7933] innerCandidates806_2 14150 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan806_3

def innerTail806_1 : List Nat := 6679 :: innerTail806_2
def innerCandidates806_1 : List Nat := [8161]
set_option maxRecDepth 4096 in
theorem innerCandidates806_1_checked : candidates edge 4 6217 6679 innerTail806_2 best.bound = innerCandidates806_1 := by
  unfold candidates
  have row : edge 6679 = SearchData.row6679.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan806_1 : scan edge (search edge 3) 4 [6217] 6217 innerTail806_1 best = best := by
  rw [innerTail806_1, scan, if_neg (by decide), innerCandidates806_1_checked]
  change scan edge (search edge 3) 4 [6217] 6217 innerTail806_2
    (if 3 ≤ innerCandidates806_1.length then search edge 3 [6217, 6679] innerCandidates806_1 12896 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan806_2

def innerTail806_0 : List Nat := 6481 :: innerTail806_1
def innerCandidates806_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates806_0_checked : candidates edge 4 6217 6481 innerTail806_1 best.bound = innerCandidates806_0 := by
  unfold candidates
  have row : edge 6481 = SearchData.row6481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan806_0 : scan edge (search edge 3) 4 [6217] 6217 innerTail806_0 best = best := by
  rw [innerTail806_0, scan, if_neg (by decide), innerCandidates806_0_checked]
  change scan edge (search edge 3) 4 [6217] 6217 innerTail806_1
    (if 3 ≤ innerCandidates806_0.length then search edge 3 [6217, 6481] innerCandidates806_0 12698 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan806_1

theorem rootBranch806 : search edge 4 [6217] rootCandidates806 6217 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6217] 6217 innerTail806_0 best = best
  exact innerScan806_0

def rootCandidates805 : List Nat := [6277, 6547, 6883, 7753]
set_option maxRecDepth 4096 in
theorem rootCandidates805_checked : candidates edge 5 0 6211 tail806 best.bound = rootCandidates805 := by
  unfold candidates
  have row : edge 6211 = SearchData.row6211.contains := by funext q; rfl
  rw [row]
  decide

def innerTail805_4 : List Nat := []
theorem innerScan805_4 : scan edge (search edge 3) 4 [6211] 6211 innerTail805_4 best = best := rfl

def innerTail805_3 : List Nat := 7753 :: innerTail805_4
def innerCandidates805_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates805_3_checked : candidates edge 4 6211 7753 innerTail805_4 best.bound = innerCandidates805_3 := by
  unfold candidates
  have row : edge 7753 = SearchData.row7753.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan805_3 : scan edge (search edge 3) 4 [6211] 6211 innerTail805_3 best = best := by
  rw [innerTail805_3, scan, if_neg (by decide), innerCandidates805_3_checked]
  change scan edge (search edge 3) 4 [6211] 6211 innerTail805_4
    (if 3 ≤ innerCandidates805_3.length then search edge 3 [6211, 7753] innerCandidates805_3 13964 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan805_4

def innerTail805_2 : List Nat := 6883 :: innerTail805_3
def innerCandidates805_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates805_2_checked : candidates edge 4 6211 6883 innerTail805_3 best.bound = innerCandidates805_2 := by
  unfold candidates
  have row : edge 6883 = SearchData.row6883.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan805_2 : scan edge (search edge 3) 4 [6211] 6211 innerTail805_2 best = best := by
  rw [innerTail805_2, scan, if_neg (by decide), innerCandidates805_2_checked]
  change scan edge (search edge 3) 4 [6211] 6211 innerTail805_3
    (if 3 ≤ innerCandidates805_2.length then search edge 3 [6211, 6883] innerCandidates805_2 13094 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan805_3

def innerTail805_1 : List Nat := 6547 :: innerTail805_2
def innerCandidates805_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates805_1_checked : candidates edge 4 6211 6547 innerTail805_2 best.bound = innerCandidates805_1 := by
  unfold candidates
  have row : edge 6547 = SearchData.row6547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan805_1 : scan edge (search edge 3) 4 [6211] 6211 innerTail805_1 best = best := by
  rw [innerTail805_1, scan, if_neg (by decide), innerCandidates805_1_checked]
  change scan edge (search edge 3) 4 [6211] 6211 innerTail805_2
    (if 3 ≤ innerCandidates805_1.length then search edge 3 [6211, 6547] innerCandidates805_1 12758 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan805_2

def innerTail805_0 : List Nat := 6277 :: innerTail805_1
def innerCandidates805_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates805_0_checked : candidates edge 4 6211 6277 innerTail805_1 best.bound = innerCandidates805_0 := by
  unfold candidates
  have row : edge 6277 = SearchData.row6277.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan805_0 : scan edge (search edge 3) 4 [6211] 6211 innerTail805_0 best = best := by
  rw [innerTail805_0, scan, if_neg (by decide), innerCandidates805_0_checked]
  change scan edge (search edge 3) 4 [6211] 6211 innerTail805_1
    (if 3 ≤ innerCandidates805_0.length then search edge 3 [6211, 6277] innerCandidates805_0 12488 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan805_1

theorem rootBranch805 : search edge 4 [6211] rootCandidates805 6211 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6211] 6211 innerTail805_0 best = best
  exact innerScan805_0

def rootCandidates804 : List Nat := [6323, 6551, 6761, 7043, 7253, 7643, 7823]
set_option maxRecDepth 4096 in
theorem rootCandidates804_checked : candidates edge 5 0 6203 tail805 best.bound = rootCandidates804 := by
  unfold candidates
  have row : edge 6203 = SearchData.row6203.contains := by funext q; rfl
  rw [row]
  decide

def innerTail804_7 : List Nat := []
theorem innerScan804_7 : scan edge (search edge 3) 4 [6203] 6203 innerTail804_7 best = best := rfl

def innerTail804_6 : List Nat := 7823 :: innerTail804_7
def innerCandidates804_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates804_6_checked : candidates edge 4 6203 7823 innerTail804_7 best.bound = innerCandidates804_6 := by
  unfold candidates
  have row : edge 7823 = SearchData.row7823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan804_6 : scan edge (search edge 3) 4 [6203] 6203 innerTail804_6 best = best := by
  rw [innerTail804_6, scan, if_neg (by decide), innerCandidates804_6_checked]
  change scan edge (search edge 3) 4 [6203] 6203 innerTail804_7
    (if 3 ≤ innerCandidates804_6.length then search edge 3 [6203, 7823] innerCandidates804_6 14026 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan804_7

def innerTail804_5 : List Nat := 7643 :: innerTail804_6
def innerCandidates804_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates804_5_checked : candidates edge 4 6203 7643 innerTail804_6 best.bound = innerCandidates804_5 := by
  unfold candidates
  have row : edge 7643 = SearchData.row7643.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan804_5 : scan edge (search edge 3) 4 [6203] 6203 innerTail804_5 best = best := by
  rw [innerTail804_5, scan, if_neg (by decide), innerCandidates804_5_checked]
  change scan edge (search edge 3) 4 [6203] 6203 innerTail804_6
    (if 3 ≤ innerCandidates804_5.length then search edge 3 [6203, 7643] innerCandidates804_5 13846 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan804_6

def innerTail804_4 : List Nat := 7253 :: innerTail804_5
def innerCandidates804_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates804_4_checked : candidates edge 4 6203 7253 innerTail804_5 best.bound = innerCandidates804_4 := by
  unfold candidates
  have row : edge 7253 = SearchData.row7253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan804_4 : scan edge (search edge 3) 4 [6203] 6203 innerTail804_4 best = best := by
  rw [innerTail804_4, scan, if_neg (by decide), innerCandidates804_4_checked]
  change scan edge (search edge 3) 4 [6203] 6203 innerTail804_5
    (if 3 ≤ innerCandidates804_4.length then search edge 3 [6203, 7253] innerCandidates804_4 13456 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan804_5

def innerTail804_3 : List Nat := 7043 :: innerTail804_4
def innerCandidates804_3 : List Nat := [7253]
set_option maxRecDepth 4096 in
theorem innerCandidates804_3_checked : candidates edge 4 6203 7043 innerTail804_4 best.bound = innerCandidates804_3 := by
  unfold candidates
  have row : edge 7043 = SearchData.row7043.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan804_3 : scan edge (search edge 3) 4 [6203] 6203 innerTail804_3 best = best := by
  rw [innerTail804_3, scan, if_neg (by decide), innerCandidates804_3_checked]
  change scan edge (search edge 3) 4 [6203] 6203 innerTail804_4
    (if 3 ≤ innerCandidates804_3.length then search edge 3 [6203, 7043] innerCandidates804_3 13246 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan804_4

def innerTail804_2 : List Nat := 6761 :: innerTail804_3
def innerCandidates804_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates804_2_checked : candidates edge 4 6203 6761 innerTail804_3 best.bound = innerCandidates804_2 := by
  unfold candidates
  have row : edge 6761 = SearchData.row6761.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan804_2 : scan edge (search edge 3) 4 [6203] 6203 innerTail804_2 best = best := by
  rw [innerTail804_2, scan, if_neg (by decide), innerCandidates804_2_checked]
  change scan edge (search edge 3) 4 [6203] 6203 innerTail804_3
    (if 3 ≤ innerCandidates804_2.length then search edge 3 [6203, 6761] innerCandidates804_2 12964 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan804_3

def innerTail804_1 : List Nat := 6551 :: innerTail804_2
def innerCandidates804_1 : List Nat := [7253]
set_option maxRecDepth 4096 in
theorem innerCandidates804_1_checked : candidates edge 4 6203 6551 innerTail804_2 best.bound = innerCandidates804_1 := by
  unfold candidates
  have row : edge 6551 = SearchData.row6551.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan804_1 : scan edge (search edge 3) 4 [6203] 6203 innerTail804_1 best = best := by
  rw [innerTail804_1, scan, if_neg (by decide), innerCandidates804_1_checked]
  change scan edge (search edge 3) 4 [6203] 6203 innerTail804_2
    (if 3 ≤ innerCandidates804_1.length then search edge 3 [6203, 6551] innerCandidates804_1 12754 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan804_2

def innerTail804_0 : List Nat := 6323 :: innerTail804_1
def innerCandidates804_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates804_0_checked : candidates edge 4 6203 6323 innerTail804_1 best.bound = innerCandidates804_0 := by
  unfold candidates
  have row : edge 6323 = SearchData.row6323.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan804_0 : scan edge (search edge 3) 4 [6203] 6203 innerTail804_0 best = best := by
  rw [innerTail804_0, scan, if_neg (by decide), innerCandidates804_0_checked]
  change scan edge (search edge 3) 4 [6203] 6203 innerTail804_1
    (if 3 ≤ innerCandidates804_0.length then search edge 3 [6203, 6323] innerCandidates804_0 12526 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan804_1

theorem rootBranch804 : search edge 4 [6203] rootCandidates804 6203 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6203] 6203 innerTail804_0 best = best
  exact innerScan804_0

def rootCandidates803 : List Nat := [6673, 7219]
set_option maxRecDepth 4096 in
theorem rootCandidates803_checked : candidates edge 5 0 6199 tail804 best.bound = rootCandidates803 := by
  unfold candidates
  have row : edge 6199 = SearchData.row6199.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates802 : List Nat := [6389, 6983, 7151, 7547, 7793]
set_option maxRecDepth 4096 in
theorem rootCandidates802_checked : candidates edge 5 0 6197 tail803 best.bound = rootCandidates802 := by
  unfold candidates
  have row : edge 6197 = SearchData.row6197.contains := by funext q; rfl
  rw [row]
  decide

def innerTail802_5 : List Nat := []
theorem innerScan802_5 : scan edge (search edge 3) 4 [6197] 6197 innerTail802_5 best = best := rfl

def innerTail802_4 : List Nat := 7793 :: innerTail802_5
def innerCandidates802_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates802_4_checked : candidates edge 4 6197 7793 innerTail802_5 best.bound = innerCandidates802_4 := by
  unfold candidates
  have row : edge 7793 = SearchData.row7793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan802_4 : scan edge (search edge 3) 4 [6197] 6197 innerTail802_4 best = best := by
  rw [innerTail802_4, scan, if_neg (by decide), innerCandidates802_4_checked]
  change scan edge (search edge 3) 4 [6197] 6197 innerTail802_5
    (if 3 ≤ innerCandidates802_4.length then search edge 3 [6197, 7793] innerCandidates802_4 13990 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan802_5

def innerTail802_3 : List Nat := 7547 :: innerTail802_4
def innerCandidates802_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates802_3_checked : candidates edge 4 6197 7547 innerTail802_4 best.bound = innerCandidates802_3 := by
  unfold candidates
  have row : edge 7547 = SearchData.row7547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan802_3 : scan edge (search edge 3) 4 [6197] 6197 innerTail802_3 best = best := by
  rw [innerTail802_3, scan, if_neg (by decide), innerCandidates802_3_checked]
  change scan edge (search edge 3) 4 [6197] 6197 innerTail802_4
    (if 3 ≤ innerCandidates802_3.length then search edge 3 [6197, 7547] innerCandidates802_3 13744 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan802_4

def innerTail802_2 : List Nat := 7151 :: innerTail802_3
def innerCandidates802_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates802_2_checked : candidates edge 4 6197 7151 innerTail802_3 best.bound = innerCandidates802_2 := by
  unfold candidates
  have row : edge 7151 = SearchData.row7151.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan802_2 : scan edge (search edge 3) 4 [6197] 6197 innerTail802_2 best = best := by
  rw [innerTail802_2, scan, if_neg (by decide), innerCandidates802_2_checked]
  change scan edge (search edge 3) 4 [6197] 6197 innerTail802_3
    (if 3 ≤ innerCandidates802_2.length then search edge 3 [6197, 7151] innerCandidates802_2 13348 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan802_3

def innerTail802_1 : List Nat := 6983 :: innerTail802_2
def innerCandidates802_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates802_1_checked : candidates edge 4 6197 6983 innerTail802_2 best.bound = innerCandidates802_1 := by
  unfold candidates
  have row : edge 6983 = SearchData.row6983.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan802_1 : scan edge (search edge 3) 4 [6197] 6197 innerTail802_1 best = best := by
  rw [innerTail802_1, scan, if_neg (by decide), innerCandidates802_1_checked]
  change scan edge (search edge 3) 4 [6197] 6197 innerTail802_2
    (if 3 ≤ innerCandidates802_1.length then search edge 3 [6197, 6983] innerCandidates802_1 13180 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan802_2

def innerTail802_0 : List Nat := 6389 :: innerTail802_1
def innerCandidates802_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates802_0_checked : candidates edge 4 6197 6389 innerTail802_1 best.bound = innerCandidates802_0 := by
  unfold candidates
  have row : edge 6389 = SearchData.row6389.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan802_0 : scan edge (search edge 3) 4 [6197] 6197 innerTail802_0 best = best := by
  rw [innerTail802_0, scan, if_neg (by decide), innerCandidates802_0_checked]
  change scan edge (search edge 3) 4 [6197] 6197 innerTail802_1
    (if 3 ≤ innerCandidates802_0.length then search edge 3 [6197, 6389] innerCandidates802_0 12586 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan802_1

theorem rootBranch802 : search edge 4 [6197] rootCandidates802 6197 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6197] 6197 innerTail802_0 best = best
  exact innerScan802_0

def rootCandidates801 : List Nat := [6581, 7283, 7691, 7853, 7901, 8231]
set_option maxRecDepth 4096 in
theorem rootCandidates801_checked : candidates edge 5 0 6173 tail802 best.bound = rootCandidates801 := by
  unfold candidates
  have row : edge 6173 = SearchData.row6173.contains := by funext q; rfl
  rw [row]
  decide

def innerTail801_6 : List Nat := []
theorem innerScan801_6 : scan edge (search edge 3) 4 [6173] 6173 innerTail801_6 best = best := rfl

def innerTail801_5 : List Nat := 8231 :: innerTail801_6
def innerCandidates801_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates801_5_checked : candidates edge 4 6173 8231 innerTail801_6 best.bound = innerCandidates801_5 := by
  unfold candidates
  have row : edge 8231 = SearchData.row8231.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan801_5 : scan edge (search edge 3) 4 [6173] 6173 innerTail801_5 best = best := by
  rw [innerTail801_5, scan, if_neg (by decide), innerCandidates801_5_checked]
  change scan edge (search edge 3) 4 [6173] 6173 innerTail801_6
    (if 3 ≤ innerCandidates801_5.length then search edge 3 [6173, 8231] innerCandidates801_5 14404 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan801_6

def innerTail801_4 : List Nat := 7901 :: innerTail801_5
def innerCandidates801_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates801_4_checked : candidates edge 4 6173 7901 innerTail801_5 best.bound = innerCandidates801_4 := by
  unfold candidates
  have row : edge 7901 = SearchData.row7901.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan801_4 : scan edge (search edge 3) 4 [6173] 6173 innerTail801_4 best = best := by
  rw [innerTail801_4, scan, if_neg (by decide), innerCandidates801_4_checked]
  change scan edge (search edge 3) 4 [6173] 6173 innerTail801_5
    (if 3 ≤ innerCandidates801_4.length then search edge 3 [6173, 7901] innerCandidates801_4 14074 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan801_5

def innerTail801_3 : List Nat := 7853 :: innerTail801_4
def innerCandidates801_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates801_3_checked : candidates edge 4 6173 7853 innerTail801_4 best.bound = innerCandidates801_3 := by
  unfold candidates
  have row : edge 7853 = SearchData.row7853.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan801_3 : scan edge (search edge 3) 4 [6173] 6173 innerTail801_3 best = best := by
  rw [innerTail801_3, scan, if_neg (by decide), innerCandidates801_3_checked]
  change scan edge (search edge 3) 4 [6173] 6173 innerTail801_4
    (if 3 ≤ innerCandidates801_3.length then search edge 3 [6173, 7853] innerCandidates801_3 14026 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan801_4

def innerTail801_2 : List Nat := 7691 :: innerTail801_3
def innerCandidates801_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates801_2_checked : candidates edge 4 6173 7691 innerTail801_3 best.bound = innerCandidates801_2 := by
  unfold candidates
  have row : edge 7691 = SearchData.row7691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan801_2 : scan edge (search edge 3) 4 [6173] 6173 innerTail801_2 best = best := by
  rw [innerTail801_2, scan, if_neg (by decide), innerCandidates801_2_checked]
  change scan edge (search edge 3) 4 [6173] 6173 innerTail801_3
    (if 3 ≤ innerCandidates801_2.length then search edge 3 [6173, 7691] innerCandidates801_2 13864 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan801_3

def innerTail801_1 : List Nat := 7283 :: innerTail801_2
def innerCandidates801_1 : List Nat := [7901]
set_option maxRecDepth 4096 in
theorem innerCandidates801_1_checked : candidates edge 4 6173 7283 innerTail801_2 best.bound = innerCandidates801_1 := by
  unfold candidates
  have row : edge 7283 = SearchData.row7283.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan801_1 : scan edge (search edge 3) 4 [6173] 6173 innerTail801_1 best = best := by
  rw [innerTail801_1, scan, if_neg (by decide), innerCandidates801_1_checked]
  change scan edge (search edge 3) 4 [6173] 6173 innerTail801_2
    (if 3 ≤ innerCandidates801_1.length then search edge 3 [6173, 7283] innerCandidates801_1 13456 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan801_2

def innerTail801_0 : List Nat := 6581 :: innerTail801_1
def innerCandidates801_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates801_0_checked : candidates edge 4 6173 6581 innerTail801_1 best.bound = innerCandidates801_0 := by
  unfold candidates
  have row : edge 6581 = SearchData.row6581.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan801_0 : scan edge (search edge 3) 4 [6173] 6173 innerTail801_0 best = best := by
  rw [innerTail801_0, scan, if_neg (by decide), innerCandidates801_0_checked]
  change scan edge (search edge 3) 4 [6173] 6173 innerTail801_1
    (if 3 ≤ innerCandidates801_0.length then search edge 3 [6173, 6581] innerCandidates801_0 12754 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan801_1

theorem rootBranch801 : search edge 4 [6173] rootCandidates801 6173 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6173] 6173 innerTail801_0 best = best
  exact innerScan801_0

def rootCandidates800 : List Nat := [6451, 6793, 7867, 8167]
set_option maxRecDepth 4096 in
theorem rootCandidates800_checked : candidates edge 5 0 6163 tail801 best.bound = rootCandidates800 := by
  unfold candidates
  have row : edge 6163 = SearchData.row6163.contains := by funext q; rfl
  rw [row]
  decide

def innerTail800_4 : List Nat := []
theorem innerScan800_4 : scan edge (search edge 3) 4 [6163] 6163 innerTail800_4 best = best := rfl

def innerTail800_3 : List Nat := 8167 :: innerTail800_4
def innerCandidates800_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates800_3_checked : candidates edge 4 6163 8167 innerTail800_4 best.bound = innerCandidates800_3 := by
  unfold candidates
  have row : edge 8167 = SearchData.row8167.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan800_3 : scan edge (search edge 3) 4 [6163] 6163 innerTail800_3 best = best := by
  rw [innerTail800_3, scan, if_neg (by decide), innerCandidates800_3_checked]
  change scan edge (search edge 3) 4 [6163] 6163 innerTail800_4
    (if 3 ≤ innerCandidates800_3.length then search edge 3 [6163, 8167] innerCandidates800_3 14330 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan800_4

def innerTail800_2 : List Nat := 7867 :: innerTail800_3
def innerCandidates800_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates800_2_checked : candidates edge 4 6163 7867 innerTail800_3 best.bound = innerCandidates800_2 := by
  unfold candidates
  have row : edge 7867 = SearchData.row7867.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan800_2 : scan edge (search edge 3) 4 [6163] 6163 innerTail800_2 best = best := by
  rw [innerTail800_2, scan, if_neg (by decide), innerCandidates800_2_checked]
  change scan edge (search edge 3) 4 [6163] 6163 innerTail800_3
    (if 3 ≤ innerCandidates800_2.length then search edge 3 [6163, 7867] innerCandidates800_2 14030 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan800_3

def innerTail800_1 : List Nat := 6793 :: innerTail800_2
def innerCandidates800_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates800_1_checked : candidates edge 4 6163 6793 innerTail800_2 best.bound = innerCandidates800_1 := by
  unfold candidates
  have row : edge 6793 = SearchData.row6793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan800_1 : scan edge (search edge 3) 4 [6163] 6163 innerTail800_1 best = best := by
  rw [innerTail800_1, scan, if_neg (by decide), innerCandidates800_1_checked]
  change scan edge (search edge 3) 4 [6163] 6163 innerTail800_2
    (if 3 ≤ innerCandidates800_1.length then search edge 3 [6163, 6793] innerCandidates800_1 12956 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan800_2

def innerTail800_0 : List Nat := 6451 :: innerTail800_1
def innerCandidates800_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates800_0_checked : candidates edge 4 6163 6451 innerTail800_1 best.bound = innerCandidates800_0 := by
  unfold candidates
  have row : edge 6451 = SearchData.row6451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan800_0 : scan edge (search edge 3) 4 [6163] 6163 innerTail800_0 best = best := by
  rw [innerTail800_0, scan, if_neg (by decide), innerCandidates800_0_checked]
  change scan edge (search edge 3) 4 [6163] 6163 innerTail800_1
    (if 3 ≤ innerCandidates800_0.length then search edge 3 [6163, 6451] innerCandidates800_0 12614 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan800_1

theorem rootBranch800 : search edge 4 [6163] rootCandidates800 6163 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6163] 6163 innerTail800_0 best = best
  exact innerScan800_0

end ProjectEulerSolutions.PrimePairUpper.Search
