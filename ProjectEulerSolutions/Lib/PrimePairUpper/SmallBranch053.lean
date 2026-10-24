import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates632 : List Nat := [5531, 5657, 6131, 6221, 6257, 6329, 6359, 6653, 8081]
set_option maxRecDepth 4096 in
theorem rootCandidates632_checked : candidates edge 5 0 4703 tail633 best.bound = rootCandidates632 := by
  unfold candidates
  have row : edge 4703 = SearchData.row4703.contains := by funext q; rfl
  rw [row]
  decide

def innerTail632_9 : List Nat := []
theorem innerScan632_9 : scan edge (search edge 3) 4 [4703] 4703 innerTail632_9 best = best := rfl

def innerTail632_8 : List Nat := 8081 :: innerTail632_9
def innerCandidates632_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates632_8_checked : candidates edge 4 4703 8081 innerTail632_9 best.bound = innerCandidates632_8 := by
  unfold candidates
  have row : edge 8081 = SearchData.row8081.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan632_8 : scan edge (search edge 3) 4 [4703] 4703 innerTail632_8 best = best := by
  rw [innerTail632_8, scan, if_neg (by decide), innerCandidates632_8_checked]
  change scan edge (search edge 3) 4 [4703] 4703 innerTail632_9
    (if 3 ≤ innerCandidates632_8.length then search edge 3 [4703, 8081] innerCandidates632_8 12784 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan632_9

def innerTail632_7 : List Nat := 6653 :: innerTail632_8
def innerCandidates632_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates632_7_checked : candidates edge 4 4703 6653 innerTail632_8 best.bound = innerCandidates632_7 := by
  unfold candidates
  have row : edge 6653 = SearchData.row6653.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan632_7 : scan edge (search edge 3) 4 [4703] 4703 innerTail632_7 best = best := by
  rw [innerTail632_7, scan, if_neg (by decide), innerCandidates632_7_checked]
  change scan edge (search edge 3) 4 [4703] 4703 innerTail632_8
    (if 3 ≤ innerCandidates632_7.length then search edge 3 [4703, 6653] innerCandidates632_7 11356 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan632_8

def innerTail632_6 : List Nat := 6359 :: innerTail632_7
def innerCandidates632_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates632_6_checked : candidates edge 4 4703 6359 innerTail632_7 best.bound = innerCandidates632_6 := by
  unfold candidates
  have row : edge 6359 = SearchData.row6359.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan632_6 : scan edge (search edge 3) 4 [4703] 4703 innerTail632_6 best = best := by
  rw [innerTail632_6, scan, if_neg (by decide), innerCandidates632_6_checked]
  change scan edge (search edge 3) 4 [4703] 4703 innerTail632_7
    (if 3 ≤ innerCandidates632_6.length then search edge 3 [4703, 6359] innerCandidates632_6 11062 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan632_7

def innerTail632_5 : List Nat := 6329 :: innerTail632_6
def innerCandidates632_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates632_5_checked : candidates edge 4 4703 6329 innerTail632_6 best.bound = innerCandidates632_5 := by
  unfold candidates
  have row : edge 6329 = SearchData.row6329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan632_5 : scan edge (search edge 3) 4 [4703] 4703 innerTail632_5 best = best := by
  rw [innerTail632_5, scan, if_neg (by decide), innerCandidates632_5_checked]
  change scan edge (search edge 3) 4 [4703] 4703 innerTail632_6
    (if 3 ≤ innerCandidates632_5.length then search edge 3 [4703, 6329] innerCandidates632_5 11032 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan632_6

def innerTail632_4 : List Nat := 6257 :: innerTail632_5
def innerCandidates632_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates632_4_checked : candidates edge 4 4703 6257 innerTail632_5 best.bound = innerCandidates632_4 := by
  unfold candidates
  have row : edge 6257 = SearchData.row6257.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan632_4 : scan edge (search edge 3) 4 [4703] 4703 innerTail632_4 best = best := by
  rw [innerTail632_4, scan, if_neg (by decide), innerCandidates632_4_checked]
  change scan edge (search edge 3) 4 [4703] 4703 innerTail632_5
    (if 3 ≤ innerCandidates632_4.length then search edge 3 [4703, 6257] innerCandidates632_4 10960 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan632_5

def innerTail632_3 : List Nat := 6221 :: innerTail632_4
def innerCandidates632_3 : List Nat := [6653]
set_option maxRecDepth 4096 in
theorem innerCandidates632_3_checked : candidates edge 4 4703 6221 innerTail632_4 best.bound = innerCandidates632_3 := by
  unfold candidates
  have row : edge 6221 = SearchData.row6221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan632_3 : scan edge (search edge 3) 4 [4703] 4703 innerTail632_3 best = best := by
  rw [innerTail632_3, scan, if_neg (by decide), innerCandidates632_3_checked]
  change scan edge (search edge 3) 4 [4703] 4703 innerTail632_4
    (if 3 ≤ innerCandidates632_3.length then search edge 3 [4703, 6221] innerCandidates632_3 10924 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan632_4

def innerTail632_2 : List Nat := 6131 :: innerTail632_3
def innerCandidates632_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates632_2_checked : candidates edge 4 4703 6131 innerTail632_3 best.bound = innerCandidates632_2 := by
  unfold candidates
  have row : edge 6131 = SearchData.row6131.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan632_2 : scan edge (search edge 3) 4 [4703] 4703 innerTail632_2 best = best := by
  rw [innerTail632_2, scan, if_neg (by decide), innerCandidates632_2_checked]
  change scan edge (search edge 3) 4 [4703] 4703 innerTail632_3
    (if 3 ≤ innerCandidates632_2.length then search edge 3 [4703, 6131] innerCandidates632_2 10834 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan632_3

def innerTail632_1 : List Nat := 5657 :: innerTail632_2
def innerCandidates632_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates632_1_checked : candidates edge 4 4703 5657 innerTail632_2 best.bound = innerCandidates632_1 := by
  unfold candidates
  have row : edge 5657 = SearchData.row5657.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan632_1 : scan edge (search edge 3) 4 [4703] 4703 innerTail632_1 best = best := by
  rw [innerTail632_1, scan, if_neg (by decide), innerCandidates632_1_checked]
  change scan edge (search edge 3) 4 [4703] 4703 innerTail632_2
    (if 3 ≤ innerCandidates632_1.length then search edge 3 [4703, 5657] innerCandidates632_1 10360 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan632_2

def innerTail632_0 : List Nat := 5531 :: innerTail632_1
def innerCandidates632_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates632_0_checked : candidates edge 4 4703 5531 innerTail632_1 best.bound = innerCandidates632_0 := by
  unfold candidates
  have row : edge 5531 = SearchData.row5531.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan632_0 : scan edge (search edge 3) 4 [4703] 4703 innerTail632_0 best = best := by
  rw [innerTail632_0, scan, if_neg (by decide), innerCandidates632_0_checked]
  change scan edge (search edge 3) 4 [4703] 4703 innerTail632_1
    (if 3 ≤ innerCandidates632_0.length then search edge 3 [4703, 5531] innerCandidates632_0 10234 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan632_1

theorem rootBranch632 : search edge 4 [4703] rootCandidates632 4703 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4703] 4703 innerTail632_0 best = best
  exact innerScan632_0

def rootCandidates631 : List Nat := [4817, 5987, 6521, 6581, 6911, 7457]
set_option maxRecDepth 4096 in
theorem rootCandidates631_checked : candidates edge 5 0 4691 tail632 best.bound = rootCandidates631 := by
  unfold candidates
  have row : edge 4691 = SearchData.row4691.contains := by funext q; rfl
  rw [row]
  decide

def innerTail631_6 : List Nat := []
theorem innerScan631_6 : scan edge (search edge 3) 4 [4691] 4691 innerTail631_6 best = best := rfl

def innerTail631_5 : List Nat := 7457 :: innerTail631_6
def innerCandidates631_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates631_5_checked : candidates edge 4 4691 7457 innerTail631_6 best.bound = innerCandidates631_5 := by
  unfold candidates
  have row : edge 7457 = SearchData.row7457.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan631_5 : scan edge (search edge 3) 4 [4691] 4691 innerTail631_5 best = best := by
  rw [innerTail631_5, scan, if_neg (by decide), innerCandidates631_5_checked]
  change scan edge (search edge 3) 4 [4691] 4691 innerTail631_6
    (if 3 ≤ innerCandidates631_5.length then search edge 3 [4691, 7457] innerCandidates631_5 12148 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan631_6

def innerTail631_4 : List Nat := 6911 :: innerTail631_5
def innerCandidates631_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates631_4_checked : candidates edge 4 4691 6911 innerTail631_5 best.bound = innerCandidates631_4 := by
  unfold candidates
  have row : edge 6911 = SearchData.row6911.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan631_4 : scan edge (search edge 3) 4 [4691] 4691 innerTail631_4 best = best := by
  rw [innerTail631_4, scan, if_neg (by decide), innerCandidates631_4_checked]
  change scan edge (search edge 3) 4 [4691] 4691 innerTail631_5
    (if 3 ≤ innerCandidates631_4.length then search edge 3 [4691, 6911] innerCandidates631_4 11602 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan631_5

def innerTail631_3 : List Nat := 6581 :: innerTail631_4
def innerCandidates631_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates631_3_checked : candidates edge 4 4691 6581 innerTail631_4 best.bound = innerCandidates631_3 := by
  unfold candidates
  have row : edge 6581 = SearchData.row6581.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan631_3 : scan edge (search edge 3) 4 [4691] 4691 innerTail631_3 best = best := by
  rw [innerTail631_3, scan, if_neg (by decide), innerCandidates631_3_checked]
  change scan edge (search edge 3) 4 [4691] 4691 innerTail631_4
    (if 3 ≤ innerCandidates631_3.length then search edge 3 [4691, 6581] innerCandidates631_3 11272 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan631_4

def innerTail631_2 : List Nat := 6521 :: innerTail631_3
def innerCandidates631_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates631_2_checked : candidates edge 4 4691 6521 innerTail631_3 best.bound = innerCandidates631_2 := by
  unfold candidates
  have row : edge 6521 = SearchData.row6521.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan631_2 : scan edge (search edge 3) 4 [4691] 4691 innerTail631_2 best = best := by
  rw [innerTail631_2, scan, if_neg (by decide), innerCandidates631_2_checked]
  change scan edge (search edge 3) 4 [4691] 4691 innerTail631_3
    (if 3 ≤ innerCandidates631_2.length then search edge 3 [4691, 6521] innerCandidates631_2 11212 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan631_3

def innerTail631_1 : List Nat := 5987 :: innerTail631_2
def innerCandidates631_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates631_1_checked : candidates edge 4 4691 5987 innerTail631_2 best.bound = innerCandidates631_1 := by
  unfold candidates
  have row : edge 5987 = SearchData.row5987.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan631_1 : scan edge (search edge 3) 4 [4691] 4691 innerTail631_1 best = best := by
  rw [innerTail631_1, scan, if_neg (by decide), innerCandidates631_1_checked]
  change scan edge (search edge 3) 4 [4691] 4691 innerTail631_2
    (if 3 ≤ innerCandidates631_1.length then search edge 3 [4691, 5987] innerCandidates631_1 10678 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan631_2

def innerTail631_0 : List Nat := 4817 :: innerTail631_1
def innerCandidates631_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates631_0_checked : candidates edge 4 4691 4817 innerTail631_1 best.bound = innerCandidates631_0 := by
  unfold candidates
  have row : edge 4817 = SearchData.row4817.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan631_0 : scan edge (search edge 3) 4 [4691] 4691 innerTail631_0 best = best := by
  rw [innerTail631_0, scan, if_neg (by decide), innerCandidates631_0_checked]
  change scan edge (search edge 3) 4 [4691] 4691 innerTail631_1
    (if 3 ≤ innerCandidates631_0.length then search edge 3 [4691, 4817] innerCandidates631_0 9508 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan631_1

theorem rootBranch631 : search edge 4 [4691] rootCandidates631 4691 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4691] 4691 innerTail631_0 best = best
  exact innerScan631_0

def rootCandidates630 : List Nat := [4889, 5003, 5393, 5717, 5927, 6521, 6569, 6653]
set_option maxRecDepth 4096 in
theorem rootCandidates630_checked : candidates edge 5 0 4679 tail631 best.bound = rootCandidates630 := by
  unfold candidates
  have row : edge 4679 = SearchData.row4679.contains := by funext q; rfl
  rw [row]
  decide

def innerTail630_8 : List Nat := []
theorem innerScan630_8 : scan edge (search edge 3) 4 [4679] 4679 innerTail630_8 best = best := rfl

def innerTail630_7 : List Nat := 6653 :: innerTail630_8
def innerCandidates630_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates630_7_checked : candidates edge 4 4679 6653 innerTail630_8 best.bound = innerCandidates630_7 := by
  unfold candidates
  have row : edge 6653 = SearchData.row6653.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan630_7 : scan edge (search edge 3) 4 [4679] 4679 innerTail630_7 best = best := by
  rw [innerTail630_7, scan, if_neg (by decide), innerCandidates630_7_checked]
  change scan edge (search edge 3) 4 [4679] 4679 innerTail630_8
    (if 3 ≤ innerCandidates630_7.length then search edge 3 [4679, 6653] innerCandidates630_7 11332 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan630_8

def innerTail630_6 : List Nat := 6569 :: innerTail630_7
def innerCandidates630_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates630_6_checked : candidates edge 4 4679 6569 innerTail630_7 best.bound = innerCandidates630_6 := by
  unfold candidates
  have row : edge 6569 = SearchData.row6569.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan630_6 : scan edge (search edge 3) 4 [4679] 4679 innerTail630_6 best = best := by
  rw [innerTail630_6, scan, if_neg (by decide), innerCandidates630_6_checked]
  change scan edge (search edge 3) 4 [4679] 4679 innerTail630_7
    (if 3 ≤ innerCandidates630_6.length then search edge 3 [4679, 6569] innerCandidates630_6 11248 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan630_7

def innerTail630_5 : List Nat := 6521 :: innerTail630_6
def innerCandidates630_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates630_5_checked : candidates edge 4 4679 6521 innerTail630_6 best.bound = innerCandidates630_5 := by
  unfold candidates
  have row : edge 6521 = SearchData.row6521.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan630_5 : scan edge (search edge 3) 4 [4679] 4679 innerTail630_5 best = best := by
  rw [innerTail630_5, scan, if_neg (by decide), innerCandidates630_5_checked]
  change scan edge (search edge 3) 4 [4679] 4679 innerTail630_6
    (if 3 ≤ innerCandidates630_5.length then search edge 3 [4679, 6521] innerCandidates630_5 11200 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan630_6

def innerTail630_4 : List Nat := 5927 :: innerTail630_5
def innerCandidates630_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates630_4_checked : candidates edge 4 4679 5927 innerTail630_5 best.bound = innerCandidates630_4 := by
  unfold candidates
  have row : edge 5927 = SearchData.row5927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan630_4 : scan edge (search edge 3) 4 [4679] 4679 innerTail630_4 best = best := by
  rw [innerTail630_4, scan, if_neg (by decide), innerCandidates630_4_checked]
  change scan edge (search edge 3) 4 [4679] 4679 innerTail630_5
    (if 3 ≤ innerCandidates630_4.length then search edge 3 [4679, 5927] innerCandidates630_4 10606 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan630_5

def innerTail630_3 : List Nat := 5717 :: innerTail630_4
def innerCandidates630_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates630_3_checked : candidates edge 4 4679 5717 innerTail630_4 best.bound = innerCandidates630_3 := by
  unfold candidates
  have row : edge 5717 = SearchData.row5717.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan630_3 : scan edge (search edge 3) 4 [4679] 4679 innerTail630_3 best = best := by
  rw [innerTail630_3, scan, if_neg (by decide), innerCandidates630_3_checked]
  change scan edge (search edge 3) 4 [4679] 4679 innerTail630_4
    (if 3 ≤ innerCandidates630_3.length then search edge 3 [4679, 5717] innerCandidates630_3 10396 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan630_4

def innerTail630_2 : List Nat := 5393 :: innerTail630_3
def innerCandidates630_2 : List Nat := [6521]
set_option maxRecDepth 4096 in
theorem innerCandidates630_2_checked : candidates edge 4 4679 5393 innerTail630_3 best.bound = innerCandidates630_2 := by
  unfold candidates
  have row : edge 5393 = SearchData.row5393.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan630_2 : scan edge (search edge 3) 4 [4679] 4679 innerTail630_2 best = best := by
  rw [innerTail630_2, scan, if_neg (by decide), innerCandidates630_2_checked]
  change scan edge (search edge 3) 4 [4679] 4679 innerTail630_3
    (if 3 ≤ innerCandidates630_2.length then search edge 3 [4679, 5393] innerCandidates630_2 10072 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan630_3

def innerTail630_1 : List Nat := 5003 :: innerTail630_2
def innerCandidates630_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates630_1_checked : candidates edge 4 4679 5003 innerTail630_2 best.bound = innerCandidates630_1 := by
  unfold candidates
  have row : edge 5003 = SearchData.row5003.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan630_1 : scan edge (search edge 3) 4 [4679] 4679 innerTail630_1 best = best := by
  rw [innerTail630_1, scan, if_neg (by decide), innerCandidates630_1_checked]
  change scan edge (search edge 3) 4 [4679] 4679 innerTail630_2
    (if 3 ≤ innerCandidates630_1.length then search edge 3 [4679, 5003] innerCandidates630_1 9682 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan630_2

def innerTail630_0 : List Nat := 4889 :: innerTail630_1
def innerCandidates630_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates630_0_checked : candidates edge 4 4679 4889 innerTail630_1 best.bound = innerCandidates630_0 := by
  unfold candidates
  have row : edge 4889 = SearchData.row4889.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan630_0 : scan edge (search edge 3) 4 [4679] 4679 innerTail630_0 best = best := by
  rw [innerTail630_0, scan, if_neg (by decide), innerCandidates630_0_checked]
  change scan edge (search edge 3) 4 [4679] 4679 innerTail630_1
    (if 3 ≤ innerCandidates630_0.length then search edge 3 [4679, 4889] innerCandidates630_0 9568 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan630_1

theorem rootBranch630 : search edge 4 [4679] rootCandidates630 4679 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4679] 4679 innerTail630_0 best = best
  exact innerScan630_0

def rootCandidates629 : List Nat := [4799, 5309, 5519, 6359, 6737, 6779, 6947, 7451, 8009]
set_option maxRecDepth 4096 in
theorem rootCandidates629_checked : candidates edge 5 0 4673 tail630 best.bound = rootCandidates629 := by
  unfold candidates
  have row : edge 4673 = SearchData.row4673.contains := by funext q; rfl
  rw [row]
  decide

def innerTail629_9 : List Nat := []
theorem innerScan629_9 : scan edge (search edge 3) 4 [4673] 4673 innerTail629_9 best = best := rfl

def innerTail629_8 : List Nat := 8009 :: innerTail629_9
def innerCandidates629_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates629_8_checked : candidates edge 4 4673 8009 innerTail629_9 best.bound = innerCandidates629_8 := by
  unfold candidates
  have row : edge 8009 = SearchData.row8009.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan629_8 : scan edge (search edge 3) 4 [4673] 4673 innerTail629_8 best = best := by
  rw [innerTail629_8, scan, if_neg (by decide), innerCandidates629_8_checked]
  change scan edge (search edge 3) 4 [4673] 4673 innerTail629_9
    (if 3 ≤ innerCandidates629_8.length then search edge 3 [4673, 8009] innerCandidates629_8 12682 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan629_9

def innerTail629_7 : List Nat := 7451 :: innerTail629_8
def innerCandidates629_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates629_7_checked : candidates edge 4 4673 7451 innerTail629_8 best.bound = innerCandidates629_7 := by
  unfold candidates
  have row : edge 7451 = SearchData.row7451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan629_7 : scan edge (search edge 3) 4 [4673] 4673 innerTail629_7 best = best := by
  rw [innerTail629_7, scan, if_neg (by decide), innerCandidates629_7_checked]
  change scan edge (search edge 3) 4 [4673] 4673 innerTail629_8
    (if 3 ≤ innerCandidates629_7.length then search edge 3 [4673, 7451] innerCandidates629_7 12124 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan629_8

def innerTail629_6 : List Nat := 6947 :: innerTail629_7
def innerCandidates629_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates629_6_checked : candidates edge 4 4673 6947 innerTail629_7 best.bound = innerCandidates629_6 := by
  unfold candidates
  have row : edge 6947 = SearchData.row6947.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan629_6 : scan edge (search edge 3) 4 [4673] 4673 innerTail629_6 best = best := by
  rw [innerTail629_6, scan, if_neg (by decide), innerCandidates629_6_checked]
  change scan edge (search edge 3) 4 [4673] 4673 innerTail629_7
    (if 3 ≤ innerCandidates629_6.length then search edge 3 [4673, 6947] innerCandidates629_6 11620 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan629_7

def innerTail629_5 : List Nat := 6779 :: innerTail629_6
def innerCandidates629_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates629_5_checked : candidates edge 4 4673 6779 innerTail629_6 best.bound = innerCandidates629_5 := by
  unfold candidates
  have row : edge 6779 = SearchData.row6779.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan629_5 : scan edge (search edge 3) 4 [4673] 4673 innerTail629_5 best = best := by
  rw [innerTail629_5, scan, if_neg (by decide), innerCandidates629_5_checked]
  change scan edge (search edge 3) 4 [4673] 4673 innerTail629_6
    (if 3 ≤ innerCandidates629_5.length then search edge 3 [4673, 6779] innerCandidates629_5 11452 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan629_6

def innerTail629_4 : List Nat := 6737 :: innerTail629_5
def innerCandidates629_4 : List Nat := [6779, 7451]
set_option maxRecDepth 4096 in
theorem innerCandidates629_4_checked : candidates edge 4 4673 6737 innerTail629_5 best.bound = innerCandidates629_4 := by
  unfold candidates
  have row : edge 6737 = SearchData.row6737.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan629_4 : scan edge (search edge 3) 4 [4673] 4673 innerTail629_4 best = best := by
  rw [innerTail629_4, scan, if_neg (by decide), innerCandidates629_4_checked]
  change scan edge (search edge 3) 4 [4673] 4673 innerTail629_5
    (if 3 ≤ innerCandidates629_4.length then search edge 3 [4673, 6737] innerCandidates629_4 11410 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan629_5

def innerTail629_3 : List Nat := 6359 :: innerTail629_4
def innerCandidates629_3 : List Nat := [6737]
set_option maxRecDepth 4096 in
theorem innerCandidates629_3_checked : candidates edge 4 4673 6359 innerTail629_4 best.bound = innerCandidates629_3 := by
  unfold candidates
  have row : edge 6359 = SearchData.row6359.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan629_3 : scan edge (search edge 3) 4 [4673] 4673 innerTail629_3 best = best := by
  rw [innerTail629_3, scan, if_neg (by decide), innerCandidates629_3_checked]
  change scan edge (search edge 3) 4 [4673] 4673 innerTail629_4
    (if 3 ≤ innerCandidates629_3.length then search edge 3 [4673, 6359] innerCandidates629_3 11032 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan629_4

def innerTail629_2 : List Nat := 5519 :: innerTail629_3
def innerCandidates629_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates629_2_checked : candidates edge 4 4673 5519 innerTail629_3 best.bound = innerCandidates629_2 := by
  unfold candidates
  have row : edge 5519 = SearchData.row5519.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan629_2 : scan edge (search edge 3) 4 [4673] 4673 innerTail629_2 best = best := by
  rw [innerTail629_2, scan, if_neg (by decide), innerCandidates629_2_checked]
  change scan edge (search edge 3) 4 [4673] 4673 innerTail629_3
    (if 3 ≤ innerCandidates629_2.length then search edge 3 [4673, 5519] innerCandidates629_2 10192 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan629_3

def innerTail629_1 : List Nat := 5309 :: innerTail629_2
def innerCandidates629_1 : List Nat := [6359]
set_option maxRecDepth 4096 in
theorem innerCandidates629_1_checked : candidates edge 4 4673 5309 innerTail629_2 best.bound = innerCandidates629_1 := by
  unfold candidates
  have row : edge 5309 = SearchData.row5309.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan629_1 : scan edge (search edge 3) 4 [4673] 4673 innerTail629_1 best = best := by
  rw [innerTail629_1, scan, if_neg (by decide), innerCandidates629_1_checked]
  change scan edge (search edge 3) 4 [4673] 4673 innerTail629_2
    (if 3 ≤ innerCandidates629_1.length then search edge 3 [4673, 5309] innerCandidates629_1 9982 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan629_2

def innerTail629_0 : List Nat := 4799 :: innerTail629_1
def innerCandidates629_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates629_0_checked : candidates edge 4 4673 4799 innerTail629_1 best.bound = innerCandidates629_0 := by
  unfold candidates
  have row : edge 4799 = SearchData.row4799.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan629_0 : scan edge (search edge 3) 4 [4673] 4673 innerTail629_0 best = best := by
  rw [innerTail629_0, scan, if_neg (by decide), innerCandidates629_0_checked]
  change scan edge (search edge 3) 4 [4673] 4673 innerTail629_1
    (if 3 ≤ innerCandidates629_0.length then search edge 3 [4673, 4799] innerCandidates629_0 9472 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan629_1

theorem rootBranch629 : search edge 4 [4673] rootCandidates629 4673 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4673] 4673 innerTail629_0 best = best
  exact innerScan629_0

def rootCandidates628 : List Nat := [4993, 5011, 5839, 6619, 6637, 7603, 7753, 7879, 8017, 8233]
set_option maxRecDepth 4096 in
theorem rootCandidates628_checked : candidates edge 5 0 4663 tail629 best.bound = rootCandidates628 := by
  unfold candidates
  have row : edge 4663 = SearchData.row4663.contains := by funext q; rfl
  rw [row]
  decide

def innerTail628_10 : List Nat := []
theorem innerScan628_10 : scan edge (search edge 3) 4 [4663] 4663 innerTail628_10 best = best := rfl

def innerTail628_9 : List Nat := 8233 :: innerTail628_10
def innerCandidates628_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates628_9_checked : candidates edge 4 4663 8233 innerTail628_10 best.bound = innerCandidates628_9 := by
  unfold candidates
  have row : edge 8233 = SearchData.row8233.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan628_9 : scan edge (search edge 3) 4 [4663] 4663 innerTail628_9 best = best := by
  rw [innerTail628_9, scan, if_neg (by decide), innerCandidates628_9_checked]
  change scan edge (search edge 3) 4 [4663] 4663 innerTail628_10
    (if 3 ≤ innerCandidates628_9.length then search edge 3 [4663, 8233] innerCandidates628_9 12896 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan628_10

def innerTail628_8 : List Nat := 8017 :: innerTail628_9
def innerCandidates628_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates628_8_checked : candidates edge 4 4663 8017 innerTail628_9 best.bound = innerCandidates628_8 := by
  unfold candidates
  have row : edge 8017 = SearchData.row8017.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan628_8 : scan edge (search edge 3) 4 [4663] 4663 innerTail628_8 best = best := by
  rw [innerTail628_8, scan, if_neg (by decide), innerCandidates628_8_checked]
  change scan edge (search edge 3) 4 [4663] 4663 innerTail628_9
    (if 3 ≤ innerCandidates628_8.length then search edge 3 [4663, 8017] innerCandidates628_8 12680 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan628_9

def innerTail628_7 : List Nat := 7879 :: innerTail628_8
def innerCandidates628_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates628_7_checked : candidates edge 4 4663 7879 innerTail628_8 best.bound = innerCandidates628_7 := by
  unfold candidates
  have row : edge 7879 = SearchData.row7879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan628_7 : scan edge (search edge 3) 4 [4663] 4663 innerTail628_7 best = best := by
  rw [innerTail628_7, scan, if_neg (by decide), innerCandidates628_7_checked]
  change scan edge (search edge 3) 4 [4663] 4663 innerTail628_8
    (if 3 ≤ innerCandidates628_7.length then search edge 3 [4663, 7879] innerCandidates628_7 12542 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan628_8

def innerTail628_6 : List Nat := 7753 :: innerTail628_7
def innerCandidates628_6 : List Nat := [8233]
set_option maxRecDepth 4096 in
theorem innerCandidates628_6_checked : candidates edge 4 4663 7753 innerTail628_7 best.bound = innerCandidates628_6 := by
  unfold candidates
  have row : edge 7753 = SearchData.row7753.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan628_6 : scan edge (search edge 3) 4 [4663] 4663 innerTail628_6 best = best := by
  rw [innerTail628_6, scan, if_neg (by decide), innerCandidates628_6_checked]
  change scan edge (search edge 3) 4 [4663] 4663 innerTail628_7
    (if 3 ≤ innerCandidates628_6.length then search edge 3 [4663, 7753] innerCandidates628_6 12416 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan628_7

def innerTail628_5 : List Nat := 7603 :: innerTail628_6
def innerCandidates628_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates628_5_checked : candidates edge 4 4663 7603 innerTail628_6 best.bound = innerCandidates628_5 := by
  unfold candidates
  have row : edge 7603 = SearchData.row7603.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan628_5 : scan edge (search edge 3) 4 [4663] 4663 innerTail628_5 best = best := by
  rw [innerTail628_5, scan, if_neg (by decide), innerCandidates628_5_checked]
  change scan edge (search edge 3) 4 [4663] 4663 innerTail628_6
    (if 3 ≤ innerCandidates628_5.length then search edge 3 [4663, 7603] innerCandidates628_5 12266 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan628_6

def innerTail628_4 : List Nat := 6637 :: innerTail628_5
def innerCandidates628_4 : List Nat := [8233]
set_option maxRecDepth 4096 in
theorem innerCandidates628_4_checked : candidates edge 4 4663 6637 innerTail628_5 best.bound = innerCandidates628_4 := by
  unfold candidates
  have row : edge 6637 = SearchData.row6637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan628_4 : scan edge (search edge 3) 4 [4663] 4663 innerTail628_4 best = best := by
  rw [innerTail628_4, scan, if_neg (by decide), innerCandidates628_4_checked]
  change scan edge (search edge 3) 4 [4663] 4663 innerTail628_5
    (if 3 ≤ innerCandidates628_4.length then search edge 3 [4663, 6637] innerCandidates628_4 11300 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan628_5

def innerTail628_3 : List Nat := 6619 :: innerTail628_4
def innerCandidates628_3 : List Nat := [7603]
set_option maxRecDepth 4096 in
theorem innerCandidates628_3_checked : candidates edge 4 4663 6619 innerTail628_4 best.bound = innerCandidates628_3 := by
  unfold candidates
  have row : edge 6619 = SearchData.row6619.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan628_3 : scan edge (search edge 3) 4 [4663] 4663 innerTail628_3 best = best := by
  rw [innerTail628_3, scan, if_neg (by decide), innerCandidates628_3_checked]
  change scan edge (search edge 3) 4 [4663] 4663 innerTail628_4
    (if 3 ≤ innerCandidates628_3.length then search edge 3 [4663, 6619] innerCandidates628_3 11282 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan628_4

def innerTail628_2 : List Nat := 5839 :: innerTail628_3
def innerCandidates628_2 : List Nat := [7753, 7879, 8233]
set_option maxRecDepth 4096 in
theorem innerCandidates628_2_checked : candidates edge 4 4663 5839 innerTail628_3 best.bound = innerCandidates628_2 := by
  unfold candidates
  have row : edge 5839 = SearchData.row5839.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild628_2 : search edge 3 [4663, 5839] innerCandidates628_2 10502 best = best := by decide

theorem innerScan628_2 : scan edge (search edge 3) 4 [4663] 4663 innerTail628_2 best = best := by
  rw [innerTail628_2, scan, if_neg (by decide), innerCandidates628_2_checked]
  change scan edge (search edge 3) 4 [4663] 4663 innerTail628_3
    (if 3 ≤ innerCandidates628_2.length then search edge 3 [4663, 5839] innerCandidates628_2 10502 best else best) = best
  rw [if_pos (by decide), innerChild628_2]
  exact innerScan628_3

def innerTail628_1 : List Nat := 5011 :: innerTail628_2
def innerCandidates628_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates628_1_checked : candidates edge 4 4663 5011 innerTail628_2 best.bound = innerCandidates628_1 := by
  unfold candidates
  have row : edge 5011 = SearchData.row5011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan628_1 : scan edge (search edge 3) 4 [4663] 4663 innerTail628_1 best = best := by
  rw [innerTail628_1, scan, if_neg (by decide), innerCandidates628_1_checked]
  change scan edge (search edge 3) 4 [4663] 4663 innerTail628_2
    (if 3 ≤ innerCandidates628_1.length then search edge 3 [4663, 5011] innerCandidates628_1 9674 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan628_2

def innerTail628_0 : List Nat := 4993 :: innerTail628_1
def innerCandidates628_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates628_0_checked : candidates edge 4 4663 4993 innerTail628_1 best.bound = innerCandidates628_0 := by
  unfold candidates
  have row : edge 4993 = SearchData.row4993.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan628_0 : scan edge (search edge 3) 4 [4663] 4663 innerTail628_0 best = best := by
  rw [innerTail628_0, scan, if_neg (by decide), innerCandidates628_0_checked]
  change scan edge (search edge 3) 4 [4663] 4663 innerTail628_1
    (if 3 ≤ innerCandidates628_0.length then search edge 3 [4663, 4993] innerCandidates628_0 9656 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan628_1

theorem rootBranch628 : search edge 4 [4663] rootCandidates628 4663 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4663] 4663 innerTail628_0 best = best
  exact innerScan628_0

def rootCandidates627 : List Nat := [4789, 5227, 6217, 7177, 7243, 7753, 7933, 8209]
set_option maxRecDepth 4096 in
theorem rootCandidates627_checked : candidates edge 5 0 4657 tail628 best.bound = rootCandidates627 := by
  unfold candidates
  have row : edge 4657 = SearchData.row4657.contains := by funext q; rfl
  rw [row]
  decide

def innerTail627_8 : List Nat := []
theorem innerScan627_8 : scan edge (search edge 3) 4 [4657] 4657 innerTail627_8 best = best := rfl

def innerTail627_7 : List Nat := 8209 :: innerTail627_8
def innerCandidates627_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates627_7_checked : candidates edge 4 4657 8209 innerTail627_8 best.bound = innerCandidates627_7 := by
  unfold candidates
  have row : edge 8209 = SearchData.row8209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan627_7 : scan edge (search edge 3) 4 [4657] 4657 innerTail627_7 best = best := by
  rw [innerTail627_7, scan, if_neg (by decide), innerCandidates627_7_checked]
  change scan edge (search edge 3) 4 [4657] 4657 innerTail627_8
    (if 3 ≤ innerCandidates627_7.length then search edge 3 [4657, 8209] innerCandidates627_7 12866 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan627_8

def innerTail627_6 : List Nat := 7933 :: innerTail627_7
def innerCandidates627_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates627_6_checked : candidates edge 4 4657 7933 innerTail627_7 best.bound = innerCandidates627_6 := by
  unfold candidates
  have row : edge 7933 = SearchData.row7933.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan627_6 : scan edge (search edge 3) 4 [4657] 4657 innerTail627_6 best = best := by
  rw [innerTail627_6, scan, if_neg (by decide), innerCandidates627_6_checked]
  change scan edge (search edge 3) 4 [4657] 4657 innerTail627_7
    (if 3 ≤ innerCandidates627_6.length then search edge 3 [4657, 7933] innerCandidates627_6 12590 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan627_7

def innerTail627_5 : List Nat := 7753 :: innerTail627_6
def innerCandidates627_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates627_5_checked : candidates edge 4 4657 7753 innerTail627_6 best.bound = innerCandidates627_5 := by
  unfold candidates
  have row : edge 7753 = SearchData.row7753.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan627_5 : scan edge (search edge 3) 4 [4657] 4657 innerTail627_5 best = best := by
  rw [innerTail627_5, scan, if_neg (by decide), innerCandidates627_5_checked]
  change scan edge (search edge 3) 4 [4657] 4657 innerTail627_6
    (if 3 ≤ innerCandidates627_5.length then search edge 3 [4657, 7753] innerCandidates627_5 12410 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan627_6

def innerTail627_4 : List Nat := 7243 :: innerTail627_5
def innerCandidates627_4 : List Nat := [8209]
set_option maxRecDepth 4096 in
theorem innerCandidates627_4_checked : candidates edge 4 4657 7243 innerTail627_5 best.bound = innerCandidates627_4 := by
  unfold candidates
  have row : edge 7243 = SearchData.row7243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan627_4 : scan edge (search edge 3) 4 [4657] 4657 innerTail627_4 best = best := by
  rw [innerTail627_4, scan, if_neg (by decide), innerCandidates627_4_checked]
  change scan edge (search edge 3) 4 [4657] 4657 innerTail627_5
    (if 3 ≤ innerCandidates627_4.length then search edge 3 [4657, 7243] innerCandidates627_4 11900 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan627_5

def innerTail627_3 : List Nat := 7177 :: innerTail627_4
def innerCandidates627_3 : List Nat := [7933]
set_option maxRecDepth 4096 in
theorem innerCandidates627_3_checked : candidates edge 4 4657 7177 innerTail627_4 best.bound = innerCandidates627_3 := by
  unfold candidates
  have row : edge 7177 = SearchData.row7177.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan627_3 : scan edge (search edge 3) 4 [4657] 4657 innerTail627_3 best = best := by
  rw [innerTail627_3, scan, if_neg (by decide), innerCandidates627_3_checked]
  change scan edge (search edge 3) 4 [4657] 4657 innerTail627_4
    (if 3 ≤ innerCandidates627_3.length then search edge 3 [4657, 7177] innerCandidates627_3 11834 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan627_4

def innerTail627_2 : List Nat := 6217 :: innerTail627_3
def innerCandidates627_2 : List Nat := [7933]
set_option maxRecDepth 4096 in
theorem innerCandidates627_2_checked : candidates edge 4 4657 6217 innerTail627_3 best.bound = innerCandidates627_2 := by
  unfold candidates
  have row : edge 6217 = SearchData.row6217.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan627_2 : scan edge (search edge 3) 4 [4657] 4657 innerTail627_2 best = best := by
  rw [innerTail627_2, scan, if_neg (by decide), innerCandidates627_2_checked]
  change scan edge (search edge 3) 4 [4657] 4657 innerTail627_3
    (if 3 ≤ innerCandidates627_2.length then search edge 3 [4657, 6217] innerCandidates627_2 10874 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan627_3

def innerTail627_1 : List Nat := 5227 :: innerTail627_2
def innerCandidates627_1 : List Nat := [7243]
set_option maxRecDepth 4096 in
theorem innerCandidates627_1_checked : candidates edge 4 4657 5227 innerTail627_2 best.bound = innerCandidates627_1 := by
  unfold candidates
  have row : edge 5227 = SearchData.row5227.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan627_1 : scan edge (search edge 3) 4 [4657] 4657 innerTail627_1 best = best := by
  rw [innerTail627_1, scan, if_neg (by decide), innerCandidates627_1_checked]
  change scan edge (search edge 3) 4 [4657] 4657 innerTail627_2
    (if 3 ≤ innerCandidates627_1.length then search edge 3 [4657, 5227] innerCandidates627_1 9884 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan627_2

def innerTail627_0 : List Nat := 4789 :: innerTail627_1
def innerCandidates627_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates627_0_checked : candidates edge 4 4657 4789 innerTail627_1 best.bound = innerCandidates627_0 := by
  unfold candidates
  have row : edge 4789 = SearchData.row4789.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan627_0 : scan edge (search edge 3) 4 [4657] 4657 innerTail627_0 best = best := by
  rw [innerTail627_0, scan, if_neg (by decide), innerCandidates627_0_checked]
  change scan edge (search edge 3) 4 [4657] 4657 innerTail627_1
    (if 3 ≤ innerCandidates627_0.length then search edge 3 [4657, 4789] innerCandidates627_0 9446 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan627_1

theorem rootBranch627 : search edge 4 [4657] rootCandidates627 4657 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4657] 4657 innerTail627_0 best = best
  exact innerScan627_0

def rootCandidates626 : List Nat := [4723, 6991, 7243, 8293]
set_option maxRecDepth 4096 in
theorem rootCandidates626_checked : candidates edge 5 0 4651 tail627 best.bound = rootCandidates626 := by
  unfold candidates
  have row : edge 4651 = SearchData.row4651.contains := by funext q; rfl
  rw [row]
  decide

def innerTail626_4 : List Nat := []
theorem innerScan626_4 : scan edge (search edge 3) 4 [4651] 4651 innerTail626_4 best = best := rfl

def innerTail626_3 : List Nat := 8293 :: innerTail626_4
def innerCandidates626_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates626_3_checked : candidates edge 4 4651 8293 innerTail626_4 best.bound = innerCandidates626_3 := by
  unfold candidates
  have row : edge 8293 = SearchData.row8293.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan626_3 : scan edge (search edge 3) 4 [4651] 4651 innerTail626_3 best = best := by
  rw [innerTail626_3, scan, if_neg (by decide), innerCandidates626_3_checked]
  change scan edge (search edge 3) 4 [4651] 4651 innerTail626_4
    (if 3 ≤ innerCandidates626_3.length then search edge 3 [4651, 8293] innerCandidates626_3 12944 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan626_4

def innerTail626_2 : List Nat := 7243 :: innerTail626_3
def innerCandidates626_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates626_2_checked : candidates edge 4 4651 7243 innerTail626_3 best.bound = innerCandidates626_2 := by
  unfold candidates
  have row : edge 7243 = SearchData.row7243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan626_2 : scan edge (search edge 3) 4 [4651] 4651 innerTail626_2 best = best := by
  rw [innerTail626_2, scan, if_neg (by decide), innerCandidates626_2_checked]
  change scan edge (search edge 3) 4 [4651] 4651 innerTail626_3
    (if 3 ≤ innerCandidates626_2.length then search edge 3 [4651, 7243] innerCandidates626_2 11894 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan626_3

def innerTail626_1 : List Nat := 6991 :: innerTail626_2
def innerCandidates626_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates626_1_checked : candidates edge 4 4651 6991 innerTail626_2 best.bound = innerCandidates626_1 := by
  unfold candidates
  have row : edge 6991 = SearchData.row6991.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan626_1 : scan edge (search edge 3) 4 [4651] 4651 innerTail626_1 best = best := by
  rw [innerTail626_1, scan, if_neg (by decide), innerCandidates626_1_checked]
  change scan edge (search edge 3) 4 [4651] 4651 innerTail626_2
    (if 3 ≤ innerCandidates626_1.length then search edge 3 [4651, 6991] innerCandidates626_1 11642 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan626_2

def innerTail626_0 : List Nat := 4723 :: innerTail626_1
def innerCandidates626_0 : List Nat := [7243]
set_option maxRecDepth 4096 in
theorem innerCandidates626_0_checked : candidates edge 4 4651 4723 innerTail626_1 best.bound = innerCandidates626_0 := by
  unfold candidates
  have row : edge 4723 = SearchData.row4723.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan626_0 : scan edge (search edge 3) 4 [4651] 4651 innerTail626_0 best = best := by
  rw [innerTail626_0, scan, if_neg (by decide), innerCandidates626_0_checked]
  change scan edge (search edge 3) 4 [4651] 4651 innerTail626_1
    (if 3 ≤ innerCandidates626_0.length then search edge 3 [4651, 4723] innerCandidates626_0 9374 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan626_1

theorem rootBranch626 : search edge 4 [4651] rootCandidates626 4651 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4651] 4651 innerTail626_0 best = best
  exact innerScan626_0

def rootCandidates625 : List Nat := [5399, 5987, 6323, 6719, 7727, 7853, 8117]
set_option maxRecDepth 4096 in
theorem rootCandidates625_checked : candidates edge 5 0 4649 tail626 best.bound = rootCandidates625 := by
  unfold candidates
  have row : edge 4649 = SearchData.row4649.contains := by funext q; rfl
  rw [row]
  decide

def innerTail625_7 : List Nat := []
theorem innerScan625_7 : scan edge (search edge 3) 4 [4649] 4649 innerTail625_7 best = best := rfl

def innerTail625_6 : List Nat := 8117 :: innerTail625_7
def innerCandidates625_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates625_6_checked : candidates edge 4 4649 8117 innerTail625_7 best.bound = innerCandidates625_6 := by
  unfold candidates
  have row : edge 8117 = SearchData.row8117.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan625_6 : scan edge (search edge 3) 4 [4649] 4649 innerTail625_6 best = best := by
  rw [innerTail625_6, scan, if_neg (by decide), innerCandidates625_6_checked]
  change scan edge (search edge 3) 4 [4649] 4649 innerTail625_7
    (if 3 ≤ innerCandidates625_6.length then search edge 3 [4649, 8117] innerCandidates625_6 12766 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan625_7

def innerTail625_5 : List Nat := 7853 :: innerTail625_6
def innerCandidates625_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates625_5_checked : candidates edge 4 4649 7853 innerTail625_6 best.bound = innerCandidates625_5 := by
  unfold candidates
  have row : edge 7853 = SearchData.row7853.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan625_5 : scan edge (search edge 3) 4 [4649] 4649 innerTail625_5 best = best := by
  rw [innerTail625_5, scan, if_neg (by decide), innerCandidates625_5_checked]
  change scan edge (search edge 3) 4 [4649] 4649 innerTail625_6
    (if 3 ≤ innerCandidates625_5.length then search edge 3 [4649, 7853] innerCandidates625_5 12502 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan625_6

def innerTail625_4 : List Nat := 7727 :: innerTail625_5
def innerCandidates625_4 : List Nat := [7853]
set_option maxRecDepth 4096 in
theorem innerCandidates625_4_checked : candidates edge 4 4649 7727 innerTail625_5 best.bound = innerCandidates625_4 := by
  unfold candidates
  have row : edge 7727 = SearchData.row7727.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan625_4 : scan edge (search edge 3) 4 [4649] 4649 innerTail625_4 best = best := by
  rw [innerTail625_4, scan, if_neg (by decide), innerCandidates625_4_checked]
  change scan edge (search edge 3) 4 [4649] 4649 innerTail625_5
    (if 3 ≤ innerCandidates625_4.length then search edge 3 [4649, 7727] innerCandidates625_4 12376 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan625_5

def innerTail625_3 : List Nat := 6719 :: innerTail625_4
def innerCandidates625_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates625_3_checked : candidates edge 4 4649 6719 innerTail625_4 best.bound = innerCandidates625_3 := by
  unfold candidates
  have row : edge 6719 = SearchData.row6719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan625_3 : scan edge (search edge 3) 4 [4649] 4649 innerTail625_3 best = best := by
  rw [innerTail625_3, scan, if_neg (by decide), innerCandidates625_3_checked]
  change scan edge (search edge 3) 4 [4649] 4649 innerTail625_4
    (if 3 ≤ innerCandidates625_3.length then search edge 3 [4649, 6719] innerCandidates625_3 11368 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan625_4

def innerTail625_2 : List Nat := 6323 :: innerTail625_3
def innerCandidates625_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates625_2_checked : candidates edge 4 4649 6323 innerTail625_3 best.bound = innerCandidates625_2 := by
  unfold candidates
  have row : edge 6323 = SearchData.row6323.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan625_2 : scan edge (search edge 3) 4 [4649] 4649 innerTail625_2 best = best := by
  rw [innerTail625_2, scan, if_neg (by decide), innerCandidates625_2_checked]
  change scan edge (search edge 3) 4 [4649] 4649 innerTail625_3
    (if 3 ≤ innerCandidates625_2.length then search edge 3 [4649, 6323] innerCandidates625_2 10972 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan625_3

def innerTail625_1 : List Nat := 5987 :: innerTail625_2
def innerCandidates625_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates625_1_checked : candidates edge 4 4649 5987 innerTail625_2 best.bound = innerCandidates625_1 := by
  unfold candidates
  have row : edge 5987 = SearchData.row5987.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan625_1 : scan edge (search edge 3) 4 [4649] 4649 innerTail625_1 best = best := by
  rw [innerTail625_1, scan, if_neg (by decide), innerCandidates625_1_checked]
  change scan edge (search edge 3) 4 [4649] 4649 innerTail625_2
    (if 3 ≤ innerCandidates625_1.length then search edge 3 [4649, 5987] innerCandidates625_1 10636 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan625_2

def innerTail625_0 : List Nat := 5399 :: innerTail625_1
def innerCandidates625_0 : List Nat := [8117]
set_option maxRecDepth 4096 in
theorem innerCandidates625_0_checked : candidates edge 4 4649 5399 innerTail625_1 best.bound = innerCandidates625_0 := by
  unfold candidates
  have row : edge 5399 = SearchData.row5399.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan625_0 : scan edge (search edge 3) 4 [4649] 4649 innerTail625_0 best = best := by
  rw [innerTail625_0, scan, if_neg (by decide), innerCandidates625_0_checked]
  change scan edge (search edge 3) 4 [4649] 4649 innerTail625_1
    (if 3 ≤ innerCandidates625_0.length then search edge 3 [4649, 5399] innerCandidates625_0 10048 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan625_1

theorem rootBranch625 : search edge 4 [4649] rootCandidates625 4649 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4649] 4649 innerTail625_0 best = best
  exact innerScan625_0

end ProjectEulerSolutions.PrimePairUpper.Search
