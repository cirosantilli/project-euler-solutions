import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates395 : List Nat := [3023, 3191, 4409, 5813, 6131, 6473, 6551, 8081]
set_option maxRecDepth 4096 in
theorem rootCandidates395_checked : candidates edge 5 0 2729 tail396 best.bound = rootCandidates395 := by
  unfold candidates
  have row : edge 2729 = SearchData.row2729.contains := by funext q; rfl
  rw [row]
  decide

def innerTail395_8 : List Nat := []
theorem innerScan395_8 : scan edge (search edge 3) 4 [2729] 2729 innerTail395_8 best = best := rfl

def innerTail395_7 : List Nat := 8081 :: innerTail395_8
def innerCandidates395_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates395_7_checked : candidates edge 4 2729 8081 innerTail395_8 best.bound = innerCandidates395_7 := by
  unfold candidates
  have row : edge 8081 = SearchData.row8081.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan395_7 : scan edge (search edge 3) 4 [2729] 2729 innerTail395_7 best = best := by
  rw [innerTail395_7, scan, if_neg (by decide), innerCandidates395_7_checked]
  change scan edge (search edge 3) 4 [2729] 2729 innerTail395_8
    (if 3 ≤ innerCandidates395_7.length then search edge 3 [2729, 8081] innerCandidates395_7 10810 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan395_8

def innerTail395_6 : List Nat := 6551 :: innerTail395_7
def innerCandidates395_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates395_6_checked : candidates edge 4 2729 6551 innerTail395_7 best.bound = innerCandidates395_6 := by
  unfold candidates
  have row : edge 6551 = SearchData.row6551.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan395_6 : scan edge (search edge 3) 4 [2729] 2729 innerTail395_6 best = best := by
  rw [innerTail395_6, scan, if_neg (by decide), innerCandidates395_6_checked]
  change scan edge (search edge 3) 4 [2729] 2729 innerTail395_7
    (if 3 ≤ innerCandidates395_6.length then search edge 3 [2729, 6551] innerCandidates395_6 9280 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan395_7

def innerTail395_5 : List Nat := 6473 :: innerTail395_6
def innerCandidates395_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates395_5_checked : candidates edge 4 2729 6473 innerTail395_6 best.bound = innerCandidates395_5 := by
  unfold candidates
  have row : edge 6473 = SearchData.row6473.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan395_5 : scan edge (search edge 3) 4 [2729] 2729 innerTail395_5 best = best := by
  rw [innerTail395_5, scan, if_neg (by decide), innerCandidates395_5_checked]
  change scan edge (search edge 3) 4 [2729] 2729 innerTail395_6
    (if 3 ≤ innerCandidates395_5.length then search edge 3 [2729, 6473] innerCandidates395_5 9202 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan395_6

def innerTail395_4 : List Nat := 6131 :: innerTail395_5
def innerCandidates395_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates395_4_checked : candidates edge 4 2729 6131 innerTail395_5 best.bound = innerCandidates395_4 := by
  unfold candidates
  have row : edge 6131 = SearchData.row6131.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan395_4 : scan edge (search edge 3) 4 [2729] 2729 innerTail395_4 best = best := by
  rw [innerTail395_4, scan, if_neg (by decide), innerCandidates395_4_checked]
  change scan edge (search edge 3) 4 [2729] 2729 innerTail395_5
    (if 3 ≤ innerCandidates395_4.length then search edge 3 [2729, 6131] innerCandidates395_4 8860 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan395_5

def innerTail395_3 : List Nat := 5813 :: innerTail395_4
def innerCandidates395_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates395_3_checked : candidates edge 4 2729 5813 innerTail395_4 best.bound = innerCandidates395_3 := by
  unfold candidates
  have row : edge 5813 = SearchData.row5813.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan395_3 : scan edge (search edge 3) 4 [2729] 2729 innerTail395_3 best = best := by
  rw [innerTail395_3, scan, if_neg (by decide), innerCandidates395_3_checked]
  change scan edge (search edge 3) 4 [2729] 2729 innerTail395_4
    (if 3 ≤ innerCandidates395_3.length then search edge 3 [2729, 5813] innerCandidates395_3 8542 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan395_4

def innerTail395_2 : List Nat := 4409 :: innerTail395_3
def innerCandidates395_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates395_2_checked : candidates edge 4 2729 4409 innerTail395_3 best.bound = innerCandidates395_2 := by
  unfold candidates
  have row : edge 4409 = SearchData.row4409.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan395_2 : scan edge (search edge 3) 4 [2729] 2729 innerTail395_2 best = best := by
  rw [innerTail395_2, scan, if_neg (by decide), innerCandidates395_2_checked]
  change scan edge (search edge 3) 4 [2729] 2729 innerTail395_3
    (if 3 ≤ innerCandidates395_2.length then search edge 3 [2729, 4409] innerCandidates395_2 7138 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan395_3

def innerTail395_1 : List Nat := 3191 :: innerTail395_2
def innerCandidates395_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates395_1_checked : candidates edge 4 2729 3191 innerTail395_2 best.bound = innerCandidates395_1 := by
  unfold candidates
  have row : edge 3191 = SearchData.row3191.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan395_1 : scan edge (search edge 3) 4 [2729] 2729 innerTail395_1 best = best := by
  rw [innerTail395_1, scan, if_neg (by decide), innerCandidates395_1_checked]
  change scan edge (search edge 3) 4 [2729] 2729 innerTail395_2
    (if 3 ≤ innerCandidates395_1.length then search edge 3 [2729, 3191] innerCandidates395_1 5920 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan395_2

def innerTail395_0 : List Nat := 3023 :: innerTail395_1
def innerCandidates395_0 : List Nat := [6551]
set_option maxRecDepth 4096 in
theorem innerCandidates395_0_checked : candidates edge 4 2729 3023 innerTail395_1 best.bound = innerCandidates395_0 := by
  unfold candidates
  have row : edge 3023 = SearchData.row3023.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan395_0 : scan edge (search edge 3) 4 [2729] 2729 innerTail395_0 best = best := by
  rw [innerTail395_0, scan, if_neg (by decide), innerCandidates395_0_checked]
  change scan edge (search edge 3) 4 [2729] 2729 innerTail395_1
    (if 3 ≤ innerCandidates395_0.length then search edge 3 [2729, 3023] innerCandidates395_0 5752 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan395_1

theorem rootBranch395 : search edge 4 [2729] rootCandidates395 2729 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2729] 2729 innerTail395_0 best = best
  exact innerScan395_0

def rootCandidates394 : List Nat := [3391, 4549, 6121, 7129, 7669]
set_option maxRecDepth 4096 in
theorem rootCandidates394_checked : candidates edge 5 0 2719 tail395 best.bound = rootCandidates394 := by
  unfold candidates
  have row : edge 2719 = SearchData.row2719.contains := by funext q; rfl
  rw [row]
  decide

def innerTail394_5 : List Nat := []
theorem innerScan394_5 : scan edge (search edge 3) 4 [2719] 2719 innerTail394_5 best = best := rfl

def innerTail394_4 : List Nat := 7669 :: innerTail394_5
def innerCandidates394_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates394_4_checked : candidates edge 4 2719 7669 innerTail394_5 best.bound = innerCandidates394_4 := by
  unfold candidates
  have row : edge 7669 = SearchData.row7669.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan394_4 : scan edge (search edge 3) 4 [2719] 2719 innerTail394_4 best = best := by
  rw [innerTail394_4, scan, if_neg (by decide), innerCandidates394_4_checked]
  change scan edge (search edge 3) 4 [2719] 2719 innerTail394_5
    (if 3 ≤ innerCandidates394_4.length then search edge 3 [2719, 7669] innerCandidates394_4 10388 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan394_5

def innerTail394_3 : List Nat := 7129 :: innerTail394_4
def innerCandidates394_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates394_3_checked : candidates edge 4 2719 7129 innerTail394_4 best.bound = innerCandidates394_3 := by
  unfold candidates
  have row : edge 7129 = SearchData.row7129.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan394_3 : scan edge (search edge 3) 4 [2719] 2719 innerTail394_3 best = best := by
  rw [innerTail394_3, scan, if_neg (by decide), innerCandidates394_3_checked]
  change scan edge (search edge 3) 4 [2719] 2719 innerTail394_4
    (if 3 ≤ innerCandidates394_3.length then search edge 3 [2719, 7129] innerCandidates394_3 9848 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan394_4

def innerTail394_2 : List Nat := 6121 :: innerTail394_3
def innerCandidates394_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates394_2_checked : candidates edge 4 2719 6121 innerTail394_3 best.bound = innerCandidates394_2 := by
  unfold candidates
  have row : edge 6121 = SearchData.row6121.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan394_2 : scan edge (search edge 3) 4 [2719] 2719 innerTail394_2 best = best := by
  rw [innerTail394_2, scan, if_neg (by decide), innerCandidates394_2_checked]
  change scan edge (search edge 3) 4 [2719] 2719 innerTail394_3
    (if 3 ≤ innerCandidates394_2.length then search edge 3 [2719, 6121] innerCandidates394_2 8840 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan394_3

def innerTail394_1 : List Nat := 4549 :: innerTail394_2
def innerCandidates394_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates394_1_checked : candidates edge 4 2719 4549 innerTail394_2 best.bound = innerCandidates394_1 := by
  unfold candidates
  have row : edge 4549 = SearchData.row4549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan394_1 : scan edge (search edge 3) 4 [2719] 2719 innerTail394_1 best = best := by
  rw [innerTail394_1, scan, if_neg (by decide), innerCandidates394_1_checked]
  change scan edge (search edge 3) 4 [2719] 2719 innerTail394_2
    (if 3 ≤ innerCandidates394_1.length then search edge 3 [2719, 4549] innerCandidates394_1 7268 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan394_2

def innerTail394_0 : List Nat := 3391 :: innerTail394_1
def innerCandidates394_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates394_0_checked : candidates edge 4 2719 3391 innerTail394_1 best.bound = innerCandidates394_0 := by
  unfold candidates
  have row : edge 3391 = SearchData.row3391.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan394_0 : scan edge (search edge 3) 4 [2719] 2719 innerTail394_0 best = best := by
  rw [innerTail394_0, scan, if_neg (by decide), innerCandidates394_0_checked]
  change scan edge (search edge 3) 4 [2719] 2719 innerTail394_1
    (if 3 ≤ innerCandidates394_0.length then search edge 3 [2719, 3391] innerCandidates394_0 6110 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan394_1

theorem rootBranch394 : search edge 4 [2719] rootCandidates394 2719 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2719] 2719 innerTail394_0 best = best
  exact innerScan394_0

def rootCandidates393 : List Nat := [2797, 3433, 3517, 3739, 3889, 5443, 5749, 5869, 6247, 6469, 7207, 7213, 8191]
set_option maxRecDepth 4096 in
theorem rootCandidates393_checked : candidates edge 5 0 2713 tail394 best.bound = rootCandidates393 := by
  unfold candidates
  have row : edge 2713 = SearchData.row2713.contains := by funext q; rfl
  rw [row]
  decide

def innerTail393_13 : List Nat := []
theorem innerScan393_13 : scan edge (search edge 3) 4 [2713] 2713 innerTail393_13 best = best := rfl

def innerTail393_12 : List Nat := 8191 :: innerTail393_13
def innerCandidates393_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates393_12_checked : candidates edge 4 2713 8191 innerTail393_13 best.bound = innerCandidates393_12 := by
  unfold candidates
  have row : edge 8191 = SearchData.row8191.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan393_12 : scan edge (search edge 3) 4 [2713] 2713 innerTail393_12 best = best := by
  rw [innerTail393_12, scan, if_neg (by decide), innerCandidates393_12_checked]
  change scan edge (search edge 3) 4 [2713] 2713 innerTail393_13
    (if 3 ≤ innerCandidates393_12.length then search edge 3 [2713, 8191] innerCandidates393_12 10904 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan393_13

def innerTail393_11 : List Nat := 7213 :: innerTail393_12
def innerCandidates393_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates393_11_checked : candidates edge 4 2713 7213 innerTail393_12 best.bound = innerCandidates393_11 := by
  unfold candidates
  have row : edge 7213 = SearchData.row7213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan393_11 : scan edge (search edge 3) 4 [2713] 2713 innerTail393_11 best = best := by
  rw [innerTail393_11, scan, if_neg (by decide), innerCandidates393_11_checked]
  change scan edge (search edge 3) 4 [2713] 2713 innerTail393_12
    (if 3 ≤ innerCandidates393_11.length then search edge 3 [2713, 7213] innerCandidates393_11 9926 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan393_12

def innerTail393_10 : List Nat := 7207 :: innerTail393_11
def innerCandidates393_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates393_10_checked : candidates edge 4 2713 7207 innerTail393_11 best.bound = innerCandidates393_10 := by
  unfold candidates
  have row : edge 7207 = SearchData.row7207.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan393_10 : scan edge (search edge 3) 4 [2713] 2713 innerTail393_10 best = best := by
  rw [innerTail393_10, scan, if_neg (by decide), innerCandidates393_10_checked]
  change scan edge (search edge 3) 4 [2713] 2713 innerTail393_11
    (if 3 ≤ innerCandidates393_10.length then search edge 3 [2713, 7207] innerCandidates393_10 9920 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan393_11

def innerTail393_9 : List Nat := 6469 :: innerTail393_10
def innerCandidates393_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates393_9_checked : candidates edge 4 2713 6469 innerTail393_10 best.bound = innerCandidates393_9 := by
  unfold candidates
  have row : edge 6469 = SearchData.row6469.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan393_9 : scan edge (search edge 3) 4 [2713] 2713 innerTail393_9 best = best := by
  rw [innerTail393_9, scan, if_neg (by decide), innerCandidates393_9_checked]
  change scan edge (search edge 3) 4 [2713] 2713 innerTail393_10
    (if 3 ≤ innerCandidates393_9.length then search edge 3 [2713, 6469] innerCandidates393_9 9182 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan393_10

def innerTail393_8 : List Nat := 6247 :: innerTail393_9
def innerCandidates393_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates393_8_checked : candidates edge 4 2713 6247 innerTail393_9 best.bound = innerCandidates393_8 := by
  unfold candidates
  have row : edge 6247 = SearchData.row6247.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan393_8 : scan edge (search edge 3) 4 [2713] 2713 innerTail393_8 best = best := by
  rw [innerTail393_8, scan, if_neg (by decide), innerCandidates393_8_checked]
  change scan edge (search edge 3) 4 [2713] 2713 innerTail393_9
    (if 3 ≤ innerCandidates393_8.length then search edge 3 [2713, 6247] innerCandidates393_8 8960 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan393_9

def innerTail393_7 : List Nat := 5869 :: innerTail393_8
def innerCandidates393_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates393_7_checked : candidates edge 4 2713 5869 innerTail393_8 best.bound = innerCandidates393_7 := by
  unfold candidates
  have row : edge 5869 = SearchData.row5869.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan393_7 : scan edge (search edge 3) 4 [2713] 2713 innerTail393_7 best = best := by
  rw [innerTail393_7, scan, if_neg (by decide), innerCandidates393_7_checked]
  change scan edge (search edge 3) 4 [2713] 2713 innerTail393_8
    (if 3 ≤ innerCandidates393_7.length then search edge 3 [2713, 5869] innerCandidates393_7 8582 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan393_8

def innerTail393_6 : List Nat := 5749 :: innerTail393_7
def innerCandidates393_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates393_6_checked : candidates edge 4 2713 5749 innerTail393_7 best.bound = innerCandidates393_6 := by
  unfold candidates
  have row : edge 5749 = SearchData.row5749.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan393_6 : scan edge (search edge 3) 4 [2713] 2713 innerTail393_6 best = best := by
  rw [innerTail393_6, scan, if_neg (by decide), innerCandidates393_6_checked]
  change scan edge (search edge 3) 4 [2713] 2713 innerTail393_7
    (if 3 ≤ innerCandidates393_6.length then search edge 3 [2713, 5749] innerCandidates393_6 8462 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan393_7

def innerTail393_5 : List Nat := 5443 :: innerTail393_6
def innerCandidates393_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates393_5_checked : candidates edge 4 2713 5443 innerTail393_6 best.bound = innerCandidates393_5 := by
  unfold candidates
  have row : edge 5443 = SearchData.row5443.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan393_5 : scan edge (search edge 3) 4 [2713] 2713 innerTail393_5 best = best := by
  rw [innerTail393_5, scan, if_neg (by decide), innerCandidates393_5_checked]
  change scan edge (search edge 3) 4 [2713] 2713 innerTail393_6
    (if 3 ≤ innerCandidates393_5.length then search edge 3 [2713, 5443] innerCandidates393_5 8156 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan393_6

def innerTail393_4 : List Nat := 3889 :: innerTail393_5
def innerCandidates393_4 : List Nat := [5749]
set_option maxRecDepth 4096 in
theorem innerCandidates393_4_checked : candidates edge 4 2713 3889 innerTail393_5 best.bound = innerCandidates393_4 := by
  unfold candidates
  have row : edge 3889 = SearchData.row3889.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan393_4 : scan edge (search edge 3) 4 [2713] 2713 innerTail393_4 best = best := by
  rw [innerTail393_4, scan, if_neg (by decide), innerCandidates393_4_checked]
  change scan edge (search edge 3) 4 [2713] 2713 innerTail393_5
    (if 3 ≤ innerCandidates393_4.length then search edge 3 [2713, 3889] innerCandidates393_4 6602 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan393_5

def innerTail393_3 : List Nat := 3739 :: innerTail393_4
def innerCandidates393_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates393_3_checked : candidates edge 4 2713 3739 innerTail393_4 best.bound = innerCandidates393_3 := by
  unfold candidates
  have row : edge 3739 = SearchData.row3739.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan393_3 : scan edge (search edge 3) 4 [2713] 2713 innerTail393_3 best = best := by
  rw [innerTail393_3, scan, if_neg (by decide), innerCandidates393_3_checked]
  change scan edge (search edge 3) 4 [2713] 2713 innerTail393_4
    (if 3 ≤ innerCandidates393_3.length then search edge 3 [2713, 3739] innerCandidates393_3 6452 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan393_4

def innerTail393_2 : List Nat := 3517 :: innerTail393_3
def innerCandidates393_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates393_2_checked : candidates edge 4 2713 3517 innerTail393_3 best.bound = innerCandidates393_2 := by
  unfold candidates
  have row : edge 3517 = SearchData.row3517.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan393_2 : scan edge (search edge 3) 4 [2713] 2713 innerTail393_2 best = best := by
  rw [innerTail393_2, scan, if_neg (by decide), innerCandidates393_2_checked]
  change scan edge (search edge 3) 4 [2713] 2713 innerTail393_3
    (if 3 ≤ innerCandidates393_2.length then search edge 3 [2713, 3517] innerCandidates393_2 6230 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan393_3

def innerTail393_1 : List Nat := 3433 :: innerTail393_2
def innerCandidates393_1 : List Nat := [7207]
set_option maxRecDepth 4096 in
theorem innerCandidates393_1_checked : candidates edge 4 2713 3433 innerTail393_2 best.bound = innerCandidates393_1 := by
  unfold candidates
  have row : edge 3433 = SearchData.row3433.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan393_1 : scan edge (search edge 3) 4 [2713] 2713 innerTail393_1 best = best := by
  rw [innerTail393_1, scan, if_neg (by decide), innerCandidates393_1_checked]
  change scan edge (search edge 3) 4 [2713] 2713 innerTail393_2
    (if 3 ≤ innerCandidates393_1.length then search edge 3 [2713, 3433] innerCandidates393_1 6146 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan393_2

def innerTail393_0 : List Nat := 2797 :: innerTail393_1
def innerCandidates393_0 : List Nat := [7207]
set_option maxRecDepth 4096 in
theorem innerCandidates393_0_checked : candidates edge 4 2713 2797 innerTail393_1 best.bound = innerCandidates393_0 := by
  unfold candidates
  have row : edge 2797 = SearchData.row2797.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan393_0 : scan edge (search edge 3) 4 [2713] 2713 innerTail393_0 best = best := by
  rw [innerTail393_0, scan, if_neg (by decide), innerCandidates393_0_checked]
  change scan edge (search edge 3) 4 [2713] 2713 innerTail393_1
    (if 3 ≤ innerCandidates393_0.length then search edge 3 [2713, 2797] innerCandidates393_0 5510 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan393_1

theorem rootBranch393 : search edge 4 [2713] rootCandidates393 2713 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2713] 2713 innerTail393_0 best = best
  exact innerScan393_0

def rootCandidates392 : List Nat := [2963, 3137, 3413, 3851, 4211, 4421, 5441, 5651, 5741, 6113, 6491, 7151, 7457, 7589, 7817, 7877]
set_option maxRecDepth 4096 in
theorem rootCandidates392_checked : candidates edge 5 0 2711 tail393 best.bound = rootCandidates392 := by
  unfold candidates
  have row : edge 2711 = SearchData.row2711.contains := by funext q; rfl
  rw [row]
  decide

def innerTail392_16 : List Nat := []
theorem innerScan392_16 : scan edge (search edge 3) 4 [2711] 2711 innerTail392_16 best = best := rfl

def innerTail392_15 : List Nat := 7877 :: innerTail392_16
def innerCandidates392_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates392_15_checked : candidates edge 4 2711 7877 innerTail392_16 best.bound = innerCandidates392_15 := by
  unfold candidates
  have row : edge 7877 = SearchData.row7877.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan392_15 : scan edge (search edge 3) 4 [2711] 2711 innerTail392_15 best = best := by
  rw [innerTail392_15, scan, if_neg (by decide), innerCandidates392_15_checked]
  change scan edge (search edge 3) 4 [2711] 2711 innerTail392_16
    (if 3 ≤ innerCandidates392_15.length then search edge 3 [2711, 7877] innerCandidates392_15 10588 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan392_16

def innerTail392_14 : List Nat := 7817 :: innerTail392_15
def innerCandidates392_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates392_14_checked : candidates edge 4 2711 7817 innerTail392_15 best.bound = innerCandidates392_14 := by
  unfold candidates
  have row : edge 7817 = SearchData.row7817.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan392_14 : scan edge (search edge 3) 4 [2711] 2711 innerTail392_14 best = best := by
  rw [innerTail392_14, scan, if_neg (by decide), innerCandidates392_14_checked]
  change scan edge (search edge 3) 4 [2711] 2711 innerTail392_15
    (if 3 ≤ innerCandidates392_14.length then search edge 3 [2711, 7817] innerCandidates392_14 10528 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan392_15

def innerTail392_13 : List Nat := 7589 :: innerTail392_14
def innerCandidates392_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates392_13_checked : candidates edge 4 2711 7589 innerTail392_14 best.bound = innerCandidates392_13 := by
  unfold candidates
  have row : edge 7589 = SearchData.row7589.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan392_13 : scan edge (search edge 3) 4 [2711] 2711 innerTail392_13 best = best := by
  rw [innerTail392_13, scan, if_neg (by decide), innerCandidates392_13_checked]
  change scan edge (search edge 3) 4 [2711] 2711 innerTail392_14
    (if 3 ≤ innerCandidates392_13.length then search edge 3 [2711, 7589] innerCandidates392_13 10300 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan392_14

def innerTail392_12 : List Nat := 7457 :: innerTail392_13
def innerCandidates392_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates392_12_checked : candidates edge 4 2711 7457 innerTail392_13 best.bound = innerCandidates392_12 := by
  unfold candidates
  have row : edge 7457 = SearchData.row7457.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan392_12 : scan edge (search edge 3) 4 [2711] 2711 innerTail392_12 best = best := by
  rw [innerTail392_12, scan, if_neg (by decide), innerCandidates392_12_checked]
  change scan edge (search edge 3) 4 [2711] 2711 innerTail392_13
    (if 3 ≤ innerCandidates392_12.length then search edge 3 [2711, 7457] innerCandidates392_12 10168 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan392_13

def innerTail392_11 : List Nat := 7151 :: innerTail392_12
def innerCandidates392_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates392_11_checked : candidates edge 4 2711 7151 innerTail392_12 best.bound = innerCandidates392_11 := by
  unfold candidates
  have row : edge 7151 = SearchData.row7151.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan392_11 : scan edge (search edge 3) 4 [2711] 2711 innerTail392_11 best = best := by
  rw [innerTail392_11, scan, if_neg (by decide), innerCandidates392_11_checked]
  change scan edge (search edge 3) 4 [2711] 2711 innerTail392_12
    (if 3 ≤ innerCandidates392_11.length then search edge 3 [2711, 7151] innerCandidates392_11 9862 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan392_12

def innerTail392_10 : List Nat := 6491 :: innerTail392_11
def innerCandidates392_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates392_10_checked : candidates edge 4 2711 6491 innerTail392_11 best.bound = innerCandidates392_10 := by
  unfold candidates
  have row : edge 6491 = SearchData.row6491.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan392_10 : scan edge (search edge 3) 4 [2711] 2711 innerTail392_10 best = best := by
  rw [innerTail392_10, scan, if_neg (by decide), innerCandidates392_10_checked]
  change scan edge (search edge 3) 4 [2711] 2711 innerTail392_11
    (if 3 ≤ innerCandidates392_10.length then search edge 3 [2711, 6491] innerCandidates392_10 9202 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan392_11

def innerTail392_9 : List Nat := 6113 :: innerTail392_10
def innerCandidates392_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates392_9_checked : candidates edge 4 2711 6113 innerTail392_10 best.bound = innerCandidates392_9 := by
  unfold candidates
  have row : edge 6113 = SearchData.row6113.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan392_9 : scan edge (search edge 3) 4 [2711] 2711 innerTail392_9 best = best := by
  rw [innerTail392_9, scan, if_neg (by decide), innerCandidates392_9_checked]
  change scan edge (search edge 3) 4 [2711] 2711 innerTail392_10
    (if 3 ≤ innerCandidates392_9.length then search edge 3 [2711, 6113] innerCandidates392_9 8824 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan392_10

def innerTail392_8 : List Nat := 5741 :: innerTail392_9
def innerCandidates392_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates392_8_checked : candidates edge 4 2711 5741 innerTail392_9 best.bound = innerCandidates392_8 := by
  unfold candidates
  have row : edge 5741 = SearchData.row5741.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan392_8 : scan edge (search edge 3) 4 [2711] 2711 innerTail392_8 best = best := by
  rw [innerTail392_8, scan, if_neg (by decide), innerCandidates392_8_checked]
  change scan edge (search edge 3) 4 [2711] 2711 innerTail392_9
    (if 3 ≤ innerCandidates392_8.length then search edge 3 [2711, 5741] innerCandidates392_8 8452 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan392_9

def innerTail392_7 : List Nat := 5651 :: innerTail392_8
def innerCandidates392_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates392_7_checked : candidates edge 4 2711 5651 innerTail392_8 best.bound = innerCandidates392_7 := by
  unfold candidates
  have row : edge 5651 = SearchData.row5651.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan392_7 : scan edge (search edge 3) 4 [2711] 2711 innerTail392_7 best = best := by
  rw [innerTail392_7, scan, if_neg (by decide), innerCandidates392_7_checked]
  change scan edge (search edge 3) 4 [2711] 2711 innerTail392_8
    (if 3 ≤ innerCandidates392_7.length then search edge 3 [2711, 5651] innerCandidates392_7 8362 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan392_8

def innerTail392_6 : List Nat := 5441 :: innerTail392_7
def innerCandidates392_6 : List Nat := [7457]
set_option maxRecDepth 4096 in
theorem innerCandidates392_6_checked : candidates edge 4 2711 5441 innerTail392_7 best.bound = innerCandidates392_6 := by
  unfold candidates
  have row : edge 5441 = SearchData.row5441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan392_6 : scan edge (search edge 3) 4 [2711] 2711 innerTail392_6 best = best := by
  rw [innerTail392_6, scan, if_neg (by decide), innerCandidates392_6_checked]
  change scan edge (search edge 3) 4 [2711] 2711 innerTail392_7
    (if 3 ≤ innerCandidates392_6.length then search edge 3 [2711, 5441] innerCandidates392_6 8152 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan392_7

def innerTail392_5 : List Nat := 4421 :: innerTail392_6
def innerCandidates392_5 : List Nat := [6113]
set_option maxRecDepth 4096 in
theorem innerCandidates392_5_checked : candidates edge 4 2711 4421 innerTail392_6 best.bound = innerCandidates392_5 := by
  unfold candidates
  have row : edge 4421 = SearchData.row4421.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan392_5 : scan edge (search edge 3) 4 [2711] 2711 innerTail392_5 best = best := by
  rw [innerTail392_5, scan, if_neg (by decide), innerCandidates392_5_checked]
  change scan edge (search edge 3) 4 [2711] 2711 innerTail392_6
    (if 3 ≤ innerCandidates392_5.length then search edge 3 [2711, 4421] innerCandidates392_5 7132 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan392_6

def innerTail392_4 : List Nat := 4211 :: innerTail392_5
def innerCandidates392_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates392_4_checked : candidates edge 4 2711 4211 innerTail392_5 best.bound = innerCandidates392_4 := by
  unfold candidates
  have row : edge 4211 = SearchData.row4211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan392_4 : scan edge (search edge 3) 4 [2711] 2711 innerTail392_4 best = best := by
  rw [innerTail392_4, scan, if_neg (by decide), innerCandidates392_4_checked]
  change scan edge (search edge 3) 4 [2711] 2711 innerTail392_5
    (if 3 ≤ innerCandidates392_4.length then search edge 3 [2711, 4211] innerCandidates392_4 6922 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan392_5

def innerTail392_3 : List Nat := 3851 :: innerTail392_4
def innerCandidates392_3 : List Nat := [5651, 7151]
set_option maxRecDepth 4096 in
theorem innerCandidates392_3_checked : candidates edge 4 2711 3851 innerTail392_4 best.bound = innerCandidates392_3 := by
  unfold candidates
  have row : edge 3851 = SearchData.row3851.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan392_3 : scan edge (search edge 3) 4 [2711] 2711 innerTail392_3 best = best := by
  rw [innerTail392_3, scan, if_neg (by decide), innerCandidates392_3_checked]
  change scan edge (search edge 3) 4 [2711] 2711 innerTail392_4
    (if 3 ≤ innerCandidates392_3.length then search edge 3 [2711, 3851] innerCandidates392_3 6562 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan392_4

def innerTail392_2 : List Nat := 3413 :: innerTail392_3
def innerCandidates392_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates392_2_checked : candidates edge 4 2711 3413 innerTail392_3 best.bound = innerCandidates392_2 := by
  unfold candidates
  have row : edge 3413 = SearchData.row3413.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan392_2 : scan edge (search edge 3) 4 [2711] 2711 innerTail392_2 best = best := by
  rw [innerTail392_2, scan, if_neg (by decide), innerCandidates392_2_checked]
  change scan edge (search edge 3) 4 [2711] 2711 innerTail392_3
    (if 3 ≤ innerCandidates392_2.length then search edge 3 [2711, 3413] innerCandidates392_2 6124 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan392_3

def innerTail392_1 : List Nat := 3137 :: innerTail392_2
def innerCandidates392_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates392_1_checked : candidates edge 4 2711 3137 innerTail392_2 best.bound = innerCandidates392_1 := by
  unfold candidates
  have row : edge 3137 = SearchData.row3137.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan392_1 : scan edge (search edge 3) 4 [2711] 2711 innerTail392_1 best = best := by
  rw [innerTail392_1, scan, if_neg (by decide), innerCandidates392_1_checked]
  change scan edge (search edge 3) 4 [2711] 2711 innerTail392_2
    (if 3 ≤ innerCandidates392_1.length then search edge 3 [2711, 3137] innerCandidates392_1 5848 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan392_2

def innerTail392_0 : List Nat := 2963 :: innerTail392_1
def innerCandidates392_0 : List Nat := [4421, 7151]
set_option maxRecDepth 4096 in
theorem innerCandidates392_0_checked : candidates edge 4 2711 2963 innerTail392_1 best.bound = innerCandidates392_0 := by
  unfold candidates
  have row : edge 2963 = SearchData.row2963.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan392_0 : scan edge (search edge 3) 4 [2711] 2711 innerTail392_0 best = best := by
  rw [innerTail392_0, scan, if_neg (by decide), innerCandidates392_0_checked]
  change scan edge (search edge 3) 4 [2711] 2711 innerTail392_1
    (if 3 ≤ innerCandidates392_0.length then search edge 3 [2711, 2963] innerCandidates392_0 5674 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan392_1

theorem rootBranch392 : search edge 4 [2711] rootCandidates392 2711 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2711] 2711 innerTail392_0 best = best
  exact innerScan392_0

def rootCandidates391 : List Nat := [2791, 2887, 3253, 3463, 3517, 3793, 4261, 4273, 4507, 4909, 5413, 5557, 5701, 5857, 6079, 6229, 6607, 6961, 7759, 7789, 8053]
set_option maxRecDepth 4096 in
theorem rootCandidates391_checked : candidates edge 5 0 2707 tail392 best.bound = rootCandidates391 := by
  unfold candidates
  have row : edge 2707 = SearchData.row2707.contains := by funext q; rfl
  rw [row]
  decide

def innerTail391_21 : List Nat := []
theorem innerScan391_21 : scan edge (search edge 3) 4 [2707] 2707 innerTail391_21 best = best := rfl

def innerTail391_20 : List Nat := 8053 :: innerTail391_21
def innerCandidates391_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates391_20_checked : candidates edge 4 2707 8053 innerTail391_21 best.bound = innerCandidates391_20 := by
  unfold candidates
  have row : edge 8053 = SearchData.row8053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan391_20 : scan edge (search edge 3) 4 [2707] 2707 innerTail391_20 best = best := by
  rw [innerTail391_20, scan, if_neg (by decide), innerCandidates391_20_checked]
  change scan edge (search edge 3) 4 [2707] 2707 innerTail391_21
    (if 3 ≤ innerCandidates391_20.length then search edge 3 [2707, 8053] innerCandidates391_20 10760 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan391_21

def innerTail391_19 : List Nat := 7789 :: innerTail391_20
def innerCandidates391_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates391_19_checked : candidates edge 4 2707 7789 innerTail391_20 best.bound = innerCandidates391_19 := by
  unfold candidates
  have row : edge 7789 = SearchData.row7789.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan391_19 : scan edge (search edge 3) 4 [2707] 2707 innerTail391_19 best = best := by
  rw [innerTail391_19, scan, if_neg (by decide), innerCandidates391_19_checked]
  change scan edge (search edge 3) 4 [2707] 2707 innerTail391_20
    (if 3 ≤ innerCandidates391_19.length then search edge 3 [2707, 7789] innerCandidates391_19 10496 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan391_20

def innerTail391_18 : List Nat := 7759 :: innerTail391_19
def innerCandidates391_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates391_18_checked : candidates edge 4 2707 7759 innerTail391_19 best.bound = innerCandidates391_18 := by
  unfold candidates
  have row : edge 7759 = SearchData.row7759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan391_18 : scan edge (search edge 3) 4 [2707] 2707 innerTail391_18 best = best := by
  rw [innerTail391_18, scan, if_neg (by decide), innerCandidates391_18_checked]
  change scan edge (search edge 3) 4 [2707] 2707 innerTail391_19
    (if 3 ≤ innerCandidates391_18.length then search edge 3 [2707, 7759] innerCandidates391_18 10466 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan391_19

def innerTail391_17 : List Nat := 6961 :: innerTail391_18
def innerCandidates391_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates391_17_checked : candidates edge 4 2707 6961 innerTail391_18 best.bound = innerCandidates391_17 := by
  unfold candidates
  have row : edge 6961 = SearchData.row6961.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan391_17 : scan edge (search edge 3) 4 [2707] 2707 innerTail391_17 best = best := by
  rw [innerTail391_17, scan, if_neg (by decide), innerCandidates391_17_checked]
  change scan edge (search edge 3) 4 [2707] 2707 innerTail391_18
    (if 3 ≤ innerCandidates391_17.length then search edge 3 [2707, 6961] innerCandidates391_17 9668 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan391_18

def innerTail391_16 : List Nat := 6607 :: innerTail391_17
def innerCandidates391_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates391_16_checked : candidates edge 4 2707 6607 innerTail391_17 best.bound = innerCandidates391_16 := by
  unfold candidates
  have row : edge 6607 = SearchData.row6607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan391_16 : scan edge (search edge 3) 4 [2707] 2707 innerTail391_16 best = best := by
  rw [innerTail391_16, scan, if_neg (by decide), innerCandidates391_16_checked]
  change scan edge (search edge 3) 4 [2707] 2707 innerTail391_17
    (if 3 ≤ innerCandidates391_16.length then search edge 3 [2707, 6607] innerCandidates391_16 9314 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan391_17

def innerTail391_15 : List Nat := 6229 :: innerTail391_16
def innerCandidates391_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates391_15_checked : candidates edge 4 2707 6229 innerTail391_16 best.bound = innerCandidates391_15 := by
  unfold candidates
  have row : edge 6229 = SearchData.row6229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan391_15 : scan edge (search edge 3) 4 [2707] 2707 innerTail391_15 best = best := by
  rw [innerTail391_15, scan, if_neg (by decide), innerCandidates391_15_checked]
  change scan edge (search edge 3) 4 [2707] 2707 innerTail391_16
    (if 3 ≤ innerCandidates391_15.length then search edge 3 [2707, 6229] innerCandidates391_15 8936 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan391_16

def innerTail391_14 : List Nat := 6079 :: innerTail391_15
def innerCandidates391_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates391_14_checked : candidates edge 4 2707 6079 innerTail391_15 best.bound = innerCandidates391_14 := by
  unfold candidates
  have row : edge 6079 = SearchData.row6079.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan391_14 : scan edge (search edge 3) 4 [2707] 2707 innerTail391_14 best = best := by
  rw [innerTail391_14, scan, if_neg (by decide), innerCandidates391_14_checked]
  change scan edge (search edge 3) 4 [2707] 2707 innerTail391_15
    (if 3 ≤ innerCandidates391_14.length then search edge 3 [2707, 6079] innerCandidates391_14 8786 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan391_15

def innerTail391_13 : List Nat := 5857 :: innerTail391_14
def innerCandidates391_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates391_13_checked : candidates edge 4 2707 5857 innerTail391_14 best.bound = innerCandidates391_13 := by
  unfold candidates
  have row : edge 5857 = SearchData.row5857.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan391_13 : scan edge (search edge 3) 4 [2707] 2707 innerTail391_13 best = best := by
  rw [innerTail391_13, scan, if_neg (by decide), innerCandidates391_13_checked]
  change scan edge (search edge 3) 4 [2707] 2707 innerTail391_14
    (if 3 ≤ innerCandidates391_13.length then search edge 3 [2707, 5857] innerCandidates391_13 8564 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan391_14

def innerTail391_12 : List Nat := 5701 :: innerTail391_13
def innerCandidates391_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates391_12_checked : candidates edge 4 2707 5701 innerTail391_13 best.bound = innerCandidates391_12 := by
  unfold candidates
  have row : edge 5701 = SearchData.row5701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan391_12 : scan edge (search edge 3) 4 [2707] 2707 innerTail391_12 best = best := by
  rw [innerTail391_12, scan, if_neg (by decide), innerCandidates391_12_checked]
  change scan edge (search edge 3) 4 [2707] 2707 innerTail391_13
    (if 3 ≤ innerCandidates391_12.length then search edge 3 [2707, 5701] innerCandidates391_12 8408 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan391_13

def innerTail391_11 : List Nat := 5557 :: innerTail391_12
def innerCandidates391_11 : List Nat := [7789, 8053]
set_option maxRecDepth 4096 in
theorem innerCandidates391_11_checked : candidates edge 4 2707 5557 innerTail391_12 best.bound = innerCandidates391_11 := by
  unfold candidates
  have row : edge 5557 = SearchData.row5557.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan391_11 : scan edge (search edge 3) 4 [2707] 2707 innerTail391_11 best = best := by
  rw [innerTail391_11, scan, if_neg (by decide), innerCandidates391_11_checked]
  change scan edge (search edge 3) 4 [2707] 2707 innerTail391_12
    (if 3 ≤ innerCandidates391_11.length then search edge 3 [2707, 5557] innerCandidates391_11 8264 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan391_12

def innerTail391_10 : List Nat := 5413 :: innerTail391_11
def innerCandidates391_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates391_10_checked : candidates edge 4 2707 5413 innerTail391_11 best.bound = innerCandidates391_10 := by
  unfold candidates
  have row : edge 5413 = SearchData.row5413.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan391_10 : scan edge (search edge 3) 4 [2707] 2707 innerTail391_10 best = best := by
  rw [innerTail391_10, scan, if_neg (by decide), innerCandidates391_10_checked]
  change scan edge (search edge 3) 4 [2707] 2707 innerTail391_11
    (if 3 ≤ innerCandidates391_10.length then search edge 3 [2707, 5413] innerCandidates391_10 8120 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan391_11

def innerTail391_9 : List Nat := 4909 :: innerTail391_10
def innerCandidates391_9 : List Nat := [5857]
set_option maxRecDepth 4096 in
theorem innerCandidates391_9_checked : candidates edge 4 2707 4909 innerTail391_10 best.bound = innerCandidates391_9 := by
  unfold candidates
  have row : edge 4909 = SearchData.row4909.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan391_9 : scan edge (search edge 3) 4 [2707] 2707 innerTail391_9 best = best := by
  rw [innerTail391_9, scan, if_neg (by decide), innerCandidates391_9_checked]
  change scan edge (search edge 3) 4 [2707] 2707 innerTail391_10
    (if 3 ≤ innerCandidates391_9.length then search edge 3 [2707, 4909] innerCandidates391_9 7616 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan391_10

def innerTail391_8 : List Nat := 4507 :: innerTail391_9
def innerCandidates391_8 : List Nat := [7759]
set_option maxRecDepth 4096 in
theorem innerCandidates391_8_checked : candidates edge 4 2707 4507 innerTail391_9 best.bound = innerCandidates391_8 := by
  unfold candidates
  have row : edge 4507 = SearchData.row4507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan391_8 : scan edge (search edge 3) 4 [2707] 2707 innerTail391_8 best = best := by
  rw [innerTail391_8, scan, if_neg (by decide), innerCandidates391_8_checked]
  change scan edge (search edge 3) 4 [2707] 2707 innerTail391_9
    (if 3 ≤ innerCandidates391_8.length then search edge 3 [2707, 4507] innerCandidates391_8 7214 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan391_9

def innerTail391_7 : List Nat := 4273 :: innerTail391_8
def innerCandidates391_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates391_7_checked : candidates edge 4 2707 4273 innerTail391_8 best.bound = innerCandidates391_7 := by
  unfold candidates
  have row : edge 4273 = SearchData.row4273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan391_7 : scan edge (search edge 3) 4 [2707] 2707 innerTail391_7 best = best := by
  rw [innerTail391_7, scan, if_neg (by decide), innerCandidates391_7_checked]
  change scan edge (search edge 3) 4 [2707] 2707 innerTail391_8
    (if 3 ≤ innerCandidates391_7.length then search edge 3 [2707, 4273] innerCandidates391_7 6980 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan391_8

def innerTail391_6 : List Nat := 4261 :: innerTail391_7
def innerCandidates391_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates391_6_checked : candidates edge 4 2707 4261 innerTail391_7 best.bound = innerCandidates391_6 := by
  unfold candidates
  have row : edge 4261 = SearchData.row4261.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan391_6 : scan edge (search edge 3) 4 [2707] 2707 innerTail391_6 best = best := by
  rw [innerTail391_6, scan, if_neg (by decide), innerCandidates391_6_checked]
  change scan edge (search edge 3) 4 [2707] 2707 innerTail391_7
    (if 3 ≤ innerCandidates391_6.length then search edge 3 [2707, 4261] innerCandidates391_6 6968 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan391_7

def innerTail391_5 : List Nat := 3793 :: innerTail391_6
def innerCandidates391_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates391_5_checked : candidates edge 4 2707 3793 innerTail391_6 best.bound = innerCandidates391_5 := by
  unfold candidates
  have row : edge 3793 = SearchData.row3793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan391_5 : scan edge (search edge 3) 4 [2707] 2707 innerTail391_5 best = best := by
  rw [innerTail391_5, scan, if_neg (by decide), innerCandidates391_5_checked]
  change scan edge (search edge 3) 4 [2707] 2707 innerTail391_6
    (if 3 ≤ innerCandidates391_5.length then search edge 3 [2707, 3793] innerCandidates391_5 6500 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan391_6

def innerTail391_4 : List Nat := 3517 :: innerTail391_5
def innerCandidates391_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates391_4_checked : candidates edge 4 2707 3517 innerTail391_5 best.bound = innerCandidates391_4 := by
  unfold candidates
  have row : edge 3517 = SearchData.row3517.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan391_4 : scan edge (search edge 3) 4 [2707] 2707 innerTail391_4 best = best := by
  rw [innerTail391_4, scan, if_neg (by decide), innerCandidates391_4_checked]
  change scan edge (search edge 3) 4 [2707] 2707 innerTail391_5
    (if 3 ≤ innerCandidates391_4.length then search edge 3 [2707, 3517] innerCandidates391_4 6224 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan391_5

def innerTail391_3 : List Nat := 3463 :: innerTail391_4
def innerCandidates391_3 : List Nat := [6079, 8053]
set_option maxRecDepth 4096 in
theorem innerCandidates391_3_checked : candidates edge 4 2707 3463 innerTail391_4 best.bound = innerCandidates391_3 := by
  unfold candidates
  have row : edge 3463 = SearchData.row3463.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan391_3 : scan edge (search edge 3) 4 [2707] 2707 innerTail391_3 best = best := by
  rw [innerTail391_3, scan, if_neg (by decide), innerCandidates391_3_checked]
  change scan edge (search edge 3) 4 [2707] 2707 innerTail391_4
    (if 3 ≤ innerCandidates391_3.length then search edge 3 [2707, 3463] innerCandidates391_3 6170 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan391_4

def innerTail391_2 : List Nat := 3253 :: innerTail391_3
def innerCandidates391_2 : List Nat := [3517, 5557]
set_option maxRecDepth 4096 in
theorem innerCandidates391_2_checked : candidates edge 4 2707 3253 innerTail391_3 best.bound = innerCandidates391_2 := by
  unfold candidates
  have row : edge 3253 = SearchData.row3253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan391_2 : scan edge (search edge 3) 4 [2707] 2707 innerTail391_2 best = best := by
  rw [innerTail391_2, scan, if_neg (by decide), innerCandidates391_2_checked]
  change scan edge (search edge 3) 4 [2707] 2707 innerTail391_3
    (if 3 ≤ innerCandidates391_2.length then search edge 3 [2707, 3253] innerCandidates391_2 5960 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan391_3

def innerTail391_1 : List Nat := 2887 :: innerTail391_2
def innerCandidates391_1 : List Nat := [4273]
set_option maxRecDepth 4096 in
theorem innerCandidates391_1_checked : candidates edge 4 2707 2887 innerTail391_2 best.bound = innerCandidates391_1 := by
  unfold candidates
  have row : edge 2887 = SearchData.row2887.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan391_1 : scan edge (search edge 3) 4 [2707] 2707 innerTail391_1 best = best := by
  rw [innerTail391_1, scan, if_neg (by decide), innerCandidates391_1_checked]
  change scan edge (search edge 3) 4 [2707] 2707 innerTail391_2
    (if 3 ≤ innerCandidates391_1.length then search edge 3 [2707, 2887] innerCandidates391_1 5594 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan391_2

def innerTail391_0 : List Nat := 2791 :: innerTail391_1
def innerCandidates391_0 : List Nat := [6607, 6961]
set_option maxRecDepth 4096 in
theorem innerCandidates391_0_checked : candidates edge 4 2707 2791 innerTail391_1 best.bound = innerCandidates391_0 := by
  unfold candidates
  have row : edge 2791 = SearchData.row2791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan391_0 : scan edge (search edge 3) 4 [2707] 2707 innerTail391_0 best = best := by
  rw [innerTail391_0, scan, if_neg (by decide), innerCandidates391_0_checked]
  change scan edge (search edge 3) 4 [2707] 2707 innerTail391_1
    (if 3 ≤ innerCandidates391_0.length then search edge 3 [2707, 2791] innerCandidates391_0 5498 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan391_1

theorem rootBranch391 : search edge 4 [2707] rootCandidates391 2707 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2707] 2707 innerTail391_0 best = best
  exact innerScan391_0

end ProjectEulerSolutions.PrimePairUpper.Search
