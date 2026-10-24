import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates791 : List Nat := [6131, 6359, 7127, 7433, 7649, 8111]
set_option maxRecDepth 4096 in
theorem rootCandidates791_checked : candidates edge 5 0 6089 tail792 best.bound = rootCandidates791 := by
  unfold candidates
  have row : edge 6089 = SearchData.row6089.contains := by funext q; rfl
  rw [row]
  decide

def innerTail791_6 : List Nat := []
theorem innerScan791_6 : scan edge (search edge 3) 4 [6089] 6089 innerTail791_6 best = best := rfl

def innerTail791_5 : List Nat := 8111 :: innerTail791_6
def innerCandidates791_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates791_5_checked : candidates edge 4 6089 8111 innerTail791_6 best.bound = innerCandidates791_5 := by
  unfold candidates
  have row : edge 8111 = SearchData.row8111.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan791_5 : scan edge (search edge 3) 4 [6089] 6089 innerTail791_5 best = best := by
  rw [innerTail791_5, scan, if_neg (by decide), innerCandidates791_5_checked]
  change scan edge (search edge 3) 4 [6089] 6089 innerTail791_6
    (if 3 ≤ innerCandidates791_5.length then search edge 3 [6089, 8111] innerCandidates791_5 14200 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan791_6

def innerTail791_4 : List Nat := 7649 :: innerTail791_5
def innerCandidates791_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates791_4_checked : candidates edge 4 6089 7649 innerTail791_5 best.bound = innerCandidates791_4 := by
  unfold candidates
  have row : edge 7649 = SearchData.row7649.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan791_4 : scan edge (search edge 3) 4 [6089] 6089 innerTail791_4 best = best := by
  rw [innerTail791_4, scan, if_neg (by decide), innerCandidates791_4_checked]
  change scan edge (search edge 3) 4 [6089] 6089 innerTail791_5
    (if 3 ≤ innerCandidates791_4.length then search edge 3 [6089, 7649] innerCandidates791_4 13738 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan791_5

def innerTail791_3 : List Nat := 7433 :: innerTail791_4
def innerCandidates791_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates791_3_checked : candidates edge 4 6089 7433 innerTail791_4 best.bound = innerCandidates791_3 := by
  unfold candidates
  have row : edge 7433 = SearchData.row7433.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan791_3 : scan edge (search edge 3) 4 [6089] 6089 innerTail791_3 best = best := by
  rw [innerTail791_3, scan, if_neg (by decide), innerCandidates791_3_checked]
  change scan edge (search edge 3) 4 [6089] 6089 innerTail791_4
    (if 3 ≤ innerCandidates791_3.length then search edge 3 [6089, 7433] innerCandidates791_3 13522 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan791_4

def innerTail791_2 : List Nat := 7127 :: innerTail791_3
def innerCandidates791_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates791_2_checked : candidates edge 4 6089 7127 innerTail791_3 best.bound = innerCandidates791_2 := by
  unfold candidates
  have row : edge 7127 = SearchData.row7127.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan791_2 : scan edge (search edge 3) 4 [6089] 6089 innerTail791_2 best = best := by
  rw [innerTail791_2, scan, if_neg (by decide), innerCandidates791_2_checked]
  change scan edge (search edge 3) 4 [6089] 6089 innerTail791_3
    (if 3 ≤ innerCandidates791_2.length then search edge 3 [6089, 7127] innerCandidates791_2 13216 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan791_3

def innerTail791_1 : List Nat := 6359 :: innerTail791_2
def innerCandidates791_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates791_1_checked : candidates edge 4 6089 6359 innerTail791_2 best.bound = innerCandidates791_1 := by
  unfold candidates
  have row : edge 6359 = SearchData.row6359.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan791_1 : scan edge (search edge 3) 4 [6089] 6089 innerTail791_1 best = best := by
  rw [innerTail791_1, scan, if_neg (by decide), innerCandidates791_1_checked]
  change scan edge (search edge 3) 4 [6089] 6089 innerTail791_2
    (if 3 ≤ innerCandidates791_1.length then search edge 3 [6089, 6359] innerCandidates791_1 12448 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan791_2

def innerTail791_0 : List Nat := 6131 :: innerTail791_1
def innerCandidates791_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates791_0_checked : candidates edge 4 6089 6131 innerTail791_1 best.bound = innerCandidates791_0 := by
  unfold candidates
  have row : edge 6131 = SearchData.row6131.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan791_0 : scan edge (search edge 3) 4 [6089] 6089 innerTail791_0 best = best := by
  rw [innerTail791_0, scan, if_neg (by decide), innerCandidates791_0_checked]
  change scan edge (search edge 3) 4 [6089] 6089 innerTail791_1
    (if 3 ≤ innerCandidates791_0.length then search edge 3 [6089, 6131] innerCandidates791_0 12220 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan791_1

theorem rootBranch791 : search edge 4 [6089] rootCandidates791 6089 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6089] 6089 innerTail791_0 best = best
  exact innerScan791_0

def rootCandidates790 : List Nat := [6451, 7621, 7753, 7927]
set_option maxRecDepth 4096 in
theorem rootCandidates790_checked : candidates edge 5 0 6079 tail791 best.bound = rootCandidates790 := by
  unfold candidates
  have row : edge 6079 = SearchData.row6079.contains := by funext q; rfl
  rw [row]
  decide

def innerTail790_4 : List Nat := []
theorem innerScan790_4 : scan edge (search edge 3) 4 [6079] 6079 innerTail790_4 best = best := rfl

def innerTail790_3 : List Nat := 7927 :: innerTail790_4
def innerCandidates790_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates790_3_checked : candidates edge 4 6079 7927 innerTail790_4 best.bound = innerCandidates790_3 := by
  unfold candidates
  have row : edge 7927 = SearchData.row7927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan790_3 : scan edge (search edge 3) 4 [6079] 6079 innerTail790_3 best = best := by
  rw [innerTail790_3, scan, if_neg (by decide), innerCandidates790_3_checked]
  change scan edge (search edge 3) 4 [6079] 6079 innerTail790_4
    (if 3 ≤ innerCandidates790_3.length then search edge 3 [6079, 7927] innerCandidates790_3 14006 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan790_4

def innerTail790_2 : List Nat := 7753 :: innerTail790_3
def innerCandidates790_2 : List Nat := [7927]
set_option maxRecDepth 4096 in
theorem innerCandidates790_2_checked : candidates edge 4 6079 7753 innerTail790_3 best.bound = innerCandidates790_2 := by
  unfold candidates
  have row : edge 7753 = SearchData.row7753.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan790_2 : scan edge (search edge 3) 4 [6079] 6079 innerTail790_2 best = best := by
  rw [innerTail790_2, scan, if_neg (by decide), innerCandidates790_2_checked]
  change scan edge (search edge 3) 4 [6079] 6079 innerTail790_3
    (if 3 ≤ innerCandidates790_2.length then search edge 3 [6079, 7753] innerCandidates790_2 13832 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan790_3

def innerTail790_1 : List Nat := 7621 :: innerTail790_2
def innerCandidates790_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates790_1_checked : candidates edge 4 6079 7621 innerTail790_2 best.bound = innerCandidates790_1 := by
  unfold candidates
  have row : edge 7621 = SearchData.row7621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan790_1 : scan edge (search edge 3) 4 [6079] 6079 innerTail790_1 best = best := by
  rw [innerTail790_1, scan, if_neg (by decide), innerCandidates790_1_checked]
  change scan edge (search edge 3) 4 [6079] 6079 innerTail790_2
    (if 3 ≤ innerCandidates790_1.length then search edge 3 [6079, 7621] innerCandidates790_1 13700 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan790_2

def innerTail790_0 : List Nat := 6451 :: innerTail790_1
def innerCandidates790_0 : List Nat := [7753]
set_option maxRecDepth 4096 in
theorem innerCandidates790_0_checked : candidates edge 4 6079 6451 innerTail790_1 best.bound = innerCandidates790_0 := by
  unfold candidates
  have row : edge 6451 = SearchData.row6451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan790_0 : scan edge (search edge 3) 4 [6079] 6079 innerTail790_0 best = best := by
  rw [innerTail790_0, scan, if_neg (by decide), innerCandidates790_0_checked]
  change scan edge (search edge 3) 4 [6079] 6079 innerTail790_1
    (if 3 ≤ innerCandidates790_0.length then search edge 3 [6079, 6451] innerCandidates790_0 12530 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan790_1

theorem rootBranch790 : search edge 4 [6079] rootCandidates790 6079 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6079] 6079 innerTail790_0 best = best
  exact innerScan790_0

def rootCandidates789 : List Nat := [6199, 6367, 7219, 7681, 7723, 7759]
set_option maxRecDepth 4096 in
theorem rootCandidates789_checked : candidates edge 5 0 6073 tail790 best.bound = rootCandidates789 := by
  unfold candidates
  have row : edge 6073 = SearchData.row6073.contains := by funext q; rfl
  rw [row]
  decide

def innerTail789_6 : List Nat := []
theorem innerScan789_6 : scan edge (search edge 3) 4 [6073] 6073 innerTail789_6 best = best := rfl

def innerTail789_5 : List Nat := 7759 :: innerTail789_6
def innerCandidates789_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates789_5_checked : candidates edge 4 6073 7759 innerTail789_6 best.bound = innerCandidates789_5 := by
  unfold candidates
  have row : edge 7759 = SearchData.row7759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan789_5 : scan edge (search edge 3) 4 [6073] 6073 innerTail789_5 best = best := by
  rw [innerTail789_5, scan, if_neg (by decide), innerCandidates789_5_checked]
  change scan edge (search edge 3) 4 [6073] 6073 innerTail789_6
    (if 3 ≤ innerCandidates789_5.length then search edge 3 [6073, 7759] innerCandidates789_5 13832 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan789_6

def innerTail789_4 : List Nat := 7723 :: innerTail789_5
def innerCandidates789_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates789_4_checked : candidates edge 4 6073 7723 innerTail789_5 best.bound = innerCandidates789_4 := by
  unfold candidates
  have row : edge 7723 = SearchData.row7723.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan789_4 : scan edge (search edge 3) 4 [6073] 6073 innerTail789_4 best = best := by
  rw [innerTail789_4, scan, if_neg (by decide), innerCandidates789_4_checked]
  change scan edge (search edge 3) 4 [6073] 6073 innerTail789_5
    (if 3 ≤ innerCandidates789_4.length then search edge 3 [6073, 7723] innerCandidates789_4 13796 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan789_5

def innerTail789_3 : List Nat := 7681 :: innerTail789_4
def innerCandidates789_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates789_3_checked : candidates edge 4 6073 7681 innerTail789_4 best.bound = innerCandidates789_3 := by
  unfold candidates
  have row : edge 7681 = SearchData.row7681.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan789_3 : scan edge (search edge 3) 4 [6073] 6073 innerTail789_3 best = best := by
  rw [innerTail789_3, scan, if_neg (by decide), innerCandidates789_3_checked]
  change scan edge (search edge 3) 4 [6073] 6073 innerTail789_4
    (if 3 ≤ innerCandidates789_3.length then search edge 3 [6073, 7681] innerCandidates789_3 13754 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan789_4

def innerTail789_2 : List Nat := 7219 :: innerTail789_3
def innerCandidates789_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates789_2_checked : candidates edge 4 6073 7219 innerTail789_3 best.bound = innerCandidates789_2 := by
  unfold candidates
  have row : edge 7219 = SearchData.row7219.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan789_2 : scan edge (search edge 3) 4 [6073] 6073 innerTail789_2 best = best := by
  rw [innerTail789_2, scan, if_neg (by decide), innerCandidates789_2_checked]
  change scan edge (search edge 3) 4 [6073] 6073 innerTail789_3
    (if 3 ≤ innerCandidates789_2.length then search edge 3 [6073, 7219] innerCandidates789_2 13292 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan789_3

def innerTail789_1 : List Nat := 6367 :: innerTail789_2
def innerCandidates789_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates789_1_checked : candidates edge 4 6073 6367 innerTail789_2 best.bound = innerCandidates789_1 := by
  unfold candidates
  have row : edge 6367 = SearchData.row6367.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan789_1 : scan edge (search edge 3) 4 [6073] 6073 innerTail789_1 best = best := by
  rw [innerTail789_1, scan, if_neg (by decide), innerCandidates789_1_checked]
  change scan edge (search edge 3) 4 [6073] 6073 innerTail789_2
    (if 3 ≤ innerCandidates789_1.length then search edge 3 [6073, 6367] innerCandidates789_1 12440 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan789_2

def innerTail789_0 : List Nat := 6199 :: innerTail789_1
def innerCandidates789_0 : List Nat := [7219]
set_option maxRecDepth 4096 in
theorem innerCandidates789_0_checked : candidates edge 4 6073 6199 innerTail789_1 best.bound = innerCandidates789_0 := by
  unfold candidates
  have row : edge 6199 = SearchData.row6199.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan789_0 : scan edge (search edge 3) 4 [6073] 6073 innerTail789_0 best = best := by
  rw [innerTail789_0, scan, if_neg (by decide), innerCandidates789_0_checked]
  change scan edge (search edge 3) 4 [6073] 6073 innerTail789_1
    (if 3 ≤ innerCandidates789_0.length then search edge 3 [6073, 6199] innerCandidates789_0 12272 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan789_1

theorem rootBranch789 : search edge 4 [6073] rootCandidates789 6073 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6073] 6073 innerTail789_0 best = best
  exact innerScan789_0

def rootCandidates788 : List Nat := [7699, 7993, 8221]
set_option maxRecDepth 4096 in
theorem rootCandidates788_checked : candidates edge 5 0 6067 tail789 best.bound = rootCandidates788 := by
  unfold candidates
  have row : edge 6067 = SearchData.row6067.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates787 : List Nat := [6173, 6977, 8171]
set_option maxRecDepth 4096 in
theorem rootCandidates787_checked : candidates edge 5 0 6053 tail788 best.bound = rootCandidates787 := by
  unfold candidates
  have row : edge 6053 = SearchData.row6053.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates786 : List Nat := [6203, 6359, 6569, 6719, 6737, 7643, 8069]
set_option maxRecDepth 4096 in
theorem rootCandidates786_checked : candidates edge 5 0 6047 tail787 best.bound = rootCandidates786 := by
  unfold candidates
  have row : edge 6047 = SearchData.row6047.contains := by funext q; rfl
  rw [row]
  decide

def innerTail786_7 : List Nat := []
theorem innerScan786_7 : scan edge (search edge 3) 4 [6047] 6047 innerTail786_7 best = best := rfl

def innerTail786_6 : List Nat := 8069 :: innerTail786_7
def innerCandidates786_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates786_6_checked : candidates edge 4 6047 8069 innerTail786_7 best.bound = innerCandidates786_6 := by
  unfold candidates
  have row : edge 8069 = SearchData.row8069.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan786_6 : scan edge (search edge 3) 4 [6047] 6047 innerTail786_6 best = best := by
  rw [innerTail786_6, scan, if_neg (by decide), innerCandidates786_6_checked]
  change scan edge (search edge 3) 4 [6047] 6047 innerTail786_7
    (if 3 ≤ innerCandidates786_6.length then search edge 3 [6047, 8069] innerCandidates786_6 14116 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan786_7

def innerTail786_5 : List Nat := 7643 :: innerTail786_6
def innerCandidates786_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates786_5_checked : candidates edge 4 6047 7643 innerTail786_6 best.bound = innerCandidates786_5 := by
  unfold candidates
  have row : edge 7643 = SearchData.row7643.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan786_5 : scan edge (search edge 3) 4 [6047] 6047 innerTail786_5 best = best := by
  rw [innerTail786_5, scan, if_neg (by decide), innerCandidates786_5_checked]
  change scan edge (search edge 3) 4 [6047] 6047 innerTail786_6
    (if 3 ≤ innerCandidates786_5.length then search edge 3 [6047, 7643] innerCandidates786_5 13690 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan786_6

def innerTail786_4 : List Nat := 6737 :: innerTail786_5
def innerCandidates786_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates786_4_checked : candidates edge 4 6047 6737 innerTail786_5 best.bound = innerCandidates786_4 := by
  unfold candidates
  have row : edge 6737 = SearchData.row6737.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan786_4 : scan edge (search edge 3) 4 [6047] 6047 innerTail786_4 best = best := by
  rw [innerTail786_4, scan, if_neg (by decide), innerCandidates786_4_checked]
  change scan edge (search edge 3) 4 [6047] 6047 innerTail786_5
    (if 3 ≤ innerCandidates786_4.length then search edge 3 [6047, 6737] innerCandidates786_4 12784 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan786_5

def innerTail786_3 : List Nat := 6719 :: innerTail786_4
def innerCandidates786_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates786_3_checked : candidates edge 4 6047 6719 innerTail786_4 best.bound = innerCandidates786_3 := by
  unfold candidates
  have row : edge 6719 = SearchData.row6719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan786_3 : scan edge (search edge 3) 4 [6047] 6047 innerTail786_3 best = best := by
  rw [innerTail786_3, scan, if_neg (by decide), innerCandidates786_3_checked]
  change scan edge (search edge 3) 4 [6047] 6047 innerTail786_4
    (if 3 ≤ innerCandidates786_3.length then search edge 3 [6047, 6719] innerCandidates786_3 12766 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan786_4

def innerTail786_2 : List Nat := 6569 :: innerTail786_3
def innerCandidates786_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates786_2_checked : candidates edge 4 6047 6569 innerTail786_3 best.bound = innerCandidates786_2 := by
  unfold candidates
  have row : edge 6569 = SearchData.row6569.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan786_2 : scan edge (search edge 3) 4 [6047] 6047 innerTail786_2 best = best := by
  rw [innerTail786_2, scan, if_neg (by decide), innerCandidates786_2_checked]
  change scan edge (search edge 3) 4 [6047] 6047 innerTail786_3
    (if 3 ≤ innerCandidates786_2.length then search edge 3 [6047, 6569] innerCandidates786_2 12616 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan786_3

def innerTail786_1 : List Nat := 6359 :: innerTail786_2
def innerCandidates786_1 : List Nat := [6737]
set_option maxRecDepth 4096 in
theorem innerCandidates786_1_checked : candidates edge 4 6047 6359 innerTail786_2 best.bound = innerCandidates786_1 := by
  unfold candidates
  have row : edge 6359 = SearchData.row6359.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan786_1 : scan edge (search edge 3) 4 [6047] 6047 innerTail786_1 best = best := by
  rw [innerTail786_1, scan, if_neg (by decide), innerCandidates786_1_checked]
  change scan edge (search edge 3) 4 [6047] 6047 innerTail786_2
    (if 3 ≤ innerCandidates786_1.length then search edge 3 [6047, 6359] innerCandidates786_1 12406 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan786_2

def innerTail786_0 : List Nat := 6203 :: innerTail786_1
def innerCandidates786_0 : List Nat := [7643]
set_option maxRecDepth 4096 in
theorem innerCandidates786_0_checked : candidates edge 4 6047 6203 innerTail786_1 best.bound = innerCandidates786_0 := by
  unfold candidates
  have row : edge 6203 = SearchData.row6203.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan786_0 : scan edge (search edge 3) 4 [6047] 6047 innerTail786_0 best = best := by
  rw [innerTail786_0, scan, if_neg (by decide), innerCandidates786_0_checked]
  change scan edge (search edge 3) 4 [6047] 6047 innerTail786_1
    (if 3 ≤ innerCandidates786_0.length then search edge 3 [6047, 6203] innerCandidates786_0 12250 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan786_1

theorem rootBranch786 : search edge 4 [6047] rootCandidates786 6047 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6047] 6047 innerTail786_0 best = best
  exact innerScan786_0

def rootCandidates785 : List Nat := [6451, 6553, 6619, 7219]
set_option maxRecDepth 4096 in
theorem rootCandidates785_checked : candidates edge 5 0 6043 tail786 best.bound = rootCandidates785 := by
  unfold candidates
  have row : edge 6043 = SearchData.row6043.contains := by funext q; rfl
  rw [row]
  decide

def innerTail785_4 : List Nat := []
theorem innerScan785_4 : scan edge (search edge 3) 4 [6043] 6043 innerTail785_4 best = best := rfl

def innerTail785_3 : List Nat := 7219 :: innerTail785_4
def innerCandidates785_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates785_3_checked : candidates edge 4 6043 7219 innerTail785_4 best.bound = innerCandidates785_3 := by
  unfold candidates
  have row : edge 7219 = SearchData.row7219.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan785_3 : scan edge (search edge 3) 4 [6043] 6043 innerTail785_3 best = best := by
  rw [innerTail785_3, scan, if_neg (by decide), innerCandidates785_3_checked]
  change scan edge (search edge 3) 4 [6043] 6043 innerTail785_4
    (if 3 ≤ innerCandidates785_3.length then search edge 3 [6043, 7219] innerCandidates785_3 13262 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan785_4

def innerTail785_2 : List Nat := 6619 :: innerTail785_3
def innerCandidates785_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates785_2_checked : candidates edge 4 6043 6619 innerTail785_3 best.bound = innerCandidates785_2 := by
  unfold candidates
  have row : edge 6619 = SearchData.row6619.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan785_2 : scan edge (search edge 3) 4 [6043] 6043 innerTail785_2 best = best := by
  rw [innerTail785_2, scan, if_neg (by decide), innerCandidates785_2_checked]
  change scan edge (search edge 3) 4 [6043] 6043 innerTail785_3
    (if 3 ≤ innerCandidates785_2.length then search edge 3 [6043, 6619] innerCandidates785_2 12662 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan785_3

def innerTail785_1 : List Nat := 6553 :: innerTail785_2
def innerCandidates785_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates785_1_checked : candidates edge 4 6043 6553 innerTail785_2 best.bound = innerCandidates785_1 := by
  unfold candidates
  have row : edge 6553 = SearchData.row6553.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan785_1 : scan edge (search edge 3) 4 [6043] 6043 innerTail785_1 best = best := by
  rw [innerTail785_1, scan, if_neg (by decide), innerCandidates785_1_checked]
  change scan edge (search edge 3) 4 [6043] 6043 innerTail785_2
    (if 3 ≤ innerCandidates785_1.length then search edge 3 [6043, 6553] innerCandidates785_1 12596 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan785_2

def innerTail785_0 : List Nat := 6451 :: innerTail785_1
def innerCandidates785_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates785_0_checked : candidates edge 4 6043 6451 innerTail785_1 best.bound = innerCandidates785_0 := by
  unfold candidates
  have row : edge 6451 = SearchData.row6451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan785_0 : scan edge (search edge 3) 4 [6043] 6043 innerTail785_0 best = best := by
  rw [innerTail785_0, scan, if_neg (by decide), innerCandidates785_0_checked]
  change scan edge (search edge 3) 4 [6043] 6043 innerTail785_1
    (if 3 ≤ innerCandidates785_0.length then search edge 3 [6043, 6451] innerCandidates785_0 12494 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan785_1

theorem rootBranch785 : search edge 4 [6043] rootCandidates785 6043 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6043] 6043 innerTail785_0 best = best
  exact innerScan785_0

def rootCandidates784 : List Nat := [6991, 8209]
set_option maxRecDepth 4096 in
theorem rootCandidates784_checked : candidates edge 5 0 6037 tail785 best.bound = rootCandidates784 := by
  unfold candidates
  have row : edge 6037 = SearchData.row6037.contains := by funext q; rfl
  rw [row]
  decide

end ProjectEulerSolutions.PrimePairUpper.Search
