import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates545 : List Nat := [4073, 4679, 4889, 5171, 5393, 5801, 6011, 6263, 6653, 6803, 7001, 8093, 8147]
set_option maxRecDepth 4096 in
theorem rootCandidates545_checked : candidates edge 5 0 3947 tail546 best.bound = rootCandidates545 := by
  unfold candidates
  have row : edge 3947 = SearchData.row3947.contains := by funext q; rfl
  rw [row]
  decide

def innerTail545_13 : List Nat := []
theorem innerScan545_13 : scan edge (search edge 3) 4 [3947] 3947 innerTail545_13 best = best := rfl

def innerTail545_12 : List Nat := 8147 :: innerTail545_13
def innerCandidates545_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates545_12_checked : candidates edge 4 3947 8147 innerTail545_13 best.bound = innerCandidates545_12 := by
  unfold candidates
  have row : edge 8147 = SearchData.row8147.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan545_12 : scan edge (search edge 3) 4 [3947] 3947 innerTail545_12 best = best := by
  rw [innerTail545_12, scan, if_neg (by decide), innerCandidates545_12_checked]
  change scan edge (search edge 3) 4 [3947] 3947 innerTail545_13
    (if 3 ≤ innerCandidates545_12.length then search edge 3 [3947, 8147] innerCandidates545_12 12094 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan545_13

def innerTail545_11 : List Nat := 8093 :: innerTail545_12
def innerCandidates545_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates545_11_checked : candidates edge 4 3947 8093 innerTail545_12 best.bound = innerCandidates545_11 := by
  unfold candidates
  have row : edge 8093 = SearchData.row8093.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan545_11 : scan edge (search edge 3) 4 [3947] 3947 innerTail545_11 best = best := by
  rw [innerTail545_11, scan, if_neg (by decide), innerCandidates545_11_checked]
  change scan edge (search edge 3) 4 [3947] 3947 innerTail545_12
    (if 3 ≤ innerCandidates545_11.length then search edge 3 [3947, 8093] innerCandidates545_11 12040 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan545_12

def innerTail545_10 : List Nat := 7001 :: innerTail545_11
def innerCandidates545_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates545_10_checked : candidates edge 4 3947 7001 innerTail545_11 best.bound = innerCandidates545_10 := by
  unfold candidates
  have row : edge 7001 = SearchData.row7001.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan545_10 : scan edge (search edge 3) 4 [3947] 3947 innerTail545_10 best = best := by
  rw [innerTail545_10, scan, if_neg (by decide), innerCandidates545_10_checked]
  change scan edge (search edge 3) 4 [3947] 3947 innerTail545_11
    (if 3 ≤ innerCandidates545_10.length then search edge 3 [3947, 7001] innerCandidates545_10 10948 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan545_11

def innerTail545_9 : List Nat := 6803 :: innerTail545_10
def innerCandidates545_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates545_9_checked : candidates edge 4 3947 6803 innerTail545_10 best.bound = innerCandidates545_9 := by
  unfold candidates
  have row : edge 6803 = SearchData.row6803.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan545_9 : scan edge (search edge 3) 4 [3947] 3947 innerTail545_9 best = best := by
  rw [innerTail545_9, scan, if_neg (by decide), innerCandidates545_9_checked]
  change scan edge (search edge 3) 4 [3947] 3947 innerTail545_10
    (if 3 ≤ innerCandidates545_9.length then search edge 3 [3947, 6803] innerCandidates545_9 10750 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan545_10

def innerTail545_8 : List Nat := 6653 :: innerTail545_9
def innerCandidates545_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates545_8_checked : candidates edge 4 3947 6653 innerTail545_9 best.bound = innerCandidates545_8 := by
  unfold candidates
  have row : edge 6653 = SearchData.row6653.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan545_8 : scan edge (search edge 3) 4 [3947] 3947 innerTail545_8 best = best := by
  rw [innerTail545_8, scan, if_neg (by decide), innerCandidates545_8_checked]
  change scan edge (search edge 3) 4 [3947] 3947 innerTail545_9
    (if 3 ≤ innerCandidates545_8.length then search edge 3 [3947, 6653] innerCandidates545_8 10600 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan545_9

def innerTail545_7 : List Nat := 6263 :: innerTail545_8
def innerCandidates545_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates545_7_checked : candidates edge 4 3947 6263 innerTail545_8 best.bound = innerCandidates545_7 := by
  unfold candidates
  have row : edge 6263 = SearchData.row6263.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan545_7 : scan edge (search edge 3) 4 [3947] 3947 innerTail545_7 best = best := by
  rw [innerTail545_7, scan, if_neg (by decide), innerCandidates545_7_checked]
  change scan edge (search edge 3) 4 [3947] 3947 innerTail545_8
    (if 3 ≤ innerCandidates545_7.length then search edge 3 [3947, 6263] innerCandidates545_7 10210 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan545_8

def innerTail545_6 : List Nat := 6011 :: innerTail545_7
def innerCandidates545_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates545_6_checked : candidates edge 4 3947 6011 innerTail545_7 best.bound = innerCandidates545_6 := by
  unfold candidates
  have row : edge 6011 = SearchData.row6011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan545_6 : scan edge (search edge 3) 4 [3947] 3947 innerTail545_6 best = best := by
  rw [innerTail545_6, scan, if_neg (by decide), innerCandidates545_6_checked]
  change scan edge (search edge 3) 4 [3947] 3947 innerTail545_7
    (if 3 ≤ innerCandidates545_6.length then search edge 3 [3947, 6011] innerCandidates545_6 9958 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan545_7

def innerTail545_5 : List Nat := 5801 :: innerTail545_6
def innerCandidates545_5 : List Nat := [6263]
set_option maxRecDepth 4096 in
theorem innerCandidates545_5_checked : candidates edge 4 3947 5801 innerTail545_6 best.bound = innerCandidates545_5 := by
  unfold candidates
  have row : edge 5801 = SearchData.row5801.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan545_5 : scan edge (search edge 3) 4 [3947] 3947 innerTail545_5 best = best := by
  rw [innerTail545_5, scan, if_neg (by decide), innerCandidates545_5_checked]
  change scan edge (search edge 3) 4 [3947] 3947 innerTail545_6
    (if 3 ≤ innerCandidates545_5.length then search edge 3 [3947, 5801] innerCandidates545_5 9748 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan545_6

def innerTail545_4 : List Nat := 5393 :: innerTail545_5
def innerCandidates545_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates545_4_checked : candidates edge 4 3947 5393 innerTail545_5 best.bound = innerCandidates545_4 := by
  unfold candidates
  have row : edge 5393 = SearchData.row5393.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan545_4 : scan edge (search edge 3) 4 [3947] 3947 innerTail545_4 best = best := by
  rw [innerTail545_4, scan, if_neg (by decide), innerCandidates545_4_checked]
  change scan edge (search edge 3) 4 [3947] 3947 innerTail545_5
    (if 3 ≤ innerCandidates545_4.length then search edge 3 [3947, 5393] innerCandidates545_4 9340 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan545_5

def innerTail545_3 : List Nat := 5171 :: innerTail545_4
def innerCandidates545_3 : List Nat := [5393]
set_option maxRecDepth 4096 in
theorem innerCandidates545_3_checked : candidates edge 4 3947 5171 innerTail545_4 best.bound = innerCandidates545_3 := by
  unfold candidates
  have row : edge 5171 = SearchData.row5171.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan545_3 : scan edge (search edge 3) 4 [3947] 3947 innerTail545_3 best = best := by
  rw [innerTail545_3, scan, if_neg (by decide), innerCandidates545_3_checked]
  change scan edge (search edge 3) 4 [3947] 3947 innerTail545_4
    (if 3 ≤ innerCandidates545_3.length then search edge 3 [3947, 5171] innerCandidates545_3 9118 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan545_4

def innerTail545_2 : List Nat := 4889 :: innerTail545_3
def innerCandidates545_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates545_2_checked : candidates edge 4 3947 4889 innerTail545_3 best.bound = innerCandidates545_2 := by
  unfold candidates
  have row : edge 4889 = SearchData.row4889.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan545_2 : scan edge (search edge 3) 4 [3947] 3947 innerTail545_2 best = best := by
  rw [innerTail545_2, scan, if_neg (by decide), innerCandidates545_2_checked]
  change scan edge (search edge 3) 4 [3947] 3947 innerTail545_3
    (if 3 ≤ innerCandidates545_2.length then search edge 3 [3947, 4889] innerCandidates545_2 8836 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan545_3

def innerTail545_1 : List Nat := 4679 :: innerTail545_2
def innerCandidates545_1 : List Nat := [4889, 5393, 6653]
set_option maxRecDepth 4096 in
theorem innerCandidates545_1_checked : candidates edge 4 3947 4679 innerTail545_2 best.bound = innerCandidates545_1 := by
  unfold candidates
  have row : edge 4679 = SearchData.row4679.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild545_1 : search edge 3 [3947, 4679] innerCandidates545_1 8626 best = best := by decide

theorem innerScan545_1 : scan edge (search edge 3) 4 [3947] 3947 innerTail545_1 best = best := by
  rw [innerTail545_1, scan, if_neg (by decide), innerCandidates545_1_checked]
  change scan edge (search edge 3) 4 [3947] 3947 innerTail545_2
    (if 3 ≤ innerCandidates545_1.length then search edge 3 [3947, 4679] innerCandidates545_1 8626 best else best) = best
  rw [if_pos (by decide), innerChild545_1]
  exact innerScan545_2

def innerTail545_0 : List Nat := 4073 :: innerTail545_1
def innerCandidates545_0 : List Nat := [5393]
set_option maxRecDepth 4096 in
theorem innerCandidates545_0_checked : candidates edge 4 3947 4073 innerTail545_1 best.bound = innerCandidates545_0 := by
  unfold candidates
  have row : edge 4073 = SearchData.row4073.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan545_0 : scan edge (search edge 3) 4 [3947] 3947 innerTail545_0 best = best := by
  rw [innerTail545_0, scan, if_neg (by decide), innerCandidates545_0_checked]
  change scan edge (search edge 3) 4 [3947] 3947 innerTail545_1
    (if 3 ≤ innerCandidates545_0.length then search edge 3 [3947, 4073] innerCandidates545_0 8020 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan545_1

theorem rootBranch545 : search edge 4 [3947] rootCandidates545 3947 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3947] 3947 innerTail545_0 best = best
  exact innerScan545_0

def rootCandidates544 : List Nat := [4621, 5167, 5413, 5479, 5569, 5581, 5779, 5839, 6829, 7789, 8059]
set_option maxRecDepth 4096 in
theorem rootCandidates544_checked : candidates edge 5 0 3943 tail545 best.bound = rootCandidates544 := by
  unfold candidates
  have row : edge 3943 = SearchData.row3943.contains := by funext q; rfl
  rw [row]
  decide

def innerTail544_11 : List Nat := []
theorem innerScan544_11 : scan edge (search edge 3) 4 [3943] 3943 innerTail544_11 best = best := rfl

def innerTail544_10 : List Nat := 8059 :: innerTail544_11
def innerCandidates544_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates544_10_checked : candidates edge 4 3943 8059 innerTail544_11 best.bound = innerCandidates544_10 := by
  unfold candidates
  have row : edge 8059 = SearchData.row8059.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan544_10 : scan edge (search edge 3) 4 [3943] 3943 innerTail544_10 best = best := by
  rw [innerTail544_10, scan, if_neg (by decide), innerCandidates544_10_checked]
  change scan edge (search edge 3) 4 [3943] 3943 innerTail544_11
    (if 3 ≤ innerCandidates544_10.length then search edge 3 [3943, 8059] innerCandidates544_10 12002 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan544_11

def innerTail544_9 : List Nat := 7789 :: innerTail544_10
def innerCandidates544_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates544_9_checked : candidates edge 4 3943 7789 innerTail544_10 best.bound = innerCandidates544_9 := by
  unfold candidates
  have row : edge 7789 = SearchData.row7789.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan544_9 : scan edge (search edge 3) 4 [3943] 3943 innerTail544_9 best = best := by
  rw [innerTail544_9, scan, if_neg (by decide), innerCandidates544_9_checked]
  change scan edge (search edge 3) 4 [3943] 3943 innerTail544_10
    (if 3 ≤ innerCandidates544_9.length then search edge 3 [3943, 7789] innerCandidates544_9 11732 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan544_10

def innerTail544_8 : List Nat := 6829 :: innerTail544_9
def innerCandidates544_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates544_8_checked : candidates edge 4 3943 6829 innerTail544_9 best.bound = innerCandidates544_8 := by
  unfold candidates
  have row : edge 6829 = SearchData.row6829.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan544_8 : scan edge (search edge 3) 4 [3943] 3943 innerTail544_8 best = best := by
  rw [innerTail544_8, scan, if_neg (by decide), innerCandidates544_8_checked]
  change scan edge (search edge 3) 4 [3943] 3943 innerTail544_9
    (if 3 ≤ innerCandidates544_8.length then search edge 3 [3943, 6829] innerCandidates544_8 10772 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan544_9

def innerTail544_7 : List Nat := 5839 :: innerTail544_8
def innerCandidates544_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates544_7_checked : candidates edge 4 3943 5839 innerTail544_8 best.bound = innerCandidates544_7 := by
  unfold candidates
  have row : edge 5839 = SearchData.row5839.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan544_7 : scan edge (search edge 3) 4 [3943] 3943 innerTail544_7 best = best := by
  rw [innerTail544_7, scan, if_neg (by decide), innerCandidates544_7_checked]
  change scan edge (search edge 3) 4 [3943] 3943 innerTail544_8
    (if 3 ≤ innerCandidates544_7.length then search edge 3 [3943, 5839] innerCandidates544_7 9782 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan544_8

def innerTail544_6 : List Nat := 5779 :: innerTail544_7
def innerCandidates544_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates544_6_checked : candidates edge 4 3943 5779 innerTail544_7 best.bound = innerCandidates544_6 := by
  unfold candidates
  have row : edge 5779 = SearchData.row5779.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan544_6 : scan edge (search edge 3) 4 [3943] 3943 innerTail544_6 best = best := by
  rw [innerTail544_6, scan, if_neg (by decide), innerCandidates544_6_checked]
  change scan edge (search edge 3) 4 [3943] 3943 innerTail544_7
    (if 3 ≤ innerCandidates544_6.length then search edge 3 [3943, 5779] innerCandidates544_6 9722 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan544_7

def innerTail544_5 : List Nat := 5581 :: innerTail544_6
def innerCandidates544_5 : List Nat := [5839]
set_option maxRecDepth 4096 in
theorem innerCandidates544_5_checked : candidates edge 4 3943 5581 innerTail544_6 best.bound = innerCandidates544_5 := by
  unfold candidates
  have row : edge 5581 = SearchData.row5581.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan544_5 : scan edge (search edge 3) 4 [3943] 3943 innerTail544_5 best = best := by
  rw [innerTail544_5, scan, if_neg (by decide), innerCandidates544_5_checked]
  change scan edge (search edge 3) 4 [3943] 3943 innerTail544_6
    (if 3 ≤ innerCandidates544_5.length then search edge 3 [3943, 5581] innerCandidates544_5 9524 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan544_6

def innerTail544_4 : List Nat := 5569 :: innerTail544_5
def innerCandidates544_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates544_4_checked : candidates edge 4 3943 5569 innerTail544_5 best.bound = innerCandidates544_4 := by
  unfold candidates
  have row : edge 5569 = SearchData.row5569.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan544_4 : scan edge (search edge 3) 4 [3943] 3943 innerTail544_4 best = best := by
  rw [innerTail544_4, scan, if_neg (by decide), innerCandidates544_4_checked]
  change scan edge (search edge 3) 4 [3943] 3943 innerTail544_5
    (if 3 ≤ innerCandidates544_4.length then search edge 3 [3943, 5569] innerCandidates544_4 9512 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan544_5

def innerTail544_3 : List Nat := 5479 :: innerTail544_4
def innerCandidates544_3 : List Nat := [5581]
set_option maxRecDepth 4096 in
theorem innerCandidates544_3_checked : candidates edge 4 3943 5479 innerTail544_4 best.bound = innerCandidates544_3 := by
  unfold candidates
  have row : edge 5479 = SearchData.row5479.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan544_3 : scan edge (search edge 3) 4 [3943] 3943 innerTail544_3 best = best := by
  rw [innerTail544_3, scan, if_neg (by decide), innerCandidates544_3_checked]
  change scan edge (search edge 3) 4 [3943] 3943 innerTail544_4
    (if 3 ≤ innerCandidates544_3.length then search edge 3 [3943, 5479] innerCandidates544_3 9422 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan544_4

def innerTail544_2 : List Nat := 5413 :: innerTail544_3
def innerCandidates544_2 : List Nat := [5569]
set_option maxRecDepth 4096 in
theorem innerCandidates544_2_checked : candidates edge 4 3943 5413 innerTail544_3 best.bound = innerCandidates544_2 := by
  unfold candidates
  have row : edge 5413 = SearchData.row5413.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan544_2 : scan edge (search edge 3) 4 [3943] 3943 innerTail544_2 best = best := by
  rw [innerTail544_2, scan, if_neg (by decide), innerCandidates544_2_checked]
  change scan edge (search edge 3) 4 [3943] 3943 innerTail544_3
    (if 3 ≤ innerCandidates544_2.length then search edge 3 [3943, 5413] innerCandidates544_2 9356 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan544_3

def innerTail544_1 : List Nat := 5167 :: innerTail544_2
def innerCandidates544_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates544_1_checked : candidates edge 4 3943 5167 innerTail544_2 best.bound = innerCandidates544_1 := by
  unfold candidates
  have row : edge 5167 = SearchData.row5167.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan544_1 : scan edge (search edge 3) 4 [3943] 3943 innerTail544_1 best = best := by
  rw [innerTail544_1, scan, if_neg (by decide), innerCandidates544_1_checked]
  change scan edge (search edge 3) 4 [3943] 3943 innerTail544_2
    (if 3 ≤ innerCandidates544_1.length then search edge 3 [3943, 5167] innerCandidates544_1 9110 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan544_2

def innerTail544_0 : List Nat := 4621 :: innerTail544_1
def innerCandidates544_0 : List Nat := [5779]
set_option maxRecDepth 4096 in
theorem innerCandidates544_0_checked : candidates edge 4 3943 4621 innerTail544_1 best.bound = innerCandidates544_0 := by
  unfold candidates
  have row : edge 4621 = SearchData.row4621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan544_0 : scan edge (search edge 3) 4 [3943] 3943 innerTail544_0 best = best := by
  rw [innerTail544_0, scan, if_neg (by decide), innerCandidates544_0_checked]
  change scan edge (search edge 3) 4 [3943] 3943 innerTail544_1
    (if 3 ≤ innerCandidates544_0.length then search edge 3 [3943, 4621] innerCandidates544_0 8564 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan544_1

theorem rootBranch544 : search edge 4 [3943] rootCandidates544 3943 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3943] 3943 innerTail544_0 best = best
  exact innerScan544_0

def rootCandidates543 : List Nat := [5413, 5449, 5527, 6553, 6661, 7459, 7507, 7561, 7639, 8089, 8233]
set_option maxRecDepth 4096 in
theorem rootCandidates543_checked : candidates edge 5 0 3931 tail544 best.bound = rootCandidates543 := by
  unfold candidates
  have row : edge 3931 = SearchData.row3931.contains := by funext q; rfl
  rw [row]
  decide

def innerTail543_11 : List Nat := []
theorem innerScan543_11 : scan edge (search edge 3) 4 [3931] 3931 innerTail543_11 best = best := rfl

def innerTail543_10 : List Nat := 8233 :: innerTail543_11
def innerCandidates543_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates543_10_checked : candidates edge 4 3931 8233 innerTail543_11 best.bound = innerCandidates543_10 := by
  unfold candidates
  have row : edge 8233 = SearchData.row8233.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan543_10 : scan edge (search edge 3) 4 [3931] 3931 innerTail543_10 best = best := by
  rw [innerTail543_10, scan, if_neg (by decide), innerCandidates543_10_checked]
  change scan edge (search edge 3) 4 [3931] 3931 innerTail543_11
    (if 3 ≤ innerCandidates543_10.length then search edge 3 [3931, 8233] innerCandidates543_10 12164 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan543_11

def innerTail543_9 : List Nat := 8089 :: innerTail543_10
def innerCandidates543_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates543_9_checked : candidates edge 4 3931 8089 innerTail543_10 best.bound = innerCandidates543_9 := by
  unfold candidates
  have row : edge 8089 = SearchData.row8089.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan543_9 : scan edge (search edge 3) 4 [3931] 3931 innerTail543_9 best = best := by
  rw [innerTail543_9, scan, if_neg (by decide), innerCandidates543_9_checked]
  change scan edge (search edge 3) 4 [3931] 3931 innerTail543_10
    (if 3 ≤ innerCandidates543_9.length then search edge 3 [3931, 8089] innerCandidates543_9 12020 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan543_10

def innerTail543_8 : List Nat := 7639 :: innerTail543_9
def innerCandidates543_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates543_8_checked : candidates edge 4 3931 7639 innerTail543_9 best.bound = innerCandidates543_8 := by
  unfold candidates
  have row : edge 7639 = SearchData.row7639.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan543_8 : scan edge (search edge 3) 4 [3931] 3931 innerTail543_8 best = best := by
  rw [innerTail543_8, scan, if_neg (by decide), innerCandidates543_8_checked]
  change scan edge (search edge 3) 4 [3931] 3931 innerTail543_9
    (if 3 ≤ innerCandidates543_8.length then search edge 3 [3931, 7639] innerCandidates543_8 11570 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan543_9

def innerTail543_7 : List Nat := 7561 :: innerTail543_8
def innerCandidates543_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates543_7_checked : candidates edge 4 3931 7561 innerTail543_8 best.bound = innerCandidates543_7 := by
  unfold candidates
  have row : edge 7561 = SearchData.row7561.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan543_7 : scan edge (search edge 3) 4 [3931] 3931 innerTail543_7 best = best := by
  rw [innerTail543_7, scan, if_neg (by decide), innerCandidates543_7_checked]
  change scan edge (search edge 3) 4 [3931] 3931 innerTail543_8
    (if 3 ≤ innerCandidates543_7.length then search edge 3 [3931, 7561] innerCandidates543_7 11492 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan543_8

def innerTail543_6 : List Nat := 7507 :: innerTail543_7
def innerCandidates543_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates543_6_checked : candidates edge 4 3931 7507 innerTail543_7 best.bound = innerCandidates543_6 := by
  unfold candidates
  have row : edge 7507 = SearchData.row7507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan543_6 : scan edge (search edge 3) 4 [3931] 3931 innerTail543_6 best = best := by
  rw [innerTail543_6, scan, if_neg (by decide), innerCandidates543_6_checked]
  change scan edge (search edge 3) 4 [3931] 3931 innerTail543_7
    (if 3 ≤ innerCandidates543_6.length then search edge 3 [3931, 7507] innerCandidates543_6 11438 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan543_7

def innerTail543_5 : List Nat := 7459 :: innerTail543_6
def innerCandidates543_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates543_5_checked : candidates edge 4 3931 7459 innerTail543_6 best.bound = innerCandidates543_5 := by
  unfold candidates
  have row : edge 7459 = SearchData.row7459.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan543_5 : scan edge (search edge 3) 4 [3931] 3931 innerTail543_5 best = best := by
  rw [innerTail543_5, scan, if_neg (by decide), innerCandidates543_5_checked]
  change scan edge (search edge 3) 4 [3931] 3931 innerTail543_6
    (if 3 ≤ innerCandidates543_5.length then search edge 3 [3931, 7459] innerCandidates543_5 11390 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan543_6

def innerTail543_4 : List Nat := 6661 :: innerTail543_5
def innerCandidates543_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates543_4_checked : candidates edge 4 3931 6661 innerTail543_5 best.bound = innerCandidates543_4 := by
  unfold candidates
  have row : edge 6661 = SearchData.row6661.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan543_4 : scan edge (search edge 3) 4 [3931] 3931 innerTail543_4 best = best := by
  rw [innerTail543_4, scan, if_neg (by decide), innerCandidates543_4_checked]
  change scan edge (search edge 3) 4 [3931] 3931 innerTail543_5
    (if 3 ≤ innerCandidates543_4.length then search edge 3 [3931, 6661] innerCandidates543_4 10592 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan543_5

def innerTail543_3 : List Nat := 6553 :: innerTail543_4
def innerCandidates543_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates543_3_checked : candidates edge 4 3931 6553 innerTail543_4 best.bound = innerCandidates543_3 := by
  unfold candidates
  have row : edge 6553 = SearchData.row6553.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan543_3 : scan edge (search edge 3) 4 [3931] 3931 innerTail543_3 best = best := by
  rw [innerTail543_3, scan, if_neg (by decide), innerCandidates543_3_checked]
  change scan edge (search edge 3) 4 [3931] 3931 innerTail543_4
    (if 3 ≤ innerCandidates543_3.length then search edge 3 [3931, 6553] innerCandidates543_3 10484 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan543_4

def innerTail543_2 : List Nat := 5527 :: innerTail543_3
def innerCandidates543_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates543_2_checked : candidates edge 4 3931 5527 innerTail543_3 best.bound = innerCandidates543_2 := by
  unfold candidates
  have row : edge 5527 = SearchData.row5527.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan543_2 : scan edge (search edge 3) 4 [3931] 3931 innerTail543_2 best = best := by
  rw [innerTail543_2, scan, if_neg (by decide), innerCandidates543_2_checked]
  change scan edge (search edge 3) 4 [3931] 3931 innerTail543_3
    (if 3 ≤ innerCandidates543_2.length then search edge 3 [3931, 5527] innerCandidates543_2 9458 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan543_3

def innerTail543_1 : List Nat := 5449 :: innerTail543_2
def innerCandidates543_1 : List Nat := [7507, 8089]
set_option maxRecDepth 4096 in
theorem innerCandidates543_1_checked : candidates edge 4 3931 5449 innerTail543_2 best.bound = innerCandidates543_1 := by
  unfold candidates
  have row : edge 5449 = SearchData.row5449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan543_1 : scan edge (search edge 3) 4 [3931] 3931 innerTail543_1 best = best := by
  rw [innerTail543_1, scan, if_neg (by decide), innerCandidates543_1_checked]
  change scan edge (search edge 3) 4 [3931] 3931 innerTail543_2
    (if 3 ≤ innerCandidates543_1.length then search edge 3 [3931, 5449] innerCandidates543_1 9380 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan543_2

def innerTail543_0 : List Nat := 5413 :: innerTail543_1
def innerCandidates543_0 : List Nat := [5527]
set_option maxRecDepth 4096 in
theorem innerCandidates543_0_checked : candidates edge 4 3931 5413 innerTail543_1 best.bound = innerCandidates543_0 := by
  unfold candidates
  have row : edge 5413 = SearchData.row5413.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan543_0 : scan edge (search edge 3) 4 [3931] 3931 innerTail543_0 best = best := by
  rw [innerTail543_0, scan, if_neg (by decide), innerCandidates543_0_checked]
  change scan edge (search edge 3) 4 [3931] 3931 innerTail543_1
    (if 3 ≤ innerCandidates543_0.length then search edge 3 [3931, 5413] innerCandidates543_0 9344 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan543_1

theorem rootBranch543 : search edge 4 [3931] rootCandidates543 3931 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3931] 3931 innerTail543_0 best = best
  exact innerScan543_0

def rootCandidates542 : List Nat := [4349, 4421, 6053, 6143, 6221, 7043, 7109, 7457, 7691]
set_option maxRecDepth 4096 in
theorem rootCandidates542_checked : candidates edge 5 0 3929 tail543 best.bound = rootCandidates542 := by
  unfold candidates
  have row : edge 3929 = SearchData.row3929.contains := by funext q; rfl
  rw [row]
  decide

def innerTail542_9 : List Nat := []
theorem innerScan542_9 : scan edge (search edge 3) 4 [3929] 3929 innerTail542_9 best = best := rfl

def innerTail542_8 : List Nat := 7691 :: innerTail542_9
def innerCandidates542_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates542_8_checked : candidates edge 4 3929 7691 innerTail542_9 best.bound = innerCandidates542_8 := by
  unfold candidates
  have row : edge 7691 = SearchData.row7691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan542_8 : scan edge (search edge 3) 4 [3929] 3929 innerTail542_8 best = best := by
  rw [innerTail542_8, scan, if_neg (by decide), innerCandidates542_8_checked]
  change scan edge (search edge 3) 4 [3929] 3929 innerTail542_9
    (if 3 ≤ innerCandidates542_8.length then search edge 3 [3929, 7691] innerCandidates542_8 11620 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan542_9

def innerTail542_7 : List Nat := 7457 :: innerTail542_8
def innerCandidates542_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates542_7_checked : candidates edge 4 3929 7457 innerTail542_8 best.bound = innerCandidates542_7 := by
  unfold candidates
  have row : edge 7457 = SearchData.row7457.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan542_7 : scan edge (search edge 3) 4 [3929] 3929 innerTail542_7 best = best := by
  rw [innerTail542_7, scan, if_neg (by decide), innerCandidates542_7_checked]
  change scan edge (search edge 3) 4 [3929] 3929 innerTail542_8
    (if 3 ≤ innerCandidates542_7.length then search edge 3 [3929, 7457] innerCandidates542_7 11386 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan542_8

def innerTail542_6 : List Nat := 7109 :: innerTail542_7
def innerCandidates542_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates542_6_checked : candidates edge 4 3929 7109 innerTail542_7 best.bound = innerCandidates542_6 := by
  unfold candidates
  have row : edge 7109 = SearchData.row7109.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan542_6 : scan edge (search edge 3) 4 [3929] 3929 innerTail542_6 best = best := by
  rw [innerTail542_6, scan, if_neg (by decide), innerCandidates542_6_checked]
  change scan edge (search edge 3) 4 [3929] 3929 innerTail542_7
    (if 3 ≤ innerCandidates542_6.length then search edge 3 [3929, 7109] innerCandidates542_6 11038 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan542_7

def innerTail542_5 : List Nat := 7043 :: innerTail542_6
def innerCandidates542_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates542_5_checked : candidates edge 4 3929 7043 innerTail542_6 best.bound = innerCandidates542_5 := by
  unfold candidates
  have row : edge 7043 = SearchData.row7043.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan542_5 : scan edge (search edge 3) 4 [3929] 3929 innerTail542_5 best = best := by
  rw [innerTail542_5, scan, if_neg (by decide), innerCandidates542_5_checked]
  change scan edge (search edge 3) 4 [3929] 3929 innerTail542_6
    (if 3 ≤ innerCandidates542_5.length then search edge 3 [3929, 7043] innerCandidates542_5 10972 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan542_6

def innerTail542_4 : List Nat := 6221 :: innerTail542_5
def innerCandidates542_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates542_4_checked : candidates edge 4 3929 6221 innerTail542_5 best.bound = innerCandidates542_4 := by
  unfold candidates
  have row : edge 6221 = SearchData.row6221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan542_4 : scan edge (search edge 3) 4 [3929] 3929 innerTail542_4 best = best := by
  rw [innerTail542_4, scan, if_neg (by decide), innerCandidates542_4_checked]
  change scan edge (search edge 3) 4 [3929] 3929 innerTail542_5
    (if 3 ≤ innerCandidates542_4.length then search edge 3 [3929, 6221] innerCandidates542_4 10150 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan542_5

def innerTail542_3 : List Nat := 6143 :: innerTail542_4
def innerCandidates542_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates542_3_checked : candidates edge 4 3929 6143 innerTail542_4 best.bound = innerCandidates542_3 := by
  unfold candidates
  have row : edge 6143 = SearchData.row6143.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan542_3 : scan edge (search edge 3) 4 [3929] 3929 innerTail542_3 best = best := by
  rw [innerTail542_3, scan, if_neg (by decide), innerCandidates542_3_checked]
  change scan edge (search edge 3) 4 [3929] 3929 innerTail542_4
    (if 3 ≤ innerCandidates542_3.length then search edge 3 [3929, 6143] innerCandidates542_3 10072 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan542_4

def innerTail542_2 : List Nat := 6053 :: innerTail542_3
def innerCandidates542_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates542_2_checked : candidates edge 4 3929 6053 innerTail542_3 best.bound = innerCandidates542_2 := by
  unfold candidates
  have row : edge 6053 = SearchData.row6053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan542_2 : scan edge (search edge 3) 4 [3929] 3929 innerTail542_2 best = best := by
  rw [innerTail542_2, scan, if_neg (by decide), innerCandidates542_2_checked]
  change scan edge (search edge 3) 4 [3929] 3929 innerTail542_3
    (if 3 ≤ innerCandidates542_2.length then search edge 3 [3929, 6053] innerCandidates542_2 9982 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan542_3

def innerTail542_1 : List Nat := 4421 :: innerTail542_2
def innerCandidates542_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates542_1_checked : candidates edge 4 3929 4421 innerTail542_2 best.bound = innerCandidates542_1 := by
  unfold candidates
  have row : edge 4421 = SearchData.row4421.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan542_1 : scan edge (search edge 3) 4 [3929] 3929 innerTail542_1 best = best := by
  rw [innerTail542_1, scan, if_neg (by decide), innerCandidates542_1_checked]
  change scan edge (search edge 3) 4 [3929] 3929 innerTail542_2
    (if 3 ≤ innerCandidates542_1.length then search edge 3 [3929, 4421] innerCandidates542_1 8350 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan542_2

def innerTail542_0 : List Nat := 4349 :: innerTail542_1
def innerCandidates542_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates542_0_checked : candidates edge 4 3929 4349 innerTail542_1 best.bound = innerCandidates542_0 := by
  unfold candidates
  have row : edge 4349 = SearchData.row4349.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan542_0 : scan edge (search edge 3) 4 [3929] 3929 innerTail542_0 best = best := by
  rw [innerTail542_0, scan, if_neg (by decide), innerCandidates542_0_checked]
  change scan edge (search edge 3) 4 [3929] 3929 innerTail542_1
    (if 3 ≤ innerCandidates542_0.length then search edge 3 [3929, 4349] innerCandidates542_0 8278 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan542_1

theorem rootBranch542 : search edge 4 [3929] rootCandidates542 3929 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3929] 3929 innerTail542_0 best = best
  exact innerScan542_0

def rootCandidates541 : List Nat := [3989, 4133, 4211, 4703, 4721, 4919, 4973, 5519, 6011, 6803, 7643, 7949, 8237]
set_option maxRecDepth 4096 in
theorem rootCandidates541_checked : candidates edge 5 0 3923 tail542 best.bound = rootCandidates541 := by
  unfold candidates
  have row : edge 3923 = SearchData.row3923.contains := by funext q; rfl
  rw [row]
  decide

def innerTail541_13 : List Nat := []
theorem innerScan541_13 : scan edge (search edge 3) 4 [3923] 3923 innerTail541_13 best = best := rfl

def innerTail541_12 : List Nat := 8237 :: innerTail541_13
def innerCandidates541_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates541_12_checked : candidates edge 4 3923 8237 innerTail541_13 best.bound = innerCandidates541_12 := by
  unfold candidates
  have row : edge 8237 = SearchData.row8237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan541_12 : scan edge (search edge 3) 4 [3923] 3923 innerTail541_12 best = best := by
  rw [innerTail541_12, scan, if_neg (by decide), innerCandidates541_12_checked]
  change scan edge (search edge 3) 4 [3923] 3923 innerTail541_13
    (if 3 ≤ innerCandidates541_12.length then search edge 3 [3923, 8237] innerCandidates541_12 12160 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan541_13

def innerTail541_11 : List Nat := 7949 :: innerTail541_12
def innerCandidates541_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates541_11_checked : candidates edge 4 3923 7949 innerTail541_12 best.bound = innerCandidates541_11 := by
  unfold candidates
  have row : edge 7949 = SearchData.row7949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan541_11 : scan edge (search edge 3) 4 [3923] 3923 innerTail541_11 best = best := by
  rw [innerTail541_11, scan, if_neg (by decide), innerCandidates541_11_checked]
  change scan edge (search edge 3) 4 [3923] 3923 innerTail541_12
    (if 3 ≤ innerCandidates541_11.length then search edge 3 [3923, 7949] innerCandidates541_11 11872 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan541_12

def innerTail541_10 : List Nat := 7643 :: innerTail541_11
def innerCandidates541_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates541_10_checked : candidates edge 4 3923 7643 innerTail541_11 best.bound = innerCandidates541_10 := by
  unfold candidates
  have row : edge 7643 = SearchData.row7643.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan541_10 : scan edge (search edge 3) 4 [3923] 3923 innerTail541_10 best = best := by
  rw [innerTail541_10, scan, if_neg (by decide), innerCandidates541_10_checked]
  change scan edge (search edge 3) 4 [3923] 3923 innerTail541_11
    (if 3 ≤ innerCandidates541_10.length then search edge 3 [3923, 7643] innerCandidates541_10 11566 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan541_11

def innerTail541_9 : List Nat := 6803 :: innerTail541_10
def innerCandidates541_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates541_9_checked : candidates edge 4 3923 6803 innerTail541_10 best.bound = innerCandidates541_9 := by
  unfold candidates
  have row : edge 6803 = SearchData.row6803.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan541_9 : scan edge (search edge 3) 4 [3923] 3923 innerTail541_9 best = best := by
  rw [innerTail541_9, scan, if_neg (by decide), innerCandidates541_9_checked]
  change scan edge (search edge 3) 4 [3923] 3923 innerTail541_10
    (if 3 ≤ innerCandidates541_9.length then search edge 3 [3923, 6803] innerCandidates541_9 10726 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan541_10

def innerTail541_8 : List Nat := 6011 :: innerTail541_9
def innerCandidates541_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates541_8_checked : candidates edge 4 3923 6011 innerTail541_9 best.bound = innerCandidates541_8 := by
  unfold candidates
  have row : edge 6011 = SearchData.row6011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan541_8 : scan edge (search edge 3) 4 [3923] 3923 innerTail541_8 best = best := by
  rw [innerTail541_8, scan, if_neg (by decide), innerCandidates541_8_checked]
  change scan edge (search edge 3) 4 [3923] 3923 innerTail541_9
    (if 3 ≤ innerCandidates541_8.length then search edge 3 [3923, 6011] innerCandidates541_8 9934 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan541_9

def innerTail541_7 : List Nat := 5519 :: innerTail541_8
def innerCandidates541_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates541_7_checked : candidates edge 4 3923 5519 innerTail541_8 best.bound = innerCandidates541_7 := by
  unfold candidates
  have row : edge 5519 = SearchData.row5519.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan541_7 : scan edge (search edge 3) 4 [3923] 3923 innerTail541_7 best = best := by
  rw [innerTail541_7, scan, if_neg (by decide), innerCandidates541_7_checked]
  change scan edge (search edge 3) 4 [3923] 3923 innerTail541_8
    (if 3 ≤ innerCandidates541_7.length then search edge 3 [3923, 5519] innerCandidates541_7 9442 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan541_8

def innerTail541_6 : List Nat := 4973 :: innerTail541_7
def innerCandidates541_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates541_6_checked : candidates edge 4 3923 4973 innerTail541_7 best.bound = innerCandidates541_6 := by
  unfold candidates
  have row : edge 4973 = SearchData.row4973.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan541_6 : scan edge (search edge 3) 4 [3923] 3923 innerTail541_6 best = best := by
  rw [innerTail541_6, scan, if_neg (by decide), innerCandidates541_6_checked]
  change scan edge (search edge 3) 4 [3923] 3923 innerTail541_7
    (if 3 ≤ innerCandidates541_6.length then search edge 3 [3923, 4973] innerCandidates541_6 8896 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan541_7

def innerTail541_5 : List Nat := 4919 :: innerTail541_6
def innerCandidates541_5 : List Nat := [6803]
set_option maxRecDepth 4096 in
theorem innerCandidates541_5_checked : candidates edge 4 3923 4919 innerTail541_6 best.bound = innerCandidates541_5 := by
  unfold candidates
  have row : edge 4919 = SearchData.row4919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan541_5 : scan edge (search edge 3) 4 [3923] 3923 innerTail541_5 best = best := by
  rw [innerTail541_5, scan, if_neg (by decide), innerCandidates541_5_checked]
  change scan edge (search edge 3) 4 [3923] 3923 innerTail541_6
    (if 3 ≤ innerCandidates541_5.length then search edge 3 [3923, 4919] innerCandidates541_5 8842 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan541_6

def innerTail541_4 : List Nat := 4721 :: innerTail541_5
def innerCandidates541_4 : List Nat := [4919, 7949]
set_option maxRecDepth 4096 in
theorem innerCandidates541_4_checked : candidates edge 4 3923 4721 innerTail541_5 best.bound = innerCandidates541_4 := by
  unfold candidates
  have row : edge 4721 = SearchData.row4721.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan541_4 : scan edge (search edge 3) 4 [3923] 3923 innerTail541_4 best = best := by
  rw [innerTail541_4, scan, if_neg (by decide), innerCandidates541_4_checked]
  change scan edge (search edge 3) 4 [3923] 3923 innerTail541_5
    (if 3 ≤ innerCandidates541_4.length then search edge 3 [3923, 4721] innerCandidates541_4 8644 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan541_5

def innerTail541_3 : List Nat := 4703 :: innerTail541_4
def innerCandidates541_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates541_3_checked : candidates edge 4 3923 4703 innerTail541_4 best.bound = innerCandidates541_3 := by
  unfold candidates
  have row : edge 4703 = SearchData.row4703.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan541_3 : scan edge (search edge 3) 4 [3923] 3923 innerTail541_3 best = best := by
  rw [innerTail541_3, scan, if_neg (by decide), innerCandidates541_3_checked]
  change scan edge (search edge 3) 4 [3923] 3923 innerTail541_4
    (if 3 ≤ innerCandidates541_3.length then search edge 3 [3923, 4703] innerCandidates541_3 8626 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan541_4

def innerTail541_2 : List Nat := 4211 :: innerTail541_3
def innerCandidates541_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates541_2_checked : candidates edge 4 3923 4211 innerTail541_3 best.bound = innerCandidates541_2 := by
  unfold candidates
  have row : edge 4211 = SearchData.row4211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan541_2 : scan edge (search edge 3) 4 [3923] 3923 innerTail541_2 best = best := by
  rw [innerTail541_2, scan, if_neg (by decide), innerCandidates541_2_checked]
  change scan edge (search edge 3) 4 [3923] 3923 innerTail541_3
    (if 3 ≤ innerCandidates541_2.length then search edge 3 [3923, 4211] innerCandidates541_2 8134 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan541_3

def innerTail541_1 : List Nat := 4133 :: innerTail541_2
def innerCandidates541_1 : List Nat := [4721]
set_option maxRecDepth 4096 in
theorem innerCandidates541_1_checked : candidates edge 4 3923 4133 innerTail541_2 best.bound = innerCandidates541_1 := by
  unfold candidates
  have row : edge 4133 = SearchData.row4133.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan541_1 : scan edge (search edge 3) 4 [3923] 3923 innerTail541_1 best = best := by
  rw [innerTail541_1, scan, if_neg (by decide), innerCandidates541_1_checked]
  change scan edge (search edge 3) 4 [3923] 3923 innerTail541_2
    (if 3 ≤ innerCandidates541_1.length then search edge 3 [3923, 4133] innerCandidates541_1 8056 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan541_2

def innerTail541_0 : List Nat := 3989 :: innerTail541_1
def innerCandidates541_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates541_0_checked : candidates edge 4 3923 3989 innerTail541_1 best.bound = innerCandidates541_0 := by
  unfold candidates
  have row : edge 3989 = SearchData.row3989.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan541_0 : scan edge (search edge 3) 4 [3923] 3923 innerTail541_0 best = best := by
  rw [innerTail541_0, scan, if_neg (by decide), innerCandidates541_0_checked]
  change scan edge (search edge 3) 4 [3923] 3923 innerTail541_1
    (if 3 ≤ innerCandidates541_0.length then search edge 3 [3923, 3989] innerCandidates541_0 7912 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan541_1

theorem rootBranch541 : search edge 4 [3923] rootCandidates541 3923 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3923] 3923 innerTail541_0 best = best
  exact innerScan541_0

end ProjectEulerSolutions.PrimePairUpper.Search
