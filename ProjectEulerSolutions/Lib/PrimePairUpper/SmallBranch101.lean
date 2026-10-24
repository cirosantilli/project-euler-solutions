import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates390 : List Nat := [3041, 3209, 3413, 3761, 4133, 4337, 5939, 6329, 6353, 7121, 7211, 7577, 7589, 8039]
set_option maxRecDepth 4096 in
theorem rootCandidates390_checked : candidates edge 5 0 2699 tail391 best.bound = rootCandidates390 := by
  unfold candidates
  have row : edge 2699 = SearchData.row2699.contains := by funext q; rfl
  rw [row]
  decide

def innerTail390_14 : List Nat := []
theorem innerScan390_14 : scan edge (search edge 3) 4 [2699] 2699 innerTail390_14 best = best := rfl

def innerTail390_13 : List Nat := 8039 :: innerTail390_14
def innerCandidates390_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates390_13_checked : candidates edge 4 2699 8039 innerTail390_14 best.bound = innerCandidates390_13 := by
  unfold candidates
  have row : edge 8039 = SearchData.row8039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan390_13 : scan edge (search edge 3) 4 [2699] 2699 innerTail390_13 best = best := by
  rw [innerTail390_13, scan, if_neg (by decide), innerCandidates390_13_checked]
  change scan edge (search edge 3) 4 [2699] 2699 innerTail390_14
    (if 3 ≤ innerCandidates390_13.length then search edge 3 [2699, 8039] innerCandidates390_13 10738 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan390_14

def innerTail390_12 : List Nat := 7589 :: innerTail390_13
def innerCandidates390_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates390_12_checked : candidates edge 4 2699 7589 innerTail390_13 best.bound = innerCandidates390_12 := by
  unfold candidates
  have row : edge 7589 = SearchData.row7589.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan390_12 : scan edge (search edge 3) 4 [2699] 2699 innerTail390_12 best = best := by
  rw [innerTail390_12, scan, if_neg (by decide), innerCandidates390_12_checked]
  change scan edge (search edge 3) 4 [2699] 2699 innerTail390_13
    (if 3 ≤ innerCandidates390_12.length then search edge 3 [2699, 7589] innerCandidates390_12 10288 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan390_13

def innerTail390_11 : List Nat := 7577 :: innerTail390_12
def innerCandidates390_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates390_11_checked : candidates edge 4 2699 7577 innerTail390_12 best.bound = innerCandidates390_11 := by
  unfold candidates
  have row : edge 7577 = SearchData.row7577.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan390_11 : scan edge (search edge 3) 4 [2699] 2699 innerTail390_11 best = best := by
  rw [innerTail390_11, scan, if_neg (by decide), innerCandidates390_11_checked]
  change scan edge (search edge 3) 4 [2699] 2699 innerTail390_12
    (if 3 ≤ innerCandidates390_11.length then search edge 3 [2699, 7577] innerCandidates390_11 10276 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan390_12

def innerTail390_10 : List Nat := 7211 :: innerTail390_11
def innerCandidates390_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates390_10_checked : candidates edge 4 2699 7211 innerTail390_11 best.bound = innerCandidates390_10 := by
  unfold candidates
  have row : edge 7211 = SearchData.row7211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan390_10 : scan edge (search edge 3) 4 [2699] 2699 innerTail390_10 best = best := by
  rw [innerTail390_10, scan, if_neg (by decide), innerCandidates390_10_checked]
  change scan edge (search edge 3) 4 [2699] 2699 innerTail390_11
    (if 3 ≤ innerCandidates390_10.length then search edge 3 [2699, 7211] innerCandidates390_10 9910 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan390_11

def innerTail390_9 : List Nat := 7121 :: innerTail390_10
def innerCandidates390_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates390_9_checked : candidates edge 4 2699 7121 innerTail390_10 best.bound = innerCandidates390_9 := by
  unfold candidates
  have row : edge 7121 = SearchData.row7121.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan390_9 : scan edge (search edge 3) 4 [2699] 2699 innerTail390_9 best = best := by
  rw [innerTail390_9, scan, if_neg (by decide), innerCandidates390_9_checked]
  change scan edge (search edge 3) 4 [2699] 2699 innerTail390_10
    (if 3 ≤ innerCandidates390_9.length then search edge 3 [2699, 7121] innerCandidates390_9 9820 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan390_10

def innerTail390_8 : List Nat := 6353 :: innerTail390_9
def innerCandidates390_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates390_8_checked : candidates edge 4 2699 6353 innerTail390_9 best.bound = innerCandidates390_8 := by
  unfold candidates
  have row : edge 6353 = SearchData.row6353.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan390_8 : scan edge (search edge 3) 4 [2699] 2699 innerTail390_8 best = best := by
  rw [innerTail390_8, scan, if_neg (by decide), innerCandidates390_8_checked]
  change scan edge (search edge 3) 4 [2699] 2699 innerTail390_9
    (if 3 ≤ innerCandidates390_8.length then search edge 3 [2699, 6353] innerCandidates390_8 9052 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan390_9

def innerTail390_7 : List Nat := 6329 :: innerTail390_8
def innerCandidates390_7 : List Nat := [7589]
set_option maxRecDepth 4096 in
theorem innerCandidates390_7_checked : candidates edge 4 2699 6329 innerTail390_8 best.bound = innerCandidates390_7 := by
  unfold candidates
  have row : edge 6329 = SearchData.row6329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan390_7 : scan edge (search edge 3) 4 [2699] 2699 innerTail390_7 best = best := by
  rw [innerTail390_7, scan, if_neg (by decide), innerCandidates390_7_checked]
  change scan edge (search edge 3) 4 [2699] 2699 innerTail390_8
    (if 3 ≤ innerCandidates390_7.length then search edge 3 [2699, 6329] innerCandidates390_7 9028 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan390_8

def innerTail390_6 : List Nat := 5939 :: innerTail390_7
def innerCandidates390_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates390_6_checked : candidates edge 4 2699 5939 innerTail390_7 best.bound = innerCandidates390_6 := by
  unfold candidates
  have row : edge 5939 = SearchData.row5939.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan390_6 : scan edge (search edge 3) 4 [2699] 2699 innerTail390_6 best = best := by
  rw [innerTail390_6, scan, if_neg (by decide), innerCandidates390_6_checked]
  change scan edge (search edge 3) 4 [2699] 2699 innerTail390_7
    (if 3 ≤ innerCandidates390_6.length then search edge 3 [2699, 5939] innerCandidates390_6 8638 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan390_7

def innerTail390_5 : List Nat := 4337 :: innerTail390_6
def innerCandidates390_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates390_5_checked : candidates edge 4 2699 4337 innerTail390_6 best.bound = innerCandidates390_5 := by
  unfold candidates
  have row : edge 4337 = SearchData.row4337.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan390_5 : scan edge (search edge 3) 4 [2699] 2699 innerTail390_5 best = best := by
  rw [innerTail390_5, scan, if_neg (by decide), innerCandidates390_5_checked]
  change scan edge (search edge 3) 4 [2699] 2699 innerTail390_6
    (if 3 ≤ innerCandidates390_5.length then search edge 3 [2699, 4337] innerCandidates390_5 7036 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan390_6

def innerTail390_4 : List Nat := 4133 :: innerTail390_5
def innerCandidates390_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates390_4_checked : candidates edge 4 2699 4133 innerTail390_5 best.bound = innerCandidates390_4 := by
  unfold candidates
  have row : edge 4133 = SearchData.row4133.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan390_4 : scan edge (search edge 3) 4 [2699] 2699 innerTail390_4 best = best := by
  rw [innerTail390_4, scan, if_neg (by decide), innerCandidates390_4_checked]
  change scan edge (search edge 3) 4 [2699] 2699 innerTail390_5
    (if 3 ≤ innerCandidates390_4.length then search edge 3 [2699, 4133] innerCandidates390_4 6832 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan390_5

def innerTail390_3 : List Nat := 3761 :: innerTail390_4
def innerCandidates390_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates390_3_checked : candidates edge 4 2699 3761 innerTail390_4 best.bound = innerCandidates390_3 := by
  unfold candidates
  have row : edge 3761 = SearchData.row3761.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan390_3 : scan edge (search edge 3) 4 [2699] 2699 innerTail390_3 best = best := by
  rw [innerTail390_3, scan, if_neg (by decide), innerCandidates390_3_checked]
  change scan edge (search edge 3) 4 [2699] 2699 innerTail390_4
    (if 3 ≤ innerCandidates390_3.length then search edge 3 [2699, 3761] innerCandidates390_3 6460 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan390_4

def innerTail390_2 : List Nat := 3413 :: innerTail390_3
def innerCandidates390_2 : List Nat := [3761, 5939, 7211]
set_option maxRecDepth 4096 in
theorem innerCandidates390_2_checked : candidates edge 4 2699 3413 innerTail390_3 best.bound = innerCandidates390_2 := by
  unfold candidates
  have row : edge 3413 = SearchData.row3413.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild390_2 : search edge 3 [2699, 3413] innerCandidates390_2 6112 best = best := by decide

theorem innerScan390_2 : scan edge (search edge 3) 4 [2699] 2699 innerTail390_2 best = best := by
  rw [innerTail390_2, scan, if_neg (by decide), innerCandidates390_2_checked]
  change scan edge (search edge 3) 4 [2699] 2699 innerTail390_3
    (if 3 ≤ innerCandidates390_2.length then search edge 3 [2699, 3413] innerCandidates390_2 6112 best else best) = best
  rw [if_pos (by decide), innerChild390_2]
  exact innerScan390_3

def innerTail390_1 : List Nat := 3209 :: innerTail390_2
def innerCandidates390_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates390_1_checked : candidates edge 4 2699 3209 innerTail390_2 best.bound = innerCandidates390_1 := by
  unfold candidates
  have row : edge 3209 = SearchData.row3209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan390_1 : scan edge (search edge 3) 4 [2699] 2699 innerTail390_1 best = best := by
  rw [innerTail390_1, scan, if_neg (by decide), innerCandidates390_1_checked]
  change scan edge (search edge 3) 4 [2699] 2699 innerTail390_2
    (if 3 ≤ innerCandidates390_1.length then search edge 3 [2699, 3209] innerCandidates390_1 5908 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan390_2

def innerTail390_0 : List Nat := 3041 :: innerTail390_1
def innerCandidates390_0 : List Nat := [3209, 4133, 4337]
set_option maxRecDepth 4096 in
theorem innerCandidates390_0_checked : candidates edge 4 2699 3041 innerTail390_1 best.bound = innerCandidates390_0 := by
  unfold candidates
  have row : edge 3041 = SearchData.row3041.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild390_0 : search edge 3 [2699, 3041] innerCandidates390_0 5740 best = best := by decide

theorem innerScan390_0 : scan edge (search edge 3) 4 [2699] 2699 innerTail390_0 best = best := by
  rw [innerTail390_0, scan, if_neg (by decide), innerCandidates390_0_checked]
  change scan edge (search edge 3) 4 [2699] 2699 innerTail390_1
    (if 3 ≤ innerCandidates390_0.length then search edge 3 [2699, 3041] innerCandidates390_0 5740 best else best) = best
  rw [if_pos (by decide), innerChild390_0]
  exact innerScan390_1

theorem rootBranch390 : search edge 4 [2699] rootCandidates390 2699 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2699] 2699 innerTail390_0 best = best
  exact innerScan390_0

def rootCandidates389 : List Nat := [4049, 4079, 4139, 4373, 4931, 5651, 5879, 5903, 6011, 6113, 6257, 6263, 6779, 7013, 7103, 7331, 7901, 8123]
set_option maxRecDepth 4096 in
theorem rootCandidates389_checked : candidates edge 5 0 2693 tail390 best.bound = rootCandidates389 := by
  unfold candidates
  have row : edge 2693 = SearchData.row2693.contains := by funext q; rfl
  rw [row]
  decide

def innerTail389_18 : List Nat := []
theorem innerScan389_18 : scan edge (search edge 3) 4 [2693] 2693 innerTail389_18 best = best := rfl

def innerTail389_17 : List Nat := 8123 :: innerTail389_18
def innerCandidates389_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates389_17_checked : candidates edge 4 2693 8123 innerTail389_18 best.bound = innerCandidates389_17 := by
  unfold candidates
  have row : edge 8123 = SearchData.row8123.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan389_17 : scan edge (search edge 3) 4 [2693] 2693 innerTail389_17 best = best := by
  rw [innerTail389_17, scan, if_neg (by decide), innerCandidates389_17_checked]
  change scan edge (search edge 3) 4 [2693] 2693 innerTail389_18
    (if 3 ≤ innerCandidates389_17.length then search edge 3 [2693, 8123] innerCandidates389_17 10816 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan389_18

def innerTail389_16 : List Nat := 7901 :: innerTail389_17
def innerCandidates389_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates389_16_checked : candidates edge 4 2693 7901 innerTail389_17 best.bound = innerCandidates389_16 := by
  unfold candidates
  have row : edge 7901 = SearchData.row7901.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan389_16 : scan edge (search edge 3) 4 [2693] 2693 innerTail389_16 best = best := by
  rw [innerTail389_16, scan, if_neg (by decide), innerCandidates389_16_checked]
  change scan edge (search edge 3) 4 [2693] 2693 innerTail389_17
    (if 3 ≤ innerCandidates389_16.length then search edge 3 [2693, 7901] innerCandidates389_16 10594 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan389_17

def innerTail389_15 : List Nat := 7331 :: innerTail389_16
def innerCandidates389_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates389_15_checked : candidates edge 4 2693 7331 innerTail389_16 best.bound = innerCandidates389_15 := by
  unfold candidates
  have row : edge 7331 = SearchData.row7331.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan389_15 : scan edge (search edge 3) 4 [2693] 2693 innerTail389_15 best = best := by
  rw [innerTail389_15, scan, if_neg (by decide), innerCandidates389_15_checked]
  change scan edge (search edge 3) 4 [2693] 2693 innerTail389_16
    (if 3 ≤ innerCandidates389_15.length then search edge 3 [2693, 7331] innerCandidates389_15 10024 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan389_16

def innerTail389_14 : List Nat := 7103 :: innerTail389_15
def innerCandidates389_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates389_14_checked : candidates edge 4 2693 7103 innerTail389_15 best.bound = innerCandidates389_14 := by
  unfold candidates
  have row : edge 7103 = SearchData.row7103.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan389_14 : scan edge (search edge 3) 4 [2693] 2693 innerTail389_14 best = best := by
  rw [innerTail389_14, scan, if_neg (by decide), innerCandidates389_14_checked]
  change scan edge (search edge 3) 4 [2693] 2693 innerTail389_15
    (if 3 ≤ innerCandidates389_14.length then search edge 3 [2693, 7103] innerCandidates389_14 9796 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan389_15

def innerTail389_13 : List Nat := 7013 :: innerTail389_14
def innerCandidates389_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates389_13_checked : candidates edge 4 2693 7013 innerTail389_14 best.bound = innerCandidates389_13 := by
  unfold candidates
  have row : edge 7013 = SearchData.row7013.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan389_13 : scan edge (search edge 3) 4 [2693] 2693 innerTail389_13 best = best := by
  rw [innerTail389_13, scan, if_neg (by decide), innerCandidates389_13_checked]
  change scan edge (search edge 3) 4 [2693] 2693 innerTail389_14
    (if 3 ≤ innerCandidates389_13.length then search edge 3 [2693, 7013] innerCandidates389_13 9706 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan389_14

def innerTail389_12 : List Nat := 6779 :: innerTail389_13
def innerCandidates389_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates389_12_checked : candidates edge 4 2693 6779 innerTail389_13 best.bound = innerCandidates389_12 := by
  unfold candidates
  have row : edge 6779 = SearchData.row6779.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan389_12 : scan edge (search edge 3) 4 [2693] 2693 innerTail389_12 best = best := by
  rw [innerTail389_12, scan, if_neg (by decide), innerCandidates389_12_checked]
  change scan edge (search edge 3) 4 [2693] 2693 innerTail389_13
    (if 3 ≤ innerCandidates389_12.length then search edge 3 [2693, 6779] innerCandidates389_12 9472 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan389_13

def innerTail389_11 : List Nat := 6263 :: innerTail389_12
def innerCandidates389_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates389_11_checked : candidates edge 4 2693 6263 innerTail389_12 best.bound = innerCandidates389_11 := by
  unfold candidates
  have row : edge 6263 = SearchData.row6263.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan389_11 : scan edge (search edge 3) 4 [2693] 2693 innerTail389_11 best = best := by
  rw [innerTail389_11, scan, if_neg (by decide), innerCandidates389_11_checked]
  change scan edge (search edge 3) 4 [2693] 2693 innerTail389_12
    (if 3 ≤ innerCandidates389_11.length then search edge 3 [2693, 6263] innerCandidates389_11 8956 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan389_12

def innerTail389_10 : List Nat := 6257 :: innerTail389_11
def innerCandidates389_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates389_10_checked : candidates edge 4 2693 6257 innerTail389_11 best.bound = innerCandidates389_10 := by
  unfold candidates
  have row : edge 6257 = SearchData.row6257.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan389_10 : scan edge (search edge 3) 4 [2693] 2693 innerTail389_10 best = best := by
  rw [innerTail389_10, scan, if_neg (by decide), innerCandidates389_10_checked]
  change scan edge (search edge 3) 4 [2693] 2693 innerTail389_11
    (if 3 ≤ innerCandidates389_10.length then search edge 3 [2693, 6257] innerCandidates389_10 8950 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan389_11

def innerTail389_9 : List Nat := 6113 :: innerTail389_10
def innerCandidates389_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates389_9_checked : candidates edge 4 2693 6113 innerTail389_10 best.bound = innerCandidates389_9 := by
  unfold candidates
  have row : edge 6113 = SearchData.row6113.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan389_9 : scan edge (search edge 3) 4 [2693] 2693 innerTail389_9 best = best := by
  rw [innerTail389_9, scan, if_neg (by decide), innerCandidates389_9_checked]
  change scan edge (search edge 3) 4 [2693] 2693 innerTail389_10
    (if 3 ≤ innerCandidates389_9.length then search edge 3 [2693, 6113] innerCandidates389_9 8806 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan389_10

def innerTail389_8 : List Nat := 6011 :: innerTail389_9
def innerCandidates389_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates389_8_checked : candidates edge 4 2693 6011 innerTail389_9 best.bound = innerCandidates389_8 := by
  unfold candidates
  have row : edge 6011 = SearchData.row6011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan389_8 : scan edge (search edge 3) 4 [2693] 2693 innerTail389_8 best = best := by
  rw [innerTail389_8, scan, if_neg (by decide), innerCandidates389_8_checked]
  change scan edge (search edge 3) 4 [2693] 2693 innerTail389_9
    (if 3 ≤ innerCandidates389_8.length then search edge 3 [2693, 6011] innerCandidates389_8 8704 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan389_9

def innerTail389_7 : List Nat := 5903 :: innerTail389_8
def innerCandidates389_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates389_7_checked : candidates edge 4 2693 5903 innerTail389_8 best.bound = innerCandidates389_7 := by
  unfold candidates
  have row : edge 5903 = SearchData.row5903.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan389_7 : scan edge (search edge 3) 4 [2693] 2693 innerTail389_7 best = best := by
  rw [innerTail389_7, scan, if_neg (by decide), innerCandidates389_7_checked]
  change scan edge (search edge 3) 4 [2693] 2693 innerTail389_8
    (if 3 ≤ innerCandidates389_7.length then search edge 3 [2693, 5903] innerCandidates389_7 8596 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan389_8

def innerTail389_6 : List Nat := 5879 :: innerTail389_7
def innerCandidates389_6 : List Nat := [6263, 6779]
set_option maxRecDepth 4096 in
theorem innerCandidates389_6_checked : candidates edge 4 2693 5879 innerTail389_7 best.bound = innerCandidates389_6 := by
  unfold candidates
  have row : edge 5879 = SearchData.row5879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan389_6 : scan edge (search edge 3) 4 [2693] 2693 innerTail389_6 best = best := by
  rw [innerTail389_6, scan, if_neg (by decide), innerCandidates389_6_checked]
  change scan edge (search edge 3) 4 [2693] 2693 innerTail389_7
    (if 3 ≤ innerCandidates389_6.length then search edge 3 [2693, 5879] innerCandidates389_6 8572 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan389_7

def innerTail389_5 : List Nat := 5651 :: innerTail389_6
def innerCandidates389_5 : List Nat := [6011]
set_option maxRecDepth 4096 in
theorem innerCandidates389_5_checked : candidates edge 4 2693 5651 innerTail389_6 best.bound = innerCandidates389_5 := by
  unfold candidates
  have row : edge 5651 = SearchData.row5651.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan389_5 : scan edge (search edge 3) 4 [2693] 2693 innerTail389_5 best = best := by
  rw [innerTail389_5, scan, if_neg (by decide), innerCandidates389_5_checked]
  change scan edge (search edge 3) 4 [2693] 2693 innerTail389_6
    (if 3 ≤ innerCandidates389_5.length then search edge 3 [2693, 5651] innerCandidates389_5 8344 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan389_6

def innerTail389_4 : List Nat := 4931 :: innerTail389_5
def innerCandidates389_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates389_4_checked : candidates edge 4 2693 4931 innerTail389_5 best.bound = innerCandidates389_4 := by
  unfold candidates
  have row : edge 4931 = SearchData.row4931.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan389_4 : scan edge (search edge 3) 4 [2693] 2693 innerTail389_4 best = best := by
  rw [innerTail389_4, scan, if_neg (by decide), innerCandidates389_4_checked]
  change scan edge (search edge 3) 4 [2693] 2693 innerTail389_5
    (if 3 ≤ innerCandidates389_4.length then search edge 3 [2693, 4931] innerCandidates389_4 7624 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan389_5

def innerTail389_3 : List Nat := 4373 :: innerTail389_4
def innerCandidates389_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates389_3_checked : candidates edge 4 2693 4373 innerTail389_4 best.bound = innerCandidates389_3 := by
  unfold candidates
  have row : edge 4373 = SearchData.row4373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan389_3 : scan edge (search edge 3) 4 [2693] 2693 innerTail389_3 best = best := by
  rw [innerTail389_3, scan, if_neg (by decide), innerCandidates389_3_checked]
  change scan edge (search edge 3) 4 [2693] 2693 innerTail389_4
    (if 3 ≤ innerCandidates389_3.length then search edge 3 [2693, 4373] innerCandidates389_3 7066 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan389_4

def innerTail389_2 : List Nat := 4139 :: innerTail389_3
def innerCandidates389_2 : List Nat := [6263]
set_option maxRecDepth 4096 in
theorem innerCandidates389_2_checked : candidates edge 4 2693 4139 innerTail389_3 best.bound = innerCandidates389_2 := by
  unfold candidates
  have row : edge 4139 = SearchData.row4139.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan389_2 : scan edge (search edge 3) 4 [2693] 2693 innerTail389_2 best = best := by
  rw [innerTail389_2, scan, if_neg (by decide), innerCandidates389_2_checked]
  change scan edge (search edge 3) 4 [2693] 2693 innerTail389_3
    (if 3 ≤ innerCandidates389_2.length then search edge 3 [2693, 4139] innerCandidates389_2 6832 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan389_3

def innerTail389_1 : List Nat := 4079 :: innerTail389_2
def innerCandidates389_1 : List Nat := [5879]
set_option maxRecDepth 4096 in
theorem innerCandidates389_1_checked : candidates edge 4 2693 4079 innerTail389_2 best.bound = innerCandidates389_1 := by
  unfold candidates
  have row : edge 4079 = SearchData.row4079.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan389_1 : scan edge (search edge 3) 4 [2693] 2693 innerTail389_1 best = best := by
  rw [innerTail389_1, scan, if_neg (by decide), innerCandidates389_1_checked]
  change scan edge (search edge 3) 4 [2693] 2693 innerTail389_2
    (if 3 ≤ innerCandidates389_1.length then search edge 3 [2693, 4079] innerCandidates389_1 6772 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan389_2

def innerTail389_0 : List Nat := 4049 :: innerTail389_1
def innerCandidates389_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates389_0_checked : candidates edge 4 2693 4049 innerTail389_1 best.bound = innerCandidates389_0 := by
  unfold candidates
  have row : edge 4049 = SearchData.row4049.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan389_0 : scan edge (search edge 3) 4 [2693] 2693 innerTail389_0 best = best := by
  rw [innerTail389_0, scan, if_neg (by decide), innerCandidates389_0_checked]
  change scan edge (search edge 3) 4 [2693] 2693 innerTail389_1
    (if 3 ≤ innerCandidates389_0.length then search edge 3 [2693, 4049] innerCandidates389_0 6742 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan389_1

theorem rootBranch389 : search edge 4 [2693] rootCandidates389 2693 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2693] 2693 innerTail389_0 best = best
  exact innerScan389_0

def rootCandidates388 : List Nat := [3271, 3343, 4423, 4507, 5347, 6043, 6367, 6379, 6679, 6763, 7477, 8089, 8161, 8191, 8317]
set_option maxRecDepth 4096 in
theorem rootCandidates388_checked : candidates edge 5 0 2689 tail389 best.bound = rootCandidates388 := by
  unfold candidates
  have row : edge 2689 = SearchData.row2689.contains := by funext q; rfl
  rw [row]
  decide

def innerTail388_15 : List Nat := []
theorem innerScan388_15 : scan edge (search edge 3) 4 [2689] 2689 innerTail388_15 best = best := rfl

def innerTail388_14 : List Nat := 8317 :: innerTail388_15
def innerCandidates388_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates388_14_checked : candidates edge 4 2689 8317 innerTail388_15 best.bound = innerCandidates388_14 := by
  unfold candidates
  have row : edge 8317 = SearchData.row8317.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan388_14 : scan edge (search edge 3) 4 [2689] 2689 innerTail388_14 best = best := by
  rw [innerTail388_14, scan, if_neg (by decide), innerCandidates388_14_checked]
  change scan edge (search edge 3) 4 [2689] 2689 innerTail388_15
    (if 3 ≤ innerCandidates388_14.length then search edge 3 [2689, 8317] innerCandidates388_14 11006 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan388_15

def innerTail388_13 : List Nat := 8191 :: innerTail388_14
def innerCandidates388_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates388_13_checked : candidates edge 4 2689 8191 innerTail388_14 best.bound = innerCandidates388_13 := by
  unfold candidates
  have row : edge 8191 = SearchData.row8191.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan388_13 : scan edge (search edge 3) 4 [2689] 2689 innerTail388_13 best = best := by
  rw [innerTail388_13, scan, if_neg (by decide), innerCandidates388_13_checked]
  change scan edge (search edge 3) 4 [2689] 2689 innerTail388_14
    (if 3 ≤ innerCandidates388_13.length then search edge 3 [2689, 8191] innerCandidates388_13 10880 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan388_14

def innerTail388_12 : List Nat := 8161 :: innerTail388_13
def innerCandidates388_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates388_12_checked : candidates edge 4 2689 8161 innerTail388_13 best.bound = innerCandidates388_12 := by
  unfold candidates
  have row : edge 8161 = SearchData.row8161.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan388_12 : scan edge (search edge 3) 4 [2689] 2689 innerTail388_12 best = best := by
  rw [innerTail388_12, scan, if_neg (by decide), innerCandidates388_12_checked]
  change scan edge (search edge 3) 4 [2689] 2689 innerTail388_13
    (if 3 ≤ innerCandidates388_12.length then search edge 3 [2689, 8161] innerCandidates388_12 10850 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan388_13

def innerTail388_11 : List Nat := 8089 :: innerTail388_12
def innerCandidates388_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates388_11_checked : candidates edge 4 2689 8089 innerTail388_12 best.bound = innerCandidates388_11 := by
  unfold candidates
  have row : edge 8089 = SearchData.row8089.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan388_11 : scan edge (search edge 3) 4 [2689] 2689 innerTail388_11 best = best := by
  rw [innerTail388_11, scan, if_neg (by decide), innerCandidates388_11_checked]
  change scan edge (search edge 3) 4 [2689] 2689 innerTail388_12
    (if 3 ≤ innerCandidates388_11.length then search edge 3 [2689, 8089] innerCandidates388_11 10778 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan388_12

def innerTail388_10 : List Nat := 7477 :: innerTail388_11
def innerCandidates388_10 : List Nat := [8191, 8317]
set_option maxRecDepth 4096 in
theorem innerCandidates388_10_checked : candidates edge 4 2689 7477 innerTail388_11 best.bound = innerCandidates388_10 := by
  unfold candidates
  have row : edge 7477 = SearchData.row7477.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan388_10 : scan edge (search edge 3) 4 [2689] 2689 innerTail388_10 best = best := by
  rw [innerTail388_10, scan, if_neg (by decide), innerCandidates388_10_checked]
  change scan edge (search edge 3) 4 [2689] 2689 innerTail388_11
    (if 3 ≤ innerCandidates388_10.length then search edge 3 [2689, 7477] innerCandidates388_10 10166 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan388_11

def innerTail388_9 : List Nat := 6763 :: innerTail388_10
def innerCandidates388_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates388_9_checked : candidates edge 4 2689 6763 innerTail388_10 best.bound = innerCandidates388_9 := by
  unfold candidates
  have row : edge 6763 = SearchData.row6763.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan388_9 : scan edge (search edge 3) 4 [2689] 2689 innerTail388_9 best = best := by
  rw [innerTail388_9, scan, if_neg (by decide), innerCandidates388_9_checked]
  change scan edge (search edge 3) 4 [2689] 2689 innerTail388_10
    (if 3 ≤ innerCandidates388_9.length then search edge 3 [2689, 6763] innerCandidates388_9 9452 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan388_10

def innerTail388_8 : List Nat := 6679 :: innerTail388_9
def innerCandidates388_8 : List Nat := [8161]
set_option maxRecDepth 4096 in
theorem innerCandidates388_8_checked : candidates edge 4 2689 6679 innerTail388_9 best.bound = innerCandidates388_8 := by
  unfold candidates
  have row : edge 6679 = SearchData.row6679.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan388_8 : scan edge (search edge 3) 4 [2689] 2689 innerTail388_8 best = best := by
  rw [innerTail388_8, scan, if_neg (by decide), innerCandidates388_8_checked]
  change scan edge (search edge 3) 4 [2689] 2689 innerTail388_9
    (if 3 ≤ innerCandidates388_8.length then search edge 3 [2689, 6679] innerCandidates388_8 9368 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan388_9

def innerTail388_7 : List Nat := 6379 :: innerTail388_8
def innerCandidates388_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates388_7_checked : candidates edge 4 2689 6379 innerTail388_8 best.bound = innerCandidates388_7 := by
  unfold candidates
  have row : edge 6379 = SearchData.row6379.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan388_7 : scan edge (search edge 3) 4 [2689] 2689 innerTail388_7 best = best := by
  rw [innerTail388_7, scan, if_neg (by decide), innerCandidates388_7_checked]
  change scan edge (search edge 3) 4 [2689] 2689 innerTail388_8
    (if 3 ≤ innerCandidates388_7.length then search edge 3 [2689, 6379] innerCandidates388_7 9068 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan388_8

def innerTail388_6 : List Nat := 6367 :: innerTail388_7
def innerCandidates388_6 : List Nat := [7477]
set_option maxRecDepth 4096 in
theorem innerCandidates388_6_checked : candidates edge 4 2689 6367 innerTail388_7 best.bound = innerCandidates388_6 := by
  unfold candidates
  have row : edge 6367 = SearchData.row6367.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan388_6 : scan edge (search edge 3) 4 [2689] 2689 innerTail388_6 best = best := by
  rw [innerTail388_6, scan, if_neg (by decide), innerCandidates388_6_checked]
  change scan edge (search edge 3) 4 [2689] 2689 innerTail388_7
    (if 3 ≤ innerCandidates388_6.length then search edge 3 [2689, 6367] innerCandidates388_6 9056 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan388_7

def innerTail388_5 : List Nat := 6043 :: innerTail388_6
def innerCandidates388_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates388_5_checked : candidates edge 4 2689 6043 innerTail388_6 best.bound = innerCandidates388_5 := by
  unfold candidates
  have row : edge 6043 = SearchData.row6043.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan388_5 : scan edge (search edge 3) 4 [2689] 2689 innerTail388_5 best = best := by
  rw [innerTail388_5, scan, if_neg (by decide), innerCandidates388_5_checked]
  change scan edge (search edge 3) 4 [2689] 2689 innerTail388_6
    (if 3 ≤ innerCandidates388_5.length then search edge 3 [2689, 6043] innerCandidates388_5 8732 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan388_6

def innerTail388_4 : List Nat := 5347 :: innerTail388_5
def innerCandidates388_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates388_4_checked : candidates edge 4 2689 5347 innerTail388_5 best.bound = innerCandidates388_4 := by
  unfold candidates
  have row : edge 5347 = SearchData.row5347.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan388_4 : scan edge (search edge 3) 4 [2689] 2689 innerTail388_4 best = best := by
  rw [innerTail388_4, scan, if_neg (by decide), innerCandidates388_4_checked]
  change scan edge (search edge 3) 4 [2689] 2689 innerTail388_5
    (if 3 ≤ innerCandidates388_4.length then search edge 3 [2689, 5347] innerCandidates388_4 8036 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan388_5

def innerTail388_3 : List Nat := 4507 :: innerTail388_4
def innerCandidates388_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates388_3_checked : candidates edge 4 2689 4507 innerTail388_4 best.bound = innerCandidates388_3 := by
  unfold candidates
  have row : edge 4507 = SearchData.row4507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan388_3 : scan edge (search edge 3) 4 [2689] 2689 innerTail388_3 best = best := by
  rw [innerTail388_3, scan, if_neg (by decide), innerCandidates388_3_checked]
  change scan edge (search edge 3) 4 [2689] 2689 innerTail388_4
    (if 3 ≤ innerCandidates388_3.length then search edge 3 [2689, 4507] innerCandidates388_3 7196 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan388_4

def innerTail388_2 : List Nat := 4423 :: innerTail388_3
def innerCandidates388_2 : List Nat := [8317]
set_option maxRecDepth 4096 in
theorem innerCandidates388_2_checked : candidates edge 4 2689 4423 innerTail388_3 best.bound = innerCandidates388_2 := by
  unfold candidates
  have row : edge 4423 = SearchData.row4423.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan388_2 : scan edge (search edge 3) 4 [2689] 2689 innerTail388_2 best = best := by
  rw [innerTail388_2, scan, if_neg (by decide), innerCandidates388_2_checked]
  change scan edge (search edge 3) 4 [2689] 2689 innerTail388_3
    (if 3 ≤ innerCandidates388_2.length then search edge 3 [2689, 4423] innerCandidates388_2 7112 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan388_3

def innerTail388_1 : List Nat := 3343 :: innerTail388_2
def innerCandidates388_1 : List Nat := [6379]
set_option maxRecDepth 4096 in
theorem innerCandidates388_1_checked : candidates edge 4 2689 3343 innerTail388_2 best.bound = innerCandidates388_1 := by
  unfold candidates
  have row : edge 3343 = SearchData.row3343.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan388_1 : scan edge (search edge 3) 4 [2689] 2689 innerTail388_1 best = best := by
  rw [innerTail388_1, scan, if_neg (by decide), innerCandidates388_1_checked]
  change scan edge (search edge 3) 4 [2689] 2689 innerTail388_2
    (if 3 ≤ innerCandidates388_1.length then search edge 3 [2689, 3343] innerCandidates388_1 6032 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan388_2

def innerTail388_0 : List Nat := 3271 :: innerTail388_1
def innerCandidates388_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates388_0_checked : candidates edge 4 2689 3271 innerTail388_1 best.bound = innerCandidates388_0 := by
  unfold candidates
  have row : edge 3271 = SearchData.row3271.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan388_0 : scan edge (search edge 3) 4 [2689] 2689 innerTail388_0 best = best := by
  rw [innerTail388_0, scan, if_neg (by decide), innerCandidates388_0_checked]
  change scan edge (search edge 3) 4 [2689] 2689 innerTail388_1
    (if 3 ≤ innerCandidates388_0.length then search edge 3 [2689, 3271] innerCandidates388_0 5960 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan388_1

theorem rootBranch388 : search edge 4 [2689] rootCandidates388 2689 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2689] 2689 innerTail388_0 best = best
  exact innerScan388_0

def rootCandidates387 : List Nat := [3023, 4007, 4481, 5237, 5897, 6221, 6599, 6737, 7703, 8039, 8069, 8231, 8363]
set_option maxRecDepth 4096 in
theorem rootCandidates387_checked : candidates edge 5 0 2687 tail388 best.bound = rootCandidates387 := by
  unfold candidates
  have row : edge 2687 = SearchData.row2687.contains := by funext q; rfl
  rw [row]
  decide

def innerTail387_13 : List Nat := []
theorem innerScan387_13 : scan edge (search edge 3) 4 [2687] 2687 innerTail387_13 best = best := rfl

def innerTail387_12 : List Nat := 8363 :: innerTail387_13
def innerCandidates387_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates387_12_checked : candidates edge 4 2687 8363 innerTail387_13 best.bound = innerCandidates387_12 := by
  unfold candidates
  have row : edge 8363 = SearchData.row8363.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan387_12 : scan edge (search edge 3) 4 [2687] 2687 innerTail387_12 best = best := by
  rw [innerTail387_12, scan, if_neg (by decide), innerCandidates387_12_checked]
  change scan edge (search edge 3) 4 [2687] 2687 innerTail387_13
    (if 3 ≤ innerCandidates387_12.length then search edge 3 [2687, 8363] innerCandidates387_12 11050 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan387_13

def innerTail387_11 : List Nat := 8231 :: innerTail387_12
def innerCandidates387_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates387_11_checked : candidates edge 4 2687 8231 innerTail387_12 best.bound = innerCandidates387_11 := by
  unfold candidates
  have row : edge 8231 = SearchData.row8231.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan387_11 : scan edge (search edge 3) 4 [2687] 2687 innerTail387_11 best = best := by
  rw [innerTail387_11, scan, if_neg (by decide), innerCandidates387_11_checked]
  change scan edge (search edge 3) 4 [2687] 2687 innerTail387_12
    (if 3 ≤ innerCandidates387_11.length then search edge 3 [2687, 8231] innerCandidates387_11 10918 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan387_12

def innerTail387_10 : List Nat := 8069 :: innerTail387_11
def innerCandidates387_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates387_10_checked : candidates edge 4 2687 8069 innerTail387_11 best.bound = innerCandidates387_10 := by
  unfold candidates
  have row : edge 8069 = SearchData.row8069.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan387_10 : scan edge (search edge 3) 4 [2687] 2687 innerTail387_10 best = best := by
  rw [innerTail387_10, scan, if_neg (by decide), innerCandidates387_10_checked]
  change scan edge (search edge 3) 4 [2687] 2687 innerTail387_11
    (if 3 ≤ innerCandidates387_10.length then search edge 3 [2687, 8069] innerCandidates387_10 10756 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan387_11

def innerTail387_9 : List Nat := 8039 :: innerTail387_10
def innerCandidates387_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates387_9_checked : candidates edge 4 2687 8039 innerTail387_10 best.bound = innerCandidates387_9 := by
  unfold candidates
  have row : edge 8039 = SearchData.row8039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan387_9 : scan edge (search edge 3) 4 [2687] 2687 innerTail387_9 best = best := by
  rw [innerTail387_9, scan, if_neg (by decide), innerCandidates387_9_checked]
  change scan edge (search edge 3) 4 [2687] 2687 innerTail387_10
    (if 3 ≤ innerCandidates387_9.length then search edge 3 [2687, 8039] innerCandidates387_9 10726 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan387_10

def innerTail387_8 : List Nat := 7703 :: innerTail387_9
def innerCandidates387_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates387_8_checked : candidates edge 4 2687 7703 innerTail387_9 best.bound = innerCandidates387_8 := by
  unfold candidates
  have row : edge 7703 = SearchData.row7703.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan387_8 : scan edge (search edge 3) 4 [2687] 2687 innerTail387_8 best = best := by
  rw [innerTail387_8, scan, if_neg (by decide), innerCandidates387_8_checked]
  change scan edge (search edge 3) 4 [2687] 2687 innerTail387_9
    (if 3 ≤ innerCandidates387_8.length then search edge 3 [2687, 7703] innerCandidates387_8 10390 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan387_9

def innerTail387_7 : List Nat := 6737 :: innerTail387_8
def innerCandidates387_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates387_7_checked : candidates edge 4 2687 6737 innerTail387_8 best.bound = innerCandidates387_7 := by
  unfold candidates
  have row : edge 6737 = SearchData.row6737.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan387_7 : scan edge (search edge 3) 4 [2687] 2687 innerTail387_7 best = best := by
  rw [innerTail387_7, scan, if_neg (by decide), innerCandidates387_7_checked]
  change scan edge (search edge 3) 4 [2687] 2687 innerTail387_8
    (if 3 ≤ innerCandidates387_7.length then search edge 3 [2687, 6737] innerCandidates387_7 9424 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan387_8

def innerTail387_6 : List Nat := 6599 :: innerTail387_7
def innerCandidates387_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates387_6_checked : candidates edge 4 2687 6599 innerTail387_7 best.bound = innerCandidates387_6 := by
  unfold candidates
  have row : edge 6599 = SearchData.row6599.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan387_6 : scan edge (search edge 3) 4 [2687] 2687 innerTail387_6 best = best := by
  rw [innerTail387_6, scan, if_neg (by decide), innerCandidates387_6_checked]
  change scan edge (search edge 3) 4 [2687] 2687 innerTail387_7
    (if 3 ≤ innerCandidates387_6.length then search edge 3 [2687, 6599] innerCandidates387_6 9286 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan387_7

def innerTail387_5 : List Nat := 6221 :: innerTail387_6
def innerCandidates387_5 : List Nat := [6599, 8363]
set_option maxRecDepth 4096 in
theorem innerCandidates387_5_checked : candidates edge 4 2687 6221 innerTail387_6 best.bound = innerCandidates387_5 := by
  unfold candidates
  have row : edge 6221 = SearchData.row6221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan387_5 : scan edge (search edge 3) 4 [2687] 2687 innerTail387_5 best = best := by
  rw [innerTail387_5, scan, if_neg (by decide), innerCandidates387_5_checked]
  change scan edge (search edge 3) 4 [2687] 2687 innerTail387_6
    (if 3 ≤ innerCandidates387_5.length then search edge 3 [2687, 6221] innerCandidates387_5 8908 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan387_6

def innerTail387_4 : List Nat := 5897 :: innerTail387_5
def innerCandidates387_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates387_4_checked : candidates edge 4 2687 5897 innerTail387_5 best.bound = innerCandidates387_4 := by
  unfold candidates
  have row : edge 5897 = SearchData.row5897.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan387_4 : scan edge (search edge 3) 4 [2687] 2687 innerTail387_4 best = best := by
  rw [innerTail387_4, scan, if_neg (by decide), innerCandidates387_4_checked]
  change scan edge (search edge 3) 4 [2687] 2687 innerTail387_5
    (if 3 ≤ innerCandidates387_4.length then search edge 3 [2687, 5897] innerCandidates387_4 8584 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan387_5

def innerTail387_3 : List Nat := 5237 :: innerTail387_4
def innerCandidates387_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates387_3_checked : candidates edge 4 2687 5237 innerTail387_4 best.bound = innerCandidates387_3 := by
  unfold candidates
  have row : edge 5237 = SearchData.row5237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan387_3 : scan edge (search edge 3) 4 [2687] 2687 innerTail387_3 best = best := by
  rw [innerTail387_3, scan, if_neg (by decide), innerCandidates387_3_checked]
  change scan edge (search edge 3) 4 [2687] 2687 innerTail387_4
    (if 3 ≤ innerCandidates387_3.length then search edge 3 [2687, 5237] innerCandidates387_3 7924 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan387_4

def innerTail387_2 : List Nat := 4481 :: innerTail387_3
def innerCandidates387_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates387_2_checked : candidates edge 4 2687 4481 innerTail387_3 best.bound = innerCandidates387_2 := by
  unfold candidates
  have row : edge 4481 = SearchData.row4481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan387_2 : scan edge (search edge 3) 4 [2687] 2687 innerTail387_2 best = best := by
  rw [innerTail387_2, scan, if_neg (by decide), innerCandidates387_2_checked]
  change scan edge (search edge 3) 4 [2687] 2687 innerTail387_3
    (if 3 ≤ innerCandidates387_2.length then search edge 3 [2687, 4481] innerCandidates387_2 7168 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan387_3

def innerTail387_1 : List Nat := 4007 :: innerTail387_2
def innerCandidates387_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates387_1_checked : candidates edge 4 2687 4007 innerTail387_2 best.bound = innerCandidates387_1 := by
  unfold candidates
  have row : edge 4007 = SearchData.row4007.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan387_1 : scan edge (search edge 3) 4 [2687] 2687 innerTail387_1 best = best := by
  rw [innerTail387_1, scan, if_neg (by decide), innerCandidates387_1_checked]
  change scan edge (search edge 3) 4 [2687] 2687 innerTail387_2
    (if 3 ≤ innerCandidates387_1.length then search edge 3 [2687, 4007] innerCandidates387_1 6694 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan387_2

def innerTail387_0 : List Nat := 3023 :: innerTail387_1
def innerCandidates387_0 : List Nat := [8069]
set_option maxRecDepth 4096 in
theorem innerCandidates387_0_checked : candidates edge 4 2687 3023 innerTail387_1 best.bound = innerCandidates387_0 := by
  unfold candidates
  have row : edge 3023 = SearchData.row3023.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan387_0 : scan edge (search edge 3) 4 [2687] 2687 innerTail387_0 best = best := by
  rw [innerTail387_0, scan, if_neg (by decide), innerCandidates387_0_checked]
  change scan edge (search edge 3) 4 [2687] 2687 innerTail387_1
    (if 3 ≤ innerCandidates387_0.length then search edge 3 [2687, 3023] innerCandidates387_0 5710 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan387_1

theorem rootBranch387 : search edge 4 [2687] rootCandidates387 2687 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2687] 2687 innerTail387_0 best = best
  exact innerScan387_0

end ProjectEulerSolutions.PrimePairUpper.Search
