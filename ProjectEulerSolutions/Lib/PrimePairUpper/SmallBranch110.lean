import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates356 : List Nat := [2741, 3449, 4001, 4229, 4253, 5153, 5303, 5903, 5927, 6389, 6563, 6599, 6857, 7019, 7103, 7691, 7877, 8237]
set_option maxRecDepth 4096 in
theorem rootCandidates356_checked : candidates edge 5 0 2417 tail357 best.bound = rootCandidates356 := by
  unfold candidates
  have row : edge 2417 = PrimePairCache.smallRow2417.contains := by funext q; rfl
  rw [row]
  decide

def innerTail356_18 : List Nat := []
theorem innerScan356_18 : scan edge (search edge 3) 4 [2417] 2417 innerTail356_18 best = best := rfl

def innerTail356_17 : List Nat := 8237 :: innerTail356_18
def innerCandidates356_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates356_17_checked : candidates edge 4 2417 8237 innerTail356_18 best.bound = innerCandidates356_17 := by
  unfold candidates
  have row : edge 8237 = SearchData.row8237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan356_17 : scan edge (search edge 3) 4 [2417] 2417 innerTail356_17 best = best := by
  rw [innerTail356_17, scan, if_neg (by decide), innerCandidates356_17_checked]
  change scan edge (search edge 3) 4 [2417] 2417 innerTail356_18
    (if 3 ≤ innerCandidates356_17.length then search edge 3 [2417, 8237] innerCandidates356_17 10654 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan356_18

def innerTail356_16 : List Nat := 7877 :: innerTail356_17
def innerCandidates356_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates356_16_checked : candidates edge 4 2417 7877 innerTail356_17 best.bound = innerCandidates356_16 := by
  unfold candidates
  have row : edge 7877 = SearchData.row7877.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan356_16 : scan edge (search edge 3) 4 [2417] 2417 innerTail356_16 best = best := by
  rw [innerTail356_16, scan, if_neg (by decide), innerCandidates356_16_checked]
  change scan edge (search edge 3) 4 [2417] 2417 innerTail356_17
    (if 3 ≤ innerCandidates356_16.length then search edge 3 [2417, 7877] innerCandidates356_16 10294 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan356_17

def innerTail356_15 : List Nat := 7691 :: innerTail356_16
def innerCandidates356_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates356_15_checked : candidates edge 4 2417 7691 innerTail356_16 best.bound = innerCandidates356_15 := by
  unfold candidates
  have row : edge 7691 = SearchData.row7691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan356_15 : scan edge (search edge 3) 4 [2417] 2417 innerTail356_15 best = best := by
  rw [innerTail356_15, scan, if_neg (by decide), innerCandidates356_15_checked]
  change scan edge (search edge 3) 4 [2417] 2417 innerTail356_16
    (if 3 ≤ innerCandidates356_15.length then search edge 3 [2417, 7691] innerCandidates356_15 10108 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan356_16

def innerTail356_14 : List Nat := 7103 :: innerTail356_15
def innerCandidates356_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates356_14_checked : candidates edge 4 2417 7103 innerTail356_15 best.bound = innerCandidates356_14 := by
  unfold candidates
  have row : edge 7103 = SearchData.row7103.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan356_14 : scan edge (search edge 3) 4 [2417] 2417 innerTail356_14 best = best := by
  rw [innerTail356_14, scan, if_neg (by decide), innerCandidates356_14_checked]
  change scan edge (search edge 3) 4 [2417] 2417 innerTail356_15
    (if 3 ≤ innerCandidates356_14.length then search edge 3 [2417, 7103] innerCandidates356_14 9520 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan356_15

def innerTail356_13 : List Nat := 7019 :: innerTail356_14
def innerCandidates356_13 : List Nat := [7877]
set_option maxRecDepth 4096 in
theorem innerCandidates356_13_checked : candidates edge 4 2417 7019 innerTail356_14 best.bound = innerCandidates356_13 := by
  unfold candidates
  have row : edge 7019 = SearchData.row7019.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan356_13 : scan edge (search edge 3) 4 [2417] 2417 innerTail356_13 best = best := by
  rw [innerTail356_13, scan, if_neg (by decide), innerCandidates356_13_checked]
  change scan edge (search edge 3) 4 [2417] 2417 innerTail356_14
    (if 3 ≤ innerCandidates356_13.length then search edge 3 [2417, 7019] innerCandidates356_13 9436 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan356_14

def innerTail356_12 : List Nat := 6857 :: innerTail356_13
def innerCandidates356_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates356_12_checked : candidates edge 4 2417 6857 innerTail356_13 best.bound = innerCandidates356_12 := by
  unfold candidates
  have row : edge 6857 = SearchData.row6857.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan356_12 : scan edge (search edge 3) 4 [2417] 2417 innerTail356_12 best = best := by
  rw [innerTail356_12, scan, if_neg (by decide), innerCandidates356_12_checked]
  change scan edge (search edge 3) 4 [2417] 2417 innerTail356_13
    (if 3 ≤ innerCandidates356_12.length then search edge 3 [2417, 6857] innerCandidates356_12 9274 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan356_13

def innerTail356_11 : List Nat := 6599 :: innerTail356_12
def innerCandidates356_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates356_11_checked : candidates edge 4 2417 6599 innerTail356_12 best.bound = innerCandidates356_11 := by
  unfold candidates
  have row : edge 6599 = SearchData.row6599.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan356_11 : scan edge (search edge 3) 4 [2417] 2417 innerTail356_11 best = best := by
  rw [innerTail356_11, scan, if_neg (by decide), innerCandidates356_11_checked]
  change scan edge (search edge 3) 4 [2417] 2417 innerTail356_12
    (if 3 ≤ innerCandidates356_11.length then search edge 3 [2417, 6599] innerCandidates356_11 9016 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan356_12

def innerTail356_10 : List Nat := 6563 :: innerTail356_11
def innerCandidates356_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates356_10_checked : candidates edge 4 2417 6563 innerTail356_11 best.bound = innerCandidates356_10 := by
  unfold candidates
  have row : edge 6563 = SearchData.row6563.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan356_10 : scan edge (search edge 3) 4 [2417] 2417 innerTail356_10 best = best := by
  rw [innerTail356_10, scan, if_neg (by decide), innerCandidates356_10_checked]
  change scan edge (search edge 3) 4 [2417] 2417 innerTail356_11
    (if 3 ≤ innerCandidates356_10.length then search edge 3 [2417, 6563] innerCandidates356_10 8980 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan356_11

def innerTail356_9 : List Nat := 6389 :: innerTail356_10
def innerCandidates356_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates356_9_checked : candidates edge 4 2417 6389 innerTail356_10 best.bound = innerCandidates356_9 := by
  unfold candidates
  have row : edge 6389 = SearchData.row6389.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan356_9 : scan edge (search edge 3) 4 [2417] 2417 innerTail356_9 best = best := by
  rw [innerTail356_9, scan, if_neg (by decide), innerCandidates356_9_checked]
  change scan edge (search edge 3) 4 [2417] 2417 innerTail356_10
    (if 3 ≤ innerCandidates356_9.length then search edge 3 [2417, 6389] innerCandidates356_9 8806 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan356_10

def innerTail356_8 : List Nat := 5927 :: innerTail356_9
def innerCandidates356_8 : List Nat := [8237]
set_option maxRecDepth 4096 in
theorem innerCandidates356_8_checked : candidates edge 4 2417 5927 innerTail356_9 best.bound = innerCandidates356_8 := by
  unfold candidates
  have row : edge 5927 = SearchData.row5927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan356_8 : scan edge (search edge 3) 4 [2417] 2417 innerTail356_8 best = best := by
  rw [innerTail356_8, scan, if_neg (by decide), innerCandidates356_8_checked]
  change scan edge (search edge 3) 4 [2417] 2417 innerTail356_9
    (if 3 ≤ innerCandidates356_8.length then search edge 3 [2417, 5927] innerCandidates356_8 8344 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan356_9

def innerTail356_7 : List Nat := 5903 :: innerTail356_8
def innerCandidates356_7 : List Nat := [7877]
set_option maxRecDepth 4096 in
theorem innerCandidates356_7_checked : candidates edge 4 2417 5903 innerTail356_8 best.bound = innerCandidates356_7 := by
  unfold candidates
  have row : edge 5903 = SearchData.row5903.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan356_7 : scan edge (search edge 3) 4 [2417] 2417 innerTail356_7 best = best := by
  rw [innerTail356_7, scan, if_neg (by decide), innerCandidates356_7_checked]
  change scan edge (search edge 3) 4 [2417] 2417 innerTail356_8
    (if 3 ≤ innerCandidates356_7.length then search edge 3 [2417, 5903] innerCandidates356_7 8320 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan356_8

def innerTail356_6 : List Nat := 5303 :: innerTail356_7
def innerCandidates356_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates356_6_checked : candidates edge 4 2417 5303 innerTail356_7 best.bound = innerCandidates356_6 := by
  unfold candidates
  have row : edge 5303 = SearchData.row5303.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan356_6 : scan edge (search edge 3) 4 [2417] 2417 innerTail356_6 best = best := by
  rw [innerTail356_6, scan, if_neg (by decide), innerCandidates356_6_checked]
  change scan edge (search edge 3) 4 [2417] 2417 innerTail356_7
    (if 3 ≤ innerCandidates356_6.length then search edge 3 [2417, 5303] innerCandidates356_6 7720 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan356_7

def innerTail356_5 : List Nat := 5153 :: innerTail356_6
def innerCandidates356_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates356_5_checked : candidates edge 4 2417 5153 innerTail356_6 best.bound = innerCandidates356_5 := by
  unfold candidates
  have row : edge 5153 = SearchData.row5153.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan356_5 : scan edge (search edge 3) 4 [2417] 2417 innerTail356_5 best = best := by
  rw [innerTail356_5, scan, if_neg (by decide), innerCandidates356_5_checked]
  change scan edge (search edge 3) 4 [2417] 2417 innerTail356_6
    (if 3 ≤ innerCandidates356_5.length then search edge 3 [2417, 5153] innerCandidates356_5 7570 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan356_6

def innerTail356_4 : List Nat := 4253 :: innerTail356_5
def innerCandidates356_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates356_4_checked : candidates edge 4 2417 4253 innerTail356_5 best.bound = innerCandidates356_4 := by
  unfold candidates
  have row : edge 4253 = SearchData.row4253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan356_4 : scan edge (search edge 3) 4 [2417] 2417 innerTail356_4 best = best := by
  rw [innerTail356_4, scan, if_neg (by decide), innerCandidates356_4_checked]
  change scan edge (search edge 3) 4 [2417] 2417 innerTail356_5
    (if 3 ≤ innerCandidates356_4.length then search edge 3 [2417, 4253] innerCandidates356_4 6670 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan356_5

def innerTail356_3 : List Nat := 4229 :: innerTail356_4
def innerCandidates356_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates356_3_checked : candidates edge 4 2417 4229 innerTail356_4 best.bound = innerCandidates356_3 := by
  unfold candidates
  have row : edge 4229 = SearchData.row4229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan356_3 : scan edge (search edge 3) 4 [2417] 2417 innerTail356_3 best = best := by
  rw [innerTail356_3, scan, if_neg (by decide), innerCandidates356_3_checked]
  change scan edge (search edge 3) 4 [2417] 2417 innerTail356_4
    (if 3 ≤ innerCandidates356_3.length then search edge 3 [2417, 4229] innerCandidates356_3 6646 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan356_4

def innerTail356_2 : List Nat := 4001 :: innerTail356_3
def innerCandidates356_2 : List Nat := [4253, 6563, 7877]
set_option maxRecDepth 4096 in
theorem innerCandidates356_2_checked : candidates edge 4 2417 4001 innerTail356_3 best.bound = innerCandidates356_2 := by
  unfold candidates
  have row : edge 4001 = SearchData.row4001.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild356_2 : search edge 3 [2417, 4001] innerCandidates356_2 6418 best = best := by decide

theorem innerScan356_2 : scan edge (search edge 3) 4 [2417] 2417 innerTail356_2 best = best := by
  rw [innerTail356_2, scan, if_neg (by decide), innerCandidates356_2_checked]
  change scan edge (search edge 3) 4 [2417] 2417 innerTail356_3
    (if 3 ≤ innerCandidates356_2.length then search edge 3 [2417, 4001] innerCandidates356_2 6418 best else best) = best
  rw [if_pos (by decide), innerChild356_2]
  exact innerScan356_3

def innerTail356_1 : List Nat := 3449 :: innerTail356_2
def innerCandidates356_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates356_1_checked : candidates edge 4 2417 3449 innerTail356_2 best.bound = innerCandidates356_1 := by
  unfold candidates
  have row : edge 3449 = SearchData.row3449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan356_1 : scan edge (search edge 3) 4 [2417] 2417 innerTail356_1 best = best := by
  rw [innerTail356_1, scan, if_neg (by decide), innerCandidates356_1_checked]
  change scan edge (search edge 3) 4 [2417] 2417 innerTail356_2
    (if 3 ≤ innerCandidates356_1.length then search edge 3 [2417, 3449] innerCandidates356_1 5866 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan356_2

def innerTail356_0 : List Nat := 2741 :: innerTail356_1
def innerCandidates356_0 : List Nat := [4253]
set_option maxRecDepth 4096 in
theorem innerCandidates356_0_checked : candidates edge 4 2417 2741 innerTail356_1 best.bound = innerCandidates356_0 := by
  unfold candidates
  have row : edge 2741 = SearchData.row2741.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan356_0 : scan edge (search edge 3) 4 [2417] 2417 innerTail356_0 best = best := by
  rw [innerTail356_0, scan, if_neg (by decide), innerCandidates356_0_checked]
  change scan edge (search edge 3) 4 [2417] 2417 innerTail356_1
    (if 3 ≤ innerCandidates356_0.length then search edge 3 [2417, 2741] innerCandidates356_0 5158 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan356_1

theorem rootBranch356 : search edge 4 [2417] rootCandidates356 2417 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2417] 2417 innerTail356_0 best = best
  exact innerScan356_0

def rootCandidates355 : List Nat := [3917, 4241, 4253, 4793, 4919, 5171, 5519, 5879, 7019, 7433, 7481, 8273]
set_option maxRecDepth 4096 in
theorem rootCandidates355_checked : candidates edge 5 0 2411 tail356 best.bound = rootCandidates355 := by
  unfold candidates
  have row : edge 2411 = PrimePairCache.smallRow2411.contains := by funext q; rfl
  rw [row]
  decide

def innerTail355_12 : List Nat := []
theorem innerScan355_12 : scan edge (search edge 3) 4 [2411] 2411 innerTail355_12 best = best := rfl

def innerTail355_11 : List Nat := 8273 :: innerTail355_12
def innerCandidates355_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates355_11_checked : candidates edge 4 2411 8273 innerTail355_12 best.bound = innerCandidates355_11 := by
  unfold candidates
  have row : edge 8273 = SearchData.row8273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan355_11 : scan edge (search edge 3) 4 [2411] 2411 innerTail355_11 best = best := by
  rw [innerTail355_11, scan, if_neg (by decide), innerCandidates355_11_checked]
  change scan edge (search edge 3) 4 [2411] 2411 innerTail355_12
    (if 3 ≤ innerCandidates355_11.length then search edge 3 [2411, 8273] innerCandidates355_11 10684 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan355_12

def innerTail355_10 : List Nat := 7481 :: innerTail355_11
def innerCandidates355_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates355_10_checked : candidates edge 4 2411 7481 innerTail355_11 best.bound = innerCandidates355_10 := by
  unfold candidates
  have row : edge 7481 = SearchData.row7481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan355_10 : scan edge (search edge 3) 4 [2411] 2411 innerTail355_10 best = best := by
  rw [innerTail355_10, scan, if_neg (by decide), innerCandidates355_10_checked]
  change scan edge (search edge 3) 4 [2411] 2411 innerTail355_11
    (if 3 ≤ innerCandidates355_10.length then search edge 3 [2411, 7481] innerCandidates355_10 9892 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan355_11

def innerTail355_9 : List Nat := 7433 :: innerTail355_10
def innerCandidates355_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates355_9_checked : candidates edge 4 2411 7433 innerTail355_10 best.bound = innerCandidates355_9 := by
  unfold candidates
  have row : edge 7433 = SearchData.row7433.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan355_9 : scan edge (search edge 3) 4 [2411] 2411 innerTail355_9 best = best := by
  rw [innerTail355_9, scan, if_neg (by decide), innerCandidates355_9_checked]
  change scan edge (search edge 3) 4 [2411] 2411 innerTail355_10
    (if 3 ≤ innerCandidates355_9.length then search edge 3 [2411, 7433] innerCandidates355_9 9844 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan355_10

def innerTail355_8 : List Nat := 7019 :: innerTail355_9
def innerCandidates355_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates355_8_checked : candidates edge 4 2411 7019 innerTail355_9 best.bound = innerCandidates355_8 := by
  unfold candidates
  have row : edge 7019 = SearchData.row7019.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan355_8 : scan edge (search edge 3) 4 [2411] 2411 innerTail355_8 best = best := by
  rw [innerTail355_8, scan, if_neg (by decide), innerCandidates355_8_checked]
  change scan edge (search edge 3) 4 [2411] 2411 innerTail355_9
    (if 3 ≤ innerCandidates355_8.length then search edge 3 [2411, 7019] innerCandidates355_8 9430 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan355_9

def innerTail355_7 : List Nat := 5879 :: innerTail355_8
def innerCandidates355_7 : List Nat := [8273]
set_option maxRecDepth 4096 in
theorem innerCandidates355_7_checked : candidates edge 4 2411 5879 innerTail355_8 best.bound = innerCandidates355_7 := by
  unfold candidates
  have row : edge 5879 = SearchData.row5879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan355_7 : scan edge (search edge 3) 4 [2411] 2411 innerTail355_7 best = best := by
  rw [innerTail355_7, scan, if_neg (by decide), innerCandidates355_7_checked]
  change scan edge (search edge 3) 4 [2411] 2411 innerTail355_8
    (if 3 ≤ innerCandidates355_7.length then search edge 3 [2411, 5879] innerCandidates355_7 8290 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan355_8

def innerTail355_6 : List Nat := 5519 :: innerTail355_7
def innerCandidates355_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates355_6_checked : candidates edge 4 2411 5519 innerTail355_7 best.bound = innerCandidates355_6 := by
  unfold candidates
  have row : edge 5519 = SearchData.row5519.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan355_6 : scan edge (search edge 3) 4 [2411] 2411 innerTail355_6 best = best := by
  rw [innerTail355_6, scan, if_neg (by decide), innerCandidates355_6_checked]
  change scan edge (search edge 3) 4 [2411] 2411 innerTail355_7
    (if 3 ≤ innerCandidates355_6.length then search edge 3 [2411, 5519] innerCandidates355_6 7930 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan355_7

def innerTail355_5 : List Nat := 5171 :: innerTail355_6
def innerCandidates355_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates355_5_checked : candidates edge 4 2411 5171 innerTail355_6 best.bound = innerCandidates355_5 := by
  unfold candidates
  have row : edge 5171 = SearchData.row5171.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan355_5 : scan edge (search edge 3) 4 [2411] 2411 innerTail355_5 best = best := by
  rw [innerTail355_5, scan, if_neg (by decide), innerCandidates355_5_checked]
  change scan edge (search edge 3) 4 [2411] 2411 innerTail355_6
    (if 3 ≤ innerCandidates355_5.length then search edge 3 [2411, 5171] innerCandidates355_5 7582 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan355_6

def innerTail355_4 : List Nat := 4919 :: innerTail355_5
def innerCandidates355_4 : List Nat := [7481]
set_option maxRecDepth 4096 in
theorem innerCandidates355_4_checked : candidates edge 4 2411 4919 innerTail355_5 best.bound = innerCandidates355_4 := by
  unfold candidates
  have row : edge 4919 = SearchData.row4919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan355_4 : scan edge (search edge 3) 4 [2411] 2411 innerTail355_4 best = best := by
  rw [innerTail355_4, scan, if_neg (by decide), innerCandidates355_4_checked]
  change scan edge (search edge 3) 4 [2411] 2411 innerTail355_5
    (if 3 ≤ innerCandidates355_4.length then search edge 3 [2411, 4919] innerCandidates355_4 7330 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan355_5

def innerTail355_3 : List Nat := 4793 :: innerTail355_4
def innerCandidates355_3 : List Nat := [7481]
set_option maxRecDepth 4096 in
theorem innerCandidates355_3_checked : candidates edge 4 2411 4793 innerTail355_4 best.bound = innerCandidates355_3 := by
  unfold candidates
  have row : edge 4793 = SearchData.row4793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan355_3 : scan edge (search edge 3) 4 [2411] 2411 innerTail355_3 best = best := by
  rw [innerTail355_3, scan, if_neg (by decide), innerCandidates355_3_checked]
  change scan edge (search edge 3) 4 [2411] 2411 innerTail355_4
    (if 3 ≤ innerCandidates355_3.length then search edge 3 [2411, 4793] innerCandidates355_3 7204 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan355_4

def innerTail355_2 : List Nat := 4253 :: innerTail355_3
def innerCandidates355_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates355_2_checked : candidates edge 4 2411 4253 innerTail355_3 best.bound = innerCandidates355_2 := by
  unfold candidates
  have row : edge 4253 = SearchData.row4253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan355_2 : scan edge (search edge 3) 4 [2411] 2411 innerTail355_2 best = best := by
  rw [innerTail355_2, scan, if_neg (by decide), innerCandidates355_2_checked]
  change scan edge (search edge 3) 4 [2411] 2411 innerTail355_3
    (if 3 ≤ innerCandidates355_2.length then search edge 3 [2411, 4253] innerCandidates355_2 6664 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan355_3

def innerTail355_1 : List Nat := 4241 :: innerTail355_2
def innerCandidates355_1 : List Nat := [4793]
set_option maxRecDepth 4096 in
theorem innerCandidates355_1_checked : candidates edge 4 2411 4241 innerTail355_2 best.bound = innerCandidates355_1 := by
  unfold candidates
  have row : edge 4241 = SearchData.row4241.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan355_1 : scan edge (search edge 3) 4 [2411] 2411 innerTail355_1 best = best := by
  rw [innerTail355_1, scan, if_neg (by decide), innerCandidates355_1_checked]
  change scan edge (search edge 3) 4 [2411] 2411 innerTail355_2
    (if 3 ≤ innerCandidates355_1.length then search edge 3 [2411, 4241] innerCandidates355_1 6652 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan355_2

def innerTail355_0 : List Nat := 3917 :: innerTail355_1
def innerCandidates355_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates355_0_checked : candidates edge 4 2411 3917 innerTail355_1 best.bound = innerCandidates355_0 := by
  unfold candidates
  have row : edge 3917 = SearchData.row3917.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan355_0 : scan edge (search edge 3) 4 [2411] 2411 innerTail355_0 best = best := by
  rw [innerTail355_0, scan, if_neg (by decide), innerCandidates355_0_checked]
  change scan edge (search edge 3) 4 [2411] 2411 innerTail355_1
    (if 3 ≤ innerCandidates355_0.length then search edge 3 [2411, 3917] innerCandidates355_0 6328 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan355_1

theorem rootBranch355 : search edge 4 [2411] rootCandidates355 2411 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2411] 2411 innerTail355_0 best = best
  exact innerScan355_0

def rootCandidates354 : List Nat := [2663, 3251, 3467, 3821, 3911, 4457, 4517, 5717, 6131, 6701, 6737, 6761, 6977, 7283, 8081, 8087, 8363]
set_option maxRecDepth 4096 in
theorem rootCandidates354_checked : candidates edge 5 0 2399 tail355 best.bound = rootCandidates354 := by
  unfold candidates
  have row : edge 2399 = PrimePairCache.smallRow2399.contains := by funext q; rfl
  rw [row]
  decide

def innerTail354_17 : List Nat := []
theorem innerScan354_17 : scan edge (search edge 3) 4 [2399] 2399 innerTail354_17 best = best := rfl

def innerTail354_16 : List Nat := 8363 :: innerTail354_17
def innerCandidates354_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates354_16_checked : candidates edge 4 2399 8363 innerTail354_17 best.bound = innerCandidates354_16 := by
  unfold candidates
  have row : edge 8363 = SearchData.row8363.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan354_16 : scan edge (search edge 3) 4 [2399] 2399 innerTail354_16 best = best := by
  rw [innerTail354_16, scan, if_neg (by decide), innerCandidates354_16_checked]
  change scan edge (search edge 3) 4 [2399] 2399 innerTail354_17
    (if 3 ≤ innerCandidates354_16.length then search edge 3 [2399, 8363] innerCandidates354_16 10762 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan354_17

def innerTail354_15 : List Nat := 8087 :: innerTail354_16
def innerCandidates354_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates354_15_checked : candidates edge 4 2399 8087 innerTail354_16 best.bound = innerCandidates354_15 := by
  unfold candidates
  have row : edge 8087 = SearchData.row8087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan354_15 : scan edge (search edge 3) 4 [2399] 2399 innerTail354_15 best = best := by
  rw [innerTail354_15, scan, if_neg (by decide), innerCandidates354_15_checked]
  change scan edge (search edge 3) 4 [2399] 2399 innerTail354_16
    (if 3 ≤ innerCandidates354_15.length then search edge 3 [2399, 8087] innerCandidates354_15 10486 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan354_16

def innerTail354_14 : List Nat := 8081 :: innerTail354_15
def innerCandidates354_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates354_14_checked : candidates edge 4 2399 8081 innerTail354_15 best.bound = innerCandidates354_14 := by
  unfold candidates
  have row : edge 8081 = SearchData.row8081.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan354_14 : scan edge (search edge 3) 4 [2399] 2399 innerTail354_14 best = best := by
  rw [innerTail354_14, scan, if_neg (by decide), innerCandidates354_14_checked]
  change scan edge (search edge 3) 4 [2399] 2399 innerTail354_15
    (if 3 ≤ innerCandidates354_14.length then search edge 3 [2399, 8081] innerCandidates354_14 10480 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan354_15

def innerTail354_13 : List Nat := 7283 :: innerTail354_14
def innerCandidates354_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates354_13_checked : candidates edge 4 2399 7283 innerTail354_14 best.bound = innerCandidates354_13 := by
  unfold candidates
  have row : edge 7283 = SearchData.row7283.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan354_13 : scan edge (search edge 3) 4 [2399] 2399 innerTail354_13 best = best := by
  rw [innerTail354_13, scan, if_neg (by decide), innerCandidates354_13_checked]
  change scan edge (search edge 3) 4 [2399] 2399 innerTail354_14
    (if 3 ≤ innerCandidates354_13.length then search edge 3 [2399, 7283] innerCandidates354_13 9682 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan354_14

def innerTail354_12 : List Nat := 6977 :: innerTail354_13
def innerCandidates354_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates354_12_checked : candidates edge 4 2399 6977 innerTail354_13 best.bound = innerCandidates354_12 := by
  unfold candidates
  have row : edge 6977 = SearchData.row6977.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan354_12 : scan edge (search edge 3) 4 [2399] 2399 innerTail354_12 best = best := by
  rw [innerTail354_12, scan, if_neg (by decide), innerCandidates354_12_checked]
  change scan edge (search edge 3) 4 [2399] 2399 innerTail354_13
    (if 3 ≤ innerCandidates354_12.length then search edge 3 [2399, 6977] innerCandidates354_12 9376 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan354_13

def innerTail354_11 : List Nat := 6761 :: innerTail354_12
def innerCandidates354_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates354_11_checked : candidates edge 4 2399 6761 innerTail354_12 best.bound = innerCandidates354_11 := by
  unfold candidates
  have row : edge 6761 = SearchData.row6761.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan354_11 : scan edge (search edge 3) 4 [2399] 2399 innerTail354_11 best = best := by
  rw [innerTail354_11, scan, if_neg (by decide), innerCandidates354_11_checked]
  change scan edge (search edge 3) 4 [2399] 2399 innerTail354_12
    (if 3 ≤ innerCandidates354_11.length then search edge 3 [2399, 6761] innerCandidates354_11 9160 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan354_12

def innerTail354_10 : List Nat := 6737 :: innerTail354_11
def innerCandidates354_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates354_10_checked : candidates edge 4 2399 6737 innerTail354_11 best.bound = innerCandidates354_10 := by
  unfold candidates
  have row : edge 6737 = SearchData.row6737.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan354_10 : scan edge (search edge 3) 4 [2399] 2399 innerTail354_10 best = best := by
  rw [innerTail354_10, scan, if_neg (by decide), innerCandidates354_10_checked]
  change scan edge (search edge 3) 4 [2399] 2399 innerTail354_11
    (if 3 ≤ innerCandidates354_10.length then search edge 3 [2399, 6737] innerCandidates354_10 9136 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan354_11

def innerTail354_9 : List Nat := 6701 :: innerTail354_10
def innerCandidates354_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates354_9_checked : candidates edge 4 2399 6701 innerTail354_10 best.bound = innerCandidates354_9 := by
  unfold candidates
  have row : edge 6701 = SearchData.row6701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan354_9 : scan edge (search edge 3) 4 [2399] 2399 innerTail354_9 best = best := by
  rw [innerTail354_9, scan, if_neg (by decide), innerCandidates354_9_checked]
  change scan edge (search edge 3) 4 [2399] 2399 innerTail354_10
    (if 3 ≤ innerCandidates354_9.length then search edge 3 [2399, 6701] innerCandidates354_9 9100 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan354_10

def innerTail354_8 : List Nat := 6131 :: innerTail354_9
def innerCandidates354_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates354_8_checked : candidates edge 4 2399 6131 innerTail354_9 best.bound = innerCandidates354_8 := by
  unfold candidates
  have row : edge 6131 = SearchData.row6131.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan354_8 : scan edge (search edge 3) 4 [2399] 2399 innerTail354_8 best = best := by
  rw [innerTail354_8, scan, if_neg (by decide), innerCandidates354_8_checked]
  change scan edge (search edge 3) 4 [2399] 2399 innerTail354_9
    (if 3 ≤ innerCandidates354_8.length then search edge 3 [2399, 6131] innerCandidates354_8 8530 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan354_9

def innerTail354_7 : List Nat := 5717 :: innerTail354_8
def innerCandidates354_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates354_7_checked : candidates edge 4 2399 5717 innerTail354_8 best.bound = innerCandidates354_7 := by
  unfold candidates
  have row : edge 5717 = SearchData.row5717.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan354_7 : scan edge (search edge 3) 4 [2399] 2399 innerTail354_7 best = best := by
  rw [innerTail354_7, scan, if_neg (by decide), innerCandidates354_7_checked]
  change scan edge (search edge 3) 4 [2399] 2399 innerTail354_8
    (if 3 ≤ innerCandidates354_7.length then search edge 3 [2399, 5717] innerCandidates354_7 8116 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan354_8

def innerTail354_6 : List Nat := 4517 :: innerTail354_7
def innerCandidates354_6 : List Nat := [8087]
set_option maxRecDepth 4096 in
theorem innerCandidates354_6_checked : candidates edge 4 2399 4517 innerTail354_7 best.bound = innerCandidates354_6 := by
  unfold candidates
  have row : edge 4517 = SearchData.row4517.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan354_6 : scan edge (search edge 3) 4 [2399] 2399 innerTail354_6 best = best := by
  rw [innerTail354_6, scan, if_neg (by decide), innerCandidates354_6_checked]
  change scan edge (search edge 3) 4 [2399] 2399 innerTail354_7
    (if 3 ≤ innerCandidates354_6.length then search edge 3 [2399, 4517] innerCandidates354_6 6916 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan354_7

def innerTail354_5 : List Nat := 4457 :: innerTail354_6
def innerCandidates354_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates354_5_checked : candidates edge 4 2399 4457 innerTail354_6 best.bound = innerCandidates354_5 := by
  unfold candidates
  have row : edge 4457 = SearchData.row4457.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan354_5 : scan edge (search edge 3) 4 [2399] 2399 innerTail354_5 best = best := by
  rw [innerTail354_5, scan, if_neg (by decide), innerCandidates354_5_checked]
  change scan edge (search edge 3) 4 [2399] 2399 innerTail354_6
    (if 3 ≤ innerCandidates354_5.length then search edge 3 [2399, 4457] innerCandidates354_5 6856 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan354_6

def innerTail354_4 : List Nat := 3911 :: innerTail354_5
def innerCandidates354_4 : List Nat := [4457, 5717]
set_option maxRecDepth 4096 in
theorem innerCandidates354_4_checked : candidates edge 4 2399 3911 innerTail354_5 best.bound = innerCandidates354_4 := by
  unfold candidates
  have row : edge 3911 = SearchData.row3911.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan354_4 : scan edge (search edge 3) 4 [2399] 2399 innerTail354_4 best = best := by
  rw [innerTail354_4, scan, if_neg (by decide), innerCandidates354_4_checked]
  change scan edge (search edge 3) 4 [2399] 2399 innerTail354_5
    (if 3 ≤ innerCandidates354_4.length then search edge 3 [2399, 3911] innerCandidates354_4 6310 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan354_5

def innerTail354_3 : List Nat := 3821 :: innerTail354_4
def innerCandidates354_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates354_3_checked : candidates edge 4 2399 3821 innerTail354_4 best.bound = innerCandidates354_3 := by
  unfold candidates
  have row : edge 3821 = SearchData.row3821.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan354_3 : scan edge (search edge 3) 4 [2399] 2399 innerTail354_3 best = best := by
  rw [innerTail354_3, scan, if_neg (by decide), innerCandidates354_3_checked]
  change scan edge (search edge 3) 4 [2399] 2399 innerTail354_4
    (if 3 ≤ innerCandidates354_3.length then search edge 3 [2399, 3821] innerCandidates354_3 6220 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan354_4

def innerTail354_2 : List Nat := 3467 :: innerTail354_3
def innerCandidates354_2 : List Nat := [4457]
set_option maxRecDepth 4096 in
theorem innerCandidates354_2_checked : candidates edge 4 2399 3467 innerTail354_3 best.bound = innerCandidates354_2 := by
  unfold candidates
  have row : edge 3467 = SearchData.row3467.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan354_2 : scan edge (search edge 3) 4 [2399] 2399 innerTail354_2 best = best := by
  rw [innerTail354_2, scan, if_neg (by decide), innerCandidates354_2_checked]
  change scan edge (search edge 3) 4 [2399] 2399 innerTail354_3
    (if 3 ≤ innerCandidates354_2.length then search edge 3 [2399, 3467] innerCandidates354_2 5866 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan354_3

def innerTail354_1 : List Nat := 3251 :: innerTail354_2
def innerCandidates354_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates354_1_checked : candidates edge 4 2399 3251 innerTail354_2 best.bound = innerCandidates354_1 := by
  unfold candidates
  have row : edge 3251 = SearchData.row3251.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan354_1 : scan edge (search edge 3) 4 [2399] 2399 innerTail354_1 best = best := by
  rw [innerTail354_1, scan, if_neg (by decide), innerCandidates354_1_checked]
  change scan edge (search edge 3) 4 [2399] 2399 innerTail354_2
    (if 3 ≤ innerCandidates354_1.length then search edge 3 [2399, 3251] innerCandidates354_1 5650 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan354_2

def innerTail354_0 : List Nat := 2663 :: innerTail354_1
def innerCandidates354_0 : List Nat := [3251, 4457]
set_option maxRecDepth 4096 in
theorem innerCandidates354_0_checked : candidates edge 4 2399 2663 innerTail354_1 best.bound = innerCandidates354_0 := by
  unfold candidates
  have row : edge 2663 = PrimePairCache.smallRow2663.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan354_0 : scan edge (search edge 3) 4 [2399] 2399 innerTail354_0 best = best := by
  rw [innerTail354_0, scan, if_neg (by decide), innerCandidates354_0_checked]
  change scan edge (search edge 3) 4 [2399] 2399 innerTail354_1
    (if 3 ≤ innerCandidates354_0.length then search edge 3 [2399, 2663] innerCandidates354_0 5062 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan354_1

theorem rootBranch354 : search edge 4 [2399] rootCandidates354 2399 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2399] 2399 innerTail354_0 best = best
  exact innerScan354_0

end ProjectEulerSolutions.PrimePairUpper.Search
