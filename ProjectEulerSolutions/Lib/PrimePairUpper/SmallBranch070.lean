import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates530 : List Nat := [3931, 4021, 4099, 5077, 5197, 5659, 5737, 6211, 6367, 6553, 6883, 8221, 8233]
set_option maxRecDepth 4096 in
theorem rootCandidates530_checked : candidates edge 5 0 3847 tail531 best.bound = rootCandidates530 := by
  unfold candidates
  have row : edge 3847 = SearchData.row3847.contains := by funext q; rfl
  rw [row]
  decide

def innerTail530_13 : List Nat := []
theorem innerScan530_13 : scan edge (search edge 3) 4 [3847] 3847 innerTail530_13 best = best := rfl

def innerTail530_12 : List Nat := 8233 :: innerTail530_13
def innerCandidates530_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates530_12_checked : candidates edge 4 3847 8233 innerTail530_13 best.bound = innerCandidates530_12 := by
  unfold candidates
  have row : edge 8233 = SearchData.row8233.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan530_12 : scan edge (search edge 3) 4 [3847] 3847 innerTail530_12 best = best := by
  rw [innerTail530_12, scan, if_neg (by decide), innerCandidates530_12_checked]
  change scan edge (search edge 3) 4 [3847] 3847 innerTail530_13
    (if 3 ≤ innerCandidates530_12.length then search edge 3 [3847, 8233] innerCandidates530_12 12080 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan530_13

def innerTail530_11 : List Nat := 8221 :: innerTail530_12
def innerCandidates530_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates530_11_checked : candidates edge 4 3847 8221 innerTail530_12 best.bound = innerCandidates530_11 := by
  unfold candidates
  have row : edge 8221 = SearchData.row8221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan530_11 : scan edge (search edge 3) 4 [3847] 3847 innerTail530_11 best = best := by
  rw [innerTail530_11, scan, if_neg (by decide), innerCandidates530_11_checked]
  change scan edge (search edge 3) 4 [3847] 3847 innerTail530_12
    (if 3 ≤ innerCandidates530_11.length then search edge 3 [3847, 8221] innerCandidates530_11 12068 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan530_12

def innerTail530_10 : List Nat := 6883 :: innerTail530_11
def innerCandidates530_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates530_10_checked : candidates edge 4 3847 6883 innerTail530_11 best.bound = innerCandidates530_10 := by
  unfold candidates
  have row : edge 6883 = SearchData.row6883.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan530_10 : scan edge (search edge 3) 4 [3847] 3847 innerTail530_10 best = best := by
  rw [innerTail530_10, scan, if_neg (by decide), innerCandidates530_10_checked]
  change scan edge (search edge 3) 4 [3847] 3847 innerTail530_11
    (if 3 ≤ innerCandidates530_10.length then search edge 3 [3847, 6883] innerCandidates530_10 10730 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan530_11

def innerTail530_9 : List Nat := 6553 :: innerTail530_10
def innerCandidates530_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates530_9_checked : candidates edge 4 3847 6553 innerTail530_10 best.bound = innerCandidates530_9 := by
  unfold candidates
  have row : edge 6553 = SearchData.row6553.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan530_9 : scan edge (search edge 3) 4 [3847] 3847 innerTail530_9 best = best := by
  rw [innerTail530_9, scan, if_neg (by decide), innerCandidates530_9_checked]
  change scan edge (search edge 3) 4 [3847] 3847 innerTail530_10
    (if 3 ≤ innerCandidates530_9.length then search edge 3 [3847, 6553] innerCandidates530_9 10400 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan530_10

def innerTail530_8 : List Nat := 6367 :: innerTail530_9
def innerCandidates530_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates530_8_checked : candidates edge 4 3847 6367 innerTail530_9 best.bound = innerCandidates530_8 := by
  unfold candidates
  have row : edge 6367 = SearchData.row6367.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan530_8 : scan edge (search edge 3) 4 [3847] 3847 innerTail530_8 best = best := by
  rw [innerTail530_8, scan, if_neg (by decide), innerCandidates530_8_checked]
  change scan edge (search edge 3) 4 [3847] 3847 innerTail530_9
    (if 3 ≤ innerCandidates530_8.length then search edge 3 [3847, 6367] innerCandidates530_8 10214 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan530_9

def innerTail530_7 : List Nat := 6211 :: innerTail530_8
def innerCandidates530_7 : List Nat := [6883]
set_option maxRecDepth 4096 in
theorem innerCandidates530_7_checked : candidates edge 4 3847 6211 innerTail530_8 best.bound = innerCandidates530_7 := by
  unfold candidates
  have row : edge 6211 = SearchData.row6211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan530_7 : scan edge (search edge 3) 4 [3847] 3847 innerTail530_7 best = best := by
  rw [innerTail530_7, scan, if_neg (by decide), innerCandidates530_7_checked]
  change scan edge (search edge 3) 4 [3847] 3847 innerTail530_8
    (if 3 ≤ innerCandidates530_7.length then search edge 3 [3847, 6211] innerCandidates530_7 10058 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan530_8

def innerTail530_6 : List Nat := 5737 :: innerTail530_7
def innerCandidates530_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates530_6_checked : candidates edge 4 3847 5737 innerTail530_7 best.bound = innerCandidates530_6 := by
  unfold candidates
  have row : edge 5737 = SearchData.row5737.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan530_6 : scan edge (search edge 3) 4 [3847] 3847 innerTail530_6 best = best := by
  rw [innerTail530_6, scan, if_neg (by decide), innerCandidates530_6_checked]
  change scan edge (search edge 3) 4 [3847] 3847 innerTail530_7
    (if 3 ≤ innerCandidates530_6.length then search edge 3 [3847, 5737] innerCandidates530_6 9584 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan530_7

def innerTail530_5 : List Nat := 5659 :: innerTail530_6
def innerCandidates530_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates530_5_checked : candidates edge 4 3847 5659 innerTail530_6 best.bound = innerCandidates530_5 := by
  unfold candidates
  have row : edge 5659 = SearchData.row5659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan530_5 : scan edge (search edge 3) 4 [3847] 3847 innerTail530_5 best = best := by
  rw [innerTail530_5, scan, if_neg (by decide), innerCandidates530_5_checked]
  change scan edge (search edge 3) 4 [3847] 3847 innerTail530_6
    (if 3 ≤ innerCandidates530_5.length then search edge 3 [3847, 5659] innerCandidates530_5 9506 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan530_6

def innerTail530_4 : List Nat := 5197 :: innerTail530_5
def innerCandidates530_4 : List Nat := [8221]
set_option maxRecDepth 4096 in
theorem innerCandidates530_4_checked : candidates edge 4 3847 5197 innerTail530_5 best.bound = innerCandidates530_4 := by
  unfold candidates
  have row : edge 5197 = SearchData.row5197.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan530_4 : scan edge (search edge 3) 4 [3847] 3847 innerTail530_4 best = best := by
  rw [innerTail530_4, scan, if_neg (by decide), innerCandidates530_4_checked]
  change scan edge (search edge 3) 4 [3847] 3847 innerTail530_5
    (if 3 ≤ innerCandidates530_4.length then search edge 3 [3847, 5197] innerCandidates530_4 9044 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan530_5

def innerTail530_3 : List Nat := 5077 :: innerTail530_4
def innerCandidates530_3 : List Nat := [6367]
set_option maxRecDepth 4096 in
theorem innerCandidates530_3_checked : candidates edge 4 3847 5077 innerTail530_4 best.bound = innerCandidates530_3 := by
  unfold candidates
  have row : edge 5077 = SearchData.row5077.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan530_3 : scan edge (search edge 3) 4 [3847] 3847 innerTail530_3 best = best := by
  rw [innerTail530_3, scan, if_neg (by decide), innerCandidates530_3_checked]
  change scan edge (search edge 3) 4 [3847] 3847 innerTail530_4
    (if 3 ≤ innerCandidates530_3.length then search edge 3 [3847, 5077] innerCandidates530_3 8924 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan530_4

def innerTail530_2 : List Nat := 4099 :: innerTail530_3
def innerCandidates530_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates530_2_checked : candidates edge 4 3847 4099 innerTail530_3 best.bound = innerCandidates530_2 := by
  unfold candidates
  have row : edge 4099 = SearchData.row4099.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan530_2 : scan edge (search edge 3) 4 [3847] 3847 innerTail530_2 best = best := by
  rw [innerTail530_2, scan, if_neg (by decide), innerCandidates530_2_checked]
  change scan edge (search edge 3) 4 [3847] 3847 innerTail530_3
    (if 3 ≤ innerCandidates530_2.length then search edge 3 [3847, 4099] innerCandidates530_2 7946 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan530_3

def innerTail530_1 : List Nat := 4021 :: innerTail530_2
def innerCandidates530_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates530_1_checked : candidates edge 4 3847 4021 innerTail530_2 best.bound = innerCandidates530_1 := by
  unfold candidates
  have row : edge 4021 = SearchData.row4021.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan530_1 : scan edge (search edge 3) 4 [3847] 3847 innerTail530_1 best = best := by
  rw [innerTail530_1, scan, if_neg (by decide), innerCandidates530_1_checked]
  change scan edge (search edge 3) 4 [3847] 3847 innerTail530_2
    (if 3 ≤ innerCandidates530_1.length then search edge 3 [3847, 4021] innerCandidates530_1 7868 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan530_2

def innerTail530_0 : List Nat := 3931 :: innerTail530_1
def innerCandidates530_0 : List Nat := [6553, 8233]
set_option maxRecDepth 4096 in
theorem innerCandidates530_0_checked : candidates edge 4 3847 3931 innerTail530_1 best.bound = innerCandidates530_0 := by
  unfold candidates
  have row : edge 3931 = SearchData.row3931.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan530_0 : scan edge (search edge 3) 4 [3847] 3847 innerTail530_0 best = best := by
  rw [innerTail530_0, scan, if_neg (by decide), innerCandidates530_0_checked]
  change scan edge (search edge 3) 4 [3847] 3847 innerTail530_1
    (if 3 ≤ innerCandidates530_0.length then search edge 3 [3847, 3931] innerCandidates530_0 7778 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan530_1

theorem rootBranch530 : search edge 4 [3847] rootCandidates530 3847 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3847] 3847 innerTail530_0 best = best
  exact innerScan530_0

def rootCandidates529 : List Nat := [4253, 4733, 4889, 5657, 5813, 6269, 6449, 6827, 6833, 6911, 6983, 7841]
set_option maxRecDepth 4096 in
theorem rootCandidates529_checked : candidates edge 5 0 3833 tail530 best.bound = rootCandidates529 := by
  unfold candidates
  have row : edge 3833 = SearchData.row3833.contains := by funext q; rfl
  rw [row]
  decide

def innerTail529_12 : List Nat := []
theorem innerScan529_12 : scan edge (search edge 3) 4 [3833] 3833 innerTail529_12 best = best := rfl

def innerTail529_11 : List Nat := 7841 :: innerTail529_12
def innerCandidates529_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates529_11_checked : candidates edge 4 3833 7841 innerTail529_12 best.bound = innerCandidates529_11 := by
  unfold candidates
  have row : edge 7841 = SearchData.row7841.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan529_11 : scan edge (search edge 3) 4 [3833] 3833 innerTail529_11 best = best := by
  rw [innerTail529_11, scan, if_neg (by decide), innerCandidates529_11_checked]
  change scan edge (search edge 3) 4 [3833] 3833 innerTail529_12
    (if 3 ≤ innerCandidates529_11.length then search edge 3 [3833, 7841] innerCandidates529_11 11674 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan529_12

def innerTail529_10 : List Nat := 6983 :: innerTail529_11
def innerCandidates529_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates529_10_checked : candidates edge 4 3833 6983 innerTail529_11 best.bound = innerCandidates529_10 := by
  unfold candidates
  have row : edge 6983 = SearchData.row6983.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan529_10 : scan edge (search edge 3) 4 [3833] 3833 innerTail529_10 best = best := by
  rw [innerTail529_10, scan, if_neg (by decide), innerCandidates529_10_checked]
  change scan edge (search edge 3) 4 [3833] 3833 innerTail529_11
    (if 3 ≤ innerCandidates529_10.length then search edge 3 [3833, 6983] innerCandidates529_10 10816 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan529_11

def innerTail529_9 : List Nat := 6911 :: innerTail529_10
def innerCandidates529_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates529_9_checked : candidates edge 4 3833 6911 innerTail529_10 best.bound = innerCandidates529_9 := by
  unfold candidates
  have row : edge 6911 = SearchData.row6911.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan529_9 : scan edge (search edge 3) 4 [3833] 3833 innerTail529_9 best = best := by
  rw [innerTail529_9, scan, if_neg (by decide), innerCandidates529_9_checked]
  change scan edge (search edge 3) 4 [3833] 3833 innerTail529_10
    (if 3 ≤ innerCandidates529_9.length then search edge 3 [3833, 6911] innerCandidates529_9 10744 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan529_10

def innerTail529_8 : List Nat := 6833 :: innerTail529_9
def innerCandidates529_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates529_8_checked : candidates edge 4 3833 6833 innerTail529_9 best.bound = innerCandidates529_8 := by
  unfold candidates
  have row : edge 6833 = SearchData.row6833.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan529_8 : scan edge (search edge 3) 4 [3833] 3833 innerTail529_8 best = best := by
  rw [innerTail529_8, scan, if_neg (by decide), innerCandidates529_8_checked]
  change scan edge (search edge 3) 4 [3833] 3833 innerTail529_9
    (if 3 ≤ innerCandidates529_8.length then search edge 3 [3833, 6833] innerCandidates529_8 10666 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan529_9

def innerTail529_7 : List Nat := 6827 :: innerTail529_8
def innerCandidates529_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates529_7_checked : candidates edge 4 3833 6827 innerTail529_8 best.bound = innerCandidates529_7 := by
  unfold candidates
  have row : edge 6827 = SearchData.row6827.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan529_7 : scan edge (search edge 3) 4 [3833] 3833 innerTail529_7 best = best := by
  rw [innerTail529_7, scan, if_neg (by decide), innerCandidates529_7_checked]
  change scan edge (search edge 3) 4 [3833] 3833 innerTail529_8
    (if 3 ≤ innerCandidates529_7.length then search edge 3 [3833, 6827] innerCandidates529_7 10660 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan529_8

def innerTail529_6 : List Nat := 6449 :: innerTail529_7
def innerCandidates529_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates529_6_checked : candidates edge 4 3833 6449 innerTail529_7 best.bound = innerCandidates529_6 := by
  unfold candidates
  have row : edge 6449 = SearchData.row6449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan529_6 : scan edge (search edge 3) 4 [3833] 3833 innerTail529_6 best = best := by
  rw [innerTail529_6, scan, if_neg (by decide), innerCandidates529_6_checked]
  change scan edge (search edge 3) 4 [3833] 3833 innerTail529_7
    (if 3 ≤ innerCandidates529_6.length then search edge 3 [3833, 6449] innerCandidates529_6 10282 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan529_7

def innerTail529_5 : List Nat := 6269 :: innerTail529_6
def innerCandidates529_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates529_5_checked : candidates edge 4 3833 6269 innerTail529_6 best.bound = innerCandidates529_5 := by
  unfold candidates
  have row : edge 6269 = SearchData.row6269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan529_5 : scan edge (search edge 3) 4 [3833] 3833 innerTail529_5 best = best := by
  rw [innerTail529_5, scan, if_neg (by decide), innerCandidates529_5_checked]
  change scan edge (search edge 3) 4 [3833] 3833 innerTail529_6
    (if 3 ≤ innerCandidates529_5.length then search edge 3 [3833, 6269] innerCandidates529_5 10102 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan529_6

def innerTail529_4 : List Nat := 5813 :: innerTail529_5
def innerCandidates529_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates529_4_checked : candidates edge 4 3833 5813 innerTail529_5 best.bound = innerCandidates529_4 := by
  unfold candidates
  have row : edge 5813 = SearchData.row5813.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan529_4 : scan edge (search edge 3) 4 [3833] 3833 innerTail529_4 best = best := by
  rw [innerTail529_4, scan, if_neg (by decide), innerCandidates529_4_checked]
  change scan edge (search edge 3) 4 [3833] 3833 innerTail529_5
    (if 3 ≤ innerCandidates529_4.length then search edge 3 [3833, 5813] innerCandidates529_4 9646 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan529_5

def innerTail529_3 : List Nat := 5657 :: innerTail529_4
def innerCandidates529_3 : List Nat := [6983]
set_option maxRecDepth 4096 in
theorem innerCandidates529_3_checked : candidates edge 4 3833 5657 innerTail529_4 best.bound = innerCandidates529_3 := by
  unfold candidates
  have row : edge 5657 = SearchData.row5657.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan529_3 : scan edge (search edge 3) 4 [3833] 3833 innerTail529_3 best = best := by
  rw [innerTail529_3, scan, if_neg (by decide), innerCandidates529_3_checked]
  change scan edge (search edge 3) 4 [3833] 3833 innerTail529_4
    (if 3 ≤ innerCandidates529_3.length then search edge 3 [3833, 5657] innerCandidates529_3 9490 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan529_4

def innerTail529_2 : List Nat := 4889 :: innerTail529_3
def innerCandidates529_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates529_2_checked : candidates edge 4 3833 4889 innerTail529_3 best.bound = innerCandidates529_2 := by
  unfold candidates
  have row : edge 4889 = SearchData.row4889.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan529_2 : scan edge (search edge 3) 4 [3833] 3833 innerTail529_2 best = best := by
  rw [innerTail529_2, scan, if_neg (by decide), innerCandidates529_2_checked]
  change scan edge (search edge 3) 4 [3833] 3833 innerTail529_3
    (if 3 ≤ innerCandidates529_2.length then search edge 3 [3833, 4889] innerCandidates529_2 8722 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan529_3

def innerTail529_1 : List Nat := 4733 :: innerTail529_2
def innerCandidates529_1 : List Nat := [6269, 6911]
set_option maxRecDepth 4096 in
theorem innerCandidates529_1_checked : candidates edge 4 3833 4733 innerTail529_2 best.bound = innerCandidates529_1 := by
  unfold candidates
  have row : edge 4733 = SearchData.row4733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan529_1 : scan edge (search edge 3) 4 [3833] 3833 innerTail529_1 best = best := by
  rw [innerTail529_1, scan, if_neg (by decide), innerCandidates529_1_checked]
  change scan edge (search edge 3) 4 [3833] 3833 innerTail529_2
    (if 3 ≤ innerCandidates529_1.length then search edge 3 [3833, 4733] innerCandidates529_1 8566 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan529_2

def innerTail529_0 : List Nat := 4253 :: innerTail529_1
def innerCandidates529_0 : List Nat := [5657]
set_option maxRecDepth 4096 in
theorem innerCandidates529_0_checked : candidates edge 4 3833 4253 innerTail529_1 best.bound = innerCandidates529_0 := by
  unfold candidates
  have row : edge 4253 = SearchData.row4253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan529_0 : scan edge (search edge 3) 4 [3833] 3833 innerTail529_0 best = best := by
  rw [innerTail529_0, scan, if_neg (by decide), innerCandidates529_0_checked]
  change scan edge (search edge 3) 4 [3833] 3833 innerTail529_1
    (if 3 ≤ innerCandidates529_0.length then search edge 3 [3833, 4253] innerCandidates529_0 8086 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan529_1

theorem rootBranch529 : search edge 4 [3833] rootCandidates529 3833 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3833] 3833 innerTail529_0 best = best
  exact innerScan529_0

def rootCandidates528 : List Nat := [4159, 4201, 4297, 4549, 4759, 5167, 5209, 5233, 5779, 6199, 6841, 7393, 7417, 7993]
set_option maxRecDepth 4096 in
theorem rootCandidates528_checked : candidates edge 5 0 3823 tail529 best.bound = rootCandidates528 := by
  unfold candidates
  have row : edge 3823 = SearchData.row3823.contains := by funext q; rfl
  rw [row]
  decide

def innerTail528_14 : List Nat := []
theorem innerScan528_14 : scan edge (search edge 3) 4 [3823] 3823 innerTail528_14 best = best := rfl

def innerTail528_13 : List Nat := 7993 :: innerTail528_14
def innerCandidates528_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates528_13_checked : candidates edge 4 3823 7993 innerTail528_14 best.bound = innerCandidates528_13 := by
  unfold candidates
  have row : edge 7993 = SearchData.row7993.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan528_13 : scan edge (search edge 3) 4 [3823] 3823 innerTail528_13 best = best := by
  rw [innerTail528_13, scan, if_neg (by decide), innerCandidates528_13_checked]
  change scan edge (search edge 3) 4 [3823] 3823 innerTail528_14
    (if 3 ≤ innerCandidates528_13.length then search edge 3 [3823, 7993] innerCandidates528_13 11816 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan528_14

def innerTail528_12 : List Nat := 7417 :: innerTail528_13
def innerCandidates528_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates528_12_checked : candidates edge 4 3823 7417 innerTail528_13 best.bound = innerCandidates528_12 := by
  unfold candidates
  have row : edge 7417 = SearchData.row7417.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan528_12 : scan edge (search edge 3) 4 [3823] 3823 innerTail528_12 best = best := by
  rw [innerTail528_12, scan, if_neg (by decide), innerCandidates528_12_checked]
  change scan edge (search edge 3) 4 [3823] 3823 innerTail528_13
    (if 3 ≤ innerCandidates528_12.length then search edge 3 [3823, 7417] innerCandidates528_12 11240 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan528_13

def innerTail528_11 : List Nat := 7393 :: innerTail528_12
def innerCandidates528_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates528_11_checked : candidates edge 4 3823 7393 innerTail528_12 best.bound = innerCandidates528_11 := by
  unfold candidates
  have row : edge 7393 = SearchData.row7393.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan528_11 : scan edge (search edge 3) 4 [3823] 3823 innerTail528_11 best = best := by
  rw [innerTail528_11, scan, if_neg (by decide), innerCandidates528_11_checked]
  change scan edge (search edge 3) 4 [3823] 3823 innerTail528_12
    (if 3 ≤ innerCandidates528_11.length then search edge 3 [3823, 7393] innerCandidates528_11 11216 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan528_12

def innerTail528_10 : List Nat := 6841 :: innerTail528_11
def innerCandidates528_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates528_10_checked : candidates edge 4 3823 6841 innerTail528_11 best.bound = innerCandidates528_10 := by
  unfold candidates
  have row : edge 6841 = SearchData.row6841.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan528_10 : scan edge (search edge 3) 4 [3823] 3823 innerTail528_10 best = best := by
  rw [innerTail528_10, scan, if_neg (by decide), innerCandidates528_10_checked]
  change scan edge (search edge 3) 4 [3823] 3823 innerTail528_11
    (if 3 ≤ innerCandidates528_10.length then search edge 3 [3823, 6841] innerCandidates528_10 10664 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan528_11

def innerTail528_9 : List Nat := 6199 :: innerTail528_10
def innerCandidates528_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates528_9_checked : candidates edge 4 3823 6199 innerTail528_10 best.bound = innerCandidates528_9 := by
  unfold candidates
  have row : edge 6199 = SearchData.row6199.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan528_9 : scan edge (search edge 3) 4 [3823] 3823 innerTail528_9 best = best := by
  rw [innerTail528_9, scan, if_neg (by decide), innerCandidates528_9_checked]
  change scan edge (search edge 3) 4 [3823] 3823 innerTail528_10
    (if 3 ≤ innerCandidates528_9.length then search edge 3 [3823, 6199] innerCandidates528_9 10022 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan528_10

def innerTail528_8 : List Nat := 5779 :: innerTail528_9
def innerCandidates528_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates528_8_checked : candidates edge 4 3823 5779 innerTail528_9 best.bound = innerCandidates528_8 := by
  unfold candidates
  have row : edge 5779 = SearchData.row5779.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan528_8 : scan edge (search edge 3) 4 [3823] 3823 innerTail528_8 best = best := by
  rw [innerTail528_8, scan, if_neg (by decide), innerCandidates528_8_checked]
  change scan edge (search edge 3) 4 [3823] 3823 innerTail528_9
    (if 3 ≤ innerCandidates528_8.length then search edge 3 [3823, 5779] innerCandidates528_8 9602 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan528_9

def innerTail528_7 : List Nat := 5233 :: innerTail528_8
def innerCandidates528_7 : List Nat := [5779]
set_option maxRecDepth 4096 in
theorem innerCandidates528_7_checked : candidates edge 4 3823 5233 innerTail528_8 best.bound = innerCandidates528_7 := by
  unfold candidates
  have row : edge 5233 = SearchData.row5233.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan528_7 : scan edge (search edge 3) 4 [3823] 3823 innerTail528_7 best = best := by
  rw [innerTail528_7, scan, if_neg (by decide), innerCandidates528_7_checked]
  change scan edge (search edge 3) 4 [3823] 3823 innerTail528_8
    (if 3 ≤ innerCandidates528_7.length then search edge 3 [3823, 5233] innerCandidates528_7 9056 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan528_8

def innerTail528_6 : List Nat := 5209 :: innerTail528_7
def innerCandidates528_6 : List Nat := [7417]
set_option maxRecDepth 4096 in
theorem innerCandidates528_6_checked : candidates edge 4 3823 5209 innerTail528_7 best.bound = innerCandidates528_6 := by
  unfold candidates
  have row : edge 5209 = SearchData.row5209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan528_6 : scan edge (search edge 3) 4 [3823] 3823 innerTail528_6 best = best := by
  rw [innerTail528_6, scan, if_neg (by decide), innerCandidates528_6_checked]
  change scan edge (search edge 3) 4 [3823] 3823 innerTail528_7
    (if 3 ≤ innerCandidates528_6.length then search edge 3 [3823, 5209] innerCandidates528_6 9032 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan528_7

def innerTail528_5 : List Nat := 5167 :: innerTail528_6
def innerCandidates528_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates528_5_checked : candidates edge 4 3823 5167 innerTail528_6 best.bound = innerCandidates528_5 := by
  unfold candidates
  have row : edge 5167 = SearchData.row5167.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan528_5 : scan edge (search edge 3) 4 [3823] 3823 innerTail528_5 best = best := by
  rw [innerTail528_5, scan, if_neg (by decide), innerCandidates528_5_checked]
  change scan edge (search edge 3) 4 [3823] 3823 innerTail528_6
    (if 3 ≤ innerCandidates528_5.length then search edge 3 [3823, 5167] innerCandidates528_5 8990 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan528_6

def innerTail528_4 : List Nat := 4759 :: innerTail528_5
def innerCandidates528_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates528_4_checked : candidates edge 4 3823 4759 innerTail528_5 best.bound = innerCandidates528_4 := by
  unfold candidates
  have row : edge 4759 = SearchData.row4759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan528_4 : scan edge (search edge 3) 4 [3823] 3823 innerTail528_4 best = best := by
  rw [innerTail528_4, scan, if_neg (by decide), innerCandidates528_4_checked]
  change scan edge (search edge 3) 4 [3823] 3823 innerTail528_5
    (if 3 ≤ innerCandidates528_4.length then search edge 3 [3823, 4759] innerCandidates528_4 8582 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan528_5

def innerTail528_3 : List Nat := 4549 :: innerTail528_4
def innerCandidates528_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates528_3_checked : candidates edge 4 3823 4549 innerTail528_4 best.bound = innerCandidates528_3 := by
  unfold candidates
  have row : edge 4549 = SearchData.row4549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan528_3 : scan edge (search edge 3) 4 [3823] 3823 innerTail528_3 best = best := by
  rw [innerTail528_3, scan, if_neg (by decide), innerCandidates528_3_checked]
  change scan edge (search edge 3) 4 [3823] 3823 innerTail528_4
    (if 3 ≤ innerCandidates528_3.length then search edge 3 [3823, 4549] innerCandidates528_3 8372 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan528_4

def innerTail528_2 : List Nat := 4297 :: innerTail528_3
def innerCandidates528_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates528_2_checked : candidates edge 4 3823 4297 innerTail528_3 best.bound = innerCandidates528_2 := by
  unfold candidates
  have row : edge 4297 = SearchData.row4297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan528_2 : scan edge (search edge 3) 4 [3823] 3823 innerTail528_2 best = best := by
  rw [innerTail528_2, scan, if_neg (by decide), innerCandidates528_2_checked]
  change scan edge (search edge 3) 4 [3823] 3823 innerTail528_3
    (if 3 ≤ innerCandidates528_2.length then search edge 3 [3823, 4297] innerCandidates528_2 8120 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan528_3

def innerTail528_1 : List Nat := 4201 :: innerTail528_2
def innerCandidates528_1 : List Nat := [4759, 6199]
set_option maxRecDepth 4096 in
theorem innerCandidates528_1_checked : candidates edge 4 3823 4201 innerTail528_2 best.bound = innerCandidates528_1 := by
  unfold candidates
  have row : edge 4201 = SearchData.row4201.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan528_1 : scan edge (search edge 3) 4 [3823] 3823 innerTail528_1 best = best := by
  rw [innerTail528_1, scan, if_neg (by decide), innerCandidates528_1_checked]
  change scan edge (search edge 3) 4 [3823] 3823 innerTail528_2
    (if 3 ≤ innerCandidates528_1.length then search edge 3 [3823, 4201] innerCandidates528_1 8024 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan528_2

def innerTail528_0 : List Nat := 4159 :: innerTail528_1
def innerCandidates528_0 : List Nat := [5779]
set_option maxRecDepth 4096 in
theorem innerCandidates528_0_checked : candidates edge 4 3823 4159 innerTail528_1 best.bound = innerCandidates528_0 := by
  unfold candidates
  have row : edge 4159 = SearchData.row4159.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan528_0 : scan edge (search edge 3) 4 [3823] 3823 innerTail528_0 best = best := by
  rw [innerTail528_0, scan, if_neg (by decide), innerCandidates528_0_checked]
  change scan edge (search edge 3) 4 [3823] 3823 innerTail528_1
    (if 3 ≤ innerCandidates528_0.length then search edge 3 [3823, 4159] innerCandidates528_0 7982 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan528_1

theorem rootBranch528 : search edge 4 [3823] rootCandidates528 3823 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3823] 3823 innerTail528_0 best = best
  exact innerScan528_0

def rootCandidates527 : List Nat := [4703, 6047, 6263, 6473, 6863, 6917, 7019, 7307]
set_option maxRecDepth 4096 in
theorem rootCandidates527_checked : candidates edge 5 0 3821 tail528 best.bound = rootCandidates527 := by
  unfold candidates
  have row : edge 3821 = SearchData.row3821.contains := by funext q; rfl
  rw [row]
  decide

def innerTail527_8 : List Nat := []
theorem innerScan527_8 : scan edge (search edge 3) 4 [3821] 3821 innerTail527_8 best = best := rfl

def innerTail527_7 : List Nat := 7307 :: innerTail527_8
def innerCandidates527_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates527_7_checked : candidates edge 4 3821 7307 innerTail527_8 best.bound = innerCandidates527_7 := by
  unfold candidates
  have row : edge 7307 = SearchData.row7307.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan527_7 : scan edge (search edge 3) 4 [3821] 3821 innerTail527_7 best = best := by
  rw [innerTail527_7, scan, if_neg (by decide), innerCandidates527_7_checked]
  change scan edge (search edge 3) 4 [3821] 3821 innerTail527_8
    (if 3 ≤ innerCandidates527_7.length then search edge 3 [3821, 7307] innerCandidates527_7 11128 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan527_8

def innerTail527_6 : List Nat := 7019 :: innerTail527_7
def innerCandidates527_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates527_6_checked : candidates edge 4 3821 7019 innerTail527_7 best.bound = innerCandidates527_6 := by
  unfold candidates
  have row : edge 7019 = SearchData.row7019.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan527_6 : scan edge (search edge 3) 4 [3821] 3821 innerTail527_6 best = best := by
  rw [innerTail527_6, scan, if_neg (by decide), innerCandidates527_6_checked]
  change scan edge (search edge 3) 4 [3821] 3821 innerTail527_7
    (if 3 ≤ innerCandidates527_6.length then search edge 3 [3821, 7019] innerCandidates527_6 10840 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan527_7

def innerTail527_5 : List Nat := 6917 :: innerTail527_6
def innerCandidates527_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates527_5_checked : candidates edge 4 3821 6917 innerTail527_6 best.bound = innerCandidates527_5 := by
  unfold candidates
  have row : edge 6917 = SearchData.row6917.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan527_5 : scan edge (search edge 3) 4 [3821] 3821 innerTail527_5 best = best := by
  rw [innerTail527_5, scan, if_neg (by decide), innerCandidates527_5_checked]
  change scan edge (search edge 3) 4 [3821] 3821 innerTail527_6
    (if 3 ≤ innerCandidates527_5.length then search edge 3 [3821, 6917] innerCandidates527_5 10738 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan527_6

def innerTail527_4 : List Nat := 6863 :: innerTail527_5
def innerCandidates527_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates527_4_checked : candidates edge 4 3821 6863 innerTail527_5 best.bound = innerCandidates527_4 := by
  unfold candidates
  have row : edge 6863 = SearchData.row6863.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan527_4 : scan edge (search edge 3) 4 [3821] 3821 innerTail527_4 best = best := by
  rw [innerTail527_4, scan, if_neg (by decide), innerCandidates527_4_checked]
  change scan edge (search edge 3) 4 [3821] 3821 innerTail527_5
    (if 3 ≤ innerCandidates527_4.length then search edge 3 [3821, 6863] innerCandidates527_4 10684 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan527_5

def innerTail527_3 : List Nat := 6473 :: innerTail527_4
def innerCandidates527_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates527_3_checked : candidates edge 4 3821 6473 innerTail527_4 best.bound = innerCandidates527_3 := by
  unfold candidates
  have row : edge 6473 = SearchData.row6473.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan527_3 : scan edge (search edge 3) 4 [3821] 3821 innerTail527_3 best = best := by
  rw [innerTail527_3, scan, if_neg (by decide), innerCandidates527_3_checked]
  change scan edge (search edge 3) 4 [3821] 3821 innerTail527_4
    (if 3 ≤ innerCandidates527_3.length then search edge 3 [3821, 6473] innerCandidates527_3 10294 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan527_4

def innerTail527_2 : List Nat := 6263 :: innerTail527_3
def innerCandidates527_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates527_2_checked : candidates edge 4 3821 6263 innerTail527_3 best.bound = innerCandidates527_2 := by
  unfold candidates
  have row : edge 6263 = SearchData.row6263.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan527_2 : scan edge (search edge 3) 4 [3821] 3821 innerTail527_2 best = best := by
  rw [innerTail527_2, scan, if_neg (by decide), innerCandidates527_2_checked]
  change scan edge (search edge 3) 4 [3821] 3821 innerTail527_3
    (if 3 ≤ innerCandidates527_2.length then search edge 3 [3821, 6263] innerCandidates527_2 10084 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan527_3

def innerTail527_1 : List Nat := 6047 :: innerTail527_2
def innerCandidates527_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates527_1_checked : candidates edge 4 3821 6047 innerTail527_2 best.bound = innerCandidates527_1 := by
  unfold candidates
  have row : edge 6047 = SearchData.row6047.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan527_1 : scan edge (search edge 3) 4 [3821] 3821 innerTail527_1 best = best := by
  rw [innerTail527_1, scan, if_neg (by decide), innerCandidates527_1_checked]
  change scan edge (search edge 3) 4 [3821] 3821 innerTail527_2
    (if 3 ≤ innerCandidates527_1.length then search edge 3 [3821, 6047] innerCandidates527_1 9868 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan527_2

def innerTail527_0 : List Nat := 4703 :: innerTail527_1
def innerCandidates527_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates527_0_checked : candidates edge 4 3821 4703 innerTail527_1 best.bound = innerCandidates527_0 := by
  unfold candidates
  have row : edge 4703 = SearchData.row4703.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan527_0 : scan edge (search edge 3) 4 [3821] 3821 innerTail527_0 best = best := by
  rw [innerTail527_0, scan, if_neg (by decide), innerCandidates527_0_checked]
  change scan edge (search edge 3) 4 [3821] 3821 innerTail527_1
    (if 3 ≤ innerCandidates527_0.length then search edge 3 [3821, 4703] innerCandidates527_0 8524 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan527_1

theorem rootBranch527 : search edge 4 [3821] rootCandidates527 3821 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3821] 3821 innerTail527_0 best = best
  exact innerScan527_0

def rootCandidates526 : List Nat := [3911, 4001, 4289, 4547, 4943, 5051, 5087, 5507, 6521, 7121, 7253, 8069]
set_option maxRecDepth 4096 in
theorem rootCandidates526_checked : candidates edge 5 0 3803 tail527 best.bound = rootCandidates526 := by
  unfold candidates
  have row : edge 3803 = SearchData.row3803.contains := by funext q; rfl
  rw [row]
  decide

def innerTail526_12 : List Nat := []
theorem innerScan526_12 : scan edge (search edge 3) 4 [3803] 3803 innerTail526_12 best = best := rfl

def innerTail526_11 : List Nat := 8069 :: innerTail526_12
def innerCandidates526_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates526_11_checked : candidates edge 4 3803 8069 innerTail526_12 best.bound = innerCandidates526_11 := by
  unfold candidates
  have row : edge 8069 = SearchData.row8069.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan526_11 : scan edge (search edge 3) 4 [3803] 3803 innerTail526_11 best = best := by
  rw [innerTail526_11, scan, if_neg (by decide), innerCandidates526_11_checked]
  change scan edge (search edge 3) 4 [3803] 3803 innerTail526_12
    (if 3 ≤ innerCandidates526_11.length then search edge 3 [3803, 8069] innerCandidates526_11 11872 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan526_12

def innerTail526_10 : List Nat := 7253 :: innerTail526_11
def innerCandidates526_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates526_10_checked : candidates edge 4 3803 7253 innerTail526_11 best.bound = innerCandidates526_10 := by
  unfold candidates
  have row : edge 7253 = SearchData.row7253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan526_10 : scan edge (search edge 3) 4 [3803] 3803 innerTail526_10 best = best := by
  rw [innerTail526_10, scan, if_neg (by decide), innerCandidates526_10_checked]
  change scan edge (search edge 3) 4 [3803] 3803 innerTail526_11
    (if 3 ≤ innerCandidates526_10.length then search edge 3 [3803, 7253] innerCandidates526_10 11056 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan526_11

def innerTail526_9 : List Nat := 7121 :: innerTail526_10
def innerCandidates526_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates526_9_checked : candidates edge 4 3803 7121 innerTail526_10 best.bound = innerCandidates526_9 := by
  unfold candidates
  have row : edge 7121 = SearchData.row7121.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan526_9 : scan edge (search edge 3) 4 [3803] 3803 innerTail526_9 best = best := by
  rw [innerTail526_9, scan, if_neg (by decide), innerCandidates526_9_checked]
  change scan edge (search edge 3) 4 [3803] 3803 innerTail526_10
    (if 3 ≤ innerCandidates526_9.length then search edge 3 [3803, 7121] innerCandidates526_9 10924 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan526_10

def innerTail526_8 : List Nat := 6521 :: innerTail526_9
def innerCandidates526_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates526_8_checked : candidates edge 4 3803 6521 innerTail526_9 best.bound = innerCandidates526_8 := by
  unfold candidates
  have row : edge 6521 = SearchData.row6521.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan526_8 : scan edge (search edge 3) 4 [3803] 3803 innerTail526_8 best = best := by
  rw [innerTail526_8, scan, if_neg (by decide), innerCandidates526_8_checked]
  change scan edge (search edge 3) 4 [3803] 3803 innerTail526_9
    (if 3 ≤ innerCandidates526_8.length then search edge 3 [3803, 6521] innerCandidates526_8 10324 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan526_9

def innerTail526_7 : List Nat := 5507 :: innerTail526_8
def innerCandidates526_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates526_7_checked : candidates edge 4 3803 5507 innerTail526_8 best.bound = innerCandidates526_7 := by
  unfold candidates
  have row : edge 5507 = SearchData.row5507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan526_7 : scan edge (search edge 3) 4 [3803] 3803 innerTail526_7 best = best := by
  rw [innerTail526_7, scan, if_neg (by decide), innerCandidates526_7_checked]
  change scan edge (search edge 3) 4 [3803] 3803 innerTail526_8
    (if 3 ≤ innerCandidates526_7.length then search edge 3 [3803, 5507] innerCandidates526_7 9310 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan526_8

def innerTail526_6 : List Nat := 5087 :: innerTail526_7
def innerCandidates526_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates526_6_checked : candidates edge 4 3803 5087 innerTail526_7 best.bound = innerCandidates526_6 := by
  unfold candidates
  have row : edge 5087 = SearchData.row5087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan526_6 : scan edge (search edge 3) 4 [3803] 3803 innerTail526_6 best = best := by
  rw [innerTail526_6, scan, if_neg (by decide), innerCandidates526_6_checked]
  change scan edge (search edge 3) 4 [3803] 3803 innerTail526_7
    (if 3 ≤ innerCandidates526_6.length then search edge 3 [3803, 5087] innerCandidates526_6 8890 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan526_7

def innerTail526_5 : List Nat := 5051 :: innerTail526_6
def innerCandidates526_5 : List Nat := [7253]
set_option maxRecDepth 4096 in
theorem innerCandidates526_5_checked : candidates edge 4 3803 5051 innerTail526_6 best.bound = innerCandidates526_5 := by
  unfold candidates
  have row : edge 5051 = SearchData.row5051.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan526_5 : scan edge (search edge 3) 4 [3803] 3803 innerTail526_5 best = best := by
  rw [innerTail526_5, scan, if_neg (by decide), innerCandidates526_5_checked]
  change scan edge (search edge 3) 4 [3803] 3803 innerTail526_6
    (if 3 ≤ innerCandidates526_5.length then search edge 3 [3803, 5051] innerCandidates526_5 8854 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan526_6

def innerTail526_4 : List Nat := 4943 :: innerTail526_5
def innerCandidates526_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates526_4_checked : candidates edge 4 3803 4943 innerTail526_5 best.bound = innerCandidates526_4 := by
  unfold candidates
  have row : edge 4943 = SearchData.row4943.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan526_4 : scan edge (search edge 3) 4 [3803] 3803 innerTail526_4 best = best := by
  rw [innerTail526_4, scan, if_neg (by decide), innerCandidates526_4_checked]
  change scan edge (search edge 3) 4 [3803] 3803 innerTail526_5
    (if 3 ≤ innerCandidates526_4.length then search edge 3 [3803, 4943] innerCandidates526_4 8746 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan526_5

def innerTail526_3 : List Nat := 4547 :: innerTail526_4
def innerCandidates526_3 : List Nat := [4943, 6521]
set_option maxRecDepth 4096 in
theorem innerCandidates526_3_checked : candidates edge 4 3803 4547 innerTail526_4 best.bound = innerCandidates526_3 := by
  unfold candidates
  have row : edge 4547 = SearchData.row4547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan526_3 : scan edge (search edge 3) 4 [3803] 3803 innerTail526_3 best = best := by
  rw [innerTail526_3, scan, if_neg (by decide), innerCandidates526_3_checked]
  change scan edge (search edge 3) 4 [3803] 3803 innerTail526_4
    (if 3 ≤ innerCandidates526_3.length then search edge 3 [3803, 4547] innerCandidates526_3 8350 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan526_4

def innerTail526_2 : List Nat := 4289 :: innerTail526_3
def innerCandidates526_2 : List Nat := [5507]
set_option maxRecDepth 4096 in
theorem innerCandidates526_2_checked : candidates edge 4 3803 4289 innerTail526_3 best.bound = innerCandidates526_2 := by
  unfold candidates
  have row : edge 4289 = SearchData.row4289.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan526_2 : scan edge (search edge 3) 4 [3803] 3803 innerTail526_2 best = best := by
  rw [innerTail526_2, scan, if_neg (by decide), innerCandidates526_2_checked]
  change scan edge (search edge 3) 4 [3803] 3803 innerTail526_3
    (if 3 ≤ innerCandidates526_2.length then search edge 3 [3803, 4289] innerCandidates526_2 8092 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan526_3

def innerTail526_1 : List Nat := 4001 :: innerTail526_2
def innerCandidates526_1 : List Nat := [7121]
set_option maxRecDepth 4096 in
theorem innerCandidates526_1_checked : candidates edge 4 3803 4001 innerTail526_2 best.bound = innerCandidates526_1 := by
  unfold candidates
  have row : edge 4001 = SearchData.row4001.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan526_1 : scan edge (search edge 3) 4 [3803] 3803 innerTail526_1 best = best := by
  rw [innerTail526_1, scan, if_neg (by decide), innerCandidates526_1_checked]
  change scan edge (search edge 3) 4 [3803] 3803 innerTail526_2
    (if 3 ≤ innerCandidates526_1.length then search edge 3 [3803, 4001] innerCandidates526_1 7804 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan526_2

def innerTail526_0 : List Nat := 3911 :: innerTail526_1
def innerCandidates526_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates526_0_checked : candidates edge 4 3803 3911 innerTail526_1 best.bound = innerCandidates526_0 := by
  unfold candidates
  have row : edge 3911 = SearchData.row3911.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan526_0 : scan edge (search edge 3) 4 [3803] 3803 innerTail526_0 best = best := by
  rw [innerTail526_0, scan, if_neg (by decide), innerCandidates526_0_checked]
  change scan edge (search edge 3) 4 [3803] 3803 innerTail526_1
    (if 3 ≤ innerCandidates526_0.length then search edge 3 [3803, 3911] innerCandidates526_0 7714 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan526_1

theorem rootBranch526 : search edge 4 [3803] rootCandidates526 3803 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3803] 3803 innerTail526_0 best = best
  exact innerScan526_0

end ProjectEulerSolutions.PrimePairUpper.Search
