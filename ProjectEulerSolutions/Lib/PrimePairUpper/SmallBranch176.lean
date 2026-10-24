import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates156 : List Nat := [1489, 1783, 2053, 2131, 2557, 2791, 3109, 3319, 3853, 3943, 4093, 4159, 4273, 5059, 5281, 5407, 5437, 5791, 5827, 7213, 8059, 8233]
set_option maxRecDepth 4096 in
theorem rootCandidates156_checked : candidates edge 5 0 937 tail157 best.bound = rootCandidates156 := by
  unfold candidates
  have row : edge 937 = PrimePairCache.smallRow937.contains := by funext q; rfl
  rw [row]
  decide

def innerTail156_22 : List Nat := []
theorem innerScan156_22 : scan edge (search edge 3) 4 [937] 937 innerTail156_22 best = best := rfl

def innerTail156_21 : List Nat := 8233 :: innerTail156_22
def innerCandidates156_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates156_21_checked : candidates edge 4 937 8233 innerTail156_22 best.bound = innerCandidates156_21 := by
  unfold candidates
  have row : edge 8233 = SearchData.row8233.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan156_21 : scan edge (search edge 3) 4 [937] 937 innerTail156_21 best = best := by
  rw [innerTail156_21, scan, if_neg (by decide), innerCandidates156_21_checked]
  change scan edge (search edge 3) 4 [937] 937 innerTail156_22
    (if 3 ≤ innerCandidates156_21.length then search edge 3 [937, 8233] innerCandidates156_21 9170 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan156_22

def innerTail156_20 : List Nat := 8059 :: innerTail156_21
def innerCandidates156_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates156_20_checked : candidates edge 4 937 8059 innerTail156_21 best.bound = innerCandidates156_20 := by
  unfold candidates
  have row : edge 8059 = SearchData.row8059.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan156_20 : scan edge (search edge 3) 4 [937] 937 innerTail156_20 best = best := by
  rw [innerTail156_20, scan, if_neg (by decide), innerCandidates156_20_checked]
  change scan edge (search edge 3) 4 [937] 937 innerTail156_21
    (if 3 ≤ innerCandidates156_20.length then search edge 3 [937, 8059] innerCandidates156_20 8996 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan156_21

def innerTail156_19 : List Nat := 7213 :: innerTail156_20
def innerCandidates156_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates156_19_checked : candidates edge 4 937 7213 innerTail156_20 best.bound = innerCandidates156_19 := by
  unfold candidates
  have row : edge 7213 = SearchData.row7213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan156_19 : scan edge (search edge 3) 4 [937] 937 innerTail156_19 best = best := by
  rw [innerTail156_19, scan, if_neg (by decide), innerCandidates156_19_checked]
  change scan edge (search edge 3) 4 [937] 937 innerTail156_20
    (if 3 ≤ innerCandidates156_19.length then search edge 3 [937, 7213] innerCandidates156_19 8150 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan156_20

def innerTail156_18 : List Nat := 5827 :: innerTail156_19
def innerCandidates156_18 : List Nat := [7213]
set_option maxRecDepth 4096 in
theorem innerCandidates156_18_checked : candidates edge 4 937 5827 innerTail156_19 best.bound = innerCandidates156_18 := by
  unfold candidates
  have row : edge 5827 = SearchData.row5827.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan156_18 : scan edge (search edge 3) 4 [937] 937 innerTail156_18 best = best := by
  rw [innerTail156_18, scan, if_neg (by decide), innerCandidates156_18_checked]
  change scan edge (search edge 3) 4 [937] 937 innerTail156_19
    (if 3 ≤ innerCandidates156_18.length then search edge 3 [937, 5827] innerCandidates156_18 6764 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan156_19

def innerTail156_17 : List Nat := 5791 :: innerTail156_18
def innerCandidates156_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates156_17_checked : candidates edge 4 937 5791 innerTail156_18 best.bound = innerCandidates156_17 := by
  unfold candidates
  have row : edge 5791 = SearchData.row5791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan156_17 : scan edge (search edge 3) 4 [937] 937 innerTail156_17 best = best := by
  rw [innerTail156_17, scan, if_neg (by decide), innerCandidates156_17_checked]
  change scan edge (search edge 3) 4 [937] 937 innerTail156_18
    (if 3 ≤ innerCandidates156_17.length then search edge 3 [937, 5791] innerCandidates156_17 6728 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan156_18

def innerTail156_16 : List Nat := 5437 :: innerTail156_17
def innerCandidates156_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates156_16_checked : candidates edge 4 937 5437 innerTail156_17 best.bound = innerCandidates156_16 := by
  unfold candidates
  have row : edge 5437 = SearchData.row5437.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan156_16 : scan edge (search edge 3) 4 [937] 937 innerTail156_16 best = best := by
  rw [innerTail156_16, scan, if_neg (by decide), innerCandidates156_16_checked]
  change scan edge (search edge 3) 4 [937] 937 innerTail156_17
    (if 3 ≤ innerCandidates156_16.length then search edge 3 [937, 5437] innerCandidates156_16 6374 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan156_17

def innerTail156_15 : List Nat := 5407 :: innerTail156_16
def innerCandidates156_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates156_15_checked : candidates edge 4 937 5407 innerTail156_16 best.bound = innerCandidates156_15 := by
  unfold candidates
  have row : edge 5407 = SearchData.row5407.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan156_15 : scan edge (search edge 3) 4 [937] 937 innerTail156_15 best = best := by
  rw [innerTail156_15, scan, if_neg (by decide), innerCandidates156_15_checked]
  change scan edge (search edge 3) 4 [937] 937 innerTail156_16
    (if 3 ≤ innerCandidates156_15.length then search edge 3 [937, 5407] innerCandidates156_15 6344 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan156_16

def innerTail156_14 : List Nat := 5281 :: innerTail156_15
def innerCandidates156_14 : List Nat := [5437]
set_option maxRecDepth 4096 in
theorem innerCandidates156_14_checked : candidates edge 4 937 5281 innerTail156_15 best.bound = innerCandidates156_14 := by
  unfold candidates
  have row : edge 5281 = SearchData.row5281.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan156_14 : scan edge (search edge 3) 4 [937] 937 innerTail156_14 best = best := by
  rw [innerTail156_14, scan, if_neg (by decide), innerCandidates156_14_checked]
  change scan edge (search edge 3) 4 [937] 937 innerTail156_15
    (if 3 ≤ innerCandidates156_14.length then search edge 3 [937, 5281] innerCandidates156_14 6218 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan156_15

def innerTail156_13 : List Nat := 5059 :: innerTail156_14
def innerCandidates156_13 : List Nat := [7213]
set_option maxRecDepth 4096 in
theorem innerCandidates156_13_checked : candidates edge 4 937 5059 innerTail156_14 best.bound = innerCandidates156_13 := by
  unfold candidates
  have row : edge 5059 = SearchData.row5059.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan156_13 : scan edge (search edge 3) 4 [937] 937 innerTail156_13 best = best := by
  rw [innerTail156_13, scan, if_neg (by decide), innerCandidates156_13_checked]
  change scan edge (search edge 3) 4 [937] 937 innerTail156_14
    (if 3 ≤ innerCandidates156_13.length then search edge 3 [937, 5059] innerCandidates156_13 5996 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan156_14

def innerTail156_12 : List Nat := 4273 :: innerTail156_13
def innerCandidates156_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates156_12_checked : candidates edge 4 937 4273 innerTail156_13 best.bound = innerCandidates156_12 := by
  unfold candidates
  have row : edge 4273 = SearchData.row4273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan156_12 : scan edge (search edge 3) 4 [937] 937 innerTail156_12 best = best := by
  rw [innerTail156_12, scan, if_neg (by decide), innerCandidates156_12_checked]
  change scan edge (search edge 3) 4 [937] 937 innerTail156_13
    (if 3 ≤ innerCandidates156_12.length then search edge 3 [937, 4273] innerCandidates156_12 5210 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan156_13

def innerTail156_11 : List Nat := 4159 :: innerTail156_12
def innerCandidates156_11 : List Nat := [8233]
set_option maxRecDepth 4096 in
theorem innerCandidates156_11_checked : candidates edge 4 937 4159 innerTail156_12 best.bound = innerCandidates156_11 := by
  unfold candidates
  have row : edge 4159 = SearchData.row4159.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan156_11 : scan edge (search edge 3) 4 [937] 937 innerTail156_11 best = best := by
  rw [innerTail156_11, scan, if_neg (by decide), innerCandidates156_11_checked]
  change scan edge (search edge 3) 4 [937] 937 innerTail156_12
    (if 3 ≤ innerCandidates156_11.length then search edge 3 [937, 4159] innerCandidates156_11 5096 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan156_12

def innerTail156_10 : List Nat := 4093 :: innerTail156_11
def innerCandidates156_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates156_10_checked : candidates edge 4 937 4093 innerTail156_11 best.bound = innerCandidates156_10 := by
  unfold candidates
  have row : edge 4093 = SearchData.row4093.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan156_10 : scan edge (search edge 3) 4 [937] 937 innerTail156_10 best = best := by
  rw [innerTail156_10, scan, if_neg (by decide), innerCandidates156_10_checked]
  change scan edge (search edge 3) 4 [937] 937 innerTail156_11
    (if 3 ≤ innerCandidates156_10.length then search edge 3 [937, 4093] innerCandidates156_10 5030 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan156_11

def innerTail156_9 : List Nat := 3943 :: innerTail156_10
def innerCandidates156_9 : List Nat := [8059]
set_option maxRecDepth 4096 in
theorem innerCandidates156_9_checked : candidates edge 4 937 3943 innerTail156_10 best.bound = innerCandidates156_9 := by
  unfold candidates
  have row : edge 3943 = SearchData.row3943.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan156_9 : scan edge (search edge 3) 4 [937] 937 innerTail156_9 best = best := by
  rw [innerTail156_9, scan, if_neg (by decide), innerCandidates156_9_checked]
  change scan edge (search edge 3) 4 [937] 937 innerTail156_10
    (if 3 ≤ innerCandidates156_9.length then search edge 3 [937, 3943] innerCandidates156_9 4880 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan156_10

def innerTail156_8 : List Nat := 3853 :: innerTail156_9
def innerCandidates156_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates156_8_checked : candidates edge 4 937 3853 innerTail156_9 best.bound = innerCandidates156_8 := by
  unfold candidates
  have row : edge 3853 = SearchData.row3853.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan156_8 : scan edge (search edge 3) 4 [937] 937 innerTail156_8 best = best := by
  rw [innerTail156_8, scan, if_neg (by decide), innerCandidates156_8_checked]
  change scan edge (search edge 3) 4 [937] 937 innerTail156_9
    (if 3 ≤ innerCandidates156_8.length then search edge 3 [937, 3853] innerCandidates156_8 4790 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan156_9

def innerTail156_7 : List Nat := 3319 :: innerTail156_8
def innerCandidates156_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates156_7_checked : candidates edge 4 937 3319 innerTail156_8 best.bound = innerCandidates156_7 := by
  unfold candidates
  have row : edge 3319 = SearchData.row3319.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan156_7 : scan edge (search edge 3) 4 [937] 937 innerTail156_7 best = best := by
  rw [innerTail156_7, scan, if_neg (by decide), innerCandidates156_7_checked]
  change scan edge (search edge 3) 4 [937] 937 innerTail156_8
    (if 3 ≤ innerCandidates156_7.length then search edge 3 [937, 3319] innerCandidates156_7 4256 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan156_8

def innerTail156_6 : List Nat := 3109 :: innerTail156_7
def innerCandidates156_6 : List Nat := [5791]
set_option maxRecDepth 4096 in
theorem innerCandidates156_6_checked : candidates edge 4 937 3109 innerTail156_7 best.bound = innerCandidates156_6 := by
  unfold candidates
  have row : edge 3109 = SearchData.row3109.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan156_6 : scan edge (search edge 3) 4 [937] 937 innerTail156_6 best = best := by
  rw [innerTail156_6, scan, if_neg (by decide), innerCandidates156_6_checked]
  change scan edge (search edge 3) 4 [937] 937 innerTail156_7
    (if 3 ≤ innerCandidates156_6.length then search edge 3 [937, 3109] innerCandidates156_6 4046 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan156_7

def innerTail156_5 : List Nat := 2791 :: innerTail156_6
def innerCandidates156_5 : List Nat := [5281]
set_option maxRecDepth 4096 in
theorem innerCandidates156_5_checked : candidates edge 4 937 2791 innerTail156_6 best.bound = innerCandidates156_5 := by
  unfold candidates
  have row : edge 2791 = SearchData.row2791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan156_5 : scan edge (search edge 3) 4 [937] 937 innerTail156_5 best = best := by
  rw [innerTail156_5, scan, if_neg (by decide), innerCandidates156_5_checked]
  change scan edge (search edge 3) 4 [937] 937 innerTail156_6
    (if 3 ≤ innerCandidates156_5.length then search edge 3 [937, 2791] innerCandidates156_5 3728 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan156_6

def innerTail156_4 : List Nat := 2557 :: innerTail156_5
def innerCandidates156_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates156_4_checked : candidates edge 4 937 2557 innerTail156_5 best.bound = innerCandidates156_4 := by
  unfold candidates
  have row : edge 2557 = PrimePairCache.smallRow2557.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan156_4 : scan edge (search edge 3) 4 [937] 937 innerTail156_4 best = best := by
  rw [innerTail156_4, scan, if_neg (by decide), innerCandidates156_4_checked]
  change scan edge (search edge 3) 4 [937] 937 innerTail156_5
    (if 3 ≤ innerCandidates156_4.length then search edge 3 [937, 2557] innerCandidates156_4 3494 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan156_5

def innerTail156_3 : List Nat := 2131 :: innerTail156_4
def innerCandidates156_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates156_3_checked : candidates edge 4 937 2131 innerTail156_4 best.bound = innerCandidates156_3 := by
  unfold candidates
  have row : edge 2131 = PrimePairCache.smallRow2131.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan156_3 : scan edge (search edge 3) 4 [937] 937 innerTail156_3 best = best := by
  rw [innerTail156_3, scan, if_neg (by decide), innerCandidates156_3_checked]
  change scan edge (search edge 3) 4 [937] 937 innerTail156_4
    (if 3 ≤ innerCandidates156_3.length then search edge 3 [937, 2131] innerCandidates156_3 3068 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan156_4

def innerTail156_2 : List Nat := 2053 :: innerTail156_3
def innerCandidates156_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates156_2_checked : candidates edge 4 937 2053 innerTail156_3 best.bound = innerCandidates156_2 := by
  unfold candidates
  have row : edge 2053 = PrimePairCache.smallRow2053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan156_2 : scan edge (search edge 3) 4 [937] 937 innerTail156_2 best = best := by
  rw [innerTail156_2, scan, if_neg (by decide), innerCandidates156_2_checked]
  change scan edge (search edge 3) 4 [937] 937 innerTail156_3
    (if 3 ≤ innerCandidates156_2.length then search edge 3 [937, 2053] innerCandidates156_2 2990 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan156_3

def innerTail156_1 : List Nat := 1783 :: innerTail156_2
def innerCandidates156_1 : List Nat := [2053, 2791, 4273]
set_option maxRecDepth 4096 in
theorem innerCandidates156_1_checked : candidates edge 4 937 1783 innerTail156_2 best.bound = innerCandidates156_1 := by
  unfold candidates
  have row : edge 1783 = PrimePairCache.smallRow1783.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild156_1 : search edge 3 [937, 1783] innerCandidates156_1 2720 best = best := by decide

theorem innerScan156_1 : scan edge (search edge 3) 4 [937] 937 innerTail156_1 best = best := by
  rw [innerTail156_1, scan, if_neg (by decide), innerCandidates156_1_checked]
  change scan edge (search edge 3) 4 [937] 937 innerTail156_2
    (if 3 ≤ innerCandidates156_1.length then search edge 3 [937, 1783] innerCandidates156_1 2720 best else best) = best
  rw [if_pos (by decide), innerChild156_1]
  exact innerScan156_2

def innerTail156_0 : List Nat := 1489 :: innerTail156_1
def innerCandidates156_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates156_0_checked : candidates edge 4 937 1489 innerTail156_1 best.bound = innerCandidates156_0 := by
  unfold candidates
  have row : edge 1489 = PrimePairCache.smallRow1489.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan156_0 : scan edge (search edge 3) 4 [937] 937 innerTail156_0 best = best := by
  rw [innerTail156_0, scan, if_neg (by decide), innerCandidates156_0_checked]
  change scan edge (search edge 3) 4 [937] 937 innerTail156_1
    (if 3 ≤ innerCandidates156_0.length then search edge 3 [937, 1489] innerCandidates156_0 2426 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan156_1

theorem rootBranch156 : search edge 4 [937] rootCandidates156 937 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [937] 937 innerTail156_0 best = best
  exact innerScan156_0

def rootCandidates155 : List Nat := [941, 953, 983, 2039, 2477, 2741, 3209, 3467, 3539, 3761, 4013, 4073, 4421, 4463, 4973, 5639, 5669, 6563, 7109, 7853, 8117, 8369]
set_option maxRecDepth 4096 in
theorem rootCandidates155_checked : candidates edge 5 0 929 tail156 best.bound = rootCandidates155 := by
  unfold candidates
  have row : edge 929 = PrimePairCache.smallRow929.contains := by funext q; rfl
  rw [row]
  decide

def innerTail155_22 : List Nat := []
theorem innerScan155_22 : scan edge (search edge 3) 4 [929] 929 innerTail155_22 best = best := rfl

def innerTail155_21 : List Nat := 8369 :: innerTail155_22
def innerCandidates155_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates155_21_checked : candidates edge 4 929 8369 innerTail155_22 best.bound = innerCandidates155_21 := by
  unfold candidates
  have row : edge 8369 = SearchData.row8369.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan155_21 : scan edge (search edge 3) 4 [929] 929 innerTail155_21 best = best := by
  rw [innerTail155_21, scan, if_neg (by decide), innerCandidates155_21_checked]
  change scan edge (search edge 3) 4 [929] 929 innerTail155_22
    (if 3 ≤ innerCandidates155_21.length then search edge 3 [929, 8369] innerCandidates155_21 9298 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan155_22

def innerTail155_20 : List Nat := 8117 :: innerTail155_21
def innerCandidates155_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates155_20_checked : candidates edge 4 929 8117 innerTail155_21 best.bound = innerCandidates155_20 := by
  unfold candidates
  have row : edge 8117 = SearchData.row8117.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan155_20 : scan edge (search edge 3) 4 [929] 929 innerTail155_20 best = best := by
  rw [innerTail155_20, scan, if_neg (by decide), innerCandidates155_20_checked]
  change scan edge (search edge 3) 4 [929] 929 innerTail155_21
    (if 3 ≤ innerCandidates155_20.length then search edge 3 [929, 8117] innerCandidates155_20 9046 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan155_21

def innerTail155_19 : List Nat := 7853 :: innerTail155_20
def innerCandidates155_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates155_19_checked : candidates edge 4 929 7853 innerTail155_20 best.bound = innerCandidates155_19 := by
  unfold candidates
  have row : edge 7853 = SearchData.row7853.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan155_19 : scan edge (search edge 3) 4 [929] 929 innerTail155_19 best = best := by
  rw [innerTail155_19, scan, if_neg (by decide), innerCandidates155_19_checked]
  change scan edge (search edge 3) 4 [929] 929 innerTail155_20
    (if 3 ≤ innerCandidates155_19.length then search edge 3 [929, 7853] innerCandidates155_19 8782 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan155_20

def innerTail155_18 : List Nat := 7109 :: innerTail155_19
def innerCandidates155_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates155_18_checked : candidates edge 4 929 7109 innerTail155_19 best.bound = innerCandidates155_18 := by
  unfold candidates
  have row : edge 7109 = SearchData.row7109.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan155_18 : scan edge (search edge 3) 4 [929] 929 innerTail155_18 best = best := by
  rw [innerTail155_18, scan, if_neg (by decide), innerCandidates155_18_checked]
  change scan edge (search edge 3) 4 [929] 929 innerTail155_19
    (if 3 ≤ innerCandidates155_18.length then search edge 3 [929, 7109] innerCandidates155_18 8038 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan155_19

def innerTail155_17 : List Nat := 6563 :: innerTail155_18
def innerCandidates155_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates155_17_checked : candidates edge 4 929 6563 innerTail155_18 best.bound = innerCandidates155_17 := by
  unfold candidates
  have row : edge 6563 = SearchData.row6563.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan155_17 : scan edge (search edge 3) 4 [929] 929 innerTail155_17 best = best := by
  rw [innerTail155_17, scan, if_neg (by decide), innerCandidates155_17_checked]
  change scan edge (search edge 3) 4 [929] 929 innerTail155_18
    (if 3 ≤ innerCandidates155_17.length then search edge 3 [929, 6563] innerCandidates155_17 7492 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan155_18

def innerTail155_16 : List Nat := 5669 :: innerTail155_17
def innerCandidates155_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates155_16_checked : candidates edge 4 929 5669 innerTail155_17 best.bound = innerCandidates155_16 := by
  unfold candidates
  have row : edge 5669 = SearchData.row5669.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan155_16 : scan edge (search edge 3) 4 [929] 929 innerTail155_16 best = best := by
  rw [innerTail155_16, scan, if_neg (by decide), innerCandidates155_16_checked]
  change scan edge (search edge 3) 4 [929] 929 innerTail155_17
    (if 3 ≤ innerCandidates155_16.length then search edge 3 [929, 5669] innerCandidates155_16 6598 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan155_17

def innerTail155_15 : List Nat := 5639 :: innerTail155_16
def innerCandidates155_15 : List Nat := [6563]
set_option maxRecDepth 4096 in
theorem innerCandidates155_15_checked : candidates edge 4 929 5639 innerTail155_16 best.bound = innerCandidates155_15 := by
  unfold candidates
  have row : edge 5639 = SearchData.row5639.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan155_15 : scan edge (search edge 3) 4 [929] 929 innerTail155_15 best = best := by
  rw [innerTail155_15, scan, if_neg (by decide), innerCandidates155_15_checked]
  change scan edge (search edge 3) 4 [929] 929 innerTail155_16
    (if 3 ≤ innerCandidates155_15.length then search edge 3 [929, 5639] innerCandidates155_15 6568 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan155_16

def innerTail155_14 : List Nat := 4973 :: innerTail155_15
def innerCandidates155_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates155_14_checked : candidates edge 4 929 4973 innerTail155_15 best.bound = innerCandidates155_14 := by
  unfold candidates
  have row : edge 4973 = SearchData.row4973.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan155_14 : scan edge (search edge 3) 4 [929] 929 innerTail155_14 best = best := by
  rw [innerTail155_14, scan, if_neg (by decide), innerCandidates155_14_checked]
  change scan edge (search edge 3) 4 [929] 929 innerTail155_15
    (if 3 ≤ innerCandidates155_14.length then search edge 3 [929, 4973] innerCandidates155_14 5902 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan155_15

def innerTail155_13 : List Nat := 4463 :: innerTail155_14
def innerCandidates155_13 : List Nat := [8369]
set_option maxRecDepth 4096 in
theorem innerCandidates155_13_checked : candidates edge 4 929 4463 innerTail155_14 best.bound = innerCandidates155_13 := by
  unfold candidates
  have row : edge 4463 = SearchData.row4463.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan155_13 : scan edge (search edge 3) 4 [929] 929 innerTail155_13 best = best := by
  rw [innerTail155_13, scan, if_neg (by decide), innerCandidates155_13_checked]
  change scan edge (search edge 3) 4 [929] 929 innerTail155_14
    (if 3 ≤ innerCandidates155_13.length then search edge 3 [929, 4463] innerCandidates155_13 5392 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan155_14

def innerTail155_12 : List Nat := 4421 :: innerTail155_13
def innerCandidates155_12 : List Nat := [6563]
set_option maxRecDepth 4096 in
theorem innerCandidates155_12_checked : candidates edge 4 929 4421 innerTail155_13 best.bound = innerCandidates155_12 := by
  unfold candidates
  have row : edge 4421 = SearchData.row4421.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan155_12 : scan edge (search edge 3) 4 [929] 929 innerTail155_12 best = best := by
  rw [innerTail155_12, scan, if_neg (by decide), innerCandidates155_12_checked]
  change scan edge (search edge 3) 4 [929] 929 innerTail155_13
    (if 3 ≤ innerCandidates155_12.length then search edge 3 [929, 4421] innerCandidates155_12 5350 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan155_13

def innerTail155_11 : List Nat := 4073 :: innerTail155_12
def innerCandidates155_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates155_11_checked : candidates edge 4 929 4073 innerTail155_12 best.bound = innerCandidates155_11 := by
  unfold candidates
  have row : edge 4073 = SearchData.row4073.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan155_11 : scan edge (search edge 3) 4 [929] 929 innerTail155_11 best = best := by
  rw [innerTail155_11, scan, if_neg (by decide), innerCandidates155_11_checked]
  change scan edge (search edge 3) 4 [929] 929 innerTail155_12
    (if 3 ≤ innerCandidates155_11.length then search edge 3 [929, 4073] innerCandidates155_11 5002 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan155_12

def innerTail155_10 : List Nat := 4013 :: innerTail155_11
def innerCandidates155_10 : List Nat := [4421]
set_option maxRecDepth 4096 in
theorem innerCandidates155_10_checked : candidates edge 4 929 4013 innerTail155_11 best.bound = innerCandidates155_10 := by
  unfold candidates
  have row : edge 4013 = SearchData.row4013.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan155_10 : scan edge (search edge 3) 4 [929] 929 innerTail155_10 best = best := by
  rw [innerTail155_10, scan, if_neg (by decide), innerCandidates155_10_checked]
  change scan edge (search edge 3) 4 [929] 929 innerTail155_11
    (if 3 ≤ innerCandidates155_10.length then search edge 3 [929, 4013] innerCandidates155_10 4942 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan155_11

def innerTail155_9 : List Nat := 3761 :: innerTail155_10
def innerCandidates155_9 : List Nat := [8369]
set_option maxRecDepth 4096 in
theorem innerCandidates155_9_checked : candidates edge 4 929 3761 innerTail155_10 best.bound = innerCandidates155_9 := by
  unfold candidates
  have row : edge 3761 = SearchData.row3761.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan155_9 : scan edge (search edge 3) 4 [929] 929 innerTail155_9 best = best := by
  rw [innerTail155_9, scan, if_neg (by decide), innerCandidates155_9_checked]
  change scan edge (search edge 3) 4 [929] 929 innerTail155_10
    (if 3 ≤ innerCandidates155_9.length then search edge 3 [929, 3761] innerCandidates155_9 4690 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan155_10

def innerTail155_8 : List Nat := 3539 :: innerTail155_9
def innerCandidates155_8 : List Nat := [4013]
set_option maxRecDepth 4096 in
theorem innerCandidates155_8_checked : candidates edge 4 929 3539 innerTail155_9 best.bound = innerCandidates155_8 := by
  unfold candidates
  have row : edge 3539 = SearchData.row3539.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan155_8 : scan edge (search edge 3) 4 [929] 929 innerTail155_8 best = best := by
  rw [innerTail155_8, scan, if_neg (by decide), innerCandidates155_8_checked]
  change scan edge (search edge 3) 4 [929] 929 innerTail155_9
    (if 3 ≤ innerCandidates155_8.length then search edge 3 [929, 3539] innerCandidates155_8 4468 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan155_9

def innerTail155_7 : List Nat := 3467 :: innerTail155_8
def innerCandidates155_7 : List Nat := [8117]
set_option maxRecDepth 4096 in
theorem innerCandidates155_7_checked : candidates edge 4 929 3467 innerTail155_8 best.bound = innerCandidates155_7 := by
  unfold candidates
  have row : edge 3467 = SearchData.row3467.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan155_7 : scan edge (search edge 3) 4 [929] 929 innerTail155_7 best = best := by
  rw [innerTail155_7, scan, if_neg (by decide), innerCandidates155_7_checked]
  change scan edge (search edge 3) 4 [929] 929 innerTail155_8
    (if 3 ≤ innerCandidates155_7.length then search edge 3 [929, 3467] innerCandidates155_7 4396 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan155_8

def innerTail155_6 : List Nat := 3209 :: innerTail155_7
def innerCandidates155_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates155_6_checked : candidates edge 4 929 3209 innerTail155_7 best.bound = innerCandidates155_6 := by
  unfold candidates
  have row : edge 3209 = SearchData.row3209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan155_6 : scan edge (search edge 3) 4 [929] 929 innerTail155_6 best = best := by
  rw [innerTail155_6, scan, if_neg (by decide), innerCandidates155_6_checked]
  change scan edge (search edge 3) 4 [929] 929 innerTail155_7
    (if 3 ≤ innerCandidates155_6.length then search edge 3 [929, 3209] innerCandidates155_6 4138 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan155_7

def innerTail155_5 : List Nat := 2741 :: innerTail155_6
def innerCandidates155_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates155_5_checked : candidates edge 4 929 2741 innerTail155_6 best.bound = innerCandidates155_5 := by
  unfold candidates
  have row : edge 2741 = SearchData.row2741.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan155_5 : scan edge (search edge 3) 4 [929] 929 innerTail155_5 best = best := by
  rw [innerTail155_5, scan, if_neg (by decide), innerCandidates155_5_checked]
  change scan edge (search edge 3) 4 [929] 929 innerTail155_6
    (if 3 ≤ innerCandidates155_5.length then search edge 3 [929, 2741] innerCandidates155_5 3670 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan155_6

def innerTail155_4 : List Nat := 2477 :: innerTail155_5
def innerCandidates155_4 : List Nat := [4073, 7853]
set_option maxRecDepth 4096 in
theorem innerCandidates155_4_checked : candidates edge 4 929 2477 innerTail155_5 best.bound = innerCandidates155_4 := by
  unfold candidates
  have row : edge 2477 = PrimePairCache.smallRow2477.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan155_4 : scan edge (search edge 3) 4 [929] 929 innerTail155_4 best = best := by
  rw [innerTail155_4, scan, if_neg (by decide), innerCandidates155_4_checked]
  change scan edge (search edge 3) 4 [929] 929 innerTail155_5
    (if 3 ≤ innerCandidates155_4.length then search edge 3 [929, 2477] innerCandidates155_4 3406 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan155_5

def innerTail155_3 : List Nat := 2039 :: innerTail155_4
def innerCandidates155_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates155_3_checked : candidates edge 4 929 2039 innerTail155_4 best.bound = innerCandidates155_3 := by
  unfold candidates
  have row : edge 2039 = PrimePairCache.smallRow2039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan155_3 : scan edge (search edge 3) 4 [929] 929 innerTail155_3 best = best := by
  rw [innerTail155_3, scan, if_neg (by decide), innerCandidates155_3_checked]
  change scan edge (search edge 3) 4 [929] 929 innerTail155_4
    (if 3 ≤ innerCandidates155_3.length then search edge 3 [929, 2039] innerCandidates155_3 2968 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan155_4

def innerTail155_2 : List Nat := 983 :: innerTail155_3
def innerCandidates155_2 : List Nat := [4073, 7109]
set_option maxRecDepth 4096 in
theorem innerCandidates155_2_checked : candidates edge 4 929 983 innerTail155_3 best.bound = innerCandidates155_2 := by
  unfold candidates
  have row : edge 983 = PrimePairCache.smallRow983.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan155_2 : scan edge (search edge 3) 4 [929] 929 innerTail155_2 best = best := by
  rw [innerTail155_2, scan, if_neg (by decide), innerCandidates155_2_checked]
  change scan edge (search edge 3) 4 [929] 929 innerTail155_3
    (if 3 ≤ innerCandidates155_2.length then search edge 3 [929, 983] innerCandidates155_2 1912 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan155_3

def innerTail155_1 : List Nat := 953 :: innerTail155_2
def innerCandidates155_1 : List Nat := [7109]
set_option maxRecDepth 4096 in
theorem innerCandidates155_1_checked : candidates edge 4 929 953 innerTail155_2 best.bound = innerCandidates155_1 := by
  unfold candidates
  have row : edge 953 = PrimePairCache.smallRow953.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan155_1 : scan edge (search edge 3) 4 [929] 929 innerTail155_1 best = best := by
  rw [innerTail155_1, scan, if_neg (by decide), innerCandidates155_1_checked]
  change scan edge (search edge 3) 4 [929] 929 innerTail155_2
    (if 3 ≤ innerCandidates155_1.length then search edge 3 [929, 953] innerCandidates155_1 1882 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan155_2

def innerTail155_0 : List Nat := 941 :: innerTail155_1
def innerCandidates155_0 : List Nat := [8369]
set_option maxRecDepth 4096 in
theorem innerCandidates155_0_checked : candidates edge 4 929 941 innerTail155_1 best.bound = innerCandidates155_0 := by
  unfold candidates
  have row : edge 941 = PrimePairCache.smallRow941.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan155_0 : scan edge (search edge 3) 4 [929] 929 innerTail155_0 best = best := by
  rw [innerTail155_0, scan, if_neg (by decide), innerCandidates155_0_checked]
  change scan edge (search edge 3) 4 [929] 929 innerTail155_1
    (if 3 ≤ innerCandidates155_0.length then search edge 3 [929, 941] innerCandidates155_0 1870 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan155_1

theorem rootBranch155 : search edge 4 [929] rootCandidates155 929 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [929] 929 innerTail155_0 best = best
  exact innerScan155_0

end ProjectEulerSolutions.PrimePairUpper.Search
