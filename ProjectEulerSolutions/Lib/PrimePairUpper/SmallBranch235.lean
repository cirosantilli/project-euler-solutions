import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates37 : List Nat := [191, 293, 347, 431, 659, 743, 1091, 2039, 2333, 3041, 3929, 4673, 4721, 4787, 4937, 5333, 5807, 5813, 7331, 7529, 7559, 8039, 8117, 8171]
set_option maxRecDepth 4096 in
theorem rootCandidates37_checked : candidates edge 5 0 173 tail38 best.bound = rootCandidates37 := by
  unfold candidates
  have row : edge 173 = PrimePairCache.smallRow173.contains := by funext q; rfl
  rw [row]
  decide

def innerTail37_24 : List Nat := []
theorem innerScan37_24 : scan edge (search edge 3) 4 [173] 173 innerTail37_24 best = best := rfl

def innerTail37_23 : List Nat := 8171 :: innerTail37_24
def innerCandidates37_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates37_23_checked : candidates edge 4 173 8171 innerTail37_24 best.bound = innerCandidates37_23 := by
  unfold candidates
  have row : edge 8171 = SearchData.row8171.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan37_23 : scan edge (search edge 3) 4 [173] 173 innerTail37_23 best = best := by
  rw [innerTail37_23, scan, if_neg (by decide), innerCandidates37_23_checked]
  change scan edge (search edge 3) 4 [173] 173 innerTail37_24
    (if 3 ≤ innerCandidates37_23.length then search edge 3 [173, 8171] innerCandidates37_23 8344 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan37_24

def innerTail37_22 : List Nat := 8117 :: innerTail37_23
def innerCandidates37_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates37_22_checked : candidates edge 4 173 8117 innerTail37_23 best.bound = innerCandidates37_22 := by
  unfold candidates
  have row : edge 8117 = SearchData.row8117.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan37_22 : scan edge (search edge 3) 4 [173] 173 innerTail37_22 best = best := by
  rw [innerTail37_22, scan, if_neg (by decide), innerCandidates37_22_checked]
  change scan edge (search edge 3) 4 [173] 173 innerTail37_23
    (if 3 ≤ innerCandidates37_22.length then search edge 3 [173, 8117] innerCandidates37_22 8290 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan37_23

def innerTail37_21 : List Nat := 8039 :: innerTail37_22
def innerCandidates37_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates37_21_checked : candidates edge 4 173 8039 innerTail37_22 best.bound = innerCandidates37_21 := by
  unfold candidates
  have row : edge 8039 = SearchData.row8039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan37_21 : scan edge (search edge 3) 4 [173] 173 innerTail37_21 best = best := by
  rw [innerTail37_21, scan, if_neg (by decide), innerCandidates37_21_checked]
  change scan edge (search edge 3) 4 [173] 173 innerTail37_22
    (if 3 ≤ innerCandidates37_21.length then search edge 3 [173, 8039] innerCandidates37_21 8212 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan37_22

def innerTail37_20 : List Nat := 7559 :: innerTail37_21
def innerCandidates37_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates37_20_checked : candidates edge 4 173 7559 innerTail37_21 best.bound = innerCandidates37_20 := by
  unfold candidates
  have row : edge 7559 = SearchData.row7559.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan37_20 : scan edge (search edge 3) 4 [173] 173 innerTail37_20 best = best := by
  rw [innerTail37_20, scan, if_neg (by decide), innerCandidates37_20_checked]
  change scan edge (search edge 3) 4 [173] 173 innerTail37_21
    (if 3 ≤ innerCandidates37_20.length then search edge 3 [173, 7559] innerCandidates37_20 7732 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan37_21

def innerTail37_19 : List Nat := 7529 :: innerTail37_20
def innerCandidates37_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates37_19_checked : candidates edge 4 173 7529 innerTail37_20 best.bound = innerCandidates37_19 := by
  unfold candidates
  have row : edge 7529 = SearchData.row7529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan37_19 : scan edge (search edge 3) 4 [173] 173 innerTail37_19 best = best := by
  rw [innerTail37_19, scan, if_neg (by decide), innerCandidates37_19_checked]
  change scan edge (search edge 3) 4 [173] 173 innerTail37_20
    (if 3 ≤ innerCandidates37_19.length then search edge 3 [173, 7529] innerCandidates37_19 7702 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan37_20

def innerTail37_18 : List Nat := 7331 :: innerTail37_19
def innerCandidates37_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates37_18_checked : candidates edge 4 173 7331 innerTail37_19 best.bound = innerCandidates37_18 := by
  unfold candidates
  have row : edge 7331 = SearchData.row7331.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan37_18 : scan edge (search edge 3) 4 [173] 173 innerTail37_18 best = best := by
  rw [innerTail37_18, scan, if_neg (by decide), innerCandidates37_18_checked]
  change scan edge (search edge 3) 4 [173] 173 innerTail37_19
    (if 3 ≤ innerCandidates37_18.length then search edge 3 [173, 7331] innerCandidates37_18 7504 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan37_19

def innerTail37_17 : List Nat := 5813 :: innerTail37_18
def innerCandidates37_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates37_17_checked : candidates edge 4 173 5813 innerTail37_18 best.bound = innerCandidates37_17 := by
  unfold candidates
  have row : edge 5813 = SearchData.row5813.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan37_17 : scan edge (search edge 3) 4 [173] 173 innerTail37_17 best = best := by
  rw [innerTail37_17, scan, if_neg (by decide), innerCandidates37_17_checked]
  change scan edge (search edge 3) 4 [173] 173 innerTail37_18
    (if 3 ≤ innerCandidates37_17.length then search edge 3 [173, 5813] innerCandidates37_17 5986 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan37_18

def innerTail37_16 : List Nat := 5807 :: innerTail37_17
def innerCandidates37_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates37_16_checked : candidates edge 4 173 5807 innerTail37_17 best.bound = innerCandidates37_16 := by
  unfold candidates
  have row : edge 5807 = SearchData.row5807.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan37_16 : scan edge (search edge 3) 4 [173] 173 innerTail37_16 best = best := by
  rw [innerTail37_16, scan, if_neg (by decide), innerCandidates37_16_checked]
  change scan edge (search edge 3) 4 [173] 173 innerTail37_17
    (if 3 ≤ innerCandidates37_16.length then search edge 3 [173, 5807] innerCandidates37_16 5980 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan37_17

def innerTail37_15 : List Nat := 5333 :: innerTail37_16
def innerCandidates37_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates37_15_checked : candidates edge 4 173 5333 innerTail37_16 best.bound = innerCandidates37_15 := by
  unfold candidates
  have row : edge 5333 = SearchData.row5333.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan37_15 : scan edge (search edge 3) 4 [173] 173 innerTail37_15 best = best := by
  rw [innerTail37_15, scan, if_neg (by decide), innerCandidates37_15_checked]
  change scan edge (search edge 3) 4 [173] 173 innerTail37_16
    (if 3 ≤ innerCandidates37_15.length then search edge 3 [173, 5333] innerCandidates37_15 5506 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan37_16

def innerTail37_14 : List Nat := 4937 :: innerTail37_15
def innerCandidates37_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates37_14_checked : candidates edge 4 173 4937 innerTail37_15 best.bound = innerCandidates37_14 := by
  unfold candidates
  have row : edge 4937 = SearchData.row4937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan37_14 : scan edge (search edge 3) 4 [173] 173 innerTail37_14 best = best := by
  rw [innerTail37_14, scan, if_neg (by decide), innerCandidates37_14_checked]
  change scan edge (search edge 3) 4 [173] 173 innerTail37_15
    (if 3 ≤ innerCandidates37_14.length then search edge 3 [173, 4937] innerCandidates37_14 5110 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan37_15

def innerTail37_13 : List Nat := 4787 :: innerTail37_14
def innerCandidates37_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates37_13_checked : candidates edge 4 173 4787 innerTail37_14 best.bound = innerCandidates37_13 := by
  unfold candidates
  have row : edge 4787 = SearchData.row4787.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan37_13 : scan edge (search edge 3) 4 [173] 173 innerTail37_13 best = best := by
  rw [innerTail37_13, scan, if_neg (by decide), innerCandidates37_13_checked]
  change scan edge (search edge 3) 4 [173] 173 innerTail37_14
    (if 3 ≤ innerCandidates37_13.length then search edge 3 [173, 4787] innerCandidates37_13 4960 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan37_14

def innerTail37_12 : List Nat := 4721 :: innerTail37_13
def innerCandidates37_12 : List Nat := [5813]
set_option maxRecDepth 4096 in
theorem innerCandidates37_12_checked : candidates edge 4 173 4721 innerTail37_13 best.bound = innerCandidates37_12 := by
  unfold candidates
  have row : edge 4721 = SearchData.row4721.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan37_12 : scan edge (search edge 3) 4 [173] 173 innerTail37_12 best = best := by
  rw [innerTail37_12, scan, if_neg (by decide), innerCandidates37_12_checked]
  change scan edge (search edge 3) 4 [173] 173 innerTail37_13
    (if 3 ≤ innerCandidates37_12.length then search edge 3 [173, 4721] innerCandidates37_12 4894 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan37_13

def innerTail37_11 : List Nat := 4673 :: innerTail37_12
def innerCandidates37_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates37_11_checked : candidates edge 4 173 4673 innerTail37_12 best.bound = innerCandidates37_11 := by
  unfold candidates
  have row : edge 4673 = SearchData.row4673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan37_11 : scan edge (search edge 3) 4 [173] 173 innerTail37_11 best = best := by
  rw [innerTail37_11, scan, if_neg (by decide), innerCandidates37_11_checked]
  change scan edge (search edge 3) 4 [173] 173 innerTail37_12
    (if 3 ≤ innerCandidates37_11.length then search edge 3 [173, 4673] innerCandidates37_11 4846 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan37_12

def innerTail37_10 : List Nat := 3929 :: innerTail37_11
def innerCandidates37_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates37_10_checked : candidates edge 4 173 3929 innerTail37_11 best.bound = innerCandidates37_10 := by
  unfold candidates
  have row : edge 3929 = SearchData.row3929.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan37_10 : scan edge (search edge 3) 4 [173] 173 innerTail37_10 best = best := by
  rw [innerTail37_10, scan, if_neg (by decide), innerCandidates37_10_checked]
  change scan edge (search edge 3) 4 [173] 173 innerTail37_11
    (if 3 ≤ innerCandidates37_10.length then search edge 3 [173, 3929] innerCandidates37_10 4102 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan37_11

def innerTail37_9 : List Nat := 3041 :: innerTail37_10
def innerCandidates37_9 : List Nat := [7529]
set_option maxRecDepth 4096 in
theorem innerCandidates37_9_checked : candidates edge 4 173 3041 innerTail37_10 best.bound = innerCandidates37_9 := by
  unfold candidates
  have row : edge 3041 = SearchData.row3041.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan37_9 : scan edge (search edge 3) 4 [173] 173 innerTail37_9 best = best := by
  rw [innerTail37_9, scan, if_neg (by decide), innerCandidates37_9_checked]
  change scan edge (search edge 3) 4 [173] 173 innerTail37_10
    (if 3 ≤ innerCandidates37_9.length then search edge 3 [173, 3041] innerCandidates37_9 3214 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan37_10

def innerTail37_8 : List Nat := 2333 :: innerTail37_9
def innerCandidates37_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates37_8_checked : candidates edge 4 173 2333 innerTail37_9 best.bound = innerCandidates37_8 := by
  unfold candidates
  have row : edge 2333 = PrimePairCache.smallRow2333.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan37_8 : scan edge (search edge 3) 4 [173] 173 innerTail37_8 best = best := by
  rw [innerTail37_8, scan, if_neg (by decide), innerCandidates37_8_checked]
  change scan edge (search edge 3) 4 [173] 173 innerTail37_9
    (if 3 ≤ innerCandidates37_8.length then search edge 3 [173, 2333] innerCandidates37_8 2506 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan37_9

def innerTail37_7 : List Nat := 2039 :: innerTail37_8
def innerCandidates37_7 : List Nat := [4937]
set_option maxRecDepth 4096 in
theorem innerCandidates37_7_checked : candidates edge 4 173 2039 innerTail37_8 best.bound = innerCandidates37_7 := by
  unfold candidates
  have row : edge 2039 = PrimePairCache.smallRow2039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan37_7 : scan edge (search edge 3) 4 [173] 173 innerTail37_7 best = best := by
  rw [innerTail37_7, scan, if_neg (by decide), innerCandidates37_7_checked]
  change scan edge (search edge 3) 4 [173] 173 innerTail37_8
    (if 3 ≤ innerCandidates37_7.length then search edge 3 [173, 2039] innerCandidates37_7 2212 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan37_8

def innerTail37_6 : List Nat := 1091 :: innerTail37_7
def innerCandidates37_6 : List Nat := [3041]
set_option maxRecDepth 4096 in
theorem innerCandidates37_6_checked : candidates edge 4 173 1091 innerTail37_7 best.bound = innerCandidates37_6 := by
  unfold candidates
  have row : edge 1091 = PrimePairCache.smallRow1091.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan37_6 : scan edge (search edge 3) 4 [173] 173 innerTail37_6 best = best := by
  rw [innerTail37_6, scan, if_neg (by decide), innerCandidates37_6_checked]
  change scan edge (search edge 3) 4 [173] 173 innerTail37_7
    (if 3 ≤ innerCandidates37_6.length then search edge 3 [173, 1091] innerCandidates37_6 1264 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan37_7

def innerTail37_5 : List Nat := 743 :: innerTail37_6
def innerCandidates37_5 : List Nat := [8117]
set_option maxRecDepth 4096 in
theorem innerCandidates37_5_checked : candidates edge 4 173 743 innerTail37_6 best.bound = innerCandidates37_5 := by
  unfold candidates
  have row : edge 743 = PrimePairCache.smallRow743.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan37_5 : scan edge (search edge 3) 4 [173] 173 innerTail37_5 best = best := by
  rw [innerTail37_5, scan, if_neg (by decide), innerCandidates37_5_checked]
  change scan edge (search edge 3) 4 [173] 173 innerTail37_6
    (if 3 ≤ innerCandidates37_5.length then search edge 3 [173, 743] innerCandidates37_5 916 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan37_6

def innerTail37_4 : List Nat := 659 :: innerTail37_5
def innerCandidates37_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates37_4_checked : candidates edge 4 173 659 innerTail37_5 best.bound = innerCandidates37_4 := by
  unfold candidates
  have row : edge 659 = PrimePairCache.smallRow659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan37_4 : scan edge (search edge 3) 4 [173] 173 innerTail37_4 best = best := by
  rw [innerTail37_4, scan, if_neg (by decide), innerCandidates37_4_checked]
  change scan edge (search edge 3) 4 [173] 173 innerTail37_5
    (if 3 ≤ innerCandidates37_4.length then search edge 3 [173, 659] innerCandidates37_4 832 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan37_5

def innerTail37_3 : List Nat := 431 :: innerTail37_4
def innerCandidates37_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates37_3_checked : candidates edge 4 173 431 innerTail37_4 best.bound = innerCandidates37_3 := by
  unfold candidates
  have row : edge 431 = PrimePairCache.smallRow431.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan37_3 : scan edge (search edge 3) 4 [173] 173 innerTail37_3 best = best := by
  rw [innerTail37_3, scan, if_neg (by decide), innerCandidates37_3_checked]
  change scan edge (search edge 3) 4 [173] 173 innerTail37_4
    (if 3 ≤ innerCandidates37_3.length then search edge 3 [173, 431] innerCandidates37_3 604 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan37_4

def innerTail37_2 : List Nat := 347 :: innerTail37_3
def innerCandidates37_2 : List Nat := [7559]
set_option maxRecDepth 4096 in
theorem innerCandidates37_2_checked : candidates edge 4 173 347 innerTail37_3 best.bound = innerCandidates37_2 := by
  unfold candidates
  have row : edge 347 = PrimePairCache.smallRow347.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan37_2 : scan edge (search edge 3) 4 [173] 173 innerTail37_2 best = best := by
  rw [innerTail37_2, scan, if_neg (by decide), innerCandidates37_2_checked]
  change scan edge (search edge 3) 4 [173] 173 innerTail37_3
    (if 3 ≤ innerCandidates37_2.length then search edge 3 [173, 347] innerCandidates37_2 520 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan37_3

def innerTail37_1 : List Nat := 293 :: innerTail37_2
def innerCandidates37_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates37_1_checked : candidates edge 4 173 293 innerTail37_2 best.bound = innerCandidates37_1 := by
  unfold candidates
  have row : edge 293 = PrimePairCache.smallRow293.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan37_1 : scan edge (search edge 3) 4 [173] 173 innerTail37_1 best = best := by
  rw [innerTail37_1, scan, if_neg (by decide), innerCandidates37_1_checked]
  change scan edge (search edge 3) 4 [173] 173 innerTail37_2
    (if 3 ≤ innerCandidates37_1.length then search edge 3 [173, 293] innerCandidates37_1 466 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan37_2

def innerTail37_0 : List Nat := 191 :: innerTail37_1
def innerCandidates37_0 : List Nat := [4673, 5813]
set_option maxRecDepth 4096 in
theorem innerCandidates37_0_checked : candidates edge 4 173 191 innerTail37_1 best.bound = innerCandidates37_0 := by
  unfold candidates
  have row : edge 191 = PrimePairCache.smallRow191.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan37_0 : scan edge (search edge 3) 4 [173] 173 innerTail37_0 best = best := by
  rw [innerTail37_0, scan, if_neg (by decide), innerCandidates37_0_checked]
  change scan edge (search edge 3) 4 [173] 173 innerTail37_1
    (if 3 ≤ innerCandidates37_0.length then search edge 3 [173, 191] innerCandidates37_0 364 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan37_1

theorem rootBranch37 : search edge 4 [173] rootCandidates37 173 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [173] 173 innerTail37_0 best = best
  exact innerScan37_0

def rootCandidates36 : List Nat := [269, 443, 491, 521, 641, 677, 911, 2063, 2087, 2129, 2441, 2963, 2999, 3011, 3461, 3719, 4271, 4733, 5441, 5981, 6053, 6827, 6947, 7253, 7451, 7673, 7877, 8009]
set_option maxRecDepth 4096 in
theorem rootCandidates36_checked : candidates edge 5 0 167 tail37 best.bound = rootCandidates36 := by
  unfold candidates
  have row : edge 167 = PrimePairCache.smallRow167.contains := by funext q; rfl
  rw [row]
  decide

def innerTail36_28 : List Nat := []
theorem innerScan36_28 : scan edge (search edge 3) 4 [167] 167 innerTail36_28 best = best := rfl

def innerTail36_27 : List Nat := 8009 :: innerTail36_28
def innerCandidates36_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates36_27_checked : candidates edge 4 167 8009 innerTail36_28 best.bound = innerCandidates36_27 := by
  unfold candidates
  have row : edge 8009 = SearchData.row8009.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan36_27 : scan edge (search edge 3) 4 [167] 167 innerTail36_27 best = best := by
  rw [innerTail36_27, scan, if_neg (by decide), innerCandidates36_27_checked]
  change scan edge (search edge 3) 4 [167] 167 innerTail36_28
    (if 3 ≤ innerCandidates36_27.length then search edge 3 [167, 8009] innerCandidates36_27 8176 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan36_28

def innerTail36_26 : List Nat := 7877 :: innerTail36_27
def innerCandidates36_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates36_26_checked : candidates edge 4 167 7877 innerTail36_27 best.bound = innerCandidates36_26 := by
  unfold candidates
  have row : edge 7877 = SearchData.row7877.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan36_26 : scan edge (search edge 3) 4 [167] 167 innerTail36_26 best = best := by
  rw [innerTail36_26, scan, if_neg (by decide), innerCandidates36_26_checked]
  change scan edge (search edge 3) 4 [167] 167 innerTail36_27
    (if 3 ≤ innerCandidates36_26.length then search edge 3 [167, 7877] innerCandidates36_26 8044 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan36_27

def innerTail36_25 : List Nat := 7673 :: innerTail36_26
def innerCandidates36_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates36_25_checked : candidates edge 4 167 7673 innerTail36_26 best.bound = innerCandidates36_25 := by
  unfold candidates
  have row : edge 7673 = SearchData.row7673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan36_25 : scan edge (search edge 3) 4 [167] 167 innerTail36_25 best = best := by
  rw [innerTail36_25, scan, if_neg (by decide), innerCandidates36_25_checked]
  change scan edge (search edge 3) 4 [167] 167 innerTail36_26
    (if 3 ≤ innerCandidates36_25.length then search edge 3 [167, 7673] innerCandidates36_25 7840 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan36_26

def innerTail36_24 : List Nat := 7451 :: innerTail36_25
def innerCandidates36_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates36_24_checked : candidates edge 4 167 7451 innerTail36_25 best.bound = innerCandidates36_24 := by
  unfold candidates
  have row : edge 7451 = SearchData.row7451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan36_24 : scan edge (search edge 3) 4 [167] 167 innerTail36_24 best = best := by
  rw [innerTail36_24, scan, if_neg (by decide), innerCandidates36_24_checked]
  change scan edge (search edge 3) 4 [167] 167 innerTail36_25
    (if 3 ≤ innerCandidates36_24.length then search edge 3 [167, 7451] innerCandidates36_24 7618 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan36_25

def innerTail36_23 : List Nat := 7253 :: innerTail36_24
def innerCandidates36_23 : List Nat := [8009]
set_option maxRecDepth 4096 in
theorem innerCandidates36_23_checked : candidates edge 4 167 7253 innerTail36_24 best.bound = innerCandidates36_23 := by
  unfold candidates
  have row : edge 7253 = SearchData.row7253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan36_23 : scan edge (search edge 3) 4 [167] 167 innerTail36_23 best = best := by
  rw [innerTail36_23, scan, if_neg (by decide), innerCandidates36_23_checked]
  change scan edge (search edge 3) 4 [167] 167 innerTail36_24
    (if 3 ≤ innerCandidates36_23.length then search edge 3 [167, 7253] innerCandidates36_23 7420 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan36_24

def innerTail36_22 : List Nat := 6947 :: innerTail36_23
def innerCandidates36_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates36_22_checked : candidates edge 4 167 6947 innerTail36_23 best.bound = innerCandidates36_22 := by
  unfold candidates
  have row : edge 6947 = SearchData.row6947.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan36_22 : scan edge (search edge 3) 4 [167] 167 innerTail36_22 best = best := by
  rw [innerTail36_22, scan, if_neg (by decide), innerCandidates36_22_checked]
  change scan edge (search edge 3) 4 [167] 167 innerTail36_23
    (if 3 ≤ innerCandidates36_22.length then search edge 3 [167, 6947] innerCandidates36_22 7114 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan36_23

def innerTail36_21 : List Nat := 6827 :: innerTail36_22
def innerCandidates36_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates36_21_checked : candidates edge 4 167 6827 innerTail36_22 best.bound = innerCandidates36_21 := by
  unfold candidates
  have row : edge 6827 = SearchData.row6827.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan36_21 : scan edge (search edge 3) 4 [167] 167 innerTail36_21 best = best := by
  rw [innerTail36_21, scan, if_neg (by decide), innerCandidates36_21_checked]
  change scan edge (search edge 3) 4 [167] 167 innerTail36_22
    (if 3 ≤ innerCandidates36_21.length then search edge 3 [167, 6827] innerCandidates36_21 6994 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan36_22

def innerTail36_20 : List Nat := 6053 :: innerTail36_21
def innerCandidates36_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates36_20_checked : candidates edge 4 167 6053 innerTail36_21 best.bound = innerCandidates36_20 := by
  unfold candidates
  have row : edge 6053 = SearchData.row6053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan36_20 : scan edge (search edge 3) 4 [167] 167 innerTail36_20 best = best := by
  rw [innerTail36_20, scan, if_neg (by decide), innerCandidates36_20_checked]
  change scan edge (search edge 3) 4 [167] 167 innerTail36_21
    (if 3 ≤ innerCandidates36_20.length then search edge 3 [167, 6053] innerCandidates36_20 6220 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan36_21

def innerTail36_19 : List Nat := 5981 :: innerTail36_20
def innerCandidates36_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates36_19_checked : candidates edge 4 167 5981 innerTail36_20 best.bound = innerCandidates36_19 := by
  unfold candidates
  have row : edge 5981 = SearchData.row5981.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan36_19 : scan edge (search edge 3) 4 [167] 167 innerTail36_19 best = best := by
  rw [innerTail36_19, scan, if_neg (by decide), innerCandidates36_19_checked]
  change scan edge (search edge 3) 4 [167] 167 innerTail36_20
    (if 3 ≤ innerCandidates36_19.length then search edge 3 [167, 5981] innerCandidates36_19 6148 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan36_20

def innerTail36_18 : List Nat := 5441 :: innerTail36_19
def innerCandidates36_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates36_18_checked : candidates edge 4 167 5441 innerTail36_19 best.bound = innerCandidates36_18 := by
  unfold candidates
  have row : edge 5441 = SearchData.row5441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan36_18 : scan edge (search edge 3) 4 [167] 167 innerTail36_18 best = best := by
  rw [innerTail36_18, scan, if_neg (by decide), innerCandidates36_18_checked]
  change scan edge (search edge 3) 4 [167] 167 innerTail36_19
    (if 3 ≤ innerCandidates36_18.length then search edge 3 [167, 5441] innerCandidates36_18 5608 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan36_19

def innerTail36_17 : List Nat := 4733 :: innerTail36_18
def innerCandidates36_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates36_17_checked : candidates edge 4 167 4733 innerTail36_18 best.bound = innerCandidates36_17 := by
  unfold candidates
  have row : edge 4733 = SearchData.row4733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan36_17 : scan edge (search edge 3) 4 [167] 167 innerTail36_17 best = best := by
  rw [innerTail36_17, scan, if_neg (by decide), innerCandidates36_17_checked]
  change scan edge (search edge 3) 4 [167] 167 innerTail36_18
    (if 3 ≤ innerCandidates36_17.length then search edge 3 [167, 4733] innerCandidates36_17 4900 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan36_18

def innerTail36_16 : List Nat := 4271 :: innerTail36_17
def innerCandidates36_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates36_16_checked : candidates edge 4 167 4271 innerTail36_17 best.bound = innerCandidates36_16 := by
  unfold candidates
  have row : edge 4271 = SearchData.row4271.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan36_16 : scan edge (search edge 3) 4 [167] 167 innerTail36_16 best = best := by
  rw [innerTail36_16, scan, if_neg (by decide), innerCandidates36_16_checked]
  change scan edge (search edge 3) 4 [167] 167 innerTail36_17
    (if 3 ≤ innerCandidates36_16.length then search edge 3 [167, 4271] innerCandidates36_16 4438 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan36_17

def innerTail36_15 : List Nat := 3719 :: innerTail36_16
def innerCandidates36_15 : List Nat := [4271]
set_option maxRecDepth 4096 in
theorem innerCandidates36_15_checked : candidates edge 4 167 3719 innerTail36_16 best.bound = innerCandidates36_15 := by
  unfold candidates
  have row : edge 3719 = SearchData.row3719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan36_15 : scan edge (search edge 3) 4 [167] 167 innerTail36_15 best = best := by
  rw [innerTail36_15, scan, if_neg (by decide), innerCandidates36_15_checked]
  change scan edge (search edge 3) 4 [167] 167 innerTail36_16
    (if 3 ≤ innerCandidates36_15.length then search edge 3 [167, 3719] innerCandidates36_15 3886 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan36_16

def innerTail36_14 : List Nat := 3461 :: innerTail36_15
def innerCandidates36_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates36_14_checked : candidates edge 4 167 3461 innerTail36_15 best.bound = innerCandidates36_14 := by
  unfold candidates
  have row : edge 3461 = SearchData.row3461.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan36_14 : scan edge (search edge 3) 4 [167] 167 innerTail36_14 best = best := by
  rw [innerTail36_14, scan, if_neg (by decide), innerCandidates36_14_checked]
  change scan edge (search edge 3) 4 [167] 167 innerTail36_15
    (if 3 ≤ innerCandidates36_14.length then search edge 3 [167, 3461] innerCandidates36_14 3628 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan36_15

def innerTail36_13 : List Nat := 3011 :: innerTail36_14
def innerCandidates36_13 : List Nat := [8009]
set_option maxRecDepth 4096 in
theorem innerCandidates36_13_checked : candidates edge 4 167 3011 innerTail36_14 best.bound = innerCandidates36_13 := by
  unfold candidates
  have row : edge 3011 = SearchData.row3011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan36_13 : scan edge (search edge 3) 4 [167] 167 innerTail36_13 best = best := by
  rw [innerTail36_13, scan, if_neg (by decide), innerCandidates36_13_checked]
  change scan edge (search edge 3) 4 [167] 167 innerTail36_14
    (if 3 ≤ innerCandidates36_13.length then search edge 3 [167, 3011] innerCandidates36_13 3178 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan36_14

def innerTail36_12 : List Nat := 2999 :: innerTail36_13
def innerCandidates36_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates36_12_checked : candidates edge 4 167 2999 innerTail36_13 best.bound = innerCandidates36_12 := by
  unfold candidates
  have row : edge 2999 = SearchData.row2999.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan36_12 : scan edge (search edge 3) 4 [167] 167 innerTail36_12 best = best := by
  rw [innerTail36_12, scan, if_neg (by decide), innerCandidates36_12_checked]
  change scan edge (search edge 3) 4 [167] 167 innerTail36_13
    (if 3 ≤ innerCandidates36_12.length then search edge 3 [167, 2999] innerCandidates36_12 3166 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan36_13

def innerTail36_11 : List Nat := 2963 :: innerTail36_12
def innerCandidates36_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates36_11_checked : candidates edge 4 167 2963 innerTail36_12 best.bound = innerCandidates36_11 := by
  unfold candidates
  have row : edge 2963 = SearchData.row2963.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan36_11 : scan edge (search edge 3) 4 [167] 167 innerTail36_11 best = best := by
  rw [innerTail36_11, scan, if_neg (by decide), innerCandidates36_11_checked]
  change scan edge (search edge 3) 4 [167] 167 innerTail36_12
    (if 3 ≤ innerCandidates36_11.length then search edge 3 [167, 2963] innerCandidates36_11 3130 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan36_12

def innerTail36_10 : List Nat := 2441 :: innerTail36_11
def innerCandidates36_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates36_10_checked : candidates edge 4 167 2441 innerTail36_11 best.bound = innerCandidates36_10 := by
  unfold candidates
  have row : edge 2441 = PrimePairCache.smallRow2441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan36_10 : scan edge (search edge 3) 4 [167] 167 innerTail36_10 best = best := by
  rw [innerTail36_10, scan, if_neg (by decide), innerCandidates36_10_checked]
  change scan edge (search edge 3) 4 [167] 167 innerTail36_11
    (if 3 ≤ innerCandidates36_10.length then search edge 3 [167, 2441] innerCandidates36_10 2608 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan36_11

def innerTail36_9 : List Nat := 2129 :: innerTail36_10
def innerCandidates36_9 : List Nat := [4271]
set_option maxRecDepth 4096 in
theorem innerCandidates36_9_checked : candidates edge 4 167 2129 innerTail36_10 best.bound = innerCandidates36_9 := by
  unfold candidates
  have row : edge 2129 = PrimePairCache.smallRow2129.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan36_9 : scan edge (search edge 3) 4 [167] 167 innerTail36_9 best = best := by
  rw [innerTail36_9, scan, if_neg (by decide), innerCandidates36_9_checked]
  change scan edge (search edge 3) 4 [167] 167 innerTail36_10
    (if 3 ≤ innerCandidates36_9.length then search edge 3 [167, 2129] innerCandidates36_9 2296 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan36_10

def innerTail36_8 : List Nat := 2087 :: innerTail36_9
def innerCandidates36_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates36_8_checked : candidates edge 4 167 2087 innerTail36_9 best.bound = innerCandidates36_8 := by
  unfold candidates
  have row : edge 2087 = PrimePairCache.smallRow2087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan36_8 : scan edge (search edge 3) 4 [167] 167 innerTail36_8 best = best := by
  rw [innerTail36_8, scan, if_neg (by decide), innerCandidates36_8_checked]
  change scan edge (search edge 3) 4 [167] 167 innerTail36_9
    (if 3 ≤ innerCandidates36_8.length then search edge 3 [167, 2087] innerCandidates36_8 2254 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan36_9

def innerTail36_7 : List Nat := 2063 :: innerTail36_8
def innerCandidates36_7 : List Nat := [5981]
set_option maxRecDepth 4096 in
theorem innerCandidates36_7_checked : candidates edge 4 167 2063 innerTail36_8 best.bound = innerCandidates36_7 := by
  unfold candidates
  have row : edge 2063 = PrimePairCache.smallRow2063.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan36_7 : scan edge (search edge 3) 4 [167] 167 innerTail36_7 best = best := by
  rw [innerTail36_7, scan, if_neg (by decide), innerCandidates36_7_checked]
  change scan edge (search edge 3) 4 [167] 167 innerTail36_8
    (if 3 ≤ innerCandidates36_7.length then search edge 3 [167, 2063] innerCandidates36_7 2230 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan36_8

def innerTail36_6 : List Nat := 911 :: innerTail36_7
def innerCandidates36_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates36_6_checked : candidates edge 4 167 911 innerTail36_7 best.bound = innerCandidates36_6 := by
  unfold candidates
  have row : edge 911 = PrimePairCache.smallRow911.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan36_6 : scan edge (search edge 3) 4 [167] 167 innerTail36_6 best = best := by
  rw [innerTail36_6, scan, if_neg (by decide), innerCandidates36_6_checked]
  change scan edge (search edge 3) 4 [167] 167 innerTail36_7
    (if 3 ≤ innerCandidates36_6.length then search edge 3 [167, 911] innerCandidates36_6 1078 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan36_7

def innerTail36_5 : List Nat := 677 :: innerTail36_6
def innerCandidates36_5 : List Nat := [7451]
set_option maxRecDepth 4096 in
theorem innerCandidates36_5_checked : candidates edge 4 167 677 innerTail36_6 best.bound = innerCandidates36_5 := by
  unfold candidates
  have row : edge 677 = PrimePairCache.smallRow677.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan36_5 : scan edge (search edge 3) 4 [167] 167 innerTail36_5 best = best := by
  rw [innerTail36_5, scan, if_neg (by decide), innerCandidates36_5_checked]
  change scan edge (search edge 3) 4 [167] 167 innerTail36_6
    (if 3 ≤ innerCandidates36_5.length then search edge 3 [167, 677] innerCandidates36_5 844 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan36_6

def innerTail36_4 : List Nat := 641 :: innerTail36_5
def innerCandidates36_4 : List Nat := [5441]
set_option maxRecDepth 4096 in
theorem innerCandidates36_4_checked : candidates edge 4 167 641 innerTail36_5 best.bound = innerCandidates36_4 := by
  unfold candidates
  have row : edge 641 = PrimePairCache.smallRow641.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan36_4 : scan edge (search edge 3) 4 [167] 167 innerTail36_4 best = best := by
  rw [innerTail36_4, scan, if_neg (by decide), innerCandidates36_4_checked]
  change scan edge (search edge 3) 4 [167] 167 innerTail36_5
    (if 3 ≤ innerCandidates36_4.length then search edge 3 [167, 641] innerCandidates36_4 808 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan36_5

def innerTail36_3 : List Nat := 521 :: innerTail36_4
def innerCandidates36_3 : List Nat := [641, 6053, 7673]
set_option maxRecDepth 4096 in
theorem innerCandidates36_3_checked : candidates edge 4 167 521 innerTail36_4 best.bound = innerCandidates36_3 := by
  unfold candidates
  have row : edge 521 = PrimePairCache.smallRow521.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild36_3 : search edge 3 [167, 521] innerCandidates36_3 688 best = best := by decide

theorem innerScan36_3 : scan edge (search edge 3) 4 [167] 167 innerTail36_3 best = best := by
  rw [innerTail36_3, scan, if_neg (by decide), innerCandidates36_3_checked]
  change scan edge (search edge 3) 4 [167] 167 innerTail36_4
    (if 3 ≤ innerCandidates36_3.length then search edge 3 [167, 521] innerCandidates36_3 688 best else best) = best
  rw [if_pos (by decide), innerChild36_3]
  exact innerScan36_4

def innerTail36_2 : List Nat := 491 :: innerTail36_3
def innerCandidates36_2 : List Nat := [6053]
set_option maxRecDepth 4096 in
theorem innerCandidates36_2_checked : candidates edge 4 167 491 innerTail36_3 best.bound = innerCandidates36_2 := by
  unfold candidates
  have row : edge 491 = PrimePairCache.smallRow491.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan36_2 : scan edge (search edge 3) 4 [167] 167 innerTail36_2 best = best := by
  rw [innerTail36_2, scan, if_neg (by decide), innerCandidates36_2_checked]
  change scan edge (search edge 3) 4 [167] 167 innerTail36_3
    (if 3 ≤ innerCandidates36_2.length then search edge 3 [167, 491] innerCandidates36_2 658 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan36_3

def innerTail36_1 : List Nat := 443 :: innerTail36_2
def innerCandidates36_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates36_1_checked : candidates edge 4 167 443 innerTail36_2 best.bound = innerCandidates36_1 := by
  unfold candidates
  have row : edge 443 = PrimePairCache.smallRow443.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan36_1 : scan edge (search edge 3) 4 [167] 167 innerTail36_1 best = best := by
  rw [innerTail36_1, scan, if_neg (by decide), innerCandidates36_1_checked]
  change scan edge (search edge 3) 4 [167] 167 innerTail36_2
    (if 3 ≤ innerCandidates36_1.length then search edge 3 [167, 443] innerCandidates36_1 610 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan36_2

def innerTail36_0 : List Nat := 269 :: innerTail36_1
def innerCandidates36_0 : List Nat := [4733, 6053]
set_option maxRecDepth 4096 in
theorem innerCandidates36_0_checked : candidates edge 4 167 269 innerTail36_1 best.bound = innerCandidates36_0 := by
  unfold candidates
  have row : edge 269 = PrimePairCache.smallRow269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan36_0 : scan edge (search edge 3) 4 [167] 167 innerTail36_0 best = best := by
  rw [innerTail36_0, scan, if_neg (by decide), innerCandidates36_0_checked]
  change scan edge (search edge 3) 4 [167] 167 innerTail36_1
    (if 3 ≤ innerCandidates36_0.length then search edge 3 [167, 269] innerCandidates36_0 436 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan36_1

theorem rootBranch36 : search edge 4 [167] rootCandidates36 167 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [167] 167 innerTail36_0 best = best
  exact innerScan36_0

end ProjectEulerSolutions.PrimePairUpper.Search
