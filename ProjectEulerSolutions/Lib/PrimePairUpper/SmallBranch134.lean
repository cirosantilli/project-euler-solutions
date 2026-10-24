import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates272 : List Nat := [2017, 2371, 2539, 2647, 2719, 3121, 3253, 3463, 3739, 4621, 5011, 5839, 6151, 6277, 7213, 7489, 7561, 7603, 7741, 7867]
set_option maxRecDepth 4096 in
theorem rootCandidates272_checked : candidates edge 5 0 1777 tail273 best.bound = rootCandidates272 := by
  unfold candidates
  have row : edge 1777 = PrimePairCache.smallRow1777.contains := by funext q; rfl
  rw [row]
  decide

def innerTail272_20 : List Nat := []
theorem innerScan272_20 : scan edge (search edge 3) 4 [1777] 1777 innerTail272_20 best = best := rfl

def innerTail272_19 : List Nat := 7867 :: innerTail272_20
def innerCandidates272_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates272_19_checked : candidates edge 4 1777 7867 innerTail272_20 best.bound = innerCandidates272_19 := by
  unfold candidates
  have row : edge 7867 = SearchData.row7867.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan272_19 : scan edge (search edge 3) 4 [1777] 1777 innerTail272_19 best = best := by
  rw [innerTail272_19, scan, if_neg (by decide), innerCandidates272_19_checked]
  change scan edge (search edge 3) 4 [1777] 1777 innerTail272_20
    (if 3 ≤ innerCandidates272_19.length then search edge 3 [1777, 7867] innerCandidates272_19 9644 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan272_20

def innerTail272_18 : List Nat := 7741 :: innerTail272_19
def innerCandidates272_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates272_18_checked : candidates edge 4 1777 7741 innerTail272_19 best.bound = innerCandidates272_18 := by
  unfold candidates
  have row : edge 7741 = SearchData.row7741.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan272_18 : scan edge (search edge 3) 4 [1777] 1777 innerTail272_18 best = best := by
  rw [innerTail272_18, scan, if_neg (by decide), innerCandidates272_18_checked]
  change scan edge (search edge 3) 4 [1777] 1777 innerTail272_19
    (if 3 ≤ innerCandidates272_18.length then search edge 3 [1777, 7741] innerCandidates272_18 9518 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan272_19

def innerTail272_17 : List Nat := 7603 :: innerTail272_18
def innerCandidates272_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates272_17_checked : candidates edge 4 1777 7603 innerTail272_18 best.bound = innerCandidates272_17 := by
  unfold candidates
  have row : edge 7603 = SearchData.row7603.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan272_17 : scan edge (search edge 3) 4 [1777] 1777 innerTail272_17 best = best := by
  rw [innerTail272_17, scan, if_neg (by decide), innerCandidates272_17_checked]
  change scan edge (search edge 3) 4 [1777] 1777 innerTail272_18
    (if 3 ≤ innerCandidates272_17.length then search edge 3 [1777, 7603] innerCandidates272_17 9380 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan272_18

def innerTail272_16 : List Nat := 7561 :: innerTail272_17
def innerCandidates272_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates272_16_checked : candidates edge 4 1777 7561 innerTail272_17 best.bound = innerCandidates272_16 := by
  unfold candidates
  have row : edge 7561 = SearchData.row7561.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan272_16 : scan edge (search edge 3) 4 [1777] 1777 innerTail272_16 best = best := by
  rw [innerTail272_16, scan, if_neg (by decide), innerCandidates272_16_checked]
  change scan edge (search edge 3) 4 [1777] 1777 innerTail272_17
    (if 3 ≤ innerCandidates272_16.length then search edge 3 [1777, 7561] innerCandidates272_16 9338 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan272_17

def innerTail272_15 : List Nat := 7489 :: innerTail272_16
def innerCandidates272_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates272_15_checked : candidates edge 4 1777 7489 innerTail272_16 best.bound = innerCandidates272_15 := by
  unfold candidates
  have row : edge 7489 = SearchData.row7489.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan272_15 : scan edge (search edge 3) 4 [1777] 1777 innerTail272_15 best = best := by
  rw [innerTail272_15, scan, if_neg (by decide), innerCandidates272_15_checked]
  change scan edge (search edge 3) 4 [1777] 1777 innerTail272_16
    (if 3 ≤ innerCandidates272_15.length then search edge 3 [1777, 7489] innerCandidates272_15 9266 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan272_16

def innerTail272_14 : List Nat := 7213 :: innerTail272_15
def innerCandidates272_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates272_14_checked : candidates edge 4 1777 7213 innerTail272_15 best.bound = innerCandidates272_14 := by
  unfold candidates
  have row : edge 7213 = SearchData.row7213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan272_14 : scan edge (search edge 3) 4 [1777] 1777 innerTail272_14 best = best := by
  rw [innerTail272_14, scan, if_neg (by decide), innerCandidates272_14_checked]
  change scan edge (search edge 3) 4 [1777] 1777 innerTail272_15
    (if 3 ≤ innerCandidates272_14.length then search edge 3 [1777, 7213] innerCandidates272_14 8990 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan272_15

def innerTail272_13 : List Nat := 6277 :: innerTail272_14
def innerCandidates272_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates272_13_checked : candidates edge 4 1777 6277 innerTail272_14 best.bound = innerCandidates272_13 := by
  unfold candidates
  have row : edge 6277 = SearchData.row6277.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan272_13 : scan edge (search edge 3) 4 [1777] 1777 innerTail272_13 best = best := by
  rw [innerTail272_13, scan, if_neg (by decide), innerCandidates272_13_checked]
  change scan edge (search edge 3) 4 [1777] 1777 innerTail272_14
    (if 3 ≤ innerCandidates272_13.length then search edge 3 [1777, 6277] innerCandidates272_13 8054 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan272_14

def innerTail272_12 : List Nat := 6151 :: innerTail272_13
def innerCandidates272_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates272_12_checked : candidates edge 4 1777 6151 innerTail272_13 best.bound = innerCandidates272_12 := by
  unfold candidates
  have row : edge 6151 = SearchData.row6151.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan272_12 : scan edge (search edge 3) 4 [1777] 1777 innerTail272_12 best = best := by
  rw [innerTail272_12, scan, if_neg (by decide), innerCandidates272_12_checked]
  change scan edge (search edge 3) 4 [1777] 1777 innerTail272_13
    (if 3 ≤ innerCandidates272_12.length then search edge 3 [1777, 6151] innerCandidates272_12 7928 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan272_13

def innerTail272_11 : List Nat := 5839 :: innerTail272_12
def innerCandidates272_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates272_11_checked : candidates edge 4 1777 5839 innerTail272_12 best.bound = innerCandidates272_11 := by
  unfold candidates
  have row : edge 5839 = SearchData.row5839.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan272_11 : scan edge (search edge 3) 4 [1777] 1777 innerTail272_11 best = best := by
  rw [innerTail272_11, scan, if_neg (by decide), innerCandidates272_11_checked]
  change scan edge (search edge 3) 4 [1777] 1777 innerTail272_12
    (if 3 ≤ innerCandidates272_11.length then search edge 3 [1777, 5839] innerCandidates272_11 7616 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan272_12

def innerTail272_10 : List Nat := 5011 :: innerTail272_11
def innerCandidates272_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates272_10_checked : candidates edge 4 1777 5011 innerTail272_11 best.bound = innerCandidates272_10 := by
  unfold candidates
  have row : edge 5011 = SearchData.row5011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan272_10 : scan edge (search edge 3) 4 [1777] 1777 innerTail272_10 best = best := by
  rw [innerTail272_10, scan, if_neg (by decide), innerCandidates272_10_checked]
  change scan edge (search edge 3) 4 [1777] 1777 innerTail272_11
    (if 3 ≤ innerCandidates272_10.length then search edge 3 [1777, 5011] innerCandidates272_10 6788 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan272_11

def innerTail272_9 : List Nat := 4621 :: innerTail272_10
def innerCandidates272_9 : List Nat := [7489]
set_option maxRecDepth 4096 in
theorem innerCandidates272_9_checked : candidates edge 4 1777 4621 innerTail272_10 best.bound = innerCandidates272_9 := by
  unfold candidates
  have row : edge 4621 = SearchData.row4621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan272_9 : scan edge (search edge 3) 4 [1777] 1777 innerTail272_9 best = best := by
  rw [innerTail272_9, scan, if_neg (by decide), innerCandidates272_9_checked]
  change scan edge (search edge 3) 4 [1777] 1777 innerTail272_10
    (if 3 ≤ innerCandidates272_9.length then search edge 3 [1777, 4621] innerCandidates272_9 6398 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan272_10

def innerTail272_8 : List Nat := 3739 :: innerTail272_9
def innerCandidates272_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates272_8_checked : candidates edge 4 1777 3739 innerTail272_9 best.bound = innerCandidates272_8 := by
  unfold candidates
  have row : edge 3739 = SearchData.row3739.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan272_8 : scan edge (search edge 3) 4 [1777] 1777 innerTail272_8 best = best := by
  rw [innerTail272_8, scan, if_neg (by decide), innerCandidates272_8_checked]
  change scan edge (search edge 3) 4 [1777] 1777 innerTail272_9
    (if 3 ≤ innerCandidates272_8.length then search edge 3 [1777, 3739] innerCandidates272_8 5516 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan272_9

def innerTail272_7 : List Nat := 3463 :: innerTail272_8
def innerCandidates272_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates272_7_checked : candidates edge 4 1777 3463 innerTail272_8 best.bound = innerCandidates272_7 := by
  unfold candidates
  have row : edge 3463 = SearchData.row3463.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan272_7 : scan edge (search edge 3) 4 [1777] 1777 innerTail272_7 best = best := by
  rw [innerTail272_7, scan, if_neg (by decide), innerCandidates272_7_checked]
  change scan edge (search edge 3) 4 [1777] 1777 innerTail272_8
    (if 3 ≤ innerCandidates272_7.length then search edge 3 [1777, 3463] innerCandidates272_7 5240 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan272_8

def innerTail272_6 : List Nat := 3253 :: innerTail272_7
def innerCandidates272_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates272_6_checked : candidates edge 4 1777 3253 innerTail272_7 best.bound = innerCandidates272_6 := by
  unfold candidates
  have row : edge 3253 = SearchData.row3253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan272_6 : scan edge (search edge 3) 4 [1777] 1777 innerTail272_6 best = best := by
  rw [innerTail272_6, scan, if_neg (by decide), innerCandidates272_6_checked]
  change scan edge (search edge 3) 4 [1777] 1777 innerTail272_7
    (if 3 ≤ innerCandidates272_6.length then search edge 3 [1777, 3253] innerCandidates272_6 5030 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan272_7

def innerTail272_5 : List Nat := 3121 :: innerTail272_6
def innerCandidates272_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates272_5_checked : candidates edge 4 1777 3121 innerTail272_6 best.bound = innerCandidates272_5 := by
  unfold candidates
  have row : edge 3121 = SearchData.row3121.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan272_5 : scan edge (search edge 3) 4 [1777] 1777 innerTail272_5 best = best := by
  rw [innerTail272_5, scan, if_neg (by decide), innerCandidates272_5_checked]
  change scan edge (search edge 3) 4 [1777] 1777 innerTail272_6
    (if 3 ≤ innerCandidates272_5.length then search edge 3 [1777, 3121] innerCandidates272_5 4898 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan272_6

def innerTail272_4 : List Nat := 2719 :: innerTail272_5
def innerCandidates272_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates272_4_checked : candidates edge 4 1777 2719 innerTail272_5 best.bound = innerCandidates272_4 := by
  unfold candidates
  have row : edge 2719 = SearchData.row2719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan272_4 : scan edge (search edge 3) 4 [1777] 1777 innerTail272_4 best = best := by
  rw [innerTail272_4, scan, if_neg (by decide), innerCandidates272_4_checked]
  change scan edge (search edge 3) 4 [1777] 1777 innerTail272_5
    (if 3 ≤ innerCandidates272_4.length then search edge 3 [1777, 2719] innerCandidates272_4 4496 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan272_5

def innerTail272_3 : List Nat := 2647 :: innerTail272_4
def innerCandidates272_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates272_3_checked : candidates edge 4 1777 2647 innerTail272_4 best.bound = innerCandidates272_3 := by
  unfold candidates
  have row : edge 2647 = PrimePairCache.smallRow2647.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan272_3 : scan edge (search edge 3) 4 [1777] 1777 innerTail272_3 best = best := by
  rw [innerTail272_3, scan, if_neg (by decide), innerCandidates272_3_checked]
  change scan edge (search edge 3) 4 [1777] 1777 innerTail272_4
    (if 3 ≤ innerCandidates272_3.length then search edge 3 [1777, 2647] innerCandidates272_3 4424 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan272_4

def innerTail272_2 : List Nat := 2539 :: innerTail272_3
def innerCandidates272_2 : List Nat := [3463]
set_option maxRecDepth 4096 in
theorem innerCandidates272_2_checked : candidates edge 4 1777 2539 innerTail272_3 best.bound = innerCandidates272_2 := by
  unfold candidates
  have row : edge 2539 = PrimePairCache.smallRow2539.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan272_2 : scan edge (search edge 3) 4 [1777] 1777 innerTail272_2 best = best := by
  rw [innerTail272_2, scan, if_neg (by decide), innerCandidates272_2_checked]
  change scan edge (search edge 3) 4 [1777] 1777 innerTail272_3
    (if 3 ≤ innerCandidates272_2.length then search edge 3 [1777, 2539] innerCandidates272_2 4316 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan272_3

def innerTail272_1 : List Nat := 2371 :: innerTail272_2
def innerCandidates272_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates272_1_checked : candidates edge 4 1777 2371 innerTail272_2 best.bound = innerCandidates272_1 := by
  unfold candidates
  have row : edge 2371 = PrimePairCache.smallRow2371.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan272_1 : scan edge (search edge 3) 4 [1777] 1777 innerTail272_1 best = best := by
  rw [innerTail272_1, scan, if_neg (by decide), innerCandidates272_1_checked]
  change scan edge (search edge 3) 4 [1777] 1777 innerTail272_2
    (if 3 ≤ innerCandidates272_1.length then search edge 3 [1777, 2371] innerCandidates272_1 4148 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan272_2

def innerTail272_0 : List Nat := 2017 :: innerTail272_1
def innerCandidates272_0 : List Nat := [4621, 7603]
set_option maxRecDepth 4096 in
theorem innerCandidates272_0_checked : candidates edge 4 1777 2017 innerTail272_1 best.bound = innerCandidates272_0 := by
  unfold candidates
  have row : edge 2017 = PrimePairCache.smallRow2017.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan272_0 : scan edge (search edge 3) 4 [1777] 1777 innerTail272_0 best = best := by
  rw [innerTail272_0, scan, if_neg (by decide), innerCandidates272_0_checked]
  change scan edge (search edge 3) 4 [1777] 1777 innerTail272_1
    (if 3 ≤ innerCandidates272_0.length then search edge 3 [1777, 2017] innerCandidates272_0 3794 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan272_1

theorem rootBranch272 : search edge 4 [1777] rootCandidates272 1777 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1777] 1777 innerTail272_0 best = best
  exact innerScan272_0

def rootCandidates271 : List Nat := [1831, 2749, 3001, 3319, 3571, 3733, 3847, 3889, 4177, 5233, 5689, 6199, 6361, 6367, 6637, 6703, 7351, 7417]
set_option maxRecDepth 4096 in
theorem rootCandidates271_checked : candidates edge 5 0 1759 tail272 best.bound = rootCandidates271 := by
  unfold candidates
  have row : edge 1759 = PrimePairCache.smallRow1759.contains := by funext q; rfl
  rw [row]
  decide

def innerTail271_18 : List Nat := []
theorem innerScan271_18 : scan edge (search edge 3) 4 [1759] 1759 innerTail271_18 best = best := rfl

def innerTail271_17 : List Nat := 7417 :: innerTail271_18
def innerCandidates271_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates271_17_checked : candidates edge 4 1759 7417 innerTail271_18 best.bound = innerCandidates271_17 := by
  unfold candidates
  have row : edge 7417 = SearchData.row7417.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan271_17 : scan edge (search edge 3) 4 [1759] 1759 innerTail271_17 best = best := by
  rw [innerTail271_17, scan, if_neg (by decide), innerCandidates271_17_checked]
  change scan edge (search edge 3) 4 [1759] 1759 innerTail271_18
    (if 3 ≤ innerCandidates271_17.length then search edge 3 [1759, 7417] innerCandidates271_17 9176 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan271_18

def innerTail271_16 : List Nat := 7351 :: innerTail271_17
def innerCandidates271_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates271_16_checked : candidates edge 4 1759 7351 innerTail271_17 best.bound = innerCandidates271_16 := by
  unfold candidates
  have row : edge 7351 = SearchData.row7351.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan271_16 : scan edge (search edge 3) 4 [1759] 1759 innerTail271_16 best = best := by
  rw [innerTail271_16, scan, if_neg (by decide), innerCandidates271_16_checked]
  change scan edge (search edge 3) 4 [1759] 1759 innerTail271_17
    (if 3 ≤ innerCandidates271_16.length then search edge 3 [1759, 7351] innerCandidates271_16 9110 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan271_17

def innerTail271_15 : List Nat := 6703 :: innerTail271_16
def innerCandidates271_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates271_15_checked : candidates edge 4 1759 6703 innerTail271_16 best.bound = innerCandidates271_15 := by
  unfold candidates
  have row : edge 6703 = SearchData.row6703.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan271_15 : scan edge (search edge 3) 4 [1759] 1759 innerTail271_15 best = best := by
  rw [innerTail271_15, scan, if_neg (by decide), innerCandidates271_15_checked]
  change scan edge (search edge 3) 4 [1759] 1759 innerTail271_16
    (if 3 ≤ innerCandidates271_15.length then search edge 3 [1759, 6703] innerCandidates271_15 8462 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan271_16

def innerTail271_14 : List Nat := 6637 :: innerTail271_15
def innerCandidates271_14 : List Nat := [7351]
set_option maxRecDepth 4096 in
theorem innerCandidates271_14_checked : candidates edge 4 1759 6637 innerTail271_15 best.bound = innerCandidates271_14 := by
  unfold candidates
  have row : edge 6637 = SearchData.row6637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan271_14 : scan edge (search edge 3) 4 [1759] 1759 innerTail271_14 best = best := by
  rw [innerTail271_14, scan, if_neg (by decide), innerCandidates271_14_checked]
  change scan edge (search edge 3) 4 [1759] 1759 innerTail271_15
    (if 3 ≤ innerCandidates271_14.length then search edge 3 [1759, 6637] innerCandidates271_14 8396 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan271_15

def innerTail271_13 : List Nat := 6367 :: innerTail271_14
def innerCandidates271_13 : List Nat := [6703]
set_option maxRecDepth 4096 in
theorem innerCandidates271_13_checked : candidates edge 4 1759 6367 innerTail271_14 best.bound = innerCandidates271_13 := by
  unfold candidates
  have row : edge 6367 = SearchData.row6367.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan271_13 : scan edge (search edge 3) 4 [1759] 1759 innerTail271_13 best = best := by
  rw [innerTail271_13, scan, if_neg (by decide), innerCandidates271_13_checked]
  change scan edge (search edge 3) 4 [1759] 1759 innerTail271_14
    (if 3 ≤ innerCandidates271_13.length then search edge 3 [1759, 6367] innerCandidates271_13 8126 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan271_14

def innerTail271_12 : List Nat := 6361 :: innerTail271_13
def innerCandidates271_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates271_12_checked : candidates edge 4 1759 6361 innerTail271_13 best.bound = innerCandidates271_12 := by
  unfold candidates
  have row : edge 6361 = SearchData.row6361.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan271_12 : scan edge (search edge 3) 4 [1759] 1759 innerTail271_12 best = best := by
  rw [innerTail271_12, scan, if_neg (by decide), innerCandidates271_12_checked]
  change scan edge (search edge 3) 4 [1759] 1759 innerTail271_13
    (if 3 ≤ innerCandidates271_12.length then search edge 3 [1759, 6361] innerCandidates271_12 8120 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan271_13

def innerTail271_11 : List Nat := 6199 :: innerTail271_12
def innerCandidates271_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates271_11_checked : candidates edge 4 1759 6199 innerTail271_12 best.bound = innerCandidates271_11 := by
  unfold candidates
  have row : edge 6199 = SearchData.row6199.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan271_11 : scan edge (search edge 3) 4 [1759] 1759 innerTail271_11 best = best := by
  rw [innerTail271_11, scan, if_neg (by decide), innerCandidates271_11_checked]
  change scan edge (search edge 3) 4 [1759] 1759 innerTail271_12
    (if 3 ≤ innerCandidates271_11.length then search edge 3 [1759, 6199] innerCandidates271_11 7958 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan271_12

def innerTail271_10 : List Nat := 5689 :: innerTail271_11
def innerCandidates271_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates271_10_checked : candidates edge 4 1759 5689 innerTail271_11 best.bound = innerCandidates271_10 := by
  unfold candidates
  have row : edge 5689 = SearchData.row5689.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan271_10 : scan edge (search edge 3) 4 [1759] 1759 innerTail271_10 best = best := by
  rw [innerTail271_10, scan, if_neg (by decide), innerCandidates271_10_checked]
  change scan edge (search edge 3) 4 [1759] 1759 innerTail271_11
    (if 3 ≤ innerCandidates271_10.length then search edge 3 [1759, 5689] innerCandidates271_10 7448 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan271_11

def innerTail271_9 : List Nat := 5233 :: innerTail271_10
def innerCandidates271_9 : List Nat := [6367]
set_option maxRecDepth 4096 in
theorem innerCandidates271_9_checked : candidates edge 4 1759 5233 innerTail271_10 best.bound = innerCandidates271_9 := by
  unfold candidates
  have row : edge 5233 = SearchData.row5233.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan271_9 : scan edge (search edge 3) 4 [1759] 1759 innerTail271_9 best = best := by
  rw [innerTail271_9, scan, if_neg (by decide), innerCandidates271_9_checked]
  change scan edge (search edge 3) 4 [1759] 1759 innerTail271_10
    (if 3 ≤ innerCandidates271_9.length then search edge 3 [1759, 5233] innerCandidates271_9 6992 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan271_10

def innerTail271_8 : List Nat := 4177 :: innerTail271_9
def innerCandidates271_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates271_8_checked : candidates edge 4 1759 4177 innerTail271_9 best.bound = innerCandidates271_8 := by
  unfold candidates
  have row : edge 4177 = SearchData.row4177.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan271_8 : scan edge (search edge 3) 4 [1759] 1759 innerTail271_8 best = best := by
  rw [innerTail271_8, scan, if_neg (by decide), innerCandidates271_8_checked]
  change scan edge (search edge 3) 4 [1759] 1759 innerTail271_9
    (if 3 ≤ innerCandidates271_8.length then search edge 3 [1759, 4177] innerCandidates271_8 5936 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan271_9

def innerTail271_7 : List Nat := 3889 :: innerTail271_8
def innerCandidates271_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates271_7_checked : candidates edge 4 1759 3889 innerTail271_8 best.bound = innerCandidates271_7 := by
  unfold candidates
  have row : edge 3889 = SearchData.row3889.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan271_7 : scan edge (search edge 3) 4 [1759] 1759 innerTail271_7 best = best := by
  rw [innerTail271_7, scan, if_neg (by decide), innerCandidates271_7_checked]
  change scan edge (search edge 3) 4 [1759] 1759 innerTail271_8
    (if 3 ≤ innerCandidates271_7.length then search edge 3 [1759, 3889] innerCandidates271_7 5648 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan271_8

def innerTail271_6 : List Nat := 3847 :: innerTail271_7
def innerCandidates271_6 : List Nat := [6367]
set_option maxRecDepth 4096 in
theorem innerCandidates271_6_checked : candidates edge 4 1759 3847 innerTail271_7 best.bound = innerCandidates271_6 := by
  unfold candidates
  have row : edge 3847 = SearchData.row3847.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan271_6 : scan edge (search edge 3) 4 [1759] 1759 innerTail271_6 best = best := by
  rw [innerTail271_6, scan, if_neg (by decide), innerCandidates271_6_checked]
  change scan edge (search edge 3) 4 [1759] 1759 innerTail271_7
    (if 3 ≤ innerCandidates271_6.length then search edge 3 [1759, 3847] innerCandidates271_6 5606 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan271_7

def innerTail271_5 : List Nat := 3733 :: innerTail271_6
def innerCandidates271_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates271_5_checked : candidates edge 4 1759 3733 innerTail271_6 best.bound = innerCandidates271_5 := by
  unfold candidates
  have row : edge 3733 = SearchData.row3733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan271_5 : scan edge (search edge 3) 4 [1759] 1759 innerTail271_5 best = best := by
  rw [innerTail271_5, scan, if_neg (by decide), innerCandidates271_5_checked]
  change scan edge (search edge 3) 4 [1759] 1759 innerTail271_6
    (if 3 ≤ innerCandidates271_5.length then search edge 3 [1759, 3733] innerCandidates271_5 5492 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan271_6

def innerTail271_4 : List Nat := 3571 :: innerTail271_5
def innerCandidates271_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates271_4_checked : candidates edge 4 1759 3571 innerTail271_5 best.bound = innerCandidates271_4 := by
  unfold candidates
  have row : edge 3571 = SearchData.row3571.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan271_4 : scan edge (search edge 3) 4 [1759] 1759 innerTail271_4 best = best := by
  rw [innerTail271_4, scan, if_neg (by decide), innerCandidates271_4_checked]
  change scan edge (search edge 3) 4 [1759] 1759 innerTail271_5
    (if 3 ≤ innerCandidates271_4.length then search edge 3 [1759, 3571] innerCandidates271_4 5330 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan271_5

def innerTail271_3 : List Nat := 3319 :: innerTail271_4
def innerCandidates271_3 : List Nat := [6367]
set_option maxRecDepth 4096 in
theorem innerCandidates271_3_checked : candidates edge 4 1759 3319 innerTail271_4 best.bound = innerCandidates271_3 := by
  unfold candidates
  have row : edge 3319 = SearchData.row3319.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan271_3 : scan edge (search edge 3) 4 [1759] 1759 innerTail271_3 best = best := by
  rw [innerTail271_3, scan, if_neg (by decide), innerCandidates271_3_checked]
  change scan edge (search edge 3) 4 [1759] 1759 innerTail271_4
    (if 3 ≤ innerCandidates271_3.length then search edge 3 [1759, 3319] innerCandidates271_3 5078 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan271_4

def innerTail271_2 : List Nat := 3001 :: innerTail271_3
def innerCandidates271_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates271_2_checked : candidates edge 4 1759 3001 innerTail271_3 best.bound = innerCandidates271_2 := by
  unfold candidates
  have row : edge 3001 = SearchData.row3001.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan271_2 : scan edge (search edge 3) 4 [1759] 1759 innerTail271_2 best = best := by
  rw [innerTail271_2, scan, if_neg (by decide), innerCandidates271_2_checked]
  change scan edge (search edge 3) 4 [1759] 1759 innerTail271_3
    (if 3 ≤ innerCandidates271_2.length then search edge 3 [1759, 3001] innerCandidates271_2 4760 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan271_3

def innerTail271_1 : List Nat := 2749 :: innerTail271_2
def innerCandidates271_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates271_1_checked : candidates edge 4 1759 2749 innerTail271_2 best.bound = innerCandidates271_1 := by
  unfold candidates
  have row : edge 2749 = SearchData.row2749.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan271_1 : scan edge (search edge 3) 4 [1759] 1759 innerTail271_1 best = best := by
  rw [innerTail271_1, scan, if_neg (by decide), innerCandidates271_1_checked]
  change scan edge (search edge 3) 4 [1759] 1759 innerTail271_2
    (if 3 ≤ innerCandidates271_1.length then search edge 3 [1759, 2749] innerCandidates271_1 4508 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan271_2

def innerTail271_0 : List Nat := 1831 :: innerTail271_1
def innerCandidates271_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates271_0_checked : candidates edge 4 1759 1831 innerTail271_1 best.bound = innerCandidates271_0 := by
  unfold candidates
  have row : edge 1831 = PrimePairCache.smallRow1831.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan271_0 : scan edge (search edge 3) 4 [1759] 1759 innerTail271_0 best = best := by
  rw [innerTail271_0, scan, if_neg (by decide), innerCandidates271_0_checked]
  change scan edge (search edge 3) 4 [1759] 1759 innerTail271_1
    (if 3 ≤ innerCandidates271_0.length then search edge 3 [1759, 1831] innerCandidates271_0 3590 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan271_1

theorem rootBranch271 : search edge 4 [1759] rootCandidates271 1759 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1759] 1759 innerTail271_0 best = best
  exact innerScan271_0

def rootCandidates270 : List Nat := [1873, 1951, 3547, 3643, 3673, 3709, 3847, 4567, 4903, 4987, 5197, 5281, 5827, 5851, 6199, 6691, 6823, 6907, 7237, 7321, 7411, 7549, 7669, 7789, 8089, 8161]
set_option maxRecDepth 4096 in
theorem rootCandidates270_checked : candidates edge 5 0 1753 tail271 best.bound = rootCandidates270 := by
  unfold candidates
  have row : edge 1753 = PrimePairCache.smallRow1753.contains := by funext q; rfl
  rw [row]
  decide

def innerTail270_26 : List Nat := []
theorem innerScan270_26 : scan edge (search edge 3) 4 [1753] 1753 innerTail270_26 best = best := rfl

def innerTail270_25 : List Nat := 8161 :: innerTail270_26
def innerCandidates270_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates270_25_checked : candidates edge 4 1753 8161 innerTail270_26 best.bound = innerCandidates270_25 := by
  unfold candidates
  have row : edge 8161 = SearchData.row8161.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan270_25 : scan edge (search edge 3) 4 [1753] 1753 innerTail270_25 best = best := by
  rw [innerTail270_25, scan, if_neg (by decide), innerCandidates270_25_checked]
  change scan edge (search edge 3) 4 [1753] 1753 innerTail270_26
    (if 3 ≤ innerCandidates270_25.length then search edge 3 [1753, 8161] innerCandidates270_25 9914 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan270_26

def innerTail270_24 : List Nat := 8089 :: innerTail270_25
def innerCandidates270_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates270_24_checked : candidates edge 4 1753 8089 innerTail270_25 best.bound = innerCandidates270_24 := by
  unfold candidates
  have row : edge 8089 = SearchData.row8089.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan270_24 : scan edge (search edge 3) 4 [1753] 1753 innerTail270_24 best = best := by
  rw [innerTail270_24, scan, if_neg (by decide), innerCandidates270_24_checked]
  change scan edge (search edge 3) 4 [1753] 1753 innerTail270_25
    (if 3 ≤ innerCandidates270_24.length then search edge 3 [1753, 8089] innerCandidates270_24 9842 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan270_25

def innerTail270_23 : List Nat := 7789 :: innerTail270_24
def innerCandidates270_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates270_23_checked : candidates edge 4 1753 7789 innerTail270_24 best.bound = innerCandidates270_23 := by
  unfold candidates
  have row : edge 7789 = SearchData.row7789.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan270_23 : scan edge (search edge 3) 4 [1753] 1753 innerTail270_23 best = best := by
  rw [innerTail270_23, scan, if_neg (by decide), innerCandidates270_23_checked]
  change scan edge (search edge 3) 4 [1753] 1753 innerTail270_24
    (if 3 ≤ innerCandidates270_23.length then search edge 3 [1753, 7789] innerCandidates270_23 9542 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan270_24

def innerTail270_22 : List Nat := 7669 :: innerTail270_23
def innerCandidates270_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates270_22_checked : candidates edge 4 1753 7669 innerTail270_23 best.bound = innerCandidates270_22 := by
  unfold candidates
  have row : edge 7669 = SearchData.row7669.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan270_22 : scan edge (search edge 3) 4 [1753] 1753 innerTail270_22 best = best := by
  rw [innerTail270_22, scan, if_neg (by decide), innerCandidates270_22_checked]
  change scan edge (search edge 3) 4 [1753] 1753 innerTail270_23
    (if 3 ≤ innerCandidates270_22.length then search edge 3 [1753, 7669] innerCandidates270_22 9422 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan270_23

def innerTail270_21 : List Nat := 7549 :: innerTail270_22
def innerCandidates270_21 : List Nat := [8089]
set_option maxRecDepth 4096 in
theorem innerCandidates270_21_checked : candidates edge 4 1753 7549 innerTail270_22 best.bound = innerCandidates270_21 := by
  unfold candidates
  have row : edge 7549 = SearchData.row7549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan270_21 : scan edge (search edge 3) 4 [1753] 1753 innerTail270_21 best = best := by
  rw [innerTail270_21, scan, if_neg (by decide), innerCandidates270_21_checked]
  change scan edge (search edge 3) 4 [1753] 1753 innerTail270_22
    (if 3 ≤ innerCandidates270_21.length then search edge 3 [1753, 7549] innerCandidates270_21 9302 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan270_22

def innerTail270_20 : List Nat := 7411 :: innerTail270_21
def innerCandidates270_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates270_20_checked : candidates edge 4 1753 7411 innerTail270_21 best.bound = innerCandidates270_20 := by
  unfold candidates
  have row : edge 7411 = SearchData.row7411.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan270_20 : scan edge (search edge 3) 4 [1753] 1753 innerTail270_20 best = best := by
  rw [innerTail270_20, scan, if_neg (by decide), innerCandidates270_20_checked]
  change scan edge (search edge 3) 4 [1753] 1753 innerTail270_21
    (if 3 ≤ innerCandidates270_20.length then search edge 3 [1753, 7411] innerCandidates270_20 9164 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan270_21

def innerTail270_19 : List Nat := 7321 :: innerTail270_20
def innerCandidates270_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates270_19_checked : candidates edge 4 1753 7321 innerTail270_20 best.bound = innerCandidates270_19 := by
  unfold candidates
  have row : edge 7321 = SearchData.row7321.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan270_19 : scan edge (search edge 3) 4 [1753] 1753 innerTail270_19 best = best := by
  rw [innerTail270_19, scan, if_neg (by decide), innerCandidates270_19_checked]
  change scan edge (search edge 3) 4 [1753] 1753 innerTail270_20
    (if 3 ≤ innerCandidates270_19.length then search edge 3 [1753, 7321] innerCandidates270_19 9074 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan270_20

def innerTail270_18 : List Nat := 7237 :: innerTail270_19
def innerCandidates270_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates270_18_checked : candidates edge 4 1753 7237 innerTail270_19 best.bound = innerCandidates270_18 := by
  unfold candidates
  have row : edge 7237 = SearchData.row7237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan270_18 : scan edge (search edge 3) 4 [1753] 1753 innerTail270_18 best = best := by
  rw [innerTail270_18, scan, if_neg (by decide), innerCandidates270_18_checked]
  change scan edge (search edge 3) 4 [1753] 1753 innerTail270_19
    (if 3 ≤ innerCandidates270_18.length then search edge 3 [1753, 7237] innerCandidates270_18 8990 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan270_19

def innerTail270_17 : List Nat := 6907 :: innerTail270_18
def innerCandidates270_17 : List Nat := [7549]
set_option maxRecDepth 4096 in
theorem innerCandidates270_17_checked : candidates edge 4 1753 6907 innerTail270_18 best.bound = innerCandidates270_17 := by
  unfold candidates
  have row : edge 6907 = SearchData.row6907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan270_17 : scan edge (search edge 3) 4 [1753] 1753 innerTail270_17 best = best := by
  rw [innerTail270_17, scan, if_neg (by decide), innerCandidates270_17_checked]
  change scan edge (search edge 3) 4 [1753] 1753 innerTail270_18
    (if 3 ≤ innerCandidates270_17.length then search edge 3 [1753, 6907] innerCandidates270_17 8660 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan270_18

def innerTail270_16 : List Nat := 6823 :: innerTail270_17
def innerCandidates270_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates270_16_checked : candidates edge 4 1753 6823 innerTail270_17 best.bound = innerCandidates270_16 := by
  unfold candidates
  have row : edge 6823 = SearchData.row6823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan270_16 : scan edge (search edge 3) 4 [1753] 1753 innerTail270_16 best = best := by
  rw [innerTail270_16, scan, if_neg (by decide), innerCandidates270_16_checked]
  change scan edge (search edge 3) 4 [1753] 1753 innerTail270_17
    (if 3 ≤ innerCandidates270_16.length then search edge 3 [1753, 6823] innerCandidates270_16 8576 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan270_17

def innerTail270_15 : List Nat := 6691 :: innerTail270_16
def innerCandidates270_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates270_15_checked : candidates edge 4 1753 6691 innerTail270_16 best.bound = innerCandidates270_15 := by
  unfold candidates
  have row : edge 6691 = SearchData.row6691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan270_15 : scan edge (search edge 3) 4 [1753] 1753 innerTail270_15 best = best := by
  rw [innerTail270_15, scan, if_neg (by decide), innerCandidates270_15_checked]
  change scan edge (search edge 3) 4 [1753] 1753 innerTail270_16
    (if 3 ≤ innerCandidates270_15.length then search edge 3 [1753, 6691] innerCandidates270_15 8444 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan270_16

def innerTail270_14 : List Nat := 6199 :: innerTail270_15
def innerCandidates270_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates270_14_checked : candidates edge 4 1753 6199 innerTail270_15 best.bound = innerCandidates270_14 := by
  unfold candidates
  have row : edge 6199 = SearchData.row6199.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan270_14 : scan edge (search edge 3) 4 [1753] 1753 innerTail270_14 best = best := by
  rw [innerTail270_14, scan, if_neg (by decide), innerCandidates270_14_checked]
  change scan edge (search edge 3) 4 [1753] 1753 innerTail270_15
    (if 3 ≤ innerCandidates270_14.length then search edge 3 [1753, 6199] innerCandidates270_14 7952 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan270_15

def innerTail270_13 : List Nat := 5851 :: innerTail270_14
def innerCandidates270_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates270_13_checked : candidates edge 4 1753 5851 innerTail270_14 best.bound = innerCandidates270_13 := by
  unfold candidates
  have row : edge 5851 = SearchData.row5851.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan270_13 : scan edge (search edge 3) 4 [1753] 1753 innerTail270_13 best = best := by
  rw [innerTail270_13, scan, if_neg (by decide), innerCandidates270_13_checked]
  change scan edge (search edge 3) 4 [1753] 1753 innerTail270_14
    (if 3 ≤ innerCandidates270_13.length then search edge 3 [1753, 5851] innerCandidates270_13 7604 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan270_14

def innerTail270_12 : List Nat := 5827 :: innerTail270_13
def innerCandidates270_12 : List Nat := [8089]
set_option maxRecDepth 4096 in
theorem innerCandidates270_12_checked : candidates edge 4 1753 5827 innerTail270_13 best.bound = innerCandidates270_12 := by
  unfold candidates
  have row : edge 5827 = SearchData.row5827.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan270_12 : scan edge (search edge 3) 4 [1753] 1753 innerTail270_12 best = best := by
  rw [innerTail270_12, scan, if_neg (by decide), innerCandidates270_12_checked]
  change scan edge (search edge 3) 4 [1753] 1753 innerTail270_13
    (if 3 ≤ innerCandidates270_12.length then search edge 3 [1753, 5827] innerCandidates270_12 7580 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan270_13

def innerTail270_11 : List Nat := 5281 :: innerTail270_12
def innerCandidates270_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates270_11_checked : candidates edge 4 1753 5281 innerTail270_12 best.bound = innerCandidates270_11 := by
  unfold candidates
  have row : edge 5281 = SearchData.row5281.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan270_11 : scan edge (search edge 3) 4 [1753] 1753 innerTail270_11 best = best := by
  rw [innerTail270_11, scan, if_neg (by decide), innerCandidates270_11_checked]
  change scan edge (search edge 3) 4 [1753] 1753 innerTail270_12
    (if 3 ≤ innerCandidates270_11.length then search edge 3 [1753, 5281] innerCandidates270_11 7034 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan270_12

def innerTail270_10 : List Nat := 5197 :: innerTail270_11
def innerCandidates270_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates270_10_checked : candidates edge 4 1753 5197 innerTail270_11 best.bound = innerCandidates270_10 := by
  unfold candidates
  have row : edge 5197 = SearchData.row5197.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan270_10 : scan edge (search edge 3) 4 [1753] 1753 innerTail270_10 best = best := by
  rw [innerTail270_10, scan, if_neg (by decide), innerCandidates270_10_checked]
  change scan edge (search edge 3) 4 [1753] 1753 innerTail270_11
    (if 3 ≤ innerCandidates270_10.length then search edge 3 [1753, 5197] innerCandidates270_10 6950 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan270_11

def innerTail270_9 : List Nat := 4987 :: innerTail270_10
def innerCandidates270_9 : List Nat := [5197]
set_option maxRecDepth 4096 in
theorem innerCandidates270_9_checked : candidates edge 4 1753 4987 innerTail270_10 best.bound = innerCandidates270_9 := by
  unfold candidates
  have row : edge 4987 = SearchData.row4987.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan270_9 : scan edge (search edge 3) 4 [1753] 1753 innerTail270_9 best = best := by
  rw [innerTail270_9, scan, if_neg (by decide), innerCandidates270_9_checked]
  change scan edge (search edge 3) 4 [1753] 1753 innerTail270_10
    (if 3 ≤ innerCandidates270_9.length then search edge 3 [1753, 4987] innerCandidates270_9 6740 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan270_10

def innerTail270_8 : List Nat := 4903 :: innerTail270_9
def innerCandidates270_8 : List Nat := [7789]
set_option maxRecDepth 4096 in
theorem innerCandidates270_8_checked : candidates edge 4 1753 4903 innerTail270_9 best.bound = innerCandidates270_8 := by
  unfold candidates
  have row : edge 4903 = SearchData.row4903.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan270_8 : scan edge (search edge 3) 4 [1753] 1753 innerTail270_8 best = best := by
  rw [innerTail270_8, scan, if_neg (by decide), innerCandidates270_8_checked]
  change scan edge (search edge 3) 4 [1753] 1753 innerTail270_9
    (if 3 ≤ innerCandidates270_8.length then search edge 3 [1753, 4903] innerCandidates270_8 6656 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan270_9

def innerTail270_7 : List Nat := 4567 :: innerTail270_8
def innerCandidates270_7 : List Nat := [7789]
set_option maxRecDepth 4096 in
theorem innerCandidates270_7_checked : candidates edge 4 1753 4567 innerTail270_8 best.bound = innerCandidates270_7 := by
  unfold candidates
  have row : edge 4567 = SearchData.row4567.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan270_7 : scan edge (search edge 3) 4 [1753] 1753 innerTail270_7 best = best := by
  rw [innerTail270_7, scan, if_neg (by decide), innerCandidates270_7_checked]
  change scan edge (search edge 3) 4 [1753] 1753 innerTail270_8
    (if 3 ≤ innerCandidates270_7.length then search edge 3 [1753, 4567] innerCandidates270_7 6320 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan270_8

def innerTail270_6 : List Nat := 3847 :: innerTail270_7
def innerCandidates270_6 : List Nat := [5197]
set_option maxRecDepth 4096 in
theorem innerCandidates270_6_checked : candidates edge 4 1753 3847 innerTail270_7 best.bound = innerCandidates270_6 := by
  unfold candidates
  have row : edge 3847 = SearchData.row3847.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan270_6 : scan edge (search edge 3) 4 [1753] 1753 innerTail270_6 best = best := by
  rw [innerTail270_6, scan, if_neg (by decide), innerCandidates270_6_checked]
  change scan edge (search edge 3) 4 [1753] 1753 innerTail270_7
    (if 3 ≤ innerCandidates270_6.length then search edge 3 [1753, 3847] innerCandidates270_6 5600 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan270_7

def innerTail270_5 : List Nat := 3709 :: innerTail270_6
def innerCandidates270_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates270_5_checked : candidates edge 4 1753 3709 innerTail270_6 best.bound = innerCandidates270_5 := by
  unfold candidates
  have row : edge 3709 = SearchData.row3709.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan270_5 : scan edge (search edge 3) 4 [1753] 1753 innerTail270_5 best = best := by
  rw [innerTail270_5, scan, if_neg (by decide), innerCandidates270_5_checked]
  change scan edge (search edge 3) 4 [1753] 1753 innerTail270_6
    (if 3 ≤ innerCandidates270_5.length then search edge 3 [1753, 3709] innerCandidates270_5 5462 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan270_6

def innerTail270_4 : List Nat := 3673 :: innerTail270_5
def innerCandidates270_4 : List Nat := [6691]
set_option maxRecDepth 4096 in
theorem innerCandidates270_4_checked : candidates edge 4 1753 3673 innerTail270_5 best.bound = innerCandidates270_4 := by
  unfold candidates
  have row : edge 3673 = SearchData.row3673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan270_4 : scan edge (search edge 3) 4 [1753] 1753 innerTail270_4 best = best := by
  rw [innerTail270_4, scan, if_neg (by decide), innerCandidates270_4_checked]
  change scan edge (search edge 3) 4 [1753] 1753 innerTail270_5
    (if 3 ≤ innerCandidates270_4.length then search edge 3 [1753, 3673] innerCandidates270_4 5426 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan270_5

def innerTail270_3 : List Nat := 3643 :: innerTail270_4
def innerCandidates270_3 : List Nat := [4987, 6199]
set_option maxRecDepth 4096 in
theorem innerCandidates270_3_checked : candidates edge 4 1753 3643 innerTail270_4 best.bound = innerCandidates270_3 := by
  unfold candidates
  have row : edge 3643 = SearchData.row3643.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan270_3 : scan edge (search edge 3) 4 [1753] 1753 innerTail270_3 best = best := by
  rw [innerTail270_3, scan, if_neg (by decide), innerCandidates270_3_checked]
  change scan edge (search edge 3) 4 [1753] 1753 innerTail270_4
    (if 3 ≤ innerCandidates270_3.length then search edge 3 [1753, 3643] innerCandidates270_3 5396 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan270_4

def innerTail270_2 : List Nat := 3547 :: innerTail270_3
def innerCandidates270_2 : List Nat := [3643, 5827]
set_option maxRecDepth 4096 in
theorem innerCandidates270_2_checked : candidates edge 4 1753 3547 innerTail270_3 best.bound = innerCandidates270_2 := by
  unfold candidates
  have row : edge 3547 = SearchData.row3547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan270_2 : scan edge (search edge 3) 4 [1753] 1753 innerTail270_2 best = best := by
  rw [innerTail270_2, scan, if_neg (by decide), innerCandidates270_2_checked]
  change scan edge (search edge 3) 4 [1753] 1753 innerTail270_3
    (if 3 ≤ innerCandidates270_2.length then search edge 3 [1753, 3547] innerCandidates270_2 5300 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan270_3

def innerTail270_1 : List Nat := 1951 :: innerTail270_2
def innerCandidates270_1 : List Nat := [3547, 3643, 6823]
set_option maxRecDepth 4096 in
theorem innerCandidates270_1_checked : candidates edge 4 1753 1951 innerTail270_2 best.bound = innerCandidates270_1 := by
  unfold candidates
  have row : edge 1951 = PrimePairCache.smallRow1951.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild270_1 : search edge 3 [1753, 1951] innerCandidates270_1 3704 best = best := by decide

theorem innerScan270_1 : scan edge (search edge 3) 4 [1753] 1753 innerTail270_1 best = best := by
  rw [innerTail270_1, scan, if_neg (by decide), innerCandidates270_1_checked]
  change scan edge (search edge 3) 4 [1753] 1753 innerTail270_2
    (if 3 ≤ innerCandidates270_1.length then search edge 3 [1753, 1951] innerCandidates270_1 3704 best else best) = best
  rw [if_pos (by decide), innerChild270_1]
  exact innerScan270_2

def innerTail270_0 : List Nat := 1873 :: innerTail270_1
def innerCandidates270_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates270_0_checked : candidates edge 4 1753 1873 innerTail270_1 best.bound = innerCandidates270_0 := by
  unfold candidates
  have row : edge 1873 = PrimePairCache.smallRow1873.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan270_0 : scan edge (search edge 3) 4 [1753] 1753 innerTail270_0 best = best := by
  rw [innerTail270_0, scan, if_neg (by decide), innerCandidates270_0_checked]
  change scan edge (search edge 3) 4 [1753] 1753 innerTail270_1
    (if 3 ≤ innerCandidates270_0.length then search edge 3 [1753, 1873] innerCandidates270_0 3626 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan270_1

theorem rootBranch270 : search edge 4 [1753] rootCandidates270 1753 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1753] 1753 innerTail270_0 best = best
  exact innerScan270_0

end ProjectEulerSolutions.PrimePairUpper.Search
