import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates353 : List Nat := [2411, 2423, 2663, 2927, 3083, 3347, 3491, 3557, 3821, 3851, 4241, 4919, 5981, 6329, 6791, 7127, 7829, 8069]
set_option maxRecDepth 4096 in
theorem rootCandidates353_checked : candidates edge 5 0 2393 tail354 best.bound = rootCandidates353 := by
  unfold candidates
  have row : edge 2393 = PrimePairCache.smallRow2393.contains := by funext q; rfl
  rw [row]
  decide

def innerTail353_18 : List Nat := []
theorem innerScan353_18 : scan edge (search edge 3) 4 [2393] 2393 innerTail353_18 best = best := rfl

def innerTail353_17 : List Nat := 8069 :: innerTail353_18
def innerCandidates353_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates353_17_checked : candidates edge 4 2393 8069 innerTail353_18 best.bound = innerCandidates353_17 := by
  unfold candidates
  have row : edge 8069 = SearchData.row8069.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan353_17 : scan edge (search edge 3) 4 [2393] 2393 innerTail353_17 best = best := by
  rw [innerTail353_17, scan, if_neg (by decide), innerCandidates353_17_checked]
  change scan edge (search edge 3) 4 [2393] 2393 innerTail353_18
    (if 3 ≤ innerCandidates353_17.length then search edge 3 [2393, 8069] innerCandidates353_17 10462 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan353_18

def innerTail353_16 : List Nat := 7829 :: innerTail353_17
def innerCandidates353_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates353_16_checked : candidates edge 4 2393 7829 innerTail353_17 best.bound = innerCandidates353_16 := by
  unfold candidates
  have row : edge 7829 = SearchData.row7829.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan353_16 : scan edge (search edge 3) 4 [2393] 2393 innerTail353_16 best = best := by
  rw [innerTail353_16, scan, if_neg (by decide), innerCandidates353_16_checked]
  change scan edge (search edge 3) 4 [2393] 2393 innerTail353_17
    (if 3 ≤ innerCandidates353_16.length then search edge 3 [2393, 7829] innerCandidates353_16 10222 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan353_17

def innerTail353_15 : List Nat := 7127 :: innerTail353_16
def innerCandidates353_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates353_15_checked : candidates edge 4 2393 7127 innerTail353_16 best.bound = innerCandidates353_15 := by
  unfold candidates
  have row : edge 7127 = SearchData.row7127.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan353_15 : scan edge (search edge 3) 4 [2393] 2393 innerTail353_15 best = best := by
  rw [innerTail353_15, scan, if_neg (by decide), innerCandidates353_15_checked]
  change scan edge (search edge 3) 4 [2393] 2393 innerTail353_16
    (if 3 ≤ innerCandidates353_15.length then search edge 3 [2393, 7127] innerCandidates353_15 9520 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan353_16

def innerTail353_14 : List Nat := 6791 :: innerTail353_15
def innerCandidates353_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates353_14_checked : candidates edge 4 2393 6791 innerTail353_15 best.bound = innerCandidates353_14 := by
  unfold candidates
  have row : edge 6791 = SearchData.row6791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan353_14 : scan edge (search edge 3) 4 [2393] 2393 innerTail353_14 best = best := by
  rw [innerTail353_14, scan, if_neg (by decide), innerCandidates353_14_checked]
  change scan edge (search edge 3) 4 [2393] 2393 innerTail353_15
    (if 3 ≤ innerCandidates353_14.length then search edge 3 [2393, 6791] innerCandidates353_14 9184 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan353_15

def innerTail353_13 : List Nat := 6329 :: innerTail353_14
def innerCandidates353_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates353_13_checked : candidates edge 4 2393 6329 innerTail353_14 best.bound = innerCandidates353_13 := by
  unfold candidates
  have row : edge 6329 = SearchData.row6329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan353_13 : scan edge (search edge 3) 4 [2393] 2393 innerTail353_13 best = best := by
  rw [innerTail353_13, scan, if_neg (by decide), innerCandidates353_13_checked]
  change scan edge (search edge 3) 4 [2393] 2393 innerTail353_14
    (if 3 ≤ innerCandidates353_13.length then search edge 3 [2393, 6329] innerCandidates353_13 8722 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan353_14

def innerTail353_12 : List Nat := 5981 :: innerTail353_13
def innerCandidates353_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates353_12_checked : candidates edge 4 2393 5981 innerTail353_13 best.bound = innerCandidates353_12 := by
  unfold candidates
  have row : edge 5981 = SearchData.row5981.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan353_12 : scan edge (search edge 3) 4 [2393] 2393 innerTail353_12 best = best := by
  rw [innerTail353_12, scan, if_neg (by decide), innerCandidates353_12_checked]
  change scan edge (search edge 3) 4 [2393] 2393 innerTail353_13
    (if 3 ≤ innerCandidates353_12.length then search edge 3 [2393, 5981] innerCandidates353_12 8374 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan353_13

def innerTail353_11 : List Nat := 4919 :: innerTail353_12
def innerCandidates353_11 : List Nat := [7829]
set_option maxRecDepth 4096 in
theorem innerCandidates353_11_checked : candidates edge 4 2393 4919 innerTail353_12 best.bound = innerCandidates353_11 := by
  unfold candidates
  have row : edge 4919 = SearchData.row4919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan353_11 : scan edge (search edge 3) 4 [2393] 2393 innerTail353_11 best = best := by
  rw [innerTail353_11, scan, if_neg (by decide), innerCandidates353_11_checked]
  change scan edge (search edge 3) 4 [2393] 2393 innerTail353_12
    (if 3 ≤ innerCandidates353_11.length then search edge 3 [2393, 4919] innerCandidates353_11 7312 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan353_12

def innerTail353_10 : List Nat := 4241 :: innerTail353_11
def innerCandidates353_10 : List Nat := [5981]
set_option maxRecDepth 4096 in
theorem innerCandidates353_10_checked : candidates edge 4 2393 4241 innerTail353_11 best.bound = innerCandidates353_10 := by
  unfold candidates
  have row : edge 4241 = SearchData.row4241.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan353_10 : scan edge (search edge 3) 4 [2393] 2393 innerTail353_10 best = best := by
  rw [innerTail353_10, scan, if_neg (by decide), innerCandidates353_10_checked]
  change scan edge (search edge 3) 4 [2393] 2393 innerTail353_11
    (if 3 ≤ innerCandidates353_10.length then search edge 3 [2393, 4241] innerCandidates353_10 6634 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan353_11

def innerTail353_9 : List Nat := 3851 :: innerTail353_10
def innerCandidates353_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates353_9_checked : candidates edge 4 2393 3851 innerTail353_10 best.bound = innerCandidates353_9 := by
  unfold candidates
  have row : edge 3851 = SearchData.row3851.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan353_9 : scan edge (search edge 3) 4 [2393] 2393 innerTail353_9 best = best := by
  rw [innerTail353_9, scan, if_neg (by decide), innerCandidates353_9_checked]
  change scan edge (search edge 3) 4 [2393] 2393 innerTail353_10
    (if 3 ≤ innerCandidates353_9.length then search edge 3 [2393, 3851] innerCandidates353_9 6244 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan353_10

def innerTail353_8 : List Nat := 3821 :: innerTail353_9
def innerCandidates353_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates353_8_checked : candidates edge 4 2393 3821 innerTail353_9 best.bound = innerCandidates353_8 := by
  unfold candidates
  have row : edge 3821 = SearchData.row3821.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan353_8 : scan edge (search edge 3) 4 [2393] 2393 innerTail353_8 best = best := by
  rw [innerTail353_8, scan, if_neg (by decide), innerCandidates353_8_checked]
  change scan edge (search edge 3) 4 [2393] 2393 innerTail353_9
    (if 3 ≤ innerCandidates353_8.length then search edge 3 [2393, 3821] innerCandidates353_8 6214 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan353_9

def innerTail353_7 : List Nat := 3557 :: innerTail353_8
def innerCandidates353_7 : List Nat := [6329]
set_option maxRecDepth 4096 in
theorem innerCandidates353_7_checked : candidates edge 4 2393 3557 innerTail353_8 best.bound = innerCandidates353_7 := by
  unfold candidates
  have row : edge 3557 = SearchData.row3557.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan353_7 : scan edge (search edge 3) 4 [2393] 2393 innerTail353_7 best = best := by
  rw [innerTail353_7, scan, if_neg (by decide), innerCandidates353_7_checked]
  change scan edge (search edge 3) 4 [2393] 2393 innerTail353_8
    (if 3 ≤ innerCandidates353_7.length then search edge 3 [2393, 3557] innerCandidates353_7 5950 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan353_8

def innerTail353_6 : List Nat := 3491 :: innerTail353_7
def innerCandidates353_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates353_6_checked : candidates edge 4 2393 3491 innerTail353_7 best.bound = innerCandidates353_6 := by
  unfold candidates
  have row : edge 3491 = SearchData.row3491.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan353_6 : scan edge (search edge 3) 4 [2393] 2393 innerTail353_6 best = best := by
  rw [innerTail353_6, scan, if_neg (by decide), innerCandidates353_6_checked]
  change scan edge (search edge 3) 4 [2393] 2393 innerTail353_7
    (if 3 ≤ innerCandidates353_6.length then search edge 3 [2393, 3491] innerCandidates353_6 5884 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan353_7

def innerTail353_5 : List Nat := 3347 :: innerTail353_6
def innerCandidates353_5 : List Nat := [3821]
set_option maxRecDepth 4096 in
theorem innerCandidates353_5_checked : candidates edge 4 2393 3347 innerTail353_6 best.bound = innerCandidates353_5 := by
  unfold candidates
  have row : edge 3347 = SearchData.row3347.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan353_5 : scan edge (search edge 3) 4 [2393] 2393 innerTail353_5 best = best := by
  rw [innerTail353_5, scan, if_neg (by decide), innerCandidates353_5_checked]
  change scan edge (search edge 3) 4 [2393] 2393 innerTail353_6
    (if 3 ≤ innerCandidates353_5.length then search edge 3 [2393, 3347] innerCandidates353_5 5740 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan353_6

def innerTail353_4 : List Nat := 3083 :: innerTail353_5
def innerCandidates353_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates353_4_checked : candidates edge 4 2393 3083 innerTail353_5 best.bound = innerCandidates353_4 := by
  unfold candidates
  have row : edge 3083 = SearchData.row3083.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan353_4 : scan edge (search edge 3) 4 [2393] 2393 innerTail353_4 best = best := by
  rw [innerTail353_4, scan, if_neg (by decide), innerCandidates353_4_checked]
  change scan edge (search edge 3) 4 [2393] 2393 innerTail353_5
    (if 3 ≤ innerCandidates353_4.length then search edge 3 [2393, 3083] innerCandidates353_4 5476 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan353_5

def innerTail353_3 : List Nat := 2927 :: innerTail353_4
def innerCandidates353_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates353_3_checked : candidates edge 4 2393 2927 innerTail353_4 best.bound = innerCandidates353_3 := by
  unfold candidates
  have row : edge 2927 = SearchData.row2927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan353_3 : scan edge (search edge 3) 4 [2393] 2393 innerTail353_3 best = best := by
  rw [innerTail353_3, scan, if_neg (by decide), innerCandidates353_3_checked]
  change scan edge (search edge 3) 4 [2393] 2393 innerTail353_4
    (if 3 ≤ innerCandidates353_3.length then search edge 3 [2393, 2927] innerCandidates353_3 5320 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan353_4

def innerTail353_2 : List Nat := 2663 :: innerTail353_3
def innerCandidates353_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates353_2_checked : candidates edge 4 2393 2663 innerTail353_3 best.bound = innerCandidates353_2 := by
  unfold candidates
  have row : edge 2663 = PrimePairCache.smallRow2663.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan353_2 : scan edge (search edge 3) 4 [2393] 2393 innerTail353_2 best = best := by
  rw [innerTail353_2, scan, if_neg (by decide), innerCandidates353_2_checked]
  change scan edge (search edge 3) 4 [2393] 2393 innerTail353_3
    (if 3 ≤ innerCandidates353_2.length then search edge 3 [2393, 2663] innerCandidates353_2 5056 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan353_3

def innerTail353_1 : List Nat := 2423 :: innerTail353_2
def innerCandidates353_1 : List Nat := [3851]
set_option maxRecDepth 4096 in
theorem innerCandidates353_1_checked : candidates edge 4 2393 2423 innerTail353_2 best.bound = innerCandidates353_1 := by
  unfold candidates
  have row : edge 2423 = PrimePairCache.smallRow2423.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan353_1 : scan edge (search edge 3) 4 [2393] 2393 innerTail353_1 best = best := by
  rw [innerTail353_1, scan, if_neg (by decide), innerCandidates353_1_checked]
  change scan edge (search edge 3) 4 [2393] 2393 innerTail353_2
    (if 3 ≤ innerCandidates353_1.length then search edge 3 [2393, 2423] innerCandidates353_1 4816 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan353_2

def innerTail353_0 : List Nat := 2411 :: innerTail353_1
def innerCandidates353_0 : List Nat := [4241, 4919]
set_option maxRecDepth 4096 in
theorem innerCandidates353_0_checked : candidates edge 4 2393 2411 innerTail353_1 best.bound = innerCandidates353_0 := by
  unfold candidates
  have row : edge 2411 = PrimePairCache.smallRow2411.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan353_0 : scan edge (search edge 3) 4 [2393] 2393 innerTail353_0 best = best := by
  rw [innerTail353_0, scan, if_neg (by decide), innerCandidates353_0_checked]
  change scan edge (search edge 3) 4 [2393] 2393 innerTail353_1
    (if 3 ≤ innerCandidates353_0.length then search edge 3 [2393, 2411] innerCandidates353_0 4804 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan353_1

theorem rootBranch353 : search edge 4 [2393] rootCandidates353 2393 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2393] 2393 innerTail353_0 best = best
  exact innerScan353_0

def rootCandidates352 : List Nat := [2503, 2521, 2767, 3733, 4243, 4957, 5023, 5167, 5479, 6427, 6529, 6703, 6871, 6907, 7351, 7477, 7561]
set_option maxRecDepth 4096 in
theorem rootCandidates352_checked : candidates edge 5 0 2389 tail353 best.bound = rootCandidates352 := by
  unfold candidates
  have row : edge 2389 = PrimePairCache.smallRow2389.contains := by funext q; rfl
  rw [row]
  decide

def innerTail352_17 : List Nat := []
theorem innerScan352_17 : scan edge (search edge 3) 4 [2389] 2389 innerTail352_17 best = best := rfl

def innerTail352_16 : List Nat := 7561 :: innerTail352_17
def innerCandidates352_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates352_16_checked : candidates edge 4 2389 7561 innerTail352_17 best.bound = innerCandidates352_16 := by
  unfold candidates
  have row : edge 7561 = SearchData.row7561.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan352_16 : scan edge (search edge 3) 4 [2389] 2389 innerTail352_16 best = best := by
  rw [innerTail352_16, scan, if_neg (by decide), innerCandidates352_16_checked]
  change scan edge (search edge 3) 4 [2389] 2389 innerTail352_17
    (if 3 ≤ innerCandidates352_16.length then search edge 3 [2389, 7561] innerCandidates352_16 9950 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan352_17

def innerTail352_15 : List Nat := 7477 :: innerTail352_16
def innerCandidates352_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates352_15_checked : candidates edge 4 2389 7477 innerTail352_16 best.bound = innerCandidates352_15 := by
  unfold candidates
  have row : edge 7477 = SearchData.row7477.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan352_15 : scan edge (search edge 3) 4 [2389] 2389 innerTail352_15 best = best := by
  rw [innerTail352_15, scan, if_neg (by decide), innerCandidates352_15_checked]
  change scan edge (search edge 3) 4 [2389] 2389 innerTail352_16
    (if 3 ≤ innerCandidates352_15.length then search edge 3 [2389, 7477] innerCandidates352_15 9866 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan352_16

def innerTail352_14 : List Nat := 7351 :: innerTail352_15
def innerCandidates352_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates352_14_checked : candidates edge 4 2389 7351 innerTail352_15 best.bound = innerCandidates352_14 := by
  unfold candidates
  have row : edge 7351 = SearchData.row7351.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan352_14 : scan edge (search edge 3) 4 [2389] 2389 innerTail352_14 best = best := by
  rw [innerTail352_14, scan, if_neg (by decide), innerCandidates352_14_checked]
  change scan edge (search edge 3) 4 [2389] 2389 innerTail352_15
    (if 3 ≤ innerCandidates352_14.length then search edge 3 [2389, 7351] innerCandidates352_14 9740 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan352_15

def innerTail352_13 : List Nat := 6907 :: innerTail352_14
def innerCandidates352_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates352_13_checked : candidates edge 4 2389 6907 innerTail352_14 best.bound = innerCandidates352_13 := by
  unfold candidates
  have row : edge 6907 = SearchData.row6907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan352_13 : scan edge (search edge 3) 4 [2389] 2389 innerTail352_13 best = best := by
  rw [innerTail352_13, scan, if_neg (by decide), innerCandidates352_13_checked]
  change scan edge (search edge 3) 4 [2389] 2389 innerTail352_14
    (if 3 ≤ innerCandidates352_13.length then search edge 3 [2389, 6907] innerCandidates352_13 9296 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan352_14

def innerTail352_12 : List Nat := 6871 :: innerTail352_13
def innerCandidates352_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates352_12_checked : candidates edge 4 2389 6871 innerTail352_13 best.bound = innerCandidates352_12 := by
  unfold candidates
  have row : edge 6871 = SearchData.row6871.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan352_12 : scan edge (search edge 3) 4 [2389] 2389 innerTail352_12 best = best := by
  rw [innerTail352_12, scan, if_neg (by decide), innerCandidates352_12_checked]
  change scan edge (search edge 3) 4 [2389] 2389 innerTail352_13
    (if 3 ≤ innerCandidates352_12.length then search edge 3 [2389, 6871] innerCandidates352_12 9260 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan352_13

def innerTail352_11 : List Nat := 6703 :: innerTail352_12
def innerCandidates352_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates352_11_checked : candidates edge 4 2389 6703 innerTail352_12 best.bound = innerCandidates352_11 := by
  unfold candidates
  have row : edge 6703 = SearchData.row6703.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan352_11 : scan edge (search edge 3) 4 [2389] 2389 innerTail352_11 best = best := by
  rw [innerTail352_11, scan, if_neg (by decide), innerCandidates352_11_checked]
  change scan edge (search edge 3) 4 [2389] 2389 innerTail352_12
    (if 3 ≤ innerCandidates352_11.length then search edge 3 [2389, 6703] innerCandidates352_11 9092 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan352_12

def innerTail352_10 : List Nat := 6529 :: innerTail352_11
def innerCandidates352_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates352_10_checked : candidates edge 4 2389 6529 innerTail352_11 best.bound = innerCandidates352_10 := by
  unfold candidates
  have row : edge 6529 = SearchData.row6529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan352_10 : scan edge (search edge 3) 4 [2389] 2389 innerTail352_10 best = best := by
  rw [innerTail352_10, scan, if_neg (by decide), innerCandidates352_10_checked]
  change scan edge (search edge 3) 4 [2389] 2389 innerTail352_11
    (if 3 ≤ innerCandidates352_10.length then search edge 3 [2389, 6529] innerCandidates352_10 8918 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan352_11

def innerTail352_9 : List Nat := 6427 :: innerTail352_10
def innerCandidates352_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates352_9_checked : candidates edge 4 2389 6427 innerTail352_10 best.bound = innerCandidates352_9 := by
  unfold candidates
  have row : edge 6427 = SearchData.row6427.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan352_9 : scan edge (search edge 3) 4 [2389] 2389 innerTail352_9 best = best := by
  rw [innerTail352_9, scan, if_neg (by decide), innerCandidates352_9_checked]
  change scan edge (search edge 3) 4 [2389] 2389 innerTail352_10
    (if 3 ≤ innerCandidates352_9.length then search edge 3 [2389, 6427] innerCandidates352_9 8816 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan352_10

def innerTail352_8 : List Nat := 5479 :: innerTail352_9
def innerCandidates352_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates352_8_checked : candidates edge 4 2389 5479 innerTail352_9 best.bound = innerCandidates352_8 := by
  unfold candidates
  have row : edge 5479 = SearchData.row5479.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan352_8 : scan edge (search edge 3) 4 [2389] 2389 innerTail352_8 best = best := by
  rw [innerTail352_8, scan, if_neg (by decide), innerCandidates352_8_checked]
  change scan edge (search edge 3) 4 [2389] 2389 innerTail352_9
    (if 3 ≤ innerCandidates352_8.length then search edge 3 [2389, 5479] innerCandidates352_8 7868 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan352_9

def innerTail352_7 : List Nat := 5167 :: innerTail352_8
def innerCandidates352_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates352_7_checked : candidates edge 4 2389 5167 innerTail352_8 best.bound = innerCandidates352_7 := by
  unfold candidates
  have row : edge 5167 = SearchData.row5167.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan352_7 : scan edge (search edge 3) 4 [2389] 2389 innerTail352_7 best = best := by
  rw [innerTail352_7, scan, if_neg (by decide), innerCandidates352_7_checked]
  change scan edge (search edge 3) 4 [2389] 2389 innerTail352_8
    (if 3 ≤ innerCandidates352_7.length then search edge 3 [2389, 5167] innerCandidates352_7 7556 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan352_8

def innerTail352_6 : List Nat := 5023 :: innerTail352_7
def innerCandidates352_6 : List Nat := [7477]
set_option maxRecDepth 4096 in
theorem innerCandidates352_6_checked : candidates edge 4 2389 5023 innerTail352_7 best.bound = innerCandidates352_6 := by
  unfold candidates
  have row : edge 5023 = SearchData.row5023.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan352_6 : scan edge (search edge 3) 4 [2389] 2389 innerTail352_6 best = best := by
  rw [innerTail352_6, scan, if_neg (by decide), innerCandidates352_6_checked]
  change scan edge (search edge 3) 4 [2389] 2389 innerTail352_7
    (if 3 ≤ innerCandidates352_6.length then search edge 3 [2389, 5023] innerCandidates352_6 7412 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan352_7

def innerTail352_5 : List Nat := 4957 :: innerTail352_6
def innerCandidates352_5 : List Nat := [5023]
set_option maxRecDepth 4096 in
theorem innerCandidates352_5_checked : candidates edge 4 2389 4957 innerTail352_6 best.bound = innerCandidates352_5 := by
  unfold candidates
  have row : edge 4957 = SearchData.row4957.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan352_5 : scan edge (search edge 3) 4 [2389] 2389 innerTail352_5 best = best := by
  rw [innerTail352_5, scan, if_neg (by decide), innerCandidates352_5_checked]
  change scan edge (search edge 3) 4 [2389] 2389 innerTail352_6
    (if 3 ≤ innerCandidates352_5.length then search edge 3 [2389, 4957] innerCandidates352_5 7346 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan352_6

def innerTail352_4 : List Nat := 4243 :: innerTail352_5
def innerCandidates352_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates352_4_checked : candidates edge 4 2389 4243 innerTail352_5 best.bound = innerCandidates352_4 := by
  unfold candidates
  have row : edge 4243 = SearchData.row4243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan352_4 : scan edge (search edge 3) 4 [2389] 2389 innerTail352_4 best = best := by
  rw [innerTail352_4, scan, if_neg (by decide), innerCandidates352_4_checked]
  change scan edge (search edge 3) 4 [2389] 2389 innerTail352_5
    (if 3 ≤ innerCandidates352_4.length then search edge 3 [2389, 4243] innerCandidates352_4 6632 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan352_5

def innerTail352_3 : List Nat := 3733 :: innerTail352_4
def innerCandidates352_3 : List Nat := [6529]
set_option maxRecDepth 4096 in
theorem innerCandidates352_3_checked : candidates edge 4 2389 3733 innerTail352_4 best.bound = innerCandidates352_3 := by
  unfold candidates
  have row : edge 3733 = SearchData.row3733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan352_3 : scan edge (search edge 3) 4 [2389] 2389 innerTail352_3 best = best := by
  rw [innerTail352_3, scan, if_neg (by decide), innerCandidates352_3_checked]
  change scan edge (search edge 3) 4 [2389] 2389 innerTail352_4
    (if 3 ≤ innerCandidates352_3.length then search edge 3 [2389, 3733] innerCandidates352_3 6122 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan352_4

def innerTail352_2 : List Nat := 2767 :: innerTail352_3
def innerCandidates352_2 : List Nat := [3733, 7477, 7561]
set_option maxRecDepth 4096 in
theorem innerCandidates352_2_checked : candidates edge 4 2389 2767 innerTail352_3 best.bound = innerCandidates352_2 := by
  unfold candidates
  have row : edge 2767 = SearchData.row2767.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild352_2 : search edge 3 [2389, 2767] innerCandidates352_2 5156 best = best := by decide

theorem innerScan352_2 : scan edge (search edge 3) 4 [2389] 2389 innerTail352_2 best = best := by
  rw [innerTail352_2, scan, if_neg (by decide), innerCandidates352_2_checked]
  change scan edge (search edge 3) 4 [2389] 2389 innerTail352_3
    (if 3 ≤ innerCandidates352_2.length then search edge 3 [2389, 2767] innerCandidates352_2 5156 best else best) = best
  rw [if_pos (by decide), innerChild352_2]
  exact innerScan352_3

def innerTail352_1 : List Nat := 2521 :: innerTail352_2
def innerCandidates352_1 : List Nat := [7561]
set_option maxRecDepth 4096 in
theorem innerCandidates352_1_checked : candidates edge 4 2389 2521 innerTail352_2 best.bound = innerCandidates352_1 := by
  unfold candidates
  have row : edge 2521 = PrimePairCache.smallRow2521.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan352_1 : scan edge (search edge 3) 4 [2389] 2389 innerTail352_1 best = best := by
  rw [innerTail352_1, scan, if_neg (by decide), innerCandidates352_1_checked]
  change scan edge (search edge 3) 4 [2389] 2389 innerTail352_2
    (if 3 ≤ innerCandidates352_1.length then search edge 3 [2389, 2521] innerCandidates352_1 4910 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan352_2

def innerTail352_0 : List Nat := 2503 :: innerTail352_1
def innerCandidates352_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates352_0_checked : candidates edge 4 2389 2503 innerTail352_1 best.bound = innerCandidates352_0 := by
  unfold candidates
  have row : edge 2503 = PrimePairCache.smallRow2503.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan352_0 : scan edge (search edge 3) 4 [2389] 2389 innerTail352_0 best = best := by
  rw [innerTail352_0, scan, if_neg (by decide), innerCandidates352_0_checked]
  change scan edge (search edge 3) 4 [2389] 2389 innerTail352_1
    (if 3 ≤ innerCandidates352_0.length then search edge 3 [2389, 2503] innerCandidates352_0 4892 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan352_1

theorem rootBranch352 : search edge 4 [2389] rootCandidates352 2389 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2389] 2389 innerTail352_0 best = best
  exact innerScan352_0

def rootCandidates351 : List Nat := [2659, 3583, 4021, 4099, 4273, 4639, 5023, 5113, 5431, 5527, 6733, 7129, 7573, 7873, 8263]
set_option maxRecDepth 4096 in
theorem rootCandidates351_checked : candidates edge 5 0 2383 tail352 best.bound = rootCandidates351 := by
  unfold candidates
  have row : edge 2383 = PrimePairCache.smallRow2383.contains := by funext q; rfl
  rw [row]
  decide

def innerTail351_15 : List Nat := []
theorem innerScan351_15 : scan edge (search edge 3) 4 [2383] 2383 innerTail351_15 best = best := rfl

def innerTail351_14 : List Nat := 8263 :: innerTail351_15
def innerCandidates351_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates351_14_checked : candidates edge 4 2383 8263 innerTail351_15 best.bound = innerCandidates351_14 := by
  unfold candidates
  have row : edge 8263 = SearchData.row8263.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan351_14 : scan edge (search edge 3) 4 [2383] 2383 innerTail351_14 best = best := by
  rw [innerTail351_14, scan, if_neg (by decide), innerCandidates351_14_checked]
  change scan edge (search edge 3) 4 [2383] 2383 innerTail351_15
    (if 3 ≤ innerCandidates351_14.length then search edge 3 [2383, 8263] innerCandidates351_14 10646 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan351_15

def innerTail351_13 : List Nat := 7873 :: innerTail351_14
def innerCandidates351_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates351_13_checked : candidates edge 4 2383 7873 innerTail351_14 best.bound = innerCandidates351_13 := by
  unfold candidates
  have row : edge 7873 = SearchData.row7873.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan351_13 : scan edge (search edge 3) 4 [2383] 2383 innerTail351_13 best = best := by
  rw [innerTail351_13, scan, if_neg (by decide), innerCandidates351_13_checked]
  change scan edge (search edge 3) 4 [2383] 2383 innerTail351_14
    (if 3 ≤ innerCandidates351_13.length then search edge 3 [2383, 7873] innerCandidates351_13 10256 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan351_14

def innerTail351_12 : List Nat := 7573 :: innerTail351_13
def innerCandidates351_12 : List Nat := [8263]
set_option maxRecDepth 4096 in
theorem innerCandidates351_12_checked : candidates edge 4 2383 7573 innerTail351_13 best.bound = innerCandidates351_12 := by
  unfold candidates
  have row : edge 7573 = SearchData.row7573.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan351_12 : scan edge (search edge 3) 4 [2383] 2383 innerTail351_12 best = best := by
  rw [innerTail351_12, scan, if_neg (by decide), innerCandidates351_12_checked]
  change scan edge (search edge 3) 4 [2383] 2383 innerTail351_13
    (if 3 ≤ innerCandidates351_12.length then search edge 3 [2383, 7573] innerCandidates351_12 9956 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan351_13

def innerTail351_11 : List Nat := 7129 :: innerTail351_12
def innerCandidates351_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates351_11_checked : candidates edge 4 2383 7129 innerTail351_12 best.bound = innerCandidates351_11 := by
  unfold candidates
  have row : edge 7129 = SearchData.row7129.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan351_11 : scan edge (search edge 3) 4 [2383] 2383 innerTail351_11 best = best := by
  rw [innerTail351_11, scan, if_neg (by decide), innerCandidates351_11_checked]
  change scan edge (search edge 3) 4 [2383] 2383 innerTail351_12
    (if 3 ≤ innerCandidates351_11.length then search edge 3 [2383, 7129] innerCandidates351_11 9512 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan351_12

def innerTail351_10 : List Nat := 6733 :: innerTail351_11
def innerCandidates351_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates351_10_checked : candidates edge 4 2383 6733 innerTail351_11 best.bound = innerCandidates351_10 := by
  unfold candidates
  have row : edge 6733 = SearchData.row6733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan351_10 : scan edge (search edge 3) 4 [2383] 2383 innerTail351_10 best = best := by
  rw [innerTail351_10, scan, if_neg (by decide), innerCandidates351_10_checked]
  change scan edge (search edge 3) 4 [2383] 2383 innerTail351_11
    (if 3 ≤ innerCandidates351_10.length then search edge 3 [2383, 6733] innerCandidates351_10 9116 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan351_11

def innerTail351_9 : List Nat := 5527 :: innerTail351_10
def innerCandidates351_9 : List Nat := [7129]
set_option maxRecDepth 4096 in
theorem innerCandidates351_9_checked : candidates edge 4 2383 5527 innerTail351_10 best.bound = innerCandidates351_9 := by
  unfold candidates
  have row : edge 5527 = SearchData.row5527.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan351_9 : scan edge (search edge 3) 4 [2383] 2383 innerTail351_9 best = best := by
  rw [innerTail351_9, scan, if_neg (by decide), innerCandidates351_9_checked]
  change scan edge (search edge 3) 4 [2383] 2383 innerTail351_10
    (if 3 ≤ innerCandidates351_9.length then search edge 3 [2383, 5527] innerCandidates351_9 7910 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan351_10

def innerTail351_8 : List Nat := 5431 :: innerTail351_9
def innerCandidates351_8 : List Nat := [7573]
set_option maxRecDepth 4096 in
theorem innerCandidates351_8_checked : candidates edge 4 2383 5431 innerTail351_9 best.bound = innerCandidates351_8 := by
  unfold candidates
  have row : edge 5431 = SearchData.row5431.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan351_8 : scan edge (search edge 3) 4 [2383] 2383 innerTail351_8 best = best := by
  rw [innerTail351_8, scan, if_neg (by decide), innerCandidates351_8_checked]
  change scan edge (search edge 3) 4 [2383] 2383 innerTail351_9
    (if 3 ≤ innerCandidates351_8.length then search edge 3 [2383, 5431] innerCandidates351_8 7814 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan351_9

def innerTail351_7 : List Nat := 5113 :: innerTail351_8
def innerCandidates351_7 : List Nat := [5431]
set_option maxRecDepth 4096 in
theorem innerCandidates351_7_checked : candidates edge 4 2383 5113 innerTail351_8 best.bound = innerCandidates351_7 := by
  unfold candidates
  have row : edge 5113 = SearchData.row5113.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan351_7 : scan edge (search edge 3) 4 [2383] 2383 innerTail351_7 best = best := by
  rw [innerTail351_7, scan, if_neg (by decide), innerCandidates351_7_checked]
  change scan edge (search edge 3) 4 [2383] 2383 innerTail351_8
    (if 3 ≤ innerCandidates351_7.length then search edge 3 [2383, 5113] innerCandidates351_7 7496 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan351_8

def innerTail351_6 : List Nat := 5023 :: innerTail351_7
def innerCandidates351_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates351_6_checked : candidates edge 4 2383 5023 innerTail351_7 best.bound = innerCandidates351_6 := by
  unfold candidates
  have row : edge 5023 = SearchData.row5023.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan351_6 : scan edge (search edge 3) 4 [2383] 2383 innerTail351_6 best = best := by
  rw [innerTail351_6, scan, if_neg (by decide), innerCandidates351_6_checked]
  change scan edge (search edge 3) 4 [2383] 2383 innerTail351_7
    (if 3 ≤ innerCandidates351_6.length then search edge 3 [2383, 5023] innerCandidates351_6 7406 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan351_7

def innerTail351_5 : List Nat := 4639 :: innerTail351_6
def innerCandidates351_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates351_5_checked : candidates edge 4 2383 4639 innerTail351_6 best.bound = innerCandidates351_5 := by
  unfold candidates
  have row : edge 4639 = SearchData.row4639.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan351_5 : scan edge (search edge 3) 4 [2383] 2383 innerTail351_5 best = best := by
  rw [innerTail351_5, scan, if_neg (by decide), innerCandidates351_5_checked]
  change scan edge (search edge 3) 4 [2383] 2383 innerTail351_6
    (if 3 ≤ innerCandidates351_5.length then search edge 3 [2383, 4639] innerCandidates351_5 7022 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan351_6

def innerTail351_4 : List Nat := 4273 :: innerTail351_5
def innerCandidates351_4 : List Nat := [5023, 7573]
set_option maxRecDepth 4096 in
theorem innerCandidates351_4_checked : candidates edge 4 2383 4273 innerTail351_5 best.bound = innerCandidates351_4 := by
  unfold candidates
  have row : edge 4273 = SearchData.row4273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan351_4 : scan edge (search edge 3) 4 [2383] 2383 innerTail351_4 best = best := by
  rw [innerTail351_4, scan, if_neg (by decide), innerCandidates351_4_checked]
  change scan edge (search edge 3) 4 [2383] 2383 innerTail351_5
    (if 3 ≤ innerCandidates351_4.length then search edge 3 [2383, 4273] innerCandidates351_4 6656 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan351_5

def innerTail351_3 : List Nat := 4099 :: innerTail351_4
def innerCandidates351_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates351_3_checked : candidates edge 4 2383 4099 innerTail351_4 best.bound = innerCandidates351_3 := by
  unfold candidates
  have row : edge 4099 = SearchData.row4099.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan351_3 : scan edge (search edge 3) 4 [2383] 2383 innerTail351_3 best = best := by
  rw [innerTail351_3, scan, if_neg (by decide), innerCandidates351_3_checked]
  change scan edge (search edge 3) 4 [2383] 2383 innerTail351_4
    (if 3 ≤ innerCandidates351_3.length then search edge 3 [2383, 4099] innerCandidates351_3 6482 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan351_4

def innerTail351_2 : List Nat := 4021 :: innerTail351_3
def innerCandidates351_2 : List Nat := [6733]
set_option maxRecDepth 4096 in
theorem innerCandidates351_2_checked : candidates edge 4 2383 4021 innerTail351_3 best.bound = innerCandidates351_2 := by
  unfold candidates
  have row : edge 4021 = SearchData.row4021.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan351_2 : scan edge (search edge 3) 4 [2383] 2383 innerTail351_2 best = best := by
  rw [innerTail351_2, scan, if_neg (by decide), innerCandidates351_2_checked]
  change scan edge (search edge 3) 4 [2383] 2383 innerTail351_3
    (if 3 ≤ innerCandidates351_2.length then search edge 3 [2383, 4021] innerCandidates351_2 6404 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan351_3

def innerTail351_1 : List Nat := 3583 :: innerTail351_2
def innerCandidates351_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates351_1_checked : candidates edge 4 2383 3583 innerTail351_2 best.bound = innerCandidates351_1 := by
  unfold candidates
  have row : edge 3583 = SearchData.row3583.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan351_1 : scan edge (search edge 3) 4 [2383] 2383 innerTail351_1 best = best := by
  rw [innerTail351_1, scan, if_neg (by decide), innerCandidates351_1_checked]
  change scan edge (search edge 3) 4 [2383] 2383 innerTail351_2
    (if 3 ≤ innerCandidates351_1.length then search edge 3 [2383, 3583] innerCandidates351_1 5966 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan351_2

def innerTail351_0 : List Nat := 2659 :: innerTail351_1
def innerCandidates351_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates351_0_checked : candidates edge 4 2383 2659 innerTail351_1 best.bound = innerCandidates351_0 := by
  unfold candidates
  have row : edge 2659 = PrimePairCache.smallRow2659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan351_0 : scan edge (search edge 3) 4 [2383] 2383 innerTail351_0 best = best := by
  rw [innerTail351_0, scan, if_neg (by decide), innerCandidates351_0_checked]
  change scan edge (search edge 3) 4 [2383] 2383 innerTail351_1
    (if 3 ≤ innerCandidates351_0.length then search edge 3 [2383, 2659] innerCandidates351_0 5042 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan351_1

theorem rootBranch351 : search edge 4 [2383] rootCandidates351 2383 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2383] 2383 innerTail351_0 best = best
  exact innerScan351_0

def rootCandidates350 : List Nat := [2447, 3221, 3539, 4073, 5483, 5501, 5639, 5807, 6563, 6857, 7487, 7853, 8117, 8369]
set_option maxRecDepth 4096 in
theorem rootCandidates350_checked : candidates edge 5 0 2381 tail351 best.bound = rootCandidates350 := by
  unfold candidates
  have row : edge 2381 = PrimePairCache.smallRow2381.contains := by funext q; rfl
  rw [row]
  decide

def innerTail350_14 : List Nat := []
theorem innerScan350_14 : scan edge (search edge 3) 4 [2381] 2381 innerTail350_14 best = best := rfl

def innerTail350_13 : List Nat := 8369 :: innerTail350_14
def innerCandidates350_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates350_13_checked : candidates edge 4 2381 8369 innerTail350_14 best.bound = innerCandidates350_13 := by
  unfold candidates
  have row : edge 8369 = SearchData.row8369.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan350_13 : scan edge (search edge 3) 4 [2381] 2381 innerTail350_13 best = best := by
  rw [innerTail350_13, scan, if_neg (by decide), innerCandidates350_13_checked]
  change scan edge (search edge 3) 4 [2381] 2381 innerTail350_14
    (if 3 ≤ innerCandidates350_13.length then search edge 3 [2381, 8369] innerCandidates350_13 10750 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan350_14

def innerTail350_12 : List Nat := 8117 :: innerTail350_13
def innerCandidates350_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates350_12_checked : candidates edge 4 2381 8117 innerTail350_13 best.bound = innerCandidates350_12 := by
  unfold candidates
  have row : edge 8117 = SearchData.row8117.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan350_12 : scan edge (search edge 3) 4 [2381] 2381 innerTail350_12 best = best := by
  rw [innerTail350_12, scan, if_neg (by decide), innerCandidates350_12_checked]
  change scan edge (search edge 3) 4 [2381] 2381 innerTail350_13
    (if 3 ≤ innerCandidates350_12.length then search edge 3 [2381, 8117] innerCandidates350_12 10498 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan350_13

def innerTail350_11 : List Nat := 7853 :: innerTail350_12
def innerCandidates350_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates350_11_checked : candidates edge 4 2381 7853 innerTail350_12 best.bound = innerCandidates350_11 := by
  unfold candidates
  have row : edge 7853 = SearchData.row7853.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan350_11 : scan edge (search edge 3) 4 [2381] 2381 innerTail350_11 best = best := by
  rw [innerTail350_11, scan, if_neg (by decide), innerCandidates350_11_checked]
  change scan edge (search edge 3) 4 [2381] 2381 innerTail350_12
    (if 3 ≤ innerCandidates350_11.length then search edge 3 [2381, 7853] innerCandidates350_11 10234 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan350_12

def innerTail350_10 : List Nat := 7487 :: innerTail350_11
def innerCandidates350_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates350_10_checked : candidates edge 4 2381 7487 innerTail350_11 best.bound = innerCandidates350_10 := by
  unfold candidates
  have row : edge 7487 = SearchData.row7487.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan350_10 : scan edge (search edge 3) 4 [2381] 2381 innerTail350_10 best = best := by
  rw [innerTail350_10, scan, if_neg (by decide), innerCandidates350_10_checked]
  change scan edge (search edge 3) 4 [2381] 2381 innerTail350_11
    (if 3 ≤ innerCandidates350_10.length then search edge 3 [2381, 7487] innerCandidates350_10 9868 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan350_11

def innerTail350_9 : List Nat := 6857 :: innerTail350_10
def innerCandidates350_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates350_9_checked : candidates edge 4 2381 6857 innerTail350_10 best.bound = innerCandidates350_9 := by
  unfold candidates
  have row : edge 6857 = SearchData.row6857.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan350_9 : scan edge (search edge 3) 4 [2381] 2381 innerTail350_9 best = best := by
  rw [innerTail350_9, scan, if_neg (by decide), innerCandidates350_9_checked]
  change scan edge (search edge 3) 4 [2381] 2381 innerTail350_10
    (if 3 ≤ innerCandidates350_9.length then search edge 3 [2381, 6857] innerCandidates350_9 9238 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan350_10

def innerTail350_8 : List Nat := 6563 :: innerTail350_9
def innerCandidates350_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates350_8_checked : candidates edge 4 2381 6563 innerTail350_9 best.bound = innerCandidates350_8 := by
  unfold candidates
  have row : edge 6563 = SearchData.row6563.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan350_8 : scan edge (search edge 3) 4 [2381] 2381 innerTail350_8 best = best := by
  rw [innerTail350_8, scan, if_neg (by decide), innerCandidates350_8_checked]
  change scan edge (search edge 3) 4 [2381] 2381 innerTail350_9
    (if 3 ≤ innerCandidates350_8.length then search edge 3 [2381, 6563] innerCandidates350_8 8944 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan350_9

def innerTail350_7 : List Nat := 5807 :: innerTail350_8
def innerCandidates350_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates350_7_checked : candidates edge 4 2381 5807 innerTail350_8 best.bound = innerCandidates350_7 := by
  unfold candidates
  have row : edge 5807 = SearchData.row5807.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan350_7 : scan edge (search edge 3) 4 [2381] 2381 innerTail350_7 best = best := by
  rw [innerTail350_7, scan, if_neg (by decide), innerCandidates350_7_checked]
  change scan edge (search edge 3) 4 [2381] 2381 innerTail350_8
    (if 3 ≤ innerCandidates350_7.length then search edge 3 [2381, 5807] innerCandidates350_7 8188 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan350_8

def innerTail350_6 : List Nat := 5639 :: innerTail350_7
def innerCandidates350_6 : List Nat := [6563]
set_option maxRecDepth 4096 in
theorem innerCandidates350_6_checked : candidates edge 4 2381 5639 innerTail350_7 best.bound = innerCandidates350_6 := by
  unfold candidates
  have row : edge 5639 = SearchData.row5639.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan350_6 : scan edge (search edge 3) 4 [2381] 2381 innerTail350_6 best = best := by
  rw [innerTail350_6, scan, if_neg (by decide), innerCandidates350_6_checked]
  change scan edge (search edge 3) 4 [2381] 2381 innerTail350_7
    (if 3 ≤ innerCandidates350_6.length then search edge 3 [2381, 5639] innerCandidates350_6 8020 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan350_7

def innerTail350_5 : List Nat := 5501 :: innerTail350_6
def innerCandidates350_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates350_5_checked : candidates edge 4 2381 5501 innerTail350_6 best.bound = innerCandidates350_5 := by
  unfold candidates
  have row : edge 5501 = SearchData.row5501.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan350_5 : scan edge (search edge 3) 4 [2381] 2381 innerTail350_5 best = best := by
  rw [innerTail350_5, scan, if_neg (by decide), innerCandidates350_5_checked]
  change scan edge (search edge 3) 4 [2381] 2381 innerTail350_6
    (if 3 ≤ innerCandidates350_5.length then search edge 3 [2381, 5501] innerCandidates350_5 7882 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan350_6

def innerTail350_4 : List Nat := 5483 :: innerTail350_5
def innerCandidates350_4 : List Nat := [7487, 8369]
set_option maxRecDepth 4096 in
theorem innerCandidates350_4_checked : candidates edge 4 2381 5483 innerTail350_5 best.bound = innerCandidates350_4 := by
  unfold candidates
  have row : edge 5483 = SearchData.row5483.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan350_4 : scan edge (search edge 3) 4 [2381] 2381 innerTail350_4 best = best := by
  rw [innerTail350_4, scan, if_neg (by decide), innerCandidates350_4_checked]
  change scan edge (search edge 3) 4 [2381] 2381 innerTail350_5
    (if 3 ≤ innerCandidates350_4.length then search edge 3 [2381, 5483] innerCandidates350_4 7864 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan350_5

def innerTail350_3 : List Nat := 4073 :: innerTail350_4
def innerCandidates350_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates350_3_checked : candidates edge 4 2381 4073 innerTail350_4 best.bound = innerCandidates350_3 := by
  unfold candidates
  have row : edge 4073 = SearchData.row4073.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan350_3 : scan edge (search edge 3) 4 [2381] 2381 innerTail350_3 best = best := by
  rw [innerTail350_3, scan, if_neg (by decide), innerCandidates350_3_checked]
  change scan edge (search edge 3) 4 [2381] 2381 innerTail350_4
    (if 3 ≤ innerCandidates350_3.length then search edge 3 [2381, 4073] innerCandidates350_3 6454 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan350_4

def innerTail350_2 : List Nat := 3539 :: innerTail350_3
def innerCandidates350_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates350_2_checked : candidates edge 4 2381 3539 innerTail350_3 best.bound = innerCandidates350_2 := by
  unfold candidates
  have row : edge 3539 = SearchData.row3539.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan350_2 : scan edge (search edge 3) 4 [2381] 2381 innerTail350_2 best = best := by
  rw [innerTail350_2, scan, if_neg (by decide), innerCandidates350_2_checked]
  change scan edge (search edge 3) 4 [2381] 2381 innerTail350_3
    (if 3 ≤ innerCandidates350_2.length then search edge 3 [2381, 3539] innerCandidates350_2 5920 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan350_3

def innerTail350_1 : List Nat := 3221 :: innerTail350_2
def innerCandidates350_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates350_1_checked : candidates edge 4 2381 3221 innerTail350_2 best.bound = innerCandidates350_1 := by
  unfold candidates
  have row : edge 3221 = SearchData.row3221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan350_1 : scan edge (search edge 3) 4 [2381] 2381 innerTail350_1 best = best := by
  rw [innerTail350_1, scan, if_neg (by decide), innerCandidates350_1_checked]
  change scan edge (search edge 3) 4 [2381] 2381 innerTail350_2
    (if 3 ≤ innerCandidates350_1.length then search edge 3 [2381, 3221] innerCandidates350_1 5602 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan350_2

def innerTail350_0 : List Nat := 2447 :: innerTail350_1
def innerCandidates350_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates350_0_checked : candidates edge 4 2381 2447 innerTail350_1 best.bound = innerCandidates350_0 := by
  unfold candidates
  have row : edge 2447 = PrimePairCache.smallRow2447.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan350_0 : scan edge (search edge 3) 4 [2381] 2381 innerTail350_0 best = best := by
  rw [innerTail350_0, scan, if_neg (by decide), innerCandidates350_0_checked]
  change scan edge (search edge 3) 4 [2381] 2381 innerTail350_1
    (if 3 ≤ innerCandidates350_0.length then search edge 3 [2381, 2447] innerCandidates350_0 4828 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan350_1

theorem rootBranch350 : search edge 4 [2381] rootCandidates350 2381 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2381] 2381 innerTail350_0 best = best
  exact innerScan350_0

end ProjectEulerSolutions.PrimePairUpper.Search
