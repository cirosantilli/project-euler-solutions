import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates90 : List Nat := [601, 727, 757, 769, 1093, 1297, 1567, 1723, 2311, 3163, 3433, 3559, 3697, 4339, 5113, 5407, 5881, 6133, 6301, 6607, 6709, 8233]
set_option maxRecDepth 4096 in
theorem rootCandidates90_checked : candidates edge 5 0 487 tail91 best.bound = rootCandidates90 := by
  unfold candidates
  have row : edge 487 = PrimePairCache.smallRow487.contains := by funext q; rfl
  rw [row]
  decide

def innerTail90_22 : List Nat := []
theorem innerScan90_22 : scan edge (search edge 3) 4 [487] 487 innerTail90_22 best = best := rfl

def innerTail90_21 : List Nat := 8233 :: innerTail90_22
def innerCandidates90_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates90_21_checked : candidates edge 4 487 8233 innerTail90_22 best.bound = innerCandidates90_21 := by
  unfold candidates
  have row : edge 8233 = SearchData.row8233.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan90_21 : scan edge (search edge 3) 4 [487] 487 innerTail90_21 best = best := by
  rw [innerTail90_21, scan, if_neg (by decide), innerCandidates90_21_checked]
  change scan edge (search edge 3) 4 [487] 487 innerTail90_22
    (if 3 ≤ innerCandidates90_21.length then search edge 3 [487, 8233] innerCandidates90_21 8720 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan90_22

def innerTail90_20 : List Nat := 6709 :: innerTail90_21
def innerCandidates90_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates90_20_checked : candidates edge 4 487 6709 innerTail90_21 best.bound = innerCandidates90_20 := by
  unfold candidates
  have row : edge 6709 = SearchData.row6709.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan90_20 : scan edge (search edge 3) 4 [487] 487 innerTail90_20 best = best := by
  rw [innerTail90_20, scan, if_neg (by decide), innerCandidates90_20_checked]
  change scan edge (search edge 3) 4 [487] 487 innerTail90_21
    (if 3 ≤ innerCandidates90_20.length then search edge 3 [487, 6709] innerCandidates90_20 7196 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan90_21

def innerTail90_19 : List Nat := 6607 :: innerTail90_20
def innerCandidates90_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates90_19_checked : candidates edge 4 487 6607 innerTail90_20 best.bound = innerCandidates90_19 := by
  unfold candidates
  have row : edge 6607 = SearchData.row6607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan90_19 : scan edge (search edge 3) 4 [487] 487 innerTail90_19 best = best := by
  rw [innerTail90_19, scan, if_neg (by decide), innerCandidates90_19_checked]
  change scan edge (search edge 3) 4 [487] 487 innerTail90_20
    (if 3 ≤ innerCandidates90_19.length then search edge 3 [487, 6607] innerCandidates90_19 7094 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan90_20

def innerTail90_18 : List Nat := 6301 :: innerTail90_19
def innerCandidates90_18 : List Nat := [6607, 8233]
set_option maxRecDepth 4096 in
theorem innerCandidates90_18_checked : candidates edge 4 487 6301 innerTail90_19 best.bound = innerCandidates90_18 := by
  unfold candidates
  have row : edge 6301 = SearchData.row6301.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan90_18 : scan edge (search edge 3) 4 [487] 487 innerTail90_18 best = best := by
  rw [innerTail90_18, scan, if_neg (by decide), innerCandidates90_18_checked]
  change scan edge (search edge 3) 4 [487] 487 innerTail90_19
    (if 3 ≤ innerCandidates90_18.length then search edge 3 [487, 6301] innerCandidates90_18 6788 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan90_19

def innerTail90_17 : List Nat := 6133 :: innerTail90_18
def innerCandidates90_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates90_17_checked : candidates edge 4 487 6133 innerTail90_18 best.bound = innerCandidates90_17 := by
  unfold candidates
  have row : edge 6133 = SearchData.row6133.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan90_17 : scan edge (search edge 3) 4 [487] 487 innerTail90_17 best = best := by
  rw [innerTail90_17, scan, if_neg (by decide), innerCandidates90_17_checked]
  change scan edge (search edge 3) 4 [487] 487 innerTail90_18
    (if 3 ≤ innerCandidates90_17.length then search edge 3 [487, 6133] innerCandidates90_17 6620 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan90_18

def innerTail90_16 : List Nat := 5881 :: innerTail90_17
def innerCandidates90_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates90_16_checked : candidates edge 4 487 5881 innerTail90_17 best.bound = innerCandidates90_16 := by
  unfold candidates
  have row : edge 5881 = SearchData.row5881.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan90_16 : scan edge (search edge 3) 4 [487] 487 innerTail90_16 best = best := by
  rw [innerTail90_16, scan, if_neg (by decide), innerCandidates90_16_checked]
  change scan edge (search edge 3) 4 [487] 487 innerTail90_17
    (if 3 ≤ innerCandidates90_16.length then search edge 3 [487, 5881] innerCandidates90_16 6368 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan90_17

def innerTail90_15 : List Nat := 5407 :: innerTail90_16
def innerCandidates90_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates90_15_checked : candidates edge 4 487 5407 innerTail90_16 best.bound = innerCandidates90_15 := by
  unfold candidates
  have row : edge 5407 = SearchData.row5407.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan90_15 : scan edge (search edge 3) 4 [487] 487 innerTail90_15 best = best := by
  rw [innerTail90_15, scan, if_neg (by decide), innerCandidates90_15_checked]
  change scan edge (search edge 3) 4 [487] 487 innerTail90_16
    (if 3 ≤ innerCandidates90_15.length then search edge 3 [487, 5407] innerCandidates90_15 5894 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan90_16

def innerTail90_14 : List Nat := 5113 :: innerTail90_15
def innerCandidates90_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates90_14_checked : candidates edge 4 487 5113 innerTail90_15 best.bound = innerCandidates90_14 := by
  unfold candidates
  have row : edge 5113 = SearchData.row5113.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan90_14 : scan edge (search edge 3) 4 [487] 487 innerTail90_14 best = best := by
  rw [innerTail90_14, scan, if_neg (by decide), innerCandidates90_14_checked]
  change scan edge (search edge 3) 4 [487] 487 innerTail90_15
    (if 3 ≤ innerCandidates90_14.length then search edge 3 [487, 5113] innerCandidates90_14 5600 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan90_15

def innerTail90_13 : List Nat := 4339 :: innerTail90_14
def innerCandidates90_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates90_13_checked : candidates edge 4 487 4339 innerTail90_14 best.bound = innerCandidates90_13 := by
  unfold candidates
  have row : edge 4339 = SearchData.row4339.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan90_13 : scan edge (search edge 3) 4 [487] 487 innerTail90_13 best = best := by
  rw [innerTail90_13, scan, if_neg (by decide), innerCandidates90_13_checked]
  change scan edge (search edge 3) 4 [487] 487 innerTail90_14
    (if 3 ≤ innerCandidates90_13.length then search edge 3 [487, 4339] innerCandidates90_13 4826 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan90_14

def innerTail90_12 : List Nat := 3697 :: innerTail90_13
def innerCandidates90_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates90_12_checked : candidates edge 4 487 3697 innerTail90_13 best.bound = innerCandidates90_12 := by
  unfold candidates
  have row : edge 3697 = SearchData.row3697.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan90_12 : scan edge (search edge 3) 4 [487] 487 innerTail90_12 best = best := by
  rw [innerTail90_12, scan, if_neg (by decide), innerCandidates90_12_checked]
  change scan edge (search edge 3) 4 [487] 487 innerTail90_13
    (if 3 ≤ innerCandidates90_12.length then search edge 3 [487, 3697] innerCandidates90_12 4184 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan90_13

def innerTail90_11 : List Nat := 3559 :: innerTail90_12
def innerCandidates90_11 : List Nat := [6607]
set_option maxRecDepth 4096 in
theorem innerCandidates90_11_checked : candidates edge 4 487 3559 innerTail90_12 best.bound = innerCandidates90_11 := by
  unfold candidates
  have row : edge 3559 = SearchData.row3559.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan90_11 : scan edge (search edge 3) 4 [487] 487 innerTail90_11 best = best := by
  rw [innerTail90_11, scan, if_neg (by decide), innerCandidates90_11_checked]
  change scan edge (search edge 3) 4 [487] 487 innerTail90_12
    (if 3 ≤ innerCandidates90_11.length then search edge 3 [487, 3559] innerCandidates90_11 4046 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan90_12

def innerTail90_10 : List Nat := 3433 :: innerTail90_11
def innerCandidates90_10 : List Nat := [5407, 6607]
set_option maxRecDepth 4096 in
theorem innerCandidates90_10_checked : candidates edge 4 487 3433 innerTail90_11 best.bound = innerCandidates90_10 := by
  unfold candidates
  have row : edge 3433 = SearchData.row3433.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan90_10 : scan edge (search edge 3) 4 [487] 487 innerTail90_10 best = best := by
  rw [innerTail90_10, scan, if_neg (by decide), innerCandidates90_10_checked]
  change scan edge (search edge 3) 4 [487] 487 innerTail90_11
    (if 3 ≤ innerCandidates90_10.length then search edge 3 [487, 3433] innerCandidates90_10 3920 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan90_11

def innerTail90_9 : List Nat := 3163 :: innerTail90_10
def innerCandidates90_9 : List Nat := [6301]
set_option maxRecDepth 4096 in
theorem innerCandidates90_9_checked : candidates edge 4 487 3163 innerTail90_10 best.bound = innerCandidates90_9 := by
  unfold candidates
  have row : edge 3163 = SearchData.row3163.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan90_9 : scan edge (search edge 3) 4 [487] 487 innerTail90_9 best = best := by
  rw [innerTail90_9, scan, if_neg (by decide), innerCandidates90_9_checked]
  change scan edge (search edge 3) 4 [487] 487 innerTail90_10
    (if 3 ≤ innerCandidates90_9.length then search edge 3 [487, 3163] innerCandidates90_9 3650 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan90_10

def innerTail90_8 : List Nat := 2311 :: innerTail90_9
def innerCandidates90_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates90_8_checked : candidates edge 4 487 2311 innerTail90_9 best.bound = innerCandidates90_8 := by
  unfold candidates
  have row : edge 2311 = PrimePairCache.smallRow2311.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan90_8 : scan edge (search edge 3) 4 [487] 487 innerTail90_8 best = best := by
  rw [innerTail90_8, scan, if_neg (by decide), innerCandidates90_8_checked]
  change scan edge (search edge 3) 4 [487] 487 innerTail90_9
    (if 3 ≤ innerCandidates90_8.length then search edge 3 [487, 2311] innerCandidates90_8 2798 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan90_9

def innerTail90_7 : List Nat := 1723 :: innerTail90_8
def innerCandidates90_7 : List Nat := [8233]
set_option maxRecDepth 4096 in
theorem innerCandidates90_7_checked : candidates edge 4 487 1723 innerTail90_8 best.bound = innerCandidates90_7 := by
  unfold candidates
  have row : edge 1723 = PrimePairCache.smallRow1723.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan90_7 : scan edge (search edge 3) 4 [487] 487 innerTail90_7 best = best := by
  rw [innerTail90_7, scan, if_neg (by decide), innerCandidates90_7_checked]
  change scan edge (search edge 3) 4 [487] 487 innerTail90_8
    (if 3 ≤ innerCandidates90_7.length then search edge 3 [487, 1723] innerCandidates90_7 2210 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan90_8

def innerTail90_6 : List Nat := 1567 :: innerTail90_7
def innerCandidates90_6 : List Nat := [2311]
set_option maxRecDepth 4096 in
theorem innerCandidates90_6_checked : candidates edge 4 487 1567 innerTail90_7 best.bound = innerCandidates90_6 := by
  unfold candidates
  have row : edge 1567 = PrimePairCache.smallRow1567.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan90_6 : scan edge (search edge 3) 4 [487] 487 innerTail90_6 best = best := by
  rw [innerTail90_6, scan, if_neg (by decide), innerCandidates90_6_checked]
  change scan edge (search edge 3) 4 [487] 487 innerTail90_7
    (if 3 ≤ innerCandidates90_6.length then search edge 3 [487, 1567] innerCandidates90_6 2054 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan90_7

def innerTail90_5 : List Nat := 1297 :: innerTail90_6
def innerCandidates90_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates90_5_checked : candidates edge 4 487 1297 innerTail90_6 best.bound = innerCandidates90_5 := by
  unfold candidates
  have row : edge 1297 = PrimePairCache.smallRow1297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan90_5 : scan edge (search edge 3) 4 [487] 487 innerTail90_5 best = best := by
  rw [innerTail90_5, scan, if_neg (by decide), innerCandidates90_5_checked]
  change scan edge (search edge 3) 4 [487] 487 innerTail90_6
    (if 3 ≤ innerCandidates90_5.length then search edge 3 [487, 1297] innerCandidates90_5 1784 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan90_6

def innerTail90_4 : List Nat := 1093 :: innerTail90_5
def innerCandidates90_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates90_4_checked : candidates edge 4 487 1093 innerTail90_5 best.bound = innerCandidates90_4 := by
  unfold candidates
  have row : edge 1093 = PrimePairCache.smallRow1093.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan90_4 : scan edge (search edge 3) 4 [487] 487 innerTail90_4 best = best := by
  rw [innerTail90_4, scan, if_neg (by decide), innerCandidates90_4_checked]
  change scan edge (search edge 3) 4 [487] 487 innerTail90_5
    (if 3 ≤ innerCandidates90_4.length then search edge 3 [487, 1093] innerCandidates90_4 1580 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan90_5

def innerTail90_3 : List Nat := 769 :: innerTail90_4
def innerCandidates90_3 : List Nat := [5113, 6133]
set_option maxRecDepth 4096 in
theorem innerCandidates90_3_checked : candidates edge 4 487 769 innerTail90_4 best.bound = innerCandidates90_3 := by
  unfold candidates
  have row : edge 769 = PrimePairCache.smallRow769.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan90_3 : scan edge (search edge 3) 4 [487] 487 innerTail90_3 best = best := by
  rw [innerTail90_3, scan, if_neg (by decide), innerCandidates90_3_checked]
  change scan edge (search edge 3) 4 [487] 487 innerTail90_4
    (if 3 ≤ innerCandidates90_3.length then search edge 3 [487, 769] innerCandidates90_3 1256 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan90_4

def innerTail90_2 : List Nat := 757 :: innerTail90_3
def innerCandidates90_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates90_2_checked : candidates edge 4 487 757 innerTail90_3 best.bound = innerCandidates90_2 := by
  unfold candidates
  have row : edge 757 = PrimePairCache.smallRow757.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan90_2 : scan edge (search edge 3) 4 [487] 487 innerTail90_2 best = best := by
  rw [innerTail90_2, scan, if_neg (by decide), innerCandidates90_2_checked]
  change scan edge (search edge 3) 4 [487] 487 innerTail90_3
    (if 3 ≤ innerCandidates90_2.length then search edge 3 [487, 757] innerCandidates90_2 1244 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan90_3

def innerTail90_1 : List Nat := 727 :: innerTail90_2
def innerCandidates90_1 : List Nat := [1723]
set_option maxRecDepth 4096 in
theorem innerCandidates90_1_checked : candidates edge 4 487 727 innerTail90_2 best.bound = innerCandidates90_1 := by
  unfold candidates
  have row : edge 727 = PrimePairCache.smallRow727.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan90_1 : scan edge (search edge 3) 4 [487] 487 innerTail90_1 best = best := by
  rw [innerTail90_1, scan, if_neg (by decide), innerCandidates90_1_checked]
  change scan edge (search edge 3) 4 [487] 487 innerTail90_2
    (if 3 ≤ innerCandidates90_1.length then search edge 3 [487, 727] innerCandidates90_1 1214 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan90_2

def innerTail90_0 : List Nat := 601 :: innerTail90_1
def innerCandidates90_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates90_0_checked : candidates edge 4 487 601 innerTail90_1 best.bound = innerCandidates90_0 := by
  unfold candidates
  have row : edge 601 = PrimePairCache.smallRow601.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan90_0 : scan edge (search edge 3) 4 [487] 487 innerTail90_0 best = best := by
  rw [innerTail90_0, scan, if_neg (by decide), innerCandidates90_0_checked]
  change scan edge (search edge 3) 4 [487] 487 innerTail90_1
    (if 3 ≤ innerCandidates90_0.length then search edge 3 [487, 601] innerCandidates90_0 1088 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan90_1

theorem rootBranch90 : search edge 4 [487] rootCandidates90 487 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [487] 487 innerTail90_0 best = best
  exact innerScan90_0

def rootCandidates89 : List Nat := [569, 593, 599, 701, 821, 881, 971, 1049, 1187, 1307, 1559, 2297, 3533, 4337, 4481, 4733, 4817, 5807, 6203, 7517]
set_option maxRecDepth 4096 in
theorem rootCandidates89_checked : candidates edge 5 0 479 tail90 best.bound = rootCandidates89 := by
  unfold candidates
  have row : edge 479 = PrimePairCache.smallRow479.contains := by funext q; rfl
  rw [row]
  decide

def innerTail89_20 : List Nat := []
theorem innerScan89_20 : scan edge (search edge 3) 4 [479] 479 innerTail89_20 best = best := rfl

def innerTail89_19 : List Nat := 7517 :: innerTail89_20
def innerCandidates89_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates89_19_checked : candidates edge 4 479 7517 innerTail89_20 best.bound = innerCandidates89_19 := by
  unfold candidates
  have row : edge 7517 = SearchData.row7517.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan89_19 : scan edge (search edge 3) 4 [479] 479 innerTail89_19 best = best := by
  rw [innerTail89_19, scan, if_neg (by decide), innerCandidates89_19_checked]
  change scan edge (search edge 3) 4 [479] 479 innerTail89_20
    (if 3 ≤ innerCandidates89_19.length then search edge 3 [479, 7517] innerCandidates89_19 7996 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan89_20

def innerTail89_18 : List Nat := 6203 :: innerTail89_19
def innerCandidates89_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates89_18_checked : candidates edge 4 479 6203 innerTail89_19 best.bound = innerCandidates89_18 := by
  unfold candidates
  have row : edge 6203 = SearchData.row6203.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan89_18 : scan edge (search edge 3) 4 [479] 479 innerTail89_18 best = best := by
  rw [innerTail89_18, scan, if_neg (by decide), innerCandidates89_18_checked]
  change scan edge (search edge 3) 4 [479] 479 innerTail89_19
    (if 3 ≤ innerCandidates89_18.length then search edge 3 [479, 6203] innerCandidates89_18 6682 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan89_19

def innerTail89_17 : List Nat := 5807 :: innerTail89_18
def innerCandidates89_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates89_17_checked : candidates edge 4 479 5807 innerTail89_18 best.bound = innerCandidates89_17 := by
  unfold candidates
  have row : edge 5807 = SearchData.row5807.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan89_17 : scan edge (search edge 3) 4 [479] 479 innerTail89_17 best = best := by
  rw [innerTail89_17, scan, if_neg (by decide), innerCandidates89_17_checked]
  change scan edge (search edge 3) 4 [479] 479 innerTail89_18
    (if 3 ≤ innerCandidates89_17.length then search edge 3 [479, 5807] innerCandidates89_17 6286 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan89_18

def innerTail89_16 : List Nat := 4817 :: innerTail89_17
def innerCandidates89_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates89_16_checked : candidates edge 4 479 4817 innerTail89_17 best.bound = innerCandidates89_16 := by
  unfold candidates
  have row : edge 4817 = SearchData.row4817.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan89_16 : scan edge (search edge 3) 4 [479] 479 innerTail89_16 best = best := by
  rw [innerTail89_16, scan, if_neg (by decide), innerCandidates89_16_checked]
  change scan edge (search edge 3) 4 [479] 479 innerTail89_17
    (if 3 ≤ innerCandidates89_16.length then search edge 3 [479, 4817] innerCandidates89_16 5296 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan89_17

def innerTail89_15 : List Nat := 4733 :: innerTail89_16
def innerCandidates89_15 : List Nat := [7517]
set_option maxRecDepth 4096 in
theorem innerCandidates89_15_checked : candidates edge 4 479 4733 innerTail89_16 best.bound = innerCandidates89_15 := by
  unfold candidates
  have row : edge 4733 = SearchData.row4733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan89_15 : scan edge (search edge 3) 4 [479] 479 innerTail89_15 best = best := by
  rw [innerTail89_15, scan, if_neg (by decide), innerCandidates89_15_checked]
  change scan edge (search edge 3) 4 [479] 479 innerTail89_16
    (if 3 ≤ innerCandidates89_15.length then search edge 3 [479, 4733] innerCandidates89_15 5212 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan89_16

def innerTail89_14 : List Nat := 4481 :: innerTail89_15
def innerCandidates89_14 : List Nat := [6203]
set_option maxRecDepth 4096 in
theorem innerCandidates89_14_checked : candidates edge 4 479 4481 innerTail89_15 best.bound = innerCandidates89_14 := by
  unfold candidates
  have row : edge 4481 = SearchData.row4481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan89_14 : scan edge (search edge 3) 4 [479] 479 innerTail89_14 best = best := by
  rw [innerTail89_14, scan, if_neg (by decide), innerCandidates89_14_checked]
  change scan edge (search edge 3) 4 [479] 479 innerTail89_15
    (if 3 ≤ innerCandidates89_14.length then search edge 3 [479, 4481] innerCandidates89_14 4960 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan89_15

def innerTail89_13 : List Nat := 4337 :: innerTail89_14
def innerCandidates89_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates89_13_checked : candidates edge 4 479 4337 innerTail89_14 best.bound = innerCandidates89_13 := by
  unfold candidates
  have row : edge 4337 = SearchData.row4337.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan89_13 : scan edge (search edge 3) 4 [479] 479 innerTail89_13 best = best := by
  rw [innerTail89_13, scan, if_neg (by decide), innerCandidates89_13_checked]
  change scan edge (search edge 3) 4 [479] 479 innerTail89_14
    (if 3 ≤ innerCandidates89_13.length then search edge 3 [479, 4337] innerCandidates89_13 4816 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan89_14

def innerTail89_12 : List Nat := 3533 :: innerTail89_13
def innerCandidates89_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates89_12_checked : candidates edge 4 479 3533 innerTail89_13 best.bound = innerCandidates89_12 := by
  unfold candidates
  have row : edge 3533 = SearchData.row3533.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan89_12 : scan edge (search edge 3) 4 [479] 479 innerTail89_12 best = best := by
  rw [innerTail89_12, scan, if_neg (by decide), innerCandidates89_12_checked]
  change scan edge (search edge 3) 4 [479] 479 innerTail89_13
    (if 3 ≤ innerCandidates89_12.length then search edge 3 [479, 3533] innerCandidates89_12 4012 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan89_13

def innerTail89_11 : List Nat := 2297 :: innerTail89_12
def innerCandidates89_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates89_11_checked : candidates edge 4 479 2297 innerTail89_12 best.bound = innerCandidates89_11 := by
  unfold candidates
  have row : edge 2297 = PrimePairCache.smallRow2297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan89_11 : scan edge (search edge 3) 4 [479] 479 innerTail89_11 best = best := by
  rw [innerTail89_11, scan, if_neg (by decide), innerCandidates89_11_checked]
  change scan edge (search edge 3) 4 [479] 479 innerTail89_12
    (if 3 ≤ innerCandidates89_11.length then search edge 3 [479, 2297] innerCandidates89_11 2776 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan89_12

def innerTail89_10 : List Nat := 1559 :: innerTail89_11
def innerCandidates89_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates89_10_checked : candidates edge 4 479 1559 innerTail89_11 best.bound = innerCandidates89_10 := by
  unfold candidates
  have row : edge 1559 = PrimePairCache.smallRow1559.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan89_10 : scan edge (search edge 3) 4 [479] 479 innerTail89_10 best = best := by
  rw [innerTail89_10, scan, if_neg (by decide), innerCandidates89_10_checked]
  change scan edge (search edge 3) 4 [479] 479 innerTail89_11
    (if 3 ≤ innerCandidates89_10.length then search edge 3 [479, 1559] innerCandidates89_10 2038 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan89_11

def innerTail89_9 : List Nat := 1307 :: innerTail89_10
def innerCandidates89_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates89_9_checked : candidates edge 4 479 1307 innerTail89_10 best.bound = innerCandidates89_9 := by
  unfold candidates
  have row : edge 1307 = PrimePairCache.smallRow1307.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan89_9 : scan edge (search edge 3) 4 [479] 479 innerTail89_9 best = best := by
  rw [innerTail89_9, scan, if_neg (by decide), innerCandidates89_9_checked]
  change scan edge (search edge 3) 4 [479] 479 innerTail89_10
    (if 3 ≤ innerCandidates89_9.length then search edge 3 [479, 1307] innerCandidates89_9 1786 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan89_10

def innerTail89_8 : List Nat := 1187 :: innerTail89_9
def innerCandidates89_8 : List Nat := [6203]
set_option maxRecDepth 4096 in
theorem innerCandidates89_8_checked : candidates edge 4 479 1187 innerTail89_9 best.bound = innerCandidates89_8 := by
  unfold candidates
  have row : edge 1187 = PrimePairCache.smallRow1187.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan89_8 : scan edge (search edge 3) 4 [479] 479 innerTail89_8 best = best := by
  rw [innerTail89_8, scan, if_neg (by decide), innerCandidates89_8_checked]
  change scan edge (search edge 3) 4 [479] 479 innerTail89_9
    (if 3 ≤ innerCandidates89_8.length then search edge 3 [479, 1187] innerCandidates89_8 1666 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan89_9

def innerTail89_7 : List Nat := 1049 :: innerTail89_8
def innerCandidates89_7 : List Nat := [2297, 6203]
set_option maxRecDepth 4096 in
theorem innerCandidates89_7_checked : candidates edge 4 479 1049 innerTail89_8 best.bound = innerCandidates89_7 := by
  unfold candidates
  have row : edge 1049 = PrimePairCache.smallRow1049.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan89_7 : scan edge (search edge 3) 4 [479] 479 innerTail89_7 best = best := by
  rw [innerTail89_7, scan, if_neg (by decide), innerCandidates89_7_checked]
  change scan edge (search edge 3) 4 [479] 479 innerTail89_8
    (if 3 ≤ innerCandidates89_7.length then search edge 3 [479, 1049] innerCandidates89_7 1528 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan89_8

def innerTail89_6 : List Nat := 971 :: innerTail89_7
def innerCandidates89_6 : List Nat := [7517]
set_option maxRecDepth 4096 in
theorem innerCandidates89_6_checked : candidates edge 4 479 971 innerTail89_7 best.bound = innerCandidates89_6 := by
  unfold candidates
  have row : edge 971 = PrimePairCache.smallRow971.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan89_6 : scan edge (search edge 3) 4 [479] 479 innerTail89_6 best = best := by
  rw [innerTail89_6, scan, if_neg (by decide), innerCandidates89_6_checked]
  change scan edge (search edge 3) 4 [479] 479 innerTail89_7
    (if 3 ≤ innerCandidates89_6.length then search edge 3 [479, 971] innerCandidates89_6 1450 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan89_7

def innerTail89_5 : List Nat := 881 :: innerTail89_6
def innerCandidates89_5 : List Nat := [4733]
set_option maxRecDepth 4096 in
theorem innerCandidates89_5_checked : candidates edge 4 479 881 innerTail89_6 best.bound = innerCandidates89_5 := by
  unfold candidates
  have row : edge 881 = PrimePairCache.smallRow881.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan89_5 : scan edge (search edge 3) 4 [479] 479 innerTail89_5 best = best := by
  rw [innerTail89_5, scan, if_neg (by decide), innerCandidates89_5_checked]
  change scan edge (search edge 3) 4 [479] 479 innerTail89_6
    (if 3 ≤ innerCandidates89_5.length then search edge 3 [479, 881] innerCandidates89_5 1360 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan89_6

def innerTail89_4 : List Nat := 821 :: innerTail89_5
def innerCandidates89_4 : List Nat := [971]
set_option maxRecDepth 4096 in
theorem innerCandidates89_4_checked : candidates edge 4 479 821 innerTail89_5 best.bound = innerCandidates89_4 := by
  unfold candidates
  have row : edge 821 = PrimePairCache.smallRow821.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan89_4 : scan edge (search edge 3) 4 [479] 479 innerTail89_4 best = best := by
  rw [innerTail89_4, scan, if_neg (by decide), innerCandidates89_4_checked]
  change scan edge (search edge 3) 4 [479] 479 innerTail89_5
    (if 3 ≤ innerCandidates89_4.length then search edge 3 [479, 821] innerCandidates89_4 1300 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan89_5

def innerTail89_3 : List Nat := 701 :: innerTail89_4
def innerCandidates89_3 : List Nat := [1187, 1307]
set_option maxRecDepth 4096 in
theorem innerCandidates89_3_checked : candidates edge 4 479 701 innerTail89_4 best.bound = innerCandidates89_3 := by
  unfold candidates
  have row : edge 701 = PrimePairCache.smallRow701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan89_3 : scan edge (search edge 3) 4 [479] 479 innerTail89_3 best = best := by
  rw [innerTail89_3, scan, if_neg (by decide), innerCandidates89_3_checked]
  change scan edge (search edge 3) 4 [479] 479 innerTail89_4
    (if 3 ≤ innerCandidates89_3.length then search edge 3 [479, 701] innerCandidates89_3 1180 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan89_4

def innerTail89_2 : List Nat := 599 :: innerTail89_3
def innerCandidates89_2 : List Nat := [7517]
set_option maxRecDepth 4096 in
theorem innerCandidates89_2_checked : candidates edge 4 479 599 innerTail89_3 best.bound = innerCandidates89_2 := by
  unfold candidates
  have row : edge 599 = PrimePairCache.smallRow599.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan89_2 : scan edge (search edge 3) 4 [479] 479 innerTail89_2 best = best := by
  rw [innerTail89_2, scan, if_neg (by decide), innerCandidates89_2_checked]
  change scan edge (search edge 3) 4 [479] 479 innerTail89_3
    (if 3 ≤ innerCandidates89_2.length then search edge 3 [479, 599] innerCandidates89_2 1078 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan89_3

def innerTail89_1 : List Nat := 593 :: innerTail89_2
def innerCandidates89_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates89_1_checked : candidates edge 4 479 593 innerTail89_2 best.bound = innerCandidates89_1 := by
  unfold candidates
  have row : edge 593 = PrimePairCache.smallRow593.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan89_1 : scan edge (search edge 3) 4 [479] 479 innerTail89_1 best = best := by
  rw [innerTail89_1, scan, if_neg (by decide), innerCandidates89_1_checked]
  change scan edge (search edge 3) 4 [479] 479 innerTail89_2
    (if 3 ≤ innerCandidates89_1.length then search edge 3 [479, 593] innerCandidates89_1 1072 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan89_2

def innerTail89_0 : List Nat := 569 :: innerTail89_1
def innerCandidates89_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates89_0_checked : candidates edge 4 479 569 innerTail89_1 best.bound = innerCandidates89_0 := by
  unfold candidates
  have row : edge 569 = PrimePairCache.smallRow569.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan89_0 : scan edge (search edge 3) 4 [479] 479 innerTail89_0 best = best := by
  rw [innerTail89_0, scan, if_neg (by decide), innerCandidates89_0_checked]
  change scan edge (search edge 3) 4 [479] 479 innerTail89_1
    (if 3 ≤ innerCandidates89_0.length then search edge 3 [479, 569] innerCandidates89_0 1048 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan89_1

theorem rootBranch89 : search edge 4 [479] rootCandidates89 479 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [479] 479 innerTail89_0 best = best
  exact innerScan89_0

end ProjectEulerSolutions.PrimePairUpper.Search
