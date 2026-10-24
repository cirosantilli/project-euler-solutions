import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates281 : List Nat := [1933, 1951, 2371, 2383, 2791, 3163, 3433, 3709, 3769, 4789, 5011, 5059, 5479, 5521, 5881, 6007, 6121, 6133, 6907, 7213, 7867]
set_option maxRecDepth 4096 in
theorem rootCandidates281_checked : candidates edge 5 0 1861 tail282 best.bound = rootCandidates281 := by
  unfold candidates
  have row : edge 1861 = PrimePairCache.smallRow1861.contains := by funext q; rfl
  rw [row]
  decide

def innerTail281_21 : List Nat := []
theorem innerScan281_21 : scan edge (search edge 3) 4 [1861] 1861 innerTail281_21 best = best := rfl

def innerTail281_20 : List Nat := 7867 :: innerTail281_21
def innerCandidates281_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates281_20_checked : candidates edge 4 1861 7867 innerTail281_21 best.bound = innerCandidates281_20 := by
  unfold candidates
  have row : edge 7867 = SearchData.row7867.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan281_20 : scan edge (search edge 3) 4 [1861] 1861 innerTail281_20 best = best := by
  rw [innerTail281_20, scan, if_neg (by decide), innerCandidates281_20_checked]
  change scan edge (search edge 3) 4 [1861] 1861 innerTail281_21
    (if 3 ≤ innerCandidates281_20.length then search edge 3 [1861, 7867] innerCandidates281_20 9728 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan281_21

def innerTail281_19 : List Nat := 7213 :: innerTail281_20
def innerCandidates281_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates281_19_checked : candidates edge 4 1861 7213 innerTail281_20 best.bound = innerCandidates281_19 := by
  unfold candidates
  have row : edge 7213 = SearchData.row7213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan281_19 : scan edge (search edge 3) 4 [1861] 1861 innerTail281_19 best = best := by
  rw [innerTail281_19, scan, if_neg (by decide), innerCandidates281_19_checked]
  change scan edge (search edge 3) 4 [1861] 1861 innerTail281_20
    (if 3 ≤ innerCandidates281_19.length then search edge 3 [1861, 7213] innerCandidates281_19 9074 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan281_20

def innerTail281_18 : List Nat := 6907 :: innerTail281_19
def innerCandidates281_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates281_18_checked : candidates edge 4 1861 6907 innerTail281_19 best.bound = innerCandidates281_18 := by
  unfold candidates
  have row : edge 6907 = SearchData.row6907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan281_18 : scan edge (search edge 3) 4 [1861] 1861 innerTail281_18 best = best := by
  rw [innerTail281_18, scan, if_neg (by decide), innerCandidates281_18_checked]
  change scan edge (search edge 3) 4 [1861] 1861 innerTail281_19
    (if 3 ≤ innerCandidates281_18.length then search edge 3 [1861, 6907] innerCandidates281_18 8768 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan281_19

def innerTail281_17 : List Nat := 6133 :: innerTail281_18
def innerCandidates281_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates281_17_checked : candidates edge 4 1861 6133 innerTail281_18 best.bound = innerCandidates281_17 := by
  unfold candidates
  have row : edge 6133 = SearchData.row6133.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan281_17 : scan edge (search edge 3) 4 [1861] 1861 innerTail281_17 best = best := by
  rw [innerTail281_17, scan, if_neg (by decide), innerCandidates281_17_checked]
  change scan edge (search edge 3) 4 [1861] 1861 innerTail281_18
    (if 3 ≤ innerCandidates281_17.length then search edge 3 [1861, 6133] innerCandidates281_17 7994 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan281_18

def innerTail281_16 : List Nat := 6121 :: innerTail281_17
def innerCandidates281_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates281_16_checked : candidates edge 4 1861 6121 innerTail281_17 best.bound = innerCandidates281_16 := by
  unfold candidates
  have row : edge 6121 = SearchData.row6121.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan281_16 : scan edge (search edge 3) 4 [1861] 1861 innerTail281_16 best = best := by
  rw [innerTail281_16, scan, if_neg (by decide), innerCandidates281_16_checked]
  change scan edge (search edge 3) 4 [1861] 1861 innerTail281_17
    (if 3 ≤ innerCandidates281_16.length then search edge 3 [1861, 6121] innerCandidates281_16 7982 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan281_17

def innerTail281_15 : List Nat := 6007 :: innerTail281_16
def innerCandidates281_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates281_15_checked : candidates edge 4 1861 6007 innerTail281_16 best.bound = innerCandidates281_15 := by
  unfold candidates
  have row : edge 6007 = SearchData.row6007.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan281_15 : scan edge (search edge 3) 4 [1861] 1861 innerTail281_15 best = best := by
  rw [innerTail281_15, scan, if_neg (by decide), innerCandidates281_15_checked]
  change scan edge (search edge 3) 4 [1861] 1861 innerTail281_16
    (if 3 ≤ innerCandidates281_15.length then search edge 3 [1861, 6007] innerCandidates281_15 7868 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan281_16

def innerTail281_14 : List Nat := 5881 :: innerTail281_15
def innerCandidates281_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates281_14_checked : candidates edge 4 1861 5881 innerTail281_15 best.bound = innerCandidates281_14 := by
  unfold candidates
  have row : edge 5881 = SearchData.row5881.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan281_14 : scan edge (search edge 3) 4 [1861] 1861 innerTail281_14 best = best := by
  rw [innerTail281_14, scan, if_neg (by decide), innerCandidates281_14_checked]
  change scan edge (search edge 3) 4 [1861] 1861 innerTail281_15
    (if 3 ≤ innerCandidates281_14.length then search edge 3 [1861, 5881] innerCandidates281_14 7742 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan281_15

def innerTail281_13 : List Nat := 5521 :: innerTail281_14
def innerCandidates281_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates281_13_checked : candidates edge 4 1861 5521 innerTail281_14 best.bound = innerCandidates281_13 := by
  unfold candidates
  have row : edge 5521 = SearchData.row5521.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan281_13 : scan edge (search edge 3) 4 [1861] 1861 innerTail281_13 best = best := by
  rw [innerTail281_13, scan, if_neg (by decide), innerCandidates281_13_checked]
  change scan edge (search edge 3) 4 [1861] 1861 innerTail281_14
    (if 3 ≤ innerCandidates281_13.length then search edge 3 [1861, 5521] innerCandidates281_13 7382 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan281_14

def innerTail281_12 : List Nat := 5479 :: innerTail281_13
def innerCandidates281_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates281_12_checked : candidates edge 4 1861 5479 innerTail281_13 best.bound = innerCandidates281_12 := by
  unfold candidates
  have row : edge 5479 = SearchData.row5479.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan281_12 : scan edge (search edge 3) 4 [1861] 1861 innerTail281_12 best = best := by
  rw [innerTail281_12, scan, if_neg (by decide), innerCandidates281_12_checked]
  change scan edge (search edge 3) 4 [1861] 1861 innerTail281_13
    (if 3 ≤ innerCandidates281_12.length then search edge 3 [1861, 5479] innerCandidates281_12 7340 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan281_13

def innerTail281_11 : List Nat := 5059 :: innerTail281_12
def innerCandidates281_11 : List Nat := [7213]
set_option maxRecDepth 4096 in
theorem innerCandidates281_11_checked : candidates edge 4 1861 5059 innerTail281_12 best.bound = innerCandidates281_11 := by
  unfold candidates
  have row : edge 5059 = SearchData.row5059.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan281_11 : scan edge (search edge 3) 4 [1861] 1861 innerTail281_11 best = best := by
  rw [innerTail281_11, scan, if_neg (by decide), innerCandidates281_11_checked]
  change scan edge (search edge 3) 4 [1861] 1861 innerTail281_12
    (if 3 ≤ innerCandidates281_11.length then search edge 3 [1861, 5059] innerCandidates281_11 6920 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan281_12

def innerTail281_10 : List Nat := 5011 :: innerTail281_11
def innerCandidates281_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates281_10_checked : candidates edge 4 1861 5011 innerTail281_11 best.bound = innerCandidates281_10 := by
  unfold candidates
  have row : edge 5011 = SearchData.row5011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan281_10 : scan edge (search edge 3) 4 [1861] 1861 innerTail281_10 best = best := by
  rw [innerTail281_10, scan, if_neg (by decide), innerCandidates281_10_checked]
  change scan edge (search edge 3) 4 [1861] 1861 innerTail281_11
    (if 3 ≤ innerCandidates281_10.length then search edge 3 [1861, 5011] innerCandidates281_10 6872 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan281_11

def innerTail281_9 : List Nat := 4789 :: innerTail281_10
def innerCandidates281_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates281_9_checked : candidates edge 4 1861 4789 innerTail281_10 best.bound = innerCandidates281_9 := by
  unfold candidates
  have row : edge 4789 = SearchData.row4789.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan281_9 : scan edge (search edge 3) 4 [1861] 1861 innerTail281_9 best = best := by
  rw [innerTail281_9, scan, if_neg (by decide), innerCandidates281_9_checked]
  change scan edge (search edge 3) 4 [1861] 1861 innerTail281_10
    (if 3 ≤ innerCandidates281_9.length then search edge 3 [1861, 4789] innerCandidates281_9 6650 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan281_10

def innerTail281_8 : List Nat := 3769 :: innerTail281_9
def innerCandidates281_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates281_8_checked : candidates edge 4 1861 3769 innerTail281_9 best.bound = innerCandidates281_8 := by
  unfold candidates
  have row : edge 3769 = SearchData.row3769.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan281_8 : scan edge (search edge 3) 4 [1861] 1861 innerTail281_8 best = best := by
  rw [innerTail281_8, scan, if_neg (by decide), innerCandidates281_8_checked]
  change scan edge (search edge 3) 4 [1861] 1861 innerTail281_9
    (if 3 ≤ innerCandidates281_8.length then search edge 3 [1861, 3769] innerCandidates281_8 5630 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan281_9

def innerTail281_7 : List Nat := 3709 :: innerTail281_8
def innerCandidates281_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates281_7_checked : candidates edge 4 1861 3709 innerTail281_8 best.bound = innerCandidates281_7 := by
  unfold candidates
  have row : edge 3709 = SearchData.row3709.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan281_7 : scan edge (search edge 3) 4 [1861] 1861 innerTail281_7 best = best := by
  rw [innerTail281_7, scan, if_neg (by decide), innerCandidates281_7_checked]
  change scan edge (search edge 3) 4 [1861] 1861 innerTail281_8
    (if 3 ≤ innerCandidates281_7.length then search edge 3 [1861, 3709] innerCandidates281_7 5570 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan281_8

def innerTail281_6 : List Nat := 3433 :: innerTail281_7
def innerCandidates281_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates281_6_checked : candidates edge 4 1861 3433 innerTail281_7 best.bound = innerCandidates281_6 := by
  unfold candidates
  have row : edge 3433 = SearchData.row3433.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan281_6 : scan edge (search edge 3) 4 [1861] 1861 innerTail281_6 best = best := by
  rw [innerTail281_6, scan, if_neg (by decide), innerCandidates281_6_checked]
  change scan edge (search edge 3) 4 [1861] 1861 innerTail281_7
    (if 3 ≤ innerCandidates281_6.length then search edge 3 [1861, 3433] innerCandidates281_6 5294 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan281_7

def innerTail281_5 : List Nat := 3163 :: innerTail281_6
def innerCandidates281_5 : List Nat := [3769]
set_option maxRecDepth 4096 in
theorem innerCandidates281_5_checked : candidates edge 4 1861 3163 innerTail281_6 best.bound = innerCandidates281_5 := by
  unfold candidates
  have row : edge 3163 = SearchData.row3163.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan281_5 : scan edge (search edge 3) 4 [1861] 1861 innerTail281_5 best = best := by
  rw [innerTail281_5, scan, if_neg (by decide), innerCandidates281_5_checked]
  change scan edge (search edge 3) 4 [1861] 1861 innerTail281_6
    (if 3 ≤ innerCandidates281_5.length then search edge 3 [1861, 3163] innerCandidates281_5 5024 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan281_6

def innerTail281_4 : List Nat := 2791 :: innerTail281_5
def innerCandidates281_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates281_4_checked : candidates edge 4 1861 2791 innerTail281_5 best.bound = innerCandidates281_4 := by
  unfold candidates
  have row : edge 2791 = SearchData.row2791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan281_4 : scan edge (search edge 3) 4 [1861] 1861 innerTail281_4 best = best := by
  rw [innerTail281_4, scan, if_neg (by decide), innerCandidates281_4_checked]
  change scan edge (search edge 3) 4 [1861] 1861 innerTail281_5
    (if 3 ≤ innerCandidates281_4.length then search edge 3 [1861, 2791] innerCandidates281_4 4652 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan281_5

def innerTail281_3 : List Nat := 2383 :: innerTail281_4
def innerCandidates281_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates281_3_checked : candidates edge 4 1861 2383 innerTail281_4 best.bound = innerCandidates281_3 := by
  unfold candidates
  have row : edge 2383 = PrimePairCache.smallRow2383.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan281_3 : scan edge (search edge 3) 4 [1861] 1861 innerTail281_3 best = best := by
  rw [innerTail281_3, scan, if_neg (by decide), innerCandidates281_3_checked]
  change scan edge (search edge 3) 4 [1861] 1861 innerTail281_4
    (if 3 ≤ innerCandidates281_3.length then search edge 3 [1861, 2383] innerCandidates281_3 4244 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan281_4

def innerTail281_2 : List Nat := 2371 :: innerTail281_3
def innerCandidates281_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates281_2_checked : candidates edge 4 1861 2371 innerTail281_3 best.bound = innerCandidates281_2 := by
  unfold candidates
  have row : edge 2371 = PrimePairCache.smallRow2371.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan281_2 : scan edge (search edge 3) 4 [1861] 1861 innerTail281_2 best = best := by
  rw [innerTail281_2, scan, if_neg (by decide), innerCandidates281_2_checked]
  change scan edge (search edge 3) 4 [1861] 1861 innerTail281_3
    (if 3 ≤ innerCandidates281_2.length then search edge 3 [1861, 2371] innerCandidates281_2 4232 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan281_3

def innerTail281_1 : List Nat := 1951 :: innerTail281_2
def innerCandidates281_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates281_1_checked : candidates edge 4 1861 1951 innerTail281_2 best.bound = innerCandidates281_1 := by
  unfold candidates
  have row : edge 1951 = PrimePairCache.smallRow1951.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan281_1 : scan edge (search edge 3) 4 [1861] 1861 innerTail281_1 best = best := by
  rw [innerTail281_1, scan, if_neg (by decide), innerCandidates281_1_checked]
  change scan edge (search edge 3) 4 [1861] 1861 innerTail281_2
    (if 3 ≤ innerCandidates281_1.length then search edge 3 [1861, 1951] innerCandidates281_1 3812 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan281_2

def innerTail281_0 : List Nat := 1933 :: innerTail281_1
def innerCandidates281_0 : List Nat := [5011]
set_option maxRecDepth 4096 in
theorem innerCandidates281_0_checked : candidates edge 4 1861 1933 innerTail281_1 best.bound = innerCandidates281_0 := by
  unfold candidates
  have row : edge 1933 = PrimePairCache.smallRow1933.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan281_0 : scan edge (search edge 3) 4 [1861] 1861 innerTail281_0 best = best := by
  rw [innerTail281_0, scan, if_neg (by decide), innerCandidates281_0_checked]
  change scan edge (search edge 3) 4 [1861] 1861 innerTail281_1
    (if 3 ≤ innerCandidates281_0.length then search edge 3 [1861, 1933] innerCandidates281_0 3794 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan281_1

theorem rootBranch281 : search edge 4 [1861] rootCandidates281 1861 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1861] 1861 innerTail281_0 best = best
  exact innerScan281_0

def rootCandidates280 : List Nat := [2423, 2687, 2969, 3011, 3671, 3779, 3863, 4007, 4283, 4373, 4457, 4691, 4919, 5519, 5801, 5927, 6173, 7307, 7523, 7703, 8069, 8123]
set_option maxRecDepth 4096 in
theorem rootCandidates280_checked : candidates edge 5 0 1847 tail281 best.bound = rootCandidates280 := by
  unfold candidates
  have row : edge 1847 = PrimePairCache.smallRow1847.contains := by funext q; rfl
  rw [row]
  decide

def innerTail280_22 : List Nat := []
theorem innerScan280_22 : scan edge (search edge 3) 4 [1847] 1847 innerTail280_22 best = best := rfl

def innerTail280_21 : List Nat := 8123 :: innerTail280_22
def innerCandidates280_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates280_21_checked : candidates edge 4 1847 8123 innerTail280_22 best.bound = innerCandidates280_21 := by
  unfold candidates
  have row : edge 8123 = SearchData.row8123.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan280_21 : scan edge (search edge 3) 4 [1847] 1847 innerTail280_21 best = best := by
  rw [innerTail280_21, scan, if_neg (by decide), innerCandidates280_21_checked]
  change scan edge (search edge 3) 4 [1847] 1847 innerTail280_22
    (if 3 ≤ innerCandidates280_21.length then search edge 3 [1847, 8123] innerCandidates280_21 9970 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan280_22

def innerTail280_20 : List Nat := 8069 :: innerTail280_21
def innerCandidates280_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates280_20_checked : candidates edge 4 1847 8069 innerTail280_21 best.bound = innerCandidates280_20 := by
  unfold candidates
  have row : edge 8069 = SearchData.row8069.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan280_20 : scan edge (search edge 3) 4 [1847] 1847 innerTail280_20 best = best := by
  rw [innerTail280_20, scan, if_neg (by decide), innerCandidates280_20_checked]
  change scan edge (search edge 3) 4 [1847] 1847 innerTail280_21
    (if 3 ≤ innerCandidates280_20.length then search edge 3 [1847, 8069] innerCandidates280_20 9916 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan280_21

def innerTail280_19 : List Nat := 7703 :: innerTail280_20
def innerCandidates280_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates280_19_checked : candidates edge 4 1847 7703 innerTail280_20 best.bound = innerCandidates280_19 := by
  unfold candidates
  have row : edge 7703 = SearchData.row7703.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan280_19 : scan edge (search edge 3) 4 [1847] 1847 innerTail280_19 best = best := by
  rw [innerTail280_19, scan, if_neg (by decide), innerCandidates280_19_checked]
  change scan edge (search edge 3) 4 [1847] 1847 innerTail280_20
    (if 3 ≤ innerCandidates280_19.length then search edge 3 [1847, 7703] innerCandidates280_19 9550 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan280_20

def innerTail280_18 : List Nat := 7523 :: innerTail280_19
def innerCandidates280_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates280_18_checked : candidates edge 4 1847 7523 innerTail280_19 best.bound = innerCandidates280_18 := by
  unfold candidates
  have row : edge 7523 = SearchData.row7523.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan280_18 : scan edge (search edge 3) 4 [1847] 1847 innerTail280_18 best = best := by
  rw [innerTail280_18, scan, if_neg (by decide), innerCandidates280_18_checked]
  change scan edge (search edge 3) 4 [1847] 1847 innerTail280_19
    (if 3 ≤ innerCandidates280_18.length then search edge 3 [1847, 7523] innerCandidates280_18 9370 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan280_19

def innerTail280_17 : List Nat := 7307 :: innerTail280_18
def innerCandidates280_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates280_17_checked : candidates edge 4 1847 7307 innerTail280_18 best.bound = innerCandidates280_17 := by
  unfold candidates
  have row : edge 7307 = SearchData.row7307.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan280_17 : scan edge (search edge 3) 4 [1847] 1847 innerTail280_17 best = best := by
  rw [innerTail280_17, scan, if_neg (by decide), innerCandidates280_17_checked]
  change scan edge (search edge 3) 4 [1847] 1847 innerTail280_18
    (if 3 ≤ innerCandidates280_17.length then search edge 3 [1847, 7307] innerCandidates280_17 9154 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan280_18

def innerTail280_16 : List Nat := 6173 :: innerTail280_17
def innerCandidates280_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates280_16_checked : candidates edge 4 1847 6173 innerTail280_17 best.bound = innerCandidates280_16 := by
  unfold candidates
  have row : edge 6173 = SearchData.row6173.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan280_16 : scan edge (search edge 3) 4 [1847] 1847 innerTail280_16 best = best := by
  rw [innerTail280_16, scan, if_neg (by decide), innerCandidates280_16_checked]
  change scan edge (search edge 3) 4 [1847] 1847 innerTail280_17
    (if 3 ≤ innerCandidates280_16.length then search edge 3 [1847, 6173] innerCandidates280_16 8020 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan280_17

def innerTail280_15 : List Nat := 5927 :: innerTail280_16
def innerCandidates280_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates280_15_checked : candidates edge 4 1847 5927 innerTail280_16 best.bound = innerCandidates280_15 := by
  unfold candidates
  have row : edge 5927 = SearchData.row5927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan280_15 : scan edge (search edge 3) 4 [1847] 1847 innerTail280_15 best = best := by
  rw [innerTail280_15, scan, if_neg (by decide), innerCandidates280_15_checked]
  change scan edge (search edge 3) 4 [1847] 1847 innerTail280_16
    (if 3 ≤ innerCandidates280_15.length then search edge 3 [1847, 5927] innerCandidates280_15 7774 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan280_16

def innerTail280_14 : List Nat := 5801 :: innerTail280_15
def innerCandidates280_14 : List Nat := [6173]
set_option maxRecDepth 4096 in
theorem innerCandidates280_14_checked : candidates edge 4 1847 5801 innerTail280_15 best.bound = innerCandidates280_14 := by
  unfold candidates
  have row : edge 5801 = SearchData.row5801.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan280_14 : scan edge (search edge 3) 4 [1847] 1847 innerTail280_14 best = best := by
  rw [innerTail280_14, scan, if_neg (by decide), innerCandidates280_14_checked]
  change scan edge (search edge 3) 4 [1847] 1847 innerTail280_15
    (if 3 ≤ innerCandidates280_14.length then search edge 3 [1847, 5801] innerCandidates280_14 7648 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan280_15

def innerTail280_13 : List Nat := 5519 :: innerTail280_14
def innerCandidates280_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates280_13_checked : candidates edge 4 1847 5519 innerTail280_14 best.bound = innerCandidates280_13 := by
  unfold candidates
  have row : edge 5519 = SearchData.row5519.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan280_13 : scan edge (search edge 3) 4 [1847] 1847 innerTail280_13 best = best := by
  rw [innerTail280_13, scan, if_neg (by decide), innerCandidates280_13_checked]
  change scan edge (search edge 3) 4 [1847] 1847 innerTail280_14
    (if 3 ≤ innerCandidates280_13.length then search edge 3 [1847, 5519] innerCandidates280_13 7366 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan280_14

def innerTail280_12 : List Nat := 4919 :: innerTail280_13
def innerCandidates280_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates280_12_checked : candidates edge 4 1847 4919 innerTail280_13 best.bound = innerCandidates280_12 := by
  unfold candidates
  have row : edge 4919 = SearchData.row4919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan280_12 : scan edge (search edge 3) 4 [1847] 1847 innerTail280_12 best = best := by
  rw [innerTail280_12, scan, if_neg (by decide), innerCandidates280_12_checked]
  change scan edge (search edge 3) 4 [1847] 1847 innerTail280_13
    (if 3 ≤ innerCandidates280_12.length then search edge 3 [1847, 4919] innerCandidates280_12 6766 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan280_13

def innerTail280_11 : List Nat := 4691 :: innerTail280_12
def innerCandidates280_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates280_11_checked : candidates edge 4 1847 4691 innerTail280_12 best.bound = innerCandidates280_11 := by
  unfold candidates
  have row : edge 4691 = SearchData.row4691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan280_11 : scan edge (search edge 3) 4 [1847] 1847 innerTail280_11 best = best := by
  rw [innerTail280_11, scan, if_neg (by decide), innerCandidates280_11_checked]
  change scan edge (search edge 3) 4 [1847] 1847 innerTail280_12
    (if 3 ≤ innerCandidates280_11.length then search edge 3 [1847, 4691] innerCandidates280_11 6538 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan280_12

def innerTail280_10 : List Nat := 4457 :: innerTail280_11
def innerCandidates280_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates280_10_checked : candidates edge 4 1847 4457 innerTail280_11 best.bound = innerCandidates280_10 := by
  unfold candidates
  have row : edge 4457 = SearchData.row4457.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan280_10 : scan edge (search edge 3) 4 [1847] 1847 innerTail280_10 best = best := by
  rw [innerTail280_10, scan, if_neg (by decide), innerCandidates280_10_checked]
  change scan edge (search edge 3) 4 [1847] 1847 innerTail280_11
    (if 3 ≤ innerCandidates280_10.length then search edge 3 [1847, 4457] innerCandidates280_10 6304 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan280_11

def innerTail280_9 : List Nat := 4373 :: innerTail280_10
def innerCandidates280_9 : List Nat := [5801, 5927]
set_option maxRecDepth 4096 in
theorem innerCandidates280_9_checked : candidates edge 4 1847 4373 innerTail280_10 best.bound = innerCandidates280_9 := by
  unfold candidates
  have row : edge 4373 = SearchData.row4373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan280_9 : scan edge (search edge 3) 4 [1847] 1847 innerTail280_9 best = best := by
  rw [innerTail280_9, scan, if_neg (by decide), innerCandidates280_9_checked]
  change scan edge (search edge 3) 4 [1847] 1847 innerTail280_10
    (if 3 ≤ innerCandidates280_9.length then search edge 3 [1847, 4373] innerCandidates280_9 6220 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan280_10

def innerTail280_8 : List Nat := 4283 :: innerTail280_9
def innerCandidates280_8 : List Nat := [8069]
set_option maxRecDepth 4096 in
theorem innerCandidates280_8_checked : candidates edge 4 1847 4283 innerTail280_9 best.bound = innerCandidates280_8 := by
  unfold candidates
  have row : edge 4283 = SearchData.row4283.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan280_8 : scan edge (search edge 3) 4 [1847] 1847 innerTail280_8 best = best := by
  rw [innerTail280_8, scan, if_neg (by decide), innerCandidates280_8_checked]
  change scan edge (search edge 3) 4 [1847] 1847 innerTail280_9
    (if 3 ≤ innerCandidates280_8.length then search edge 3 [1847, 4283] innerCandidates280_8 6130 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan280_9

def innerTail280_7 : List Nat := 4007 :: innerTail280_8
def innerCandidates280_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates280_7_checked : candidates edge 4 1847 4007 innerTail280_8 best.bound = innerCandidates280_7 := by
  unfold candidates
  have row : edge 4007 = SearchData.row4007.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan280_7 : scan edge (search edge 3) 4 [1847] 1847 innerTail280_7 best = best := by
  rw [innerTail280_7, scan, if_neg (by decide), innerCandidates280_7_checked]
  change scan edge (search edge 3) 4 [1847] 1847 innerTail280_8
    (if 3 ≤ innerCandidates280_7.length then search edge 3 [1847, 4007] innerCandidates280_7 5854 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan280_8

def innerTail280_6 : List Nat := 3863 :: innerTail280_7
def innerCandidates280_6 : List Nat := [4007]
set_option maxRecDepth 4096 in
theorem innerCandidates280_6_checked : candidates edge 4 1847 3863 innerTail280_7 best.bound = innerCandidates280_6 := by
  unfold candidates
  have row : edge 3863 = SearchData.row3863.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan280_6 : scan edge (search edge 3) 4 [1847] 1847 innerTail280_6 best = best := by
  rw [innerTail280_6, scan, if_neg (by decide), innerCandidates280_6_checked]
  change scan edge (search edge 3) 4 [1847] 1847 innerTail280_7
    (if 3 ≤ innerCandidates280_6.length then search edge 3 [1847, 3863] innerCandidates280_6 5710 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan280_7

def innerTail280_5 : List Nat := 3779 :: innerTail280_6
def innerCandidates280_5 : List Nat := [3863, 7523]
set_option maxRecDepth 4096 in
theorem innerCandidates280_5_checked : candidates edge 4 1847 3779 innerTail280_6 best.bound = innerCandidates280_5 := by
  unfold candidates
  have row : edge 3779 = SearchData.row3779.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan280_5 : scan edge (search edge 3) 4 [1847] 1847 innerTail280_5 best = best := by
  rw [innerTail280_5, scan, if_neg (by decide), innerCandidates280_5_checked]
  change scan edge (search edge 3) 4 [1847] 1847 innerTail280_6
    (if 3 ≤ innerCandidates280_5.length then search edge 3 [1847, 3779] innerCandidates280_5 5626 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan280_6

def innerTail280_4 : List Nat := 3671 :: innerTail280_5
def innerCandidates280_4 : List Nat := [7523]
set_option maxRecDepth 4096 in
theorem innerCandidates280_4_checked : candidates edge 4 1847 3671 innerTail280_5 best.bound = innerCandidates280_4 := by
  unfold candidates
  have row : edge 3671 = SearchData.row3671.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan280_4 : scan edge (search edge 3) 4 [1847] 1847 innerTail280_4 best = best := by
  rw [innerTail280_4, scan, if_neg (by decide), innerCandidates280_4_checked]
  change scan edge (search edge 3) 4 [1847] 1847 innerTail280_5
    (if 3 ≤ innerCandidates280_4.length then search edge 3 [1847, 3671] innerCandidates280_4 5518 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan280_5

def innerTail280_3 : List Nat := 3011 :: innerTail280_4
def innerCandidates280_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates280_3_checked : candidates edge 4 1847 3011 innerTail280_4 best.bound = innerCandidates280_3 := by
  unfold candidates
  have row : edge 3011 = SearchData.row3011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan280_3 : scan edge (search edge 3) 4 [1847] 1847 innerTail280_3 best = best := by
  rw [innerTail280_3, scan, if_neg (by decide), innerCandidates280_3_checked]
  change scan edge (search edge 3) 4 [1847] 1847 innerTail280_4
    (if 3 ≤ innerCandidates280_3.length then search edge 3 [1847, 3011] innerCandidates280_3 4858 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan280_4

def innerTail280_2 : List Nat := 2969 :: innerTail280_3
def innerCandidates280_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates280_2_checked : candidates edge 4 1847 2969 innerTail280_3 best.bound = innerCandidates280_2 := by
  unfold candidates
  have row : edge 2969 = SearchData.row2969.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan280_2 : scan edge (search edge 3) 4 [1847] 1847 innerTail280_2 best = best := by
  rw [innerTail280_2, scan, if_neg (by decide), innerCandidates280_2_checked]
  change scan edge (search edge 3) 4 [1847] 1847 innerTail280_3
    (if 3 ≤ innerCandidates280_2.length then search edge 3 [1847, 2969] innerCandidates280_2 4816 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan280_3

def innerTail280_1 : List Nat := 2687 :: innerTail280_2
def innerCandidates280_1 : List Nat := [4007, 7703, 8069]
set_option maxRecDepth 4096 in
theorem innerCandidates280_1_checked : candidates edge 4 1847 2687 innerTail280_2 best.bound = innerCandidates280_1 := by
  unfold candidates
  have row : edge 2687 = SearchData.row2687.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild280_1 : search edge 3 [1847, 2687] innerCandidates280_1 4534 best = best := by decide

theorem innerScan280_1 : scan edge (search edge 3) 4 [1847] 1847 innerTail280_1 best = best := by
  rw [innerTail280_1, scan, if_neg (by decide), innerCandidates280_1_checked]
  change scan edge (search edge 3) 4 [1847] 1847 innerTail280_2
    (if 3 ≤ innerCandidates280_1.length then search edge 3 [1847, 2687] innerCandidates280_1 4534 best else best) = best
  rw [if_pos (by decide), innerChild280_1]
  exact innerScan280_2

def innerTail280_0 : List Nat := 2423 :: innerTail280_1
def innerCandidates280_0 : List Nat := [3863]
set_option maxRecDepth 4096 in
theorem innerCandidates280_0_checked : candidates edge 4 1847 2423 innerTail280_1 best.bound = innerCandidates280_0 := by
  unfold candidates
  have row : edge 2423 = PrimePairCache.smallRow2423.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan280_0 : scan edge (search edge 3) 4 [1847] 1847 innerTail280_0 best = best := by
  rw [innerTail280_0, scan, if_neg (by decide), innerCandidates280_0_checked]
  change scan edge (search edge 3) 4 [1847] 1847 innerTail280_1
    (if 3 ≤ innerCandidates280_0.length then search edge 3 [1847, 2423] innerCandidates280_0 4270 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan280_1

theorem rootBranch280 : search edge 4 [1847] rootCandidates280 1847 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1847] 1847 innerTail280_0 best = best
  exact innerScan280_0

def rootCandidates279 : List Nat := [2803, 3259, 3769, 5197, 6301, 7603, 7927, 8233]
set_option maxRecDepth 4096 in
theorem rootCandidates279_checked : candidates edge 5 0 1831 tail280 best.bound = rootCandidates279 := by
  unfold candidates
  have row : edge 1831 = PrimePairCache.smallRow1831.contains := by funext q; rfl
  rw [row]
  decide

def innerTail279_8 : List Nat := []
theorem innerScan279_8 : scan edge (search edge 3) 4 [1831] 1831 innerTail279_8 best = best := rfl

def innerTail279_7 : List Nat := 8233 :: innerTail279_8
def innerCandidates279_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates279_7_checked : candidates edge 4 1831 8233 innerTail279_8 best.bound = innerCandidates279_7 := by
  unfold candidates
  have row : edge 8233 = SearchData.row8233.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan279_7 : scan edge (search edge 3) 4 [1831] 1831 innerTail279_7 best = best := by
  rw [innerTail279_7, scan, if_neg (by decide), innerCandidates279_7_checked]
  change scan edge (search edge 3) 4 [1831] 1831 innerTail279_8
    (if 3 ≤ innerCandidates279_7.length then search edge 3 [1831, 8233] innerCandidates279_7 10064 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan279_8

def innerTail279_6 : List Nat := 7927 :: innerTail279_7
def innerCandidates279_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates279_6_checked : candidates edge 4 1831 7927 innerTail279_7 best.bound = innerCandidates279_6 := by
  unfold candidates
  have row : edge 7927 = SearchData.row7927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan279_6 : scan edge (search edge 3) 4 [1831] 1831 innerTail279_6 best = best := by
  rw [innerTail279_6, scan, if_neg (by decide), innerCandidates279_6_checked]
  change scan edge (search edge 3) 4 [1831] 1831 innerTail279_7
    (if 3 ≤ innerCandidates279_6.length then search edge 3 [1831, 7927] innerCandidates279_6 9758 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan279_7

def innerTail279_5 : List Nat := 7603 :: innerTail279_6
def innerCandidates279_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates279_5_checked : candidates edge 4 1831 7603 innerTail279_6 best.bound = innerCandidates279_5 := by
  unfold candidates
  have row : edge 7603 = SearchData.row7603.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan279_5 : scan edge (search edge 3) 4 [1831] 1831 innerTail279_5 best = best := by
  rw [innerTail279_5, scan, if_neg (by decide), innerCandidates279_5_checked]
  change scan edge (search edge 3) 4 [1831] 1831 innerTail279_6
    (if 3 ≤ innerCandidates279_5.length then search edge 3 [1831, 7603] innerCandidates279_5 9434 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan279_6

def innerTail279_4 : List Nat := 6301 :: innerTail279_5
def innerCandidates279_4 : List Nat := [8233]
set_option maxRecDepth 4096 in
theorem innerCandidates279_4_checked : candidates edge 4 1831 6301 innerTail279_5 best.bound = innerCandidates279_4 := by
  unfold candidates
  have row : edge 6301 = SearchData.row6301.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan279_4 : scan edge (search edge 3) 4 [1831] 1831 innerTail279_4 best = best := by
  rw [innerTail279_4, scan, if_neg (by decide), innerCandidates279_4_checked]
  change scan edge (search edge 3) 4 [1831] 1831 innerTail279_5
    (if 3 ≤ innerCandidates279_4.length then search edge 3 [1831, 6301] innerCandidates279_4 8132 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan279_5

def innerTail279_3 : List Nat := 5197 :: innerTail279_4
def innerCandidates279_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates279_3_checked : candidates edge 4 1831 5197 innerTail279_4 best.bound = innerCandidates279_3 := by
  unfold candidates
  have row : edge 5197 = SearchData.row5197.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan279_3 : scan edge (search edge 3) 4 [1831] 1831 innerTail279_3 best = best := by
  rw [innerTail279_3, scan, if_neg (by decide), innerCandidates279_3_checked]
  change scan edge (search edge 3) 4 [1831] 1831 innerTail279_4
    (if 3 ≤ innerCandidates279_3.length then search edge 3 [1831, 5197] innerCandidates279_3 7028 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan279_4

def innerTail279_2 : List Nat := 3769 :: innerTail279_3
def innerCandidates279_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates279_2_checked : candidates edge 4 1831 3769 innerTail279_3 best.bound = innerCandidates279_2 := by
  unfold candidates
  have row : edge 3769 = SearchData.row3769.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan279_2 : scan edge (search edge 3) 4 [1831] 1831 innerTail279_2 best = best := by
  rw [innerTail279_2, scan, if_neg (by decide), innerCandidates279_2_checked]
  change scan edge (search edge 3) 4 [1831] 1831 innerTail279_3
    (if 3 ≤ innerCandidates279_2.length then search edge 3 [1831, 3769] innerCandidates279_2 5600 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan279_3

def innerTail279_1 : List Nat := 3259 :: innerTail279_2
def innerCandidates279_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates279_1_checked : candidates edge 4 1831 3259 innerTail279_2 best.bound = innerCandidates279_1 := by
  unfold candidates
  have row : edge 3259 = SearchData.row3259.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan279_1 : scan edge (search edge 3) 4 [1831] 1831 innerTail279_1 best = best := by
  rw [innerTail279_1, scan, if_neg (by decide), innerCandidates279_1_checked]
  change scan edge (search edge 3) 4 [1831] 1831 innerTail279_2
    (if 3 ≤ innerCandidates279_1.length then search edge 3 [1831, 3259] innerCandidates279_1 5090 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan279_2

def innerTail279_0 : List Nat := 2803 :: innerTail279_1
def innerCandidates279_0 : List Nat := [3769]
set_option maxRecDepth 4096 in
theorem innerCandidates279_0_checked : candidates edge 4 1831 2803 innerTail279_1 best.bound = innerCandidates279_0 := by
  unfold candidates
  have row : edge 2803 = SearchData.row2803.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan279_0 : scan edge (search edge 3) 4 [1831] 1831 innerTail279_0 best = best := by
  rw [innerTail279_0, scan, if_neg (by decide), innerCandidates279_0_checked]
  change scan edge (search edge 3) 4 [1831] 1831 innerTail279_1
    (if 3 ≤ innerCandidates279_0.length then search edge 3 [1831, 2803] innerCandidates279_0 4634 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan279_1

theorem rootBranch279 : search edge 4 [1831] rootCandidates279 1831 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1831] 1831 innerTail279_0 best = best
  exact innerScan279_0

end ProjectEulerSolutions.PrimePairUpper.Search
