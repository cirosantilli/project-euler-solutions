import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates364 : List Nat := [2663, 2801, 2939, 3797, 4073, 4133, 5039, 5381, 7703, 7853]
set_option maxRecDepth 4096 in
theorem rootCandidates364_checked : candidates edge 5 0 2477 tail365 best.bound = rootCandidates364 := by
  unfold candidates
  have row : edge 2477 = PrimePairCache.smallRow2477.contains := by funext q; rfl
  rw [row]
  decide

def innerTail364_10 : List Nat := []
theorem innerScan364_10 : scan edge (search edge 3) 4 [2477] 2477 innerTail364_10 best = best := rfl

def innerTail364_9 : List Nat := 7853 :: innerTail364_10
def innerCandidates364_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates364_9_checked : candidates edge 4 2477 7853 innerTail364_10 best.bound = innerCandidates364_9 := by
  unfold candidates
  have row : edge 7853 = SearchData.row7853.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan364_9 : scan edge (search edge 3) 4 [2477] 2477 innerTail364_9 best = best := by
  rw [innerTail364_9, scan, if_neg (by decide), innerCandidates364_9_checked]
  change scan edge (search edge 3) 4 [2477] 2477 innerTail364_10
    (if 3 ≤ innerCandidates364_9.length then search edge 3 [2477, 7853] innerCandidates364_9 10330 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan364_10

def innerTail364_8 : List Nat := 7703 :: innerTail364_9
def innerCandidates364_8 : List Nat := [7853]
set_option maxRecDepth 4096 in
theorem innerCandidates364_8_checked : candidates edge 4 2477 7703 innerTail364_9 best.bound = innerCandidates364_8 := by
  unfold candidates
  have row : edge 7703 = SearchData.row7703.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan364_8 : scan edge (search edge 3) 4 [2477] 2477 innerTail364_8 best = best := by
  rw [innerTail364_8, scan, if_neg (by decide), innerCandidates364_8_checked]
  change scan edge (search edge 3) 4 [2477] 2477 innerTail364_9
    (if 3 ≤ innerCandidates364_8.length then search edge 3 [2477, 7703] innerCandidates364_8 10180 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan364_9

def innerTail364_7 : List Nat := 5381 :: innerTail364_8
def innerCandidates364_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates364_7_checked : candidates edge 4 2477 5381 innerTail364_8 best.bound = innerCandidates364_7 := by
  unfold candidates
  have row : edge 5381 = SearchData.row5381.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan364_7 : scan edge (search edge 3) 4 [2477] 2477 innerTail364_7 best = best := by
  rw [innerTail364_7, scan, if_neg (by decide), innerCandidates364_7_checked]
  change scan edge (search edge 3) 4 [2477] 2477 innerTail364_8
    (if 3 ≤ innerCandidates364_7.length then search edge 3 [2477, 5381] innerCandidates364_7 7858 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan364_8

def innerTail364_6 : List Nat := 5039 :: innerTail364_7
def innerCandidates364_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates364_6_checked : candidates edge 4 2477 5039 innerTail364_7 best.bound = innerCandidates364_6 := by
  unfold candidates
  have row : edge 5039 = SearchData.row5039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan364_6 : scan edge (search edge 3) 4 [2477] 2477 innerTail364_6 best = best := by
  rw [innerTail364_6, scan, if_neg (by decide), innerCandidates364_6_checked]
  change scan edge (search edge 3) 4 [2477] 2477 innerTail364_7
    (if 3 ≤ innerCandidates364_6.length then search edge 3 [2477, 5039] innerCandidates364_6 7516 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan364_7

def innerTail364_5 : List Nat := 4133 :: innerTail364_6
def innerCandidates364_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates364_5_checked : candidates edge 4 2477 4133 innerTail364_6 best.bound = innerCandidates364_5 := by
  unfold candidates
  have row : edge 4133 = SearchData.row4133.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan364_5 : scan edge (search edge 3) 4 [2477] 2477 innerTail364_5 best = best := by
  rw [innerTail364_5, scan, if_neg (by decide), innerCandidates364_5_checked]
  change scan edge (search edge 3) 4 [2477] 2477 innerTail364_6
    (if 3 ≤ innerCandidates364_5.length then search edge 3 [2477, 4133] innerCandidates364_5 6610 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan364_6

def innerTail364_4 : List Nat := 4073 :: innerTail364_5
def innerCandidates364_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates364_4_checked : candidates edge 4 2477 4073 innerTail364_5 best.bound = innerCandidates364_4 := by
  unfold candidates
  have row : edge 4073 = SearchData.row4073.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan364_4 : scan edge (search edge 3) 4 [2477] 2477 innerTail364_4 best = best := by
  rw [innerTail364_4, scan, if_neg (by decide), innerCandidates364_4_checked]
  change scan edge (search edge 3) 4 [2477] 2477 innerTail364_5
    (if 3 ≤ innerCandidates364_4.length then search edge 3 [2477, 4073] innerCandidates364_4 6550 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan364_5

def innerTail364_3 : List Nat := 3797 :: innerTail364_4
def innerCandidates364_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates364_3_checked : candidates edge 4 2477 3797 innerTail364_4 best.bound = innerCandidates364_3 := by
  unfold candidates
  have row : edge 3797 = SearchData.row3797.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan364_3 : scan edge (search edge 3) 4 [2477] 2477 innerTail364_3 best = best := by
  rw [innerTail364_3, scan, if_neg (by decide), innerCandidates364_3_checked]
  change scan edge (search edge 3) 4 [2477] 2477 innerTail364_4
    (if 3 ≤ innerCandidates364_3.length then search edge 3 [2477, 3797] innerCandidates364_3 6274 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan364_4

def innerTail364_2 : List Nat := 2939 :: innerTail364_3
def innerCandidates364_2 : List Nat := [4133]
set_option maxRecDepth 4096 in
theorem innerCandidates364_2_checked : candidates edge 4 2477 2939 innerTail364_3 best.bound = innerCandidates364_2 := by
  unfold candidates
  have row : edge 2939 = SearchData.row2939.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan364_2 : scan edge (search edge 3) 4 [2477] 2477 innerTail364_2 best = best := by
  rw [innerTail364_2, scan, if_neg (by decide), innerCandidates364_2_checked]
  change scan edge (search edge 3) 4 [2477] 2477 innerTail364_3
    (if 3 ≤ innerCandidates364_2.length then search edge 3 [2477, 2939] innerCandidates364_2 5416 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan364_3

def innerTail364_1 : List Nat := 2801 :: innerTail364_2
def innerCandidates364_1 : List Nat := [4073]
set_option maxRecDepth 4096 in
theorem innerCandidates364_1_checked : candidates edge 4 2477 2801 innerTail364_2 best.bound = innerCandidates364_1 := by
  unfold candidates
  have row : edge 2801 = SearchData.row2801.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan364_1 : scan edge (search edge 3) 4 [2477] 2477 innerTail364_1 best = best := by
  rw [innerTail364_1, scan, if_neg (by decide), innerCandidates364_1_checked]
  change scan edge (search edge 3) 4 [2477] 2477 innerTail364_2
    (if 3 ≤ innerCandidates364_1.length then search edge 3 [2477, 2801] innerCandidates364_1 5278 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan364_2

def innerTail364_0 : List Nat := 2663 :: innerTail364_1
def innerCandidates364_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates364_0_checked : candidates edge 4 2477 2663 innerTail364_1 best.bound = innerCandidates364_0 := by
  unfold candidates
  have row : edge 2663 = PrimePairCache.smallRow2663.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan364_0 : scan edge (search edge 3) 4 [2477] 2477 innerTail364_0 best = best := by
  rw [innerTail364_0, scan, if_neg (by decide), innerCandidates364_0_checked]
  change scan edge (search edge 3) 4 [2477] 2477 innerTail364_1
    (if 3 ≤ innerCandidates364_0.length then search edge 3 [2477, 2663] innerCandidates364_0 5140 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan364_1

theorem rootBranch364 : search edge 4 [2477] rootCandidates364 2477 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2477] 2477 innerTail364_0 best = best
  exact innerScan364_0

def rootCandidates363 : List Nat := [2647, 2713, 2791, 3001, 3343, 3889, 4327, 5023, 5437, 5881, 6379, 6427, 7417, 8269]
set_option maxRecDepth 4096 in
theorem rootCandidates363_checked : candidates edge 5 0 2473 tail364 best.bound = rootCandidates363 := by
  unfold candidates
  have row : edge 2473 = PrimePairCache.smallRow2473.contains := by funext q; rfl
  rw [row]
  decide

def innerTail363_14 : List Nat := []
theorem innerScan363_14 : scan edge (search edge 3) 4 [2473] 2473 innerTail363_14 best = best := rfl

def innerTail363_13 : List Nat := 8269 :: innerTail363_14
def innerCandidates363_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates363_13_checked : candidates edge 4 2473 8269 innerTail363_14 best.bound = innerCandidates363_13 := by
  unfold candidates
  have row : edge 8269 = SearchData.row8269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan363_13 : scan edge (search edge 3) 4 [2473] 2473 innerTail363_13 best = best := by
  rw [innerTail363_13, scan, if_neg (by decide), innerCandidates363_13_checked]
  change scan edge (search edge 3) 4 [2473] 2473 innerTail363_14
    (if 3 ≤ innerCandidates363_13.length then search edge 3 [2473, 8269] innerCandidates363_13 10742 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan363_14

def innerTail363_12 : List Nat := 7417 :: innerTail363_13
def innerCandidates363_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates363_12_checked : candidates edge 4 2473 7417 innerTail363_13 best.bound = innerCandidates363_12 := by
  unfold candidates
  have row : edge 7417 = SearchData.row7417.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan363_12 : scan edge (search edge 3) 4 [2473] 2473 innerTail363_12 best = best := by
  rw [innerTail363_12, scan, if_neg (by decide), innerCandidates363_12_checked]
  change scan edge (search edge 3) 4 [2473] 2473 innerTail363_13
    (if 3 ≤ innerCandidates363_12.length then search edge 3 [2473, 7417] innerCandidates363_12 9890 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan363_13

def innerTail363_11 : List Nat := 6427 :: innerTail363_12
def innerCandidates363_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates363_11_checked : candidates edge 4 2473 6427 innerTail363_12 best.bound = innerCandidates363_11 := by
  unfold candidates
  have row : edge 6427 = SearchData.row6427.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan363_11 : scan edge (search edge 3) 4 [2473] 2473 innerTail363_11 best = best := by
  rw [innerTail363_11, scan, if_neg (by decide), innerCandidates363_11_checked]
  change scan edge (search edge 3) 4 [2473] 2473 innerTail363_12
    (if 3 ≤ innerCandidates363_11.length then search edge 3 [2473, 6427] innerCandidates363_11 8900 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan363_12

def innerTail363_10 : List Nat := 6379 :: innerTail363_11
def innerCandidates363_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates363_10_checked : candidates edge 4 2473 6379 innerTail363_11 best.bound = innerCandidates363_10 := by
  unfold candidates
  have row : edge 6379 = SearchData.row6379.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan363_10 : scan edge (search edge 3) 4 [2473] 2473 innerTail363_10 best = best := by
  rw [innerTail363_10, scan, if_neg (by decide), innerCandidates363_10_checked]
  change scan edge (search edge 3) 4 [2473] 2473 innerTail363_11
    (if 3 ≤ innerCandidates363_10.length then search edge 3 [2473, 6379] innerCandidates363_10 8852 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan363_11

def innerTail363_9 : List Nat := 5881 :: innerTail363_10
def innerCandidates363_9 : List Nat := [8269]
set_option maxRecDepth 4096 in
theorem innerCandidates363_9_checked : candidates edge 4 2473 5881 innerTail363_10 best.bound = innerCandidates363_9 := by
  unfold candidates
  have row : edge 5881 = SearchData.row5881.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan363_9 : scan edge (search edge 3) 4 [2473] 2473 innerTail363_9 best = best := by
  rw [innerTail363_9, scan, if_neg (by decide), innerCandidates363_9_checked]
  change scan edge (search edge 3) 4 [2473] 2473 innerTail363_10
    (if 3 ≤ innerCandidates363_9.length then search edge 3 [2473, 5881] innerCandidates363_9 8354 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan363_10

def innerTail363_8 : List Nat := 5437 :: innerTail363_9
def innerCandidates363_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates363_8_checked : candidates edge 4 2473 5437 innerTail363_9 best.bound = innerCandidates363_8 := by
  unfold candidates
  have row : edge 5437 = SearchData.row5437.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan363_8 : scan edge (search edge 3) 4 [2473] 2473 innerTail363_8 best = best := by
  rw [innerTail363_8, scan, if_neg (by decide), innerCandidates363_8_checked]
  change scan edge (search edge 3) 4 [2473] 2473 innerTail363_9
    (if 3 ≤ innerCandidates363_8.length then search edge 3 [2473, 5437] innerCandidates363_8 7910 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan363_9

def innerTail363_7 : List Nat := 5023 :: innerTail363_8
def innerCandidates363_7 : List Nat := [8269]
set_option maxRecDepth 4096 in
theorem innerCandidates363_7_checked : candidates edge 4 2473 5023 innerTail363_8 best.bound = innerCandidates363_7 := by
  unfold candidates
  have row : edge 5023 = SearchData.row5023.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan363_7 : scan edge (search edge 3) 4 [2473] 2473 innerTail363_7 best = best := by
  rw [innerTail363_7, scan, if_neg (by decide), innerCandidates363_7_checked]
  change scan edge (search edge 3) 4 [2473] 2473 innerTail363_8
    (if 3 ≤ innerCandidates363_7.length then search edge 3 [2473, 5023] innerCandidates363_7 7496 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan363_8

def innerTail363_6 : List Nat := 4327 :: innerTail363_7
def innerCandidates363_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates363_6_checked : candidates edge 4 2473 4327 innerTail363_7 best.bound = innerCandidates363_6 := by
  unfold candidates
  have row : edge 4327 = SearchData.row4327.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan363_6 : scan edge (search edge 3) 4 [2473] 2473 innerTail363_6 best = best := by
  rw [innerTail363_6, scan, if_neg (by decide), innerCandidates363_6_checked]
  change scan edge (search edge 3) 4 [2473] 2473 innerTail363_7
    (if 3 ≤ innerCandidates363_6.length then search edge 3 [2473, 4327] innerCandidates363_6 6800 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan363_7

def innerTail363_5 : List Nat := 3889 :: innerTail363_6
def innerCandidates363_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates363_5_checked : candidates edge 4 2473 3889 innerTail363_6 best.bound = innerCandidates363_5 := by
  unfold candidates
  have row : edge 3889 = SearchData.row3889.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan363_5 : scan edge (search edge 3) 4 [2473] 2473 innerTail363_5 best = best := by
  rw [innerTail363_5, scan, if_neg (by decide), innerCandidates363_5_checked]
  change scan edge (search edge 3) 4 [2473] 2473 innerTail363_6
    (if 3 ≤ innerCandidates363_5.length then search edge 3 [2473, 3889] innerCandidates363_5 6362 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan363_6

def innerTail363_4 : List Nat := 3343 :: innerTail363_5
def innerCandidates363_4 : List Nat := [6379]
set_option maxRecDepth 4096 in
theorem innerCandidates363_4_checked : candidates edge 4 2473 3343 innerTail363_5 best.bound = innerCandidates363_4 := by
  unfold candidates
  have row : edge 3343 = SearchData.row3343.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan363_4 : scan edge (search edge 3) 4 [2473] 2473 innerTail363_4 best = best := by
  rw [innerTail363_4, scan, if_neg (by decide), innerCandidates363_4_checked]
  change scan edge (search edge 3) 4 [2473] 2473 innerTail363_5
    (if 3 ≤ innerCandidates363_4.length then search edge 3 [2473, 3343] innerCandidates363_4 5816 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan363_5

def innerTail363_3 : List Nat := 3001 :: innerTail363_4
def innerCandidates363_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates363_3_checked : candidates edge 4 2473 3001 innerTail363_4 best.bound = innerCandidates363_3 := by
  unfold candidates
  have row : edge 3001 = SearchData.row3001.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan363_3 : scan edge (search edge 3) 4 [2473] 2473 innerTail363_3 best = best := by
  rw [innerTail363_3, scan, if_neg (by decide), innerCandidates363_3_checked]
  change scan edge (search edge 3) 4 [2473] 2473 innerTail363_4
    (if 3 ≤ innerCandidates363_3.length then search edge 3 [2473, 3001] innerCandidates363_3 5474 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan363_4

def innerTail363_2 : List Nat := 2791 :: innerTail363_3
def innerCandidates363_2 : List Nat := [3001, 6379]
set_option maxRecDepth 4096 in
theorem innerCandidates363_2_checked : candidates edge 4 2473 2791 innerTail363_3 best.bound = innerCandidates363_2 := by
  unfold candidates
  have row : edge 2791 = SearchData.row2791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan363_2 : scan edge (search edge 3) 4 [2473] 2473 innerTail363_2 best = best := by
  rw [innerTail363_2, scan, if_neg (by decide), innerCandidates363_2_checked]
  change scan edge (search edge 3) 4 [2473] 2473 innerTail363_3
    (if 3 ≤ innerCandidates363_2.length then search edge 3 [2473, 2791] innerCandidates363_2 5264 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan363_3

def innerTail363_1 : List Nat := 2713 :: innerTail363_2
def innerCandidates363_1 : List Nat := [3889]
set_option maxRecDepth 4096 in
theorem innerCandidates363_1_checked : candidates edge 4 2473 2713 innerTail363_2 best.bound = innerCandidates363_1 := by
  unfold candidates
  have row : edge 2713 = SearchData.row2713.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan363_1 : scan edge (search edge 3) 4 [2473] 2473 innerTail363_1 best = best := by
  rw [innerTail363_1, scan, if_neg (by decide), innerCandidates363_1_checked]
  change scan edge (search edge 3) 4 [2473] 2473 innerTail363_2
    (if 3 ≤ innerCandidates363_1.length then search edge 3 [2473, 2713] innerCandidates363_1 5186 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan363_2

def innerTail363_0 : List Nat := 2647 :: innerTail363_1
def innerCandidates363_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates363_0_checked : candidates edge 4 2473 2647 innerTail363_1 best.bound = innerCandidates363_0 := by
  unfold candidates
  have row : edge 2647 = PrimePairCache.smallRow2647.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan363_0 : scan edge (search edge 3) 4 [2473] 2473 innerTail363_0 best = best := by
  rw [innerTail363_0, scan, if_neg (by decide), innerCandidates363_0_checked]
  change scan edge (search edge 3) 4 [2473] 2473 innerTail363_1
    (if 3 ≤ innerCandidates363_0.length then search edge 3 [2473, 2647] innerCandidates363_0 5120 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan363_1

theorem rootBranch363 : search edge 4 [2473] rootCandidates363 2473 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2473] 2473 innerTail363_0 best = best
  exact innerScan363_0

def rootCandidates362 : List Nat := [2539, 2551, 3217, 4051, 4723, 4861, 5407, 5683, 5689, 6037, 6277, 6529, 7621, 7963, 8167, 8377]
set_option maxRecDepth 4096 in
theorem rootCandidates362_checked : candidates edge 5 0 2467 tail363 best.bound = rootCandidates362 := by
  unfold candidates
  have row : edge 2467 = PrimePairCache.smallRow2467.contains := by funext q; rfl
  rw [row]
  decide

def innerTail362_16 : List Nat := []
theorem innerScan362_16 : scan edge (search edge 3) 4 [2467] 2467 innerTail362_16 best = best := rfl

def innerTail362_15 : List Nat := 8377 :: innerTail362_16
def innerCandidates362_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates362_15_checked : candidates edge 4 2467 8377 innerTail362_16 best.bound = innerCandidates362_15 := by
  unfold candidates
  have row : edge 8377 = SearchData.row8377.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan362_15 : scan edge (search edge 3) 4 [2467] 2467 innerTail362_15 best = best := by
  rw [innerTail362_15, scan, if_neg (by decide), innerCandidates362_15_checked]
  change scan edge (search edge 3) 4 [2467] 2467 innerTail362_16
    (if 3 ≤ innerCandidates362_15.length then search edge 3 [2467, 8377] innerCandidates362_15 10844 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan362_16

def innerTail362_14 : List Nat := 8167 :: innerTail362_15
def innerCandidates362_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates362_14_checked : candidates edge 4 2467 8167 innerTail362_15 best.bound = innerCandidates362_14 := by
  unfold candidates
  have row : edge 8167 = SearchData.row8167.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan362_14 : scan edge (search edge 3) 4 [2467] 2467 innerTail362_14 best = best := by
  rw [innerTail362_14, scan, if_neg (by decide), innerCandidates362_14_checked]
  change scan edge (search edge 3) 4 [2467] 2467 innerTail362_15
    (if 3 ≤ innerCandidates362_14.length then search edge 3 [2467, 8167] innerCandidates362_14 10634 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan362_15

def innerTail362_13 : List Nat := 7963 :: innerTail362_14
def innerCandidates362_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates362_13_checked : candidates edge 4 2467 7963 innerTail362_14 best.bound = innerCandidates362_13 := by
  unfold candidates
  have row : edge 7963 = SearchData.row7963.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan362_13 : scan edge (search edge 3) 4 [2467] 2467 innerTail362_13 best = best := by
  rw [innerTail362_13, scan, if_neg (by decide), innerCandidates362_13_checked]
  change scan edge (search edge 3) 4 [2467] 2467 innerTail362_14
    (if 3 ≤ innerCandidates362_13.length then search edge 3 [2467, 7963] innerCandidates362_13 10430 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan362_14

def innerTail362_12 : List Nat := 7621 :: innerTail362_13
def innerCandidates362_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates362_12_checked : candidates edge 4 2467 7621 innerTail362_13 best.bound = innerCandidates362_12 := by
  unfold candidates
  have row : edge 7621 = SearchData.row7621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan362_12 : scan edge (search edge 3) 4 [2467] 2467 innerTail362_12 best = best := by
  rw [innerTail362_12, scan, if_neg (by decide), innerCandidates362_12_checked]
  change scan edge (search edge 3) 4 [2467] 2467 innerTail362_13
    (if 3 ≤ innerCandidates362_12.length then search edge 3 [2467, 7621] innerCandidates362_12 10088 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan362_13

def innerTail362_11 : List Nat := 6529 :: innerTail362_12
def innerCandidates362_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates362_11_checked : candidates edge 4 2467 6529 innerTail362_12 best.bound = innerCandidates362_11 := by
  unfold candidates
  have row : edge 6529 = SearchData.row6529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan362_11 : scan edge (search edge 3) 4 [2467] 2467 innerTail362_11 best = best := by
  rw [innerTail362_11, scan, if_neg (by decide), innerCandidates362_11_checked]
  change scan edge (search edge 3) 4 [2467] 2467 innerTail362_12
    (if 3 ≤ innerCandidates362_11.length then search edge 3 [2467, 6529] innerCandidates362_11 8996 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan362_12

def innerTail362_10 : List Nat := 6277 :: innerTail362_11
def innerCandidates362_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates362_10_checked : candidates edge 4 2467 6277 innerTail362_11 best.bound = innerCandidates362_10 := by
  unfold candidates
  have row : edge 6277 = SearchData.row6277.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan362_10 : scan edge (search edge 3) 4 [2467] 2467 innerTail362_10 best = best := by
  rw [innerTail362_10, scan, if_neg (by decide), innerCandidates362_10_checked]
  change scan edge (search edge 3) 4 [2467] 2467 innerTail362_11
    (if 3 ≤ innerCandidates362_10.length then search edge 3 [2467, 6277] innerCandidates362_10 8744 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan362_11

def innerTail362_9 : List Nat := 6037 :: innerTail362_10
def innerCandidates362_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates362_9_checked : candidates edge 4 2467 6037 innerTail362_10 best.bound = innerCandidates362_9 := by
  unfold candidates
  have row : edge 6037 = SearchData.row6037.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan362_9 : scan edge (search edge 3) 4 [2467] 2467 innerTail362_9 best = best := by
  rw [innerTail362_9, scan, if_neg (by decide), innerCandidates362_9_checked]
  change scan edge (search edge 3) 4 [2467] 2467 innerTail362_10
    (if 3 ≤ innerCandidates362_9.length then search edge 3 [2467, 6037] innerCandidates362_9 8504 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan362_10

def innerTail362_8 : List Nat := 5689 :: innerTail362_9
def innerCandidates362_8 : List Nat := [6277]
set_option maxRecDepth 4096 in
theorem innerCandidates362_8_checked : candidates edge 4 2467 5689 innerTail362_9 best.bound = innerCandidates362_8 := by
  unfold candidates
  have row : edge 5689 = SearchData.row5689.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan362_8 : scan edge (search edge 3) 4 [2467] 2467 innerTail362_8 best = best := by
  rw [innerTail362_8, scan, if_neg (by decide), innerCandidates362_8_checked]
  change scan edge (search edge 3) 4 [2467] 2467 innerTail362_9
    (if 3 ≤ innerCandidates362_8.length then search edge 3 [2467, 5689] innerCandidates362_8 8156 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan362_9

def innerTail362_7 : List Nat := 5683 :: innerTail362_8
def innerCandidates362_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates362_7_checked : candidates edge 4 2467 5683 innerTail362_8 best.bound = innerCandidates362_7 := by
  unfold candidates
  have row : edge 5683 = SearchData.row5683.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan362_7 : scan edge (search edge 3) 4 [2467] 2467 innerTail362_7 best = best := by
  rw [innerTail362_7, scan, if_neg (by decide), innerCandidates362_7_checked]
  change scan edge (search edge 3) 4 [2467] 2467 innerTail362_8
    (if 3 ≤ innerCandidates362_7.length then search edge 3 [2467, 5683] innerCandidates362_7 8150 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan362_8

def innerTail362_6 : List Nat := 5407 :: innerTail362_7
def innerCandidates362_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates362_6_checked : candidates edge 4 2467 5407 innerTail362_7 best.bound = innerCandidates362_6 := by
  unfold candidates
  have row : edge 5407 = SearchData.row5407.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan362_6 : scan edge (search edge 3) 4 [2467] 2467 innerTail362_6 best = best := by
  rw [innerTail362_6, scan, if_neg (by decide), innerCandidates362_6_checked]
  change scan edge (search edge 3) 4 [2467] 2467 innerTail362_7
    (if 3 ≤ innerCandidates362_6.length then search edge 3 [2467, 5407] innerCandidates362_6 7874 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan362_7

def innerTail362_5 : List Nat := 4861 :: innerTail362_6
def innerCandidates362_5 : List Nat := [5689]
set_option maxRecDepth 4096 in
theorem innerCandidates362_5_checked : candidates edge 4 2467 4861 innerTail362_6 best.bound = innerCandidates362_5 := by
  unfold candidates
  have row : edge 4861 = SearchData.row4861.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan362_5 : scan edge (search edge 3) 4 [2467] 2467 innerTail362_5 best = best := by
  rw [innerTail362_5, scan, if_neg (by decide), innerCandidates362_5_checked]
  change scan edge (search edge 3) 4 [2467] 2467 innerTail362_6
    (if 3 ≤ innerCandidates362_5.length then search edge 3 [2467, 4861] innerCandidates362_5 7328 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan362_6

def innerTail362_4 : List Nat := 4723 :: innerTail362_5
def innerCandidates362_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates362_4_checked : candidates edge 4 2467 4723 innerTail362_5 best.bound = innerCandidates362_4 := by
  unfold candidates
  have row : edge 4723 = SearchData.row4723.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan362_4 : scan edge (search edge 3) 4 [2467] 2467 innerTail362_4 best = best := by
  rw [innerTail362_4, scan, if_neg (by decide), innerCandidates362_4_checked]
  change scan edge (search edge 3) 4 [2467] 2467 innerTail362_5
    (if 3 ≤ innerCandidates362_4.length then search edge 3 [2467, 4723] innerCandidates362_4 7190 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan362_5

def innerTail362_3 : List Nat := 4051 :: innerTail362_4
def innerCandidates362_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates362_3_checked : candidates edge 4 2467 4051 innerTail362_4 best.bound = innerCandidates362_3 := by
  unfold candidates
  have row : edge 4051 = SearchData.row4051.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan362_3 : scan edge (search edge 3) 4 [2467] 2467 innerTail362_3 best = best := by
  rw [innerTail362_3, scan, if_neg (by decide), innerCandidates362_3_checked]
  change scan edge (search edge 3) 4 [2467] 2467 innerTail362_4
    (if 3 ≤ innerCandidates362_3.length then search edge 3 [2467, 4051] innerCandidates362_3 6518 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan362_4

def innerTail362_2 : List Nat := 3217 :: innerTail362_3
def innerCandidates362_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates362_2_checked : candidates edge 4 2467 3217 innerTail362_3 best.bound = innerCandidates362_2 := by
  unfold candidates
  have row : edge 3217 = SearchData.row3217.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan362_2 : scan edge (search edge 3) 4 [2467] 2467 innerTail362_2 best = best := by
  rw [innerTail362_2, scan, if_neg (by decide), innerCandidates362_2_checked]
  change scan edge (search edge 3) 4 [2467] 2467 innerTail362_3
    (if 3 ≤ innerCandidates362_2.length then search edge 3 [2467, 3217] innerCandidates362_2 5684 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan362_3

def innerTail362_1 : List Nat := 2551 :: innerTail362_2
def innerCandidates362_1 : List Nat := [6037]
set_option maxRecDepth 4096 in
theorem innerCandidates362_1_checked : candidates edge 4 2467 2551 innerTail362_2 best.bound = innerCandidates362_1 := by
  unfold candidates
  have row : edge 2551 = PrimePairCache.smallRow2551.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan362_1 : scan edge (search edge 3) 4 [2467] 2467 innerTail362_1 best = best := by
  rw [innerTail362_1, scan, if_neg (by decide), innerCandidates362_1_checked]
  change scan edge (search edge 3) 4 [2467] 2467 innerTail362_2
    (if 3 ≤ innerCandidates362_1.length then search edge 3 [2467, 2551] innerCandidates362_1 5018 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan362_2

def innerTail362_0 : List Nat := 2539 :: innerTail362_1
def innerCandidates362_0 : List Nat := [4723]
set_option maxRecDepth 4096 in
theorem innerCandidates362_0_checked : candidates edge 4 2467 2539 innerTail362_1 best.bound = innerCandidates362_0 := by
  unfold candidates
  have row : edge 2539 = PrimePairCache.smallRow2539.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan362_0 : scan edge (search edge 3) 4 [2467] 2467 innerTail362_0 best = best := by
  rw [innerTail362_0, scan, if_neg (by decide), innerCandidates362_0_checked]
  change scan edge (search edge 3) 4 [2467] 2467 innerTail362_1
    (if 3 ≤ innerCandidates362_0.length then search edge 3 [2467, 2539] innerCandidates362_0 5006 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan362_1

theorem rootBranch362 : search edge 4 [2467] rootCandidates362 2467 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2467] 2467 innerTail362_0 best = best
  exact innerScan362_0

def rootCandidates361 : List Nat := [2657, 2927, 4139, 4289, 4463, 4919, 5171, 5843, 6101, 6353, 6899, 7211, 8009, 8069]
set_option maxRecDepth 4096 in
theorem rootCandidates361_checked : candidates edge 5 0 2459 tail362 best.bound = rootCandidates361 := by
  unfold candidates
  have row : edge 2459 = PrimePairCache.smallRow2459.contains := by funext q; rfl
  rw [row]
  decide

def innerTail361_14 : List Nat := []
theorem innerScan361_14 : scan edge (search edge 3) 4 [2459] 2459 innerTail361_14 best = best := rfl

def innerTail361_13 : List Nat := 8069 :: innerTail361_14
def innerCandidates361_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates361_13_checked : candidates edge 4 2459 8069 innerTail361_14 best.bound = innerCandidates361_13 := by
  unfold candidates
  have row : edge 8069 = SearchData.row8069.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan361_13 : scan edge (search edge 3) 4 [2459] 2459 innerTail361_13 best = best := by
  rw [innerTail361_13, scan, if_neg (by decide), innerCandidates361_13_checked]
  change scan edge (search edge 3) 4 [2459] 2459 innerTail361_14
    (if 3 ≤ innerCandidates361_13.length then search edge 3 [2459, 8069] innerCandidates361_13 10528 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan361_14

def innerTail361_12 : List Nat := 8009 :: innerTail361_13
def innerCandidates361_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates361_12_checked : candidates edge 4 2459 8009 innerTail361_13 best.bound = innerCandidates361_12 := by
  unfold candidates
  have row : edge 8009 = SearchData.row8009.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan361_12 : scan edge (search edge 3) 4 [2459] 2459 innerTail361_12 best = best := by
  rw [innerTail361_12, scan, if_neg (by decide), innerCandidates361_12_checked]
  change scan edge (search edge 3) 4 [2459] 2459 innerTail361_13
    (if 3 ≤ innerCandidates361_12.length then search edge 3 [2459, 8009] innerCandidates361_12 10468 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan361_13

def innerTail361_11 : List Nat := 7211 :: innerTail361_12
def innerCandidates361_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates361_11_checked : candidates edge 4 2459 7211 innerTail361_12 best.bound = innerCandidates361_11 := by
  unfold candidates
  have row : edge 7211 = SearchData.row7211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan361_11 : scan edge (search edge 3) 4 [2459] 2459 innerTail361_11 best = best := by
  rw [innerTail361_11, scan, if_neg (by decide), innerCandidates361_11_checked]
  change scan edge (search edge 3) 4 [2459] 2459 innerTail361_12
    (if 3 ≤ innerCandidates361_11.length then search edge 3 [2459, 7211] innerCandidates361_11 9670 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan361_12

def innerTail361_10 : List Nat := 6899 :: innerTail361_11
def innerCandidates361_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates361_10_checked : candidates edge 4 2459 6899 innerTail361_11 best.bound = innerCandidates361_10 := by
  unfold candidates
  have row : edge 6899 = SearchData.row6899.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan361_10 : scan edge (search edge 3) 4 [2459] 2459 innerTail361_10 best = best := by
  rw [innerTail361_10, scan, if_neg (by decide), innerCandidates361_10_checked]
  change scan edge (search edge 3) 4 [2459] 2459 innerTail361_11
    (if 3 ≤ innerCandidates361_10.length then search edge 3 [2459, 6899] innerCandidates361_10 9358 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan361_11

def innerTail361_9 : List Nat := 6353 :: innerTail361_10
def innerCandidates361_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates361_9_checked : candidates edge 4 2459 6353 innerTail361_10 best.bound = innerCandidates361_9 := by
  unfold candidates
  have row : edge 6353 = SearchData.row6353.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan361_9 : scan edge (search edge 3) 4 [2459] 2459 innerTail361_9 best = best := by
  rw [innerTail361_9, scan, if_neg (by decide), innerCandidates361_9_checked]
  change scan edge (search edge 3) 4 [2459] 2459 innerTail361_10
    (if 3 ≤ innerCandidates361_9.length then search edge 3 [2459, 6353] innerCandidates361_9 8812 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan361_10

def innerTail361_8 : List Nat := 6101 :: innerTail361_9
def innerCandidates361_8 : List Nat := [7211]
set_option maxRecDepth 4096 in
theorem innerCandidates361_8_checked : candidates edge 4 2459 6101 innerTail361_9 best.bound = innerCandidates361_8 := by
  unfold candidates
  have row : edge 6101 = SearchData.row6101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan361_8 : scan edge (search edge 3) 4 [2459] 2459 innerTail361_8 best = best := by
  rw [innerTail361_8, scan, if_neg (by decide), innerCandidates361_8_checked]
  change scan edge (search edge 3) 4 [2459] 2459 innerTail361_9
    (if 3 ≤ innerCandidates361_8.length then search edge 3 [2459, 6101] innerCandidates361_8 8560 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan361_9

def innerTail361_7 : List Nat := 5843 :: innerTail361_8
def innerCandidates361_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates361_7_checked : candidates edge 4 2459 5843 innerTail361_8 best.bound = innerCandidates361_7 := by
  unfold candidates
  have row : edge 5843 = SearchData.row5843.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan361_7 : scan edge (search edge 3) 4 [2459] 2459 innerTail361_7 best = best := by
  rw [innerTail361_7, scan, if_neg (by decide), innerCandidates361_7_checked]
  change scan edge (search edge 3) 4 [2459] 2459 innerTail361_8
    (if 3 ≤ innerCandidates361_7.length then search edge 3 [2459, 5843] innerCandidates361_7 8302 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan361_8

def innerTail361_6 : List Nat := 5171 :: innerTail361_7
def innerCandidates361_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates361_6_checked : candidates edge 4 2459 5171 innerTail361_7 best.bound = innerCandidates361_6 := by
  unfold candidates
  have row : edge 5171 = SearchData.row5171.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan361_6 : scan edge (search edge 3) 4 [2459] 2459 innerTail361_6 best = best := by
  rw [innerTail361_6, scan, if_neg (by decide), innerCandidates361_6_checked]
  change scan edge (search edge 3) 4 [2459] 2459 innerTail361_7
    (if 3 ≤ innerCandidates361_6.length then search edge 3 [2459, 5171] innerCandidates361_6 7630 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan361_7

def innerTail361_5 : List Nat := 4919 :: innerTail361_6
def innerCandidates361_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates361_5_checked : candidates edge 4 2459 4919 innerTail361_6 best.bound = innerCandidates361_5 := by
  unfold candidates
  have row : edge 4919 = SearchData.row4919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan361_5 : scan edge (search edge 3) 4 [2459] 2459 innerTail361_5 best = best := by
  rw [innerTail361_5, scan, if_neg (by decide), innerCandidates361_5_checked]
  change scan edge (search edge 3) 4 [2459] 2459 innerTail361_6
    (if 3 ≤ innerCandidates361_5.length then search edge 3 [2459, 4919] innerCandidates361_5 7378 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan361_6

def innerTail361_4 : List Nat := 4463 :: innerTail361_5
def innerCandidates361_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates361_4_checked : candidates edge 4 2459 4463 innerTail361_5 best.bound = innerCandidates361_4 := by
  unfold candidates
  have row : edge 4463 = SearchData.row4463.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan361_4 : scan edge (search edge 3) 4 [2459] 2459 innerTail361_4 best = best := by
  rw [innerTail361_4, scan, if_neg (by decide), innerCandidates361_4_checked]
  change scan edge (search edge 3) 4 [2459] 2459 innerTail361_5
    (if 3 ≤ innerCandidates361_4.length then search edge 3 [2459, 4463] innerCandidates361_4 6922 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan361_5

def innerTail361_3 : List Nat := 4289 :: innerTail361_4
def innerCandidates361_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates361_3_checked : candidates edge 4 2459 4289 innerTail361_4 best.bound = innerCandidates361_3 := by
  unfold candidates
  have row : edge 4289 = SearchData.row4289.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan361_3 : scan edge (search edge 3) 4 [2459] 2459 innerTail361_3 best = best := by
  rw [innerTail361_3, scan, if_neg (by decide), innerCandidates361_3_checked]
  change scan edge (search edge 3) 4 [2459] 2459 innerTail361_4
    (if 3 ≤ innerCandidates361_3.length then search edge 3 [2459, 4289] innerCandidates361_3 6748 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan361_4

def innerTail361_2 : List Nat := 4139 :: innerTail361_3
def innerCandidates361_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates361_2_checked : candidates edge 4 2459 4139 innerTail361_3 best.bound = innerCandidates361_2 := by
  unfold candidates
  have row : edge 4139 = SearchData.row4139.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan361_2 : scan edge (search edge 3) 4 [2459] 2459 innerTail361_2 best = best := by
  rw [innerTail361_2, scan, if_neg (by decide), innerCandidates361_2_checked]
  change scan edge (search edge 3) 4 [2459] 2459 innerTail361_3
    (if 3 ≤ innerCandidates361_2.length then search edge 3 [2459, 4139] innerCandidates361_2 6598 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan361_3

def innerTail361_1 : List Nat := 2927 :: innerTail361_2
def innerCandidates361_1 : List Nat := [8009]
set_option maxRecDepth 4096 in
theorem innerCandidates361_1_checked : candidates edge 4 2459 2927 innerTail361_2 best.bound = innerCandidates361_1 := by
  unfold candidates
  have row : edge 2927 = SearchData.row2927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan361_1 : scan edge (search edge 3) 4 [2459] 2459 innerTail361_1 best = best := by
  rw [innerTail361_1, scan, if_neg (by decide), innerCandidates361_1_checked]
  change scan edge (search edge 3) 4 [2459] 2459 innerTail361_2
    (if 3 ≤ innerCandidates361_1.length then search edge 3 [2459, 2927] innerCandidates361_1 5386 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan361_2

def innerTail361_0 : List Nat := 2657 :: innerTail361_1
def innerCandidates361_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates361_0_checked : candidates edge 4 2459 2657 innerTail361_1 best.bound = innerCandidates361_0 := by
  unfold candidates
  have row : edge 2657 = PrimePairCache.smallRow2657.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan361_0 : scan edge (search edge 3) 4 [2459] 2459 innerTail361_0 best = best := by
  rw [innerTail361_0, scan, if_neg (by decide), innerCandidates361_0_checked]
  change scan edge (search edge 3) 4 [2459] 2459 innerTail361_1
    (if 3 ≤ innerCandidates361_0.length then search edge 3 [2459, 2657] innerCandidates361_0 5116 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan361_1

theorem rootBranch361 : search edge 4 [2459] rootCandidates361 2459 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2459] 2459 innerTail361_0 best = best
  exact innerScan361_0

end ProjectEulerSolutions.PrimePairUpper.Search
