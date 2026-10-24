import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates349 : List Nat := [2557, 2797, 3067, 3319, 3433, 4159, 4327, 5743, 6073, 6199, 6553, 7603]
set_option maxRecDepth 4096 in
theorem rootCandidates349_checked : candidates edge 5 0 2377 tail350 best.bound = rootCandidates349 := by
  unfold candidates
  have row : edge 2377 = PrimePairCache.smallRow2377.contains := by funext q; rfl
  rw [row]
  decide

def innerTail349_12 : List Nat := []
theorem innerScan349_12 : scan edge (search edge 3) 4 [2377] 2377 innerTail349_12 best = best := rfl

def innerTail349_11 : List Nat := 7603 :: innerTail349_12
def innerCandidates349_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates349_11_checked : candidates edge 4 2377 7603 innerTail349_12 best.bound = innerCandidates349_11 := by
  unfold candidates
  have row : edge 7603 = SearchData.row7603.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan349_11 : scan edge (search edge 3) 4 [2377] 2377 innerTail349_11 best = best := by
  rw [innerTail349_11, scan, if_neg (by decide), innerCandidates349_11_checked]
  change scan edge (search edge 3) 4 [2377] 2377 innerTail349_12
    (if 3 ≤ innerCandidates349_11.length then search edge 3 [2377, 7603] innerCandidates349_11 9980 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan349_12

def innerTail349_10 : List Nat := 6553 :: innerTail349_11
def innerCandidates349_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates349_10_checked : candidates edge 4 2377 6553 innerTail349_11 best.bound = innerCandidates349_10 := by
  unfold candidates
  have row : edge 6553 = SearchData.row6553.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan349_10 : scan edge (search edge 3) 4 [2377] 2377 innerTail349_10 best = best := by
  rw [innerTail349_10, scan, if_neg (by decide), innerCandidates349_10_checked]
  change scan edge (search edge 3) 4 [2377] 2377 innerTail349_11
    (if 3 ≤ innerCandidates349_10.length then search edge 3 [2377, 6553] innerCandidates349_10 8930 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan349_11

def innerTail349_9 : List Nat := 6199 :: innerTail349_10
def innerCandidates349_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates349_9_checked : candidates edge 4 2377 6199 innerTail349_10 best.bound = innerCandidates349_9 := by
  unfold candidates
  have row : edge 6199 = SearchData.row6199.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan349_9 : scan edge (search edge 3) 4 [2377] 2377 innerTail349_9 best = best := by
  rw [innerTail349_9, scan, if_neg (by decide), innerCandidates349_9_checked]
  change scan edge (search edge 3) 4 [2377] 2377 innerTail349_10
    (if 3 ≤ innerCandidates349_9.length then search edge 3 [2377, 6199] innerCandidates349_9 8576 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan349_10

def innerTail349_8 : List Nat := 6073 :: innerTail349_9
def innerCandidates349_8 : List Nat := [6199]
set_option maxRecDepth 4096 in
theorem innerCandidates349_8_checked : candidates edge 4 2377 6073 innerTail349_9 best.bound = innerCandidates349_8 := by
  unfold candidates
  have row : edge 6073 = SearchData.row6073.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan349_8 : scan edge (search edge 3) 4 [2377] 2377 innerTail349_8 best = best := by
  rw [innerTail349_8, scan, if_neg (by decide), innerCandidates349_8_checked]
  change scan edge (search edge 3) 4 [2377] 2377 innerTail349_9
    (if 3 ≤ innerCandidates349_8.length then search edge 3 [2377, 6073] innerCandidates349_8 8450 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan349_9

def innerTail349_7 : List Nat := 5743 :: innerTail349_8
def innerCandidates349_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates349_7_checked : candidates edge 4 2377 5743 innerTail349_8 best.bound = innerCandidates349_7 := by
  unfold candidates
  have row : edge 5743 = SearchData.row5743.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan349_7 : scan edge (search edge 3) 4 [2377] 2377 innerTail349_7 best = best := by
  rw [innerTail349_7, scan, if_neg (by decide), innerCandidates349_7_checked]
  change scan edge (search edge 3) 4 [2377] 2377 innerTail349_8
    (if 3 ≤ innerCandidates349_7.length then search edge 3 [2377, 5743] innerCandidates349_7 8120 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan349_8

def innerTail349_6 : List Nat := 4327 :: innerTail349_7
def innerCandidates349_6 : List Nat := [6553]
set_option maxRecDepth 4096 in
theorem innerCandidates349_6_checked : candidates edge 4 2377 4327 innerTail349_7 best.bound = innerCandidates349_6 := by
  unfold candidates
  have row : edge 4327 = SearchData.row4327.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan349_6 : scan edge (search edge 3) 4 [2377] 2377 innerTail349_6 best = best := by
  rw [innerTail349_6, scan, if_neg (by decide), innerCandidates349_6_checked]
  change scan edge (search edge 3) 4 [2377] 2377 innerTail349_7
    (if 3 ≤ innerCandidates349_6.length then search edge 3 [2377, 4327] innerCandidates349_6 6704 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan349_7

def innerTail349_5 : List Nat := 4159 :: innerTail349_6
def innerCandidates349_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates349_5_checked : candidates edge 4 2377 4159 innerTail349_6 best.bound = innerCandidates349_5 := by
  unfold candidates
  have row : edge 4159 = SearchData.row4159.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan349_5 : scan edge (search edge 3) 4 [2377] 2377 innerTail349_5 best = best := by
  rw [innerTail349_5, scan, if_neg (by decide), innerCandidates349_5_checked]
  change scan edge (search edge 3) 4 [2377] 2377 innerTail349_6
    (if 3 ≤ innerCandidates349_5.length then search edge 3 [2377, 4159] innerCandidates349_5 6536 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan349_6

def innerTail349_4 : List Nat := 3433 :: innerTail349_5
def innerCandidates349_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates349_4_checked : candidates edge 4 2377 3433 innerTail349_5 best.bound = innerCandidates349_4 := by
  unfold candidates
  have row : edge 3433 = SearchData.row3433.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan349_4 : scan edge (search edge 3) 4 [2377] 2377 innerTail349_4 best = best := by
  rw [innerTail349_4, scan, if_neg (by decide), innerCandidates349_4_checked]
  change scan edge (search edge 3) 4 [2377] 2377 innerTail349_5
    (if 3 ≤ innerCandidates349_4.length then search edge 3 [2377, 3433] innerCandidates349_4 5810 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan349_5

def innerTail349_3 : List Nat := 3319 :: innerTail349_4
def innerCandidates349_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates349_3_checked : candidates edge 4 2377 3319 innerTail349_4 best.bound = innerCandidates349_3 := by
  unfold candidates
  have row : edge 3319 = SearchData.row3319.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan349_3 : scan edge (search edge 3) 4 [2377] 2377 innerTail349_3 best = best := by
  rw [innerTail349_3, scan, if_neg (by decide), innerCandidates349_3_checked]
  change scan edge (search edge 3) 4 [2377] 2377 innerTail349_4
    (if 3 ≤ innerCandidates349_3.length then search edge 3 [2377, 3319] innerCandidates349_3 5696 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan349_4

def innerTail349_2 : List Nat := 3067 :: innerTail349_3
def innerCandidates349_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates349_2_checked : candidates edge 4 2377 3067 innerTail349_3 best.bound = innerCandidates349_2 := by
  unfold candidates
  have row : edge 3067 = SearchData.row3067.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan349_2 : scan edge (search edge 3) 4 [2377] 2377 innerTail349_2 best = best := by
  rw [innerTail349_2, scan, if_neg (by decide), innerCandidates349_2_checked]
  change scan edge (search edge 3) 4 [2377] 2377 innerTail349_3
    (if 3 ≤ innerCandidates349_2.length then search edge 3 [2377, 3067] innerCandidates349_2 5444 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan349_3

def innerTail349_1 : List Nat := 2797 :: innerTail349_2
def innerCandidates349_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates349_1_checked : candidates edge 4 2377 2797 innerTail349_2 best.bound = innerCandidates349_1 := by
  unfold candidates
  have row : edge 2797 = SearchData.row2797.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan349_1 : scan edge (search edge 3) 4 [2377] 2377 innerTail349_1 best = best := by
  rw [innerTail349_1, scan, if_neg (by decide), innerCandidates349_1_checked]
  change scan edge (search edge 3) 4 [2377] 2377 innerTail349_2
    (if 3 ≤ innerCandidates349_1.length then search edge 3 [2377, 2797] innerCandidates349_1 5174 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan349_2

def innerTail349_0 : List Nat := 2557 :: innerTail349_1
def innerCandidates349_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates349_0_checked : candidates edge 4 2377 2557 innerTail349_1 best.bound = innerCandidates349_0 := by
  unfold candidates
  have row : edge 2557 = PrimePairCache.smallRow2557.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan349_0 : scan edge (search edge 3) 4 [2377] 2377 innerTail349_0 best = best := by
  rw [innerTail349_0, scan, if_neg (by decide), innerCandidates349_0_checked]
  change scan edge (search edge 3) 4 [2377] 2377 innerTail349_1
    (if 3 ≤ innerCandidates349_0.length then search edge 3 [2377, 2557] innerCandidates349_0 4934 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan349_1

theorem rootBranch349 : search edge 4 [2377] rootCandidates349 2377 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2377] 2377 innerTail349_0 best = best
  exact innerScan349_0

def rootCandidates348 : List Nat := [2917, 3169, 3229, 3607, 3643, 4021, 4423, 7219, 8287]
set_option maxRecDepth 4096 in
theorem rootCandidates348_checked : candidates edge 5 0 2371 tail349 best.bound = rootCandidates348 := by
  unfold candidates
  have row : edge 2371 = PrimePairCache.smallRow2371.contains := by funext q; rfl
  rw [row]
  decide

def innerTail348_9 : List Nat := []
theorem innerScan348_9 : scan edge (search edge 3) 4 [2371] 2371 innerTail348_9 best = best := rfl

def innerTail348_8 : List Nat := 8287 :: innerTail348_9
def innerCandidates348_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates348_8_checked : candidates edge 4 2371 8287 innerTail348_9 best.bound = innerCandidates348_8 := by
  unfold candidates
  have row : edge 8287 = SearchData.row8287.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan348_8 : scan edge (search edge 3) 4 [2371] 2371 innerTail348_8 best = best := by
  rw [innerTail348_8, scan, if_neg (by decide), innerCandidates348_8_checked]
  change scan edge (search edge 3) 4 [2371] 2371 innerTail348_9
    (if 3 ≤ innerCandidates348_8.length then search edge 3 [2371, 8287] innerCandidates348_8 10658 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan348_9

def innerTail348_7 : List Nat := 7219 :: innerTail348_8
def innerCandidates348_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates348_7_checked : candidates edge 4 2371 7219 innerTail348_8 best.bound = innerCandidates348_7 := by
  unfold candidates
  have row : edge 7219 = SearchData.row7219.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan348_7 : scan edge (search edge 3) 4 [2371] 2371 innerTail348_7 best = best := by
  rw [innerTail348_7, scan, if_neg (by decide), innerCandidates348_7_checked]
  change scan edge (search edge 3) 4 [2371] 2371 innerTail348_8
    (if 3 ≤ innerCandidates348_7.length then search edge 3 [2371, 7219] innerCandidates348_7 9590 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan348_8

def innerTail348_6 : List Nat := 4423 :: innerTail348_7
def innerCandidates348_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates348_6_checked : candidates edge 4 2371 4423 innerTail348_7 best.bound = innerCandidates348_6 := by
  unfold candidates
  have row : edge 4423 = SearchData.row4423.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan348_6 : scan edge (search edge 3) 4 [2371] 2371 innerTail348_6 best = best := by
  rw [innerTail348_6, scan, if_neg (by decide), innerCandidates348_6_checked]
  change scan edge (search edge 3) 4 [2371] 2371 innerTail348_7
    (if 3 ≤ innerCandidates348_6.length then search edge 3 [2371, 4423] innerCandidates348_6 6794 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan348_7

def innerTail348_5 : List Nat := 4021 :: innerTail348_6
def innerCandidates348_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates348_5_checked : candidates edge 4 2371 4021 innerTail348_6 best.bound = innerCandidates348_5 := by
  unfold candidates
  have row : edge 4021 = SearchData.row4021.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan348_5 : scan edge (search edge 3) 4 [2371] 2371 innerTail348_5 best = best := by
  rw [innerTail348_5, scan, if_neg (by decide), innerCandidates348_5_checked]
  change scan edge (search edge 3) 4 [2371] 2371 innerTail348_6
    (if 3 ≤ innerCandidates348_5.length then search edge 3 [2371, 4021] innerCandidates348_5 6392 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan348_6

def innerTail348_4 : List Nat := 3643 :: innerTail348_5
def innerCandidates348_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates348_4_checked : candidates edge 4 2371 3643 innerTail348_5 best.bound = innerCandidates348_4 := by
  unfold candidates
  have row : edge 3643 = SearchData.row3643.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan348_4 : scan edge (search edge 3) 4 [2371] 2371 innerTail348_4 best = best := by
  rw [innerTail348_4, scan, if_neg (by decide), innerCandidates348_4_checked]
  change scan edge (search edge 3) 4 [2371] 2371 innerTail348_5
    (if 3 ≤ innerCandidates348_4.length then search edge 3 [2371, 3643] innerCandidates348_4 6014 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan348_5

def innerTail348_3 : List Nat := 3607 :: innerTail348_4
def innerCandidates348_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates348_3_checked : candidates edge 4 2371 3607 innerTail348_4 best.bound = innerCandidates348_3 := by
  unfold candidates
  have row : edge 3607 = SearchData.row3607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan348_3 : scan edge (search edge 3) 4 [2371] 2371 innerTail348_3 best = best := by
  rw [innerTail348_3, scan, if_neg (by decide), innerCandidates348_3_checked]
  change scan edge (search edge 3) 4 [2371] 2371 innerTail348_4
    (if 3 ≤ innerCandidates348_3.length then search edge 3 [2371, 3607] innerCandidates348_3 5978 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan348_4

def innerTail348_2 : List Nat := 3229 :: innerTail348_3
def innerCandidates348_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates348_2_checked : candidates edge 4 2371 3229 innerTail348_3 best.bound = innerCandidates348_2 := by
  unfold candidates
  have row : edge 3229 = SearchData.row3229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan348_2 : scan edge (search edge 3) 4 [2371] 2371 innerTail348_2 best = best := by
  rw [innerTail348_2, scan, if_neg (by decide), innerCandidates348_2_checked]
  change scan edge (search edge 3) 4 [2371] 2371 innerTail348_3
    (if 3 ≤ innerCandidates348_2.length then search edge 3 [2371, 3229] innerCandidates348_2 5600 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan348_3

def innerTail348_1 : List Nat := 3169 :: innerTail348_2
def innerCandidates348_1 : List Nat := [3229]
set_option maxRecDepth 4096 in
theorem innerCandidates348_1_checked : candidates edge 4 2371 3169 innerTail348_2 best.bound = innerCandidates348_1 := by
  unfold candidates
  have row : edge 3169 = SearchData.row3169.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan348_1 : scan edge (search edge 3) 4 [2371] 2371 innerTail348_1 best = best := by
  rw [innerTail348_1, scan, if_neg (by decide), innerCandidates348_1_checked]
  change scan edge (search edge 3) 4 [2371] 2371 innerTail348_2
    (if 3 ≤ innerCandidates348_1.length then search edge 3 [2371, 3169] innerCandidates348_1 5540 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan348_2

def innerTail348_0 : List Nat := 2917 :: innerTail348_1
def innerCandidates348_0 : List Nat := [3169, 3643]
set_option maxRecDepth 4096 in
theorem innerCandidates348_0_checked : candidates edge 4 2371 2917 innerTail348_1 best.bound = innerCandidates348_0 := by
  unfold candidates
  have row : edge 2917 = SearchData.row2917.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan348_0 : scan edge (search edge 3) 4 [2371] 2371 innerTail348_0 best = best := by
  rw [innerTail348_0, scan, if_neg (by decide), innerCandidates348_0_checked]
  change scan edge (search edge 3) 4 [2371] 2371 innerTail348_1
    (if 3 ≤ innerCandidates348_0.length then search edge 3 [2371, 2917] innerCandidates348_0 5288 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan348_1

theorem rootBranch348 : search edge 4 [2371] rootCandidates348 2371 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2371] 2371 innerTail348_0 best = best
  exact innerScan348_0

def rootCandidates347 : List Nat := [2579, 2903, 3467, 3527, 3821, 4091, 4289, 5087, 5693, 5711, 6323, 6551, 6869, 7187, 7331, 7457, 7901, 7919, 7937, 8237]
set_option maxRecDepth 4096 in
theorem rootCandidates347_checked : candidates edge 5 0 2357 tail348 best.bound = rootCandidates347 := by
  unfold candidates
  have row : edge 2357 = PrimePairCache.smallRow2357.contains := by funext q; rfl
  rw [row]
  decide

def innerTail347_20 : List Nat := []
theorem innerScan347_20 : scan edge (search edge 3) 4 [2357] 2357 innerTail347_20 best = best := rfl

def innerTail347_19 : List Nat := 8237 :: innerTail347_20
def innerCandidates347_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates347_19_checked : candidates edge 4 2357 8237 innerTail347_20 best.bound = innerCandidates347_19 := by
  unfold candidates
  have row : edge 8237 = SearchData.row8237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan347_19 : scan edge (search edge 3) 4 [2357] 2357 innerTail347_19 best = best := by
  rw [innerTail347_19, scan, if_neg (by decide), innerCandidates347_19_checked]
  change scan edge (search edge 3) 4 [2357] 2357 innerTail347_20
    (if 3 ≤ innerCandidates347_19.length then search edge 3 [2357, 8237] innerCandidates347_19 10594 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan347_20

def innerTail347_18 : List Nat := 7937 :: innerTail347_19
def innerCandidates347_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates347_18_checked : candidates edge 4 2357 7937 innerTail347_19 best.bound = innerCandidates347_18 := by
  unfold candidates
  have row : edge 7937 = SearchData.row7937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan347_18 : scan edge (search edge 3) 4 [2357] 2357 innerTail347_18 best = best := by
  rw [innerTail347_18, scan, if_neg (by decide), innerCandidates347_18_checked]
  change scan edge (search edge 3) 4 [2357] 2357 innerTail347_19
    (if 3 ≤ innerCandidates347_18.length then search edge 3 [2357, 7937] innerCandidates347_18 10294 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan347_19

def innerTail347_17 : List Nat := 7919 :: innerTail347_18
def innerCandidates347_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates347_17_checked : candidates edge 4 2357 7919 innerTail347_18 best.bound = innerCandidates347_17 := by
  unfold candidates
  have row : edge 7919 = SearchData.row7919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan347_17 : scan edge (search edge 3) 4 [2357] 2357 innerTail347_17 best = best := by
  rw [innerTail347_17, scan, if_neg (by decide), innerCandidates347_17_checked]
  change scan edge (search edge 3) 4 [2357] 2357 innerTail347_18
    (if 3 ≤ innerCandidates347_17.length then search edge 3 [2357, 7919] innerCandidates347_17 10276 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan347_18

def innerTail347_16 : List Nat := 7901 :: innerTail347_17
def innerCandidates347_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates347_16_checked : candidates edge 4 2357 7901 innerTail347_17 best.bound = innerCandidates347_16 := by
  unfold candidates
  have row : edge 7901 = SearchData.row7901.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan347_16 : scan edge (search edge 3) 4 [2357] 2357 innerTail347_16 best = best := by
  rw [innerTail347_16, scan, if_neg (by decide), innerCandidates347_16_checked]
  change scan edge (search edge 3) 4 [2357] 2357 innerTail347_17
    (if 3 ≤ innerCandidates347_16.length then search edge 3 [2357, 7901] innerCandidates347_16 10258 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan347_17

def innerTail347_15 : List Nat := 7457 :: innerTail347_16
def innerCandidates347_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates347_15_checked : candidates edge 4 2357 7457 innerTail347_16 best.bound = innerCandidates347_15 := by
  unfold candidates
  have row : edge 7457 = SearchData.row7457.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan347_15 : scan edge (search edge 3) 4 [2357] 2357 innerTail347_15 best = best := by
  rw [innerTail347_15, scan, if_neg (by decide), innerCandidates347_15_checked]
  change scan edge (search edge 3) 4 [2357] 2357 innerTail347_16
    (if 3 ≤ innerCandidates347_15.length then search edge 3 [2357, 7457] innerCandidates347_15 9814 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan347_16

def innerTail347_14 : List Nat := 7331 :: innerTail347_15
def innerCandidates347_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates347_14_checked : candidates edge 4 2357 7331 innerTail347_15 best.bound = innerCandidates347_14 := by
  unfold candidates
  have row : edge 7331 = SearchData.row7331.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan347_14 : scan edge (search edge 3) 4 [2357] 2357 innerTail347_14 best = best := by
  rw [innerTail347_14, scan, if_neg (by decide), innerCandidates347_14_checked]
  change scan edge (search edge 3) 4 [2357] 2357 innerTail347_15
    (if 3 ≤ innerCandidates347_14.length then search edge 3 [2357, 7331] innerCandidates347_14 9688 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan347_15

def innerTail347_13 : List Nat := 7187 :: innerTail347_14
def innerCandidates347_13 : List Nat := [7331]
set_option maxRecDepth 4096 in
theorem innerCandidates347_13_checked : candidates edge 4 2357 7187 innerTail347_14 best.bound = innerCandidates347_13 := by
  unfold candidates
  have row : edge 7187 = SearchData.row7187.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan347_13 : scan edge (search edge 3) 4 [2357] 2357 innerTail347_13 best = best := by
  rw [innerTail347_13, scan, if_neg (by decide), innerCandidates347_13_checked]
  change scan edge (search edge 3) 4 [2357] 2357 innerTail347_14
    (if 3 ≤ innerCandidates347_13.length then search edge 3 [2357, 7187] innerCandidates347_13 9544 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan347_14

def innerTail347_12 : List Nat := 6869 :: innerTail347_13
def innerCandidates347_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates347_12_checked : candidates edge 4 2357 6869 innerTail347_13 best.bound = innerCandidates347_12 := by
  unfold candidates
  have row : edge 6869 = SearchData.row6869.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan347_12 : scan edge (search edge 3) 4 [2357] 2357 innerTail347_12 best = best := by
  rw [innerTail347_12, scan, if_neg (by decide), innerCandidates347_12_checked]
  change scan edge (search edge 3) 4 [2357] 2357 innerTail347_13
    (if 3 ≤ innerCandidates347_12.length then search edge 3 [2357, 6869] innerCandidates347_12 9226 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan347_13

def innerTail347_11 : List Nat := 6551 :: innerTail347_12
def innerCandidates347_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates347_11_checked : candidates edge 4 2357 6551 innerTail347_12 best.bound = innerCandidates347_11 := by
  unfold candidates
  have row : edge 6551 = SearchData.row6551.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan347_11 : scan edge (search edge 3) 4 [2357] 2357 innerTail347_11 best = best := by
  rw [innerTail347_11, scan, if_neg (by decide), innerCandidates347_11_checked]
  change scan edge (search edge 3) 4 [2357] 2357 innerTail347_12
    (if 3 ≤ innerCandidates347_11.length then search edge 3 [2357, 6551] innerCandidates347_11 8908 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan347_12

def innerTail347_10 : List Nat := 6323 :: innerTail347_11
def innerCandidates347_10 : List Nat := [7919, 8237]
set_option maxRecDepth 4096 in
theorem innerCandidates347_10_checked : candidates edge 4 2357 6323 innerTail347_11 best.bound = innerCandidates347_10 := by
  unfold candidates
  have row : edge 6323 = SearchData.row6323.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan347_10 : scan edge (search edge 3) 4 [2357] 2357 innerTail347_10 best = best := by
  rw [innerTail347_10, scan, if_neg (by decide), innerCandidates347_10_checked]
  change scan edge (search edge 3) 4 [2357] 2357 innerTail347_11
    (if 3 ≤ innerCandidates347_10.length then search edge 3 [2357, 6323] innerCandidates347_10 8680 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan347_11

def innerTail347_9 : List Nat := 5711 :: innerTail347_10
def innerCandidates347_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates347_9_checked : candidates edge 4 2357 5711 innerTail347_10 best.bound = innerCandidates347_9 := by
  unfold candidates
  have row : edge 5711 = SearchData.row5711.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan347_9 : scan edge (search edge 3) 4 [2357] 2357 innerTail347_9 best = best := by
  rw [innerTail347_9, scan, if_neg (by decide), innerCandidates347_9_checked]
  change scan edge (search edge 3) 4 [2357] 2357 innerTail347_10
    (if 3 ≤ innerCandidates347_9.length then search edge 3 [2357, 5711] innerCandidates347_9 8068 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan347_10

def innerTail347_8 : List Nat := 5693 :: innerTail347_9
def innerCandidates347_8 : List Nat := [7187]
set_option maxRecDepth 4096 in
theorem innerCandidates347_8_checked : candidates edge 4 2357 5693 innerTail347_9 best.bound = innerCandidates347_8 := by
  unfold candidates
  have row : edge 5693 = SearchData.row5693.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan347_8 : scan edge (search edge 3) 4 [2357] 2357 innerTail347_8 best = best := by
  rw [innerTail347_8, scan, if_neg (by decide), innerCandidates347_8_checked]
  change scan edge (search edge 3) 4 [2357] 2357 innerTail347_9
    (if 3 ≤ innerCandidates347_8.length then search edge 3 [2357, 5693] innerCandidates347_8 8050 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan347_9

def innerTail347_7 : List Nat := 5087 :: innerTail347_8
def innerCandidates347_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates347_7_checked : candidates edge 4 2357 5087 innerTail347_8 best.bound = innerCandidates347_7 := by
  unfold candidates
  have row : edge 5087 = SearchData.row5087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan347_7 : scan edge (search edge 3) 4 [2357] 2357 innerTail347_7 best = best := by
  rw [innerTail347_7, scan, if_neg (by decide), innerCandidates347_7_checked]
  change scan edge (search edge 3) 4 [2357] 2357 innerTail347_8
    (if 3 ≤ innerCandidates347_7.length then search edge 3 [2357, 5087] innerCandidates347_7 7444 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan347_8

def innerTail347_6 : List Nat := 4289 :: innerTail347_7
def innerCandidates347_6 : List Nat := [6869, 7331, 7937]
set_option maxRecDepth 4096 in
theorem innerCandidates347_6_checked : candidates edge 4 2357 4289 innerTail347_7 best.bound = innerCandidates347_6 := by
  unfold candidates
  have row : edge 4289 = SearchData.row4289.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild347_6 : search edge 3 [2357, 4289] innerCandidates347_6 6646 best = best := by decide

theorem innerScan347_6 : scan edge (search edge 3) 4 [2357] 2357 innerTail347_6 best = best := by
  rw [innerTail347_6, scan, if_neg (by decide), innerCandidates347_6_checked]
  change scan edge (search edge 3) 4 [2357] 2357 innerTail347_7
    (if 3 ≤ innerCandidates347_6.length then search edge 3 [2357, 4289] innerCandidates347_6 6646 best else best) = best
  rw [if_pos (by decide), innerChild347_6]
  exact innerScan347_7

def innerTail347_5 : List Nat := 4091 :: innerTail347_6
def innerCandidates347_5 : List Nat := [7937, 8237]
set_option maxRecDepth 4096 in
theorem innerCandidates347_5_checked : candidates edge 4 2357 4091 innerTail347_6 best.bound = innerCandidates347_5 := by
  unfold candidates
  have row : edge 4091 = SearchData.row4091.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan347_5 : scan edge (search edge 3) 4 [2357] 2357 innerTail347_5 best = best := by
  rw [innerTail347_5, scan, if_neg (by decide), innerCandidates347_5_checked]
  change scan edge (search edge 3) 4 [2357] 2357 innerTail347_6
    (if 3 ≤ innerCandidates347_5.length then search edge 3 [2357, 4091] innerCandidates347_5 6448 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan347_6

def innerTail347_4 : List Nat := 3821 :: innerTail347_5
def innerCandidates347_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates347_4_checked : candidates edge 4 2357 3821 innerTail347_5 best.bound = innerCandidates347_4 := by
  unfold candidates
  have row : edge 3821 = SearchData.row3821.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan347_4 : scan edge (search edge 3) 4 [2357] 2357 innerTail347_4 best = best := by
  rw [innerTail347_4, scan, if_neg (by decide), innerCandidates347_4_checked]
  change scan edge (search edge 3) 4 [2357] 2357 innerTail347_5
    (if 3 ≤ innerCandidates347_4.length then search edge 3 [2357, 3821] innerCandidates347_4 6178 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan347_5

def innerTail347_3 : List Nat := 3527 :: innerTail347_4
def innerCandidates347_3 : List Nat := [6551, 7901]
set_option maxRecDepth 4096 in
theorem innerCandidates347_3_checked : candidates edge 4 2357 3527 innerTail347_4 best.bound = innerCandidates347_3 := by
  unfold candidates
  have row : edge 3527 = SearchData.row3527.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan347_3 : scan edge (search edge 3) 4 [2357] 2357 innerTail347_3 best = best := by
  rw [innerTail347_3, scan, if_neg (by decide), innerCandidates347_3_checked]
  change scan edge (search edge 3) 4 [2357] 2357 innerTail347_4
    (if 3 ≤ innerCandidates347_3.length then search edge 3 [2357, 3527] innerCandidates347_3 5884 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan347_4

def innerTail347_2 : List Nat := 3467 :: innerTail347_3
def innerCandidates347_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates347_2_checked : candidates edge 4 2357 3467 innerTail347_3 best.bound = innerCandidates347_2 := by
  unfold candidates
  have row : edge 3467 = SearchData.row3467.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan347_2 : scan edge (search edge 3) 4 [2357] 2357 innerTail347_2 best = best := by
  rw [innerTail347_2, scan, if_neg (by decide), innerCandidates347_2_checked]
  change scan edge (search edge 3) 4 [2357] 2357 innerTail347_3
    (if 3 ≤ innerCandidates347_2.length then search edge 3 [2357, 3467] innerCandidates347_2 5824 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan347_3

def innerTail347_1 : List Nat := 2903 :: innerTail347_2
def innerCandidates347_1 : List Nat := [7457]
set_option maxRecDepth 4096 in
theorem innerCandidates347_1_checked : candidates edge 4 2357 2903 innerTail347_2 best.bound = innerCandidates347_1 := by
  unfold candidates
  have row : edge 2903 = SearchData.row2903.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan347_1 : scan edge (search edge 3) 4 [2357] 2357 innerTail347_1 best = best := by
  rw [innerTail347_1, scan, if_neg (by decide), innerCandidates347_1_checked]
  change scan edge (search edge 3) 4 [2357] 2357 innerTail347_2
    (if 3 ≤ innerCandidates347_1.length then search edge 3 [2357, 2903] innerCandidates347_1 5260 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan347_2

def innerTail347_0 : List Nat := 2579 :: innerTail347_1
def innerCandidates347_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates347_0_checked : candidates edge 4 2357 2579 innerTail347_1 best.bound = innerCandidates347_0 := by
  unfold candidates
  have row : edge 2579 = PrimePairCache.smallRow2579.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan347_0 : scan edge (search edge 3) 4 [2357] 2357 innerTail347_0 best = best := by
  rw [innerTail347_0, scan, if_neg (by decide), innerCandidates347_0_checked]
  change scan edge (search edge 3) 4 [2357] 2357 innerTail347_1
    (if 3 ≤ innerCandidates347_0.length then search edge 3 [2357, 2579] innerCandidates347_0 4936 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan347_1

theorem rootBranch347 : search edge 4 [2357] rootCandidates347 2357 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2357] 2357 innerTail347_0 best = best
  exact innerScan347_0

def rootCandidates346 : List Nat := [3617, 3863, 3881, 4973, 5507, 5519, 6761, 6959, 7019, 7229, 7643, 7841, 7853, 8237]
set_option maxRecDepth 4096 in
theorem rootCandidates346_checked : candidates edge 5 0 2351 tail347 best.bound = rootCandidates346 := by
  unfold candidates
  have row : edge 2351 = PrimePairCache.smallRow2351.contains := by funext q; rfl
  rw [row]
  decide

def innerTail346_14 : List Nat := []
theorem innerScan346_14 : scan edge (search edge 3) 4 [2351] 2351 innerTail346_14 best = best := rfl

def innerTail346_13 : List Nat := 8237 :: innerTail346_14
def innerCandidates346_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates346_13_checked : candidates edge 4 2351 8237 innerTail346_14 best.bound = innerCandidates346_13 := by
  unfold candidates
  have row : edge 8237 = SearchData.row8237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan346_13 : scan edge (search edge 3) 4 [2351] 2351 innerTail346_13 best = best := by
  rw [innerTail346_13, scan, if_neg (by decide), innerCandidates346_13_checked]
  change scan edge (search edge 3) 4 [2351] 2351 innerTail346_14
    (if 3 ≤ innerCandidates346_13.length then search edge 3 [2351, 8237] innerCandidates346_13 10588 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan346_14

def innerTail346_12 : List Nat := 7853 :: innerTail346_13
def innerCandidates346_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates346_12_checked : candidates edge 4 2351 7853 innerTail346_13 best.bound = innerCandidates346_12 := by
  unfold candidates
  have row : edge 7853 = SearchData.row7853.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan346_12 : scan edge (search edge 3) 4 [2351] 2351 innerTail346_12 best = best := by
  rw [innerTail346_12, scan, if_neg (by decide), innerCandidates346_12_checked]
  change scan edge (search edge 3) 4 [2351] 2351 innerTail346_13
    (if 3 ≤ innerCandidates346_12.length then search edge 3 [2351, 7853] innerCandidates346_12 10204 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan346_13

def innerTail346_11 : List Nat := 7841 :: innerTail346_12
def innerCandidates346_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates346_11_checked : candidates edge 4 2351 7841 innerTail346_12 best.bound = innerCandidates346_11 := by
  unfold candidates
  have row : edge 7841 = SearchData.row7841.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan346_11 : scan edge (search edge 3) 4 [2351] 2351 innerTail346_11 best = best := by
  rw [innerTail346_11, scan, if_neg (by decide), innerCandidates346_11_checked]
  change scan edge (search edge 3) 4 [2351] 2351 innerTail346_12
    (if 3 ≤ innerCandidates346_11.length then search edge 3 [2351, 7841] innerCandidates346_11 10192 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan346_12

def innerTail346_10 : List Nat := 7643 :: innerTail346_11
def innerCandidates346_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates346_10_checked : candidates edge 4 2351 7643 innerTail346_11 best.bound = innerCandidates346_10 := by
  unfold candidates
  have row : edge 7643 = SearchData.row7643.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan346_10 : scan edge (search edge 3) 4 [2351] 2351 innerTail346_10 best = best := by
  rw [innerTail346_10, scan, if_neg (by decide), innerCandidates346_10_checked]
  change scan edge (search edge 3) 4 [2351] 2351 innerTail346_11
    (if 3 ≤ innerCandidates346_10.length then search edge 3 [2351, 7643] innerCandidates346_10 9994 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan346_11

def innerTail346_9 : List Nat := 7229 :: innerTail346_10
def innerCandidates346_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates346_9_checked : candidates edge 4 2351 7229 innerTail346_10 best.bound = innerCandidates346_9 := by
  unfold candidates
  have row : edge 7229 = SearchData.row7229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan346_9 : scan edge (search edge 3) 4 [2351] 2351 innerTail346_9 best = best := by
  rw [innerTail346_9, scan, if_neg (by decide), innerCandidates346_9_checked]
  change scan edge (search edge 3) 4 [2351] 2351 innerTail346_10
    (if 3 ≤ innerCandidates346_9.length then search edge 3 [2351, 7229] innerCandidates346_9 9580 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan346_10

def innerTail346_8 : List Nat := 7019 :: innerTail346_9
def innerCandidates346_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates346_8_checked : candidates edge 4 2351 7019 innerTail346_9 best.bound = innerCandidates346_8 := by
  unfold candidates
  have row : edge 7019 = SearchData.row7019.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan346_8 : scan edge (search edge 3) 4 [2351] 2351 innerTail346_8 best = best := by
  rw [innerTail346_8, scan, if_neg (by decide), innerCandidates346_8_checked]
  change scan edge (search edge 3) 4 [2351] 2351 innerTail346_9
    (if 3 ≤ innerCandidates346_8.length then search edge 3 [2351, 7019] innerCandidates346_8 9370 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan346_9

def innerTail346_7 : List Nat := 6959 :: innerTail346_8
def innerCandidates346_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates346_7_checked : candidates edge 4 2351 6959 innerTail346_8 best.bound = innerCandidates346_7 := by
  unfold candidates
  have row : edge 6959 = SearchData.row6959.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan346_7 : scan edge (search edge 3) 4 [2351] 2351 innerTail346_7 best = best := by
  rw [innerTail346_7, scan, if_neg (by decide), innerCandidates346_7_checked]
  change scan edge (search edge 3) 4 [2351] 2351 innerTail346_8
    (if 3 ≤ innerCandidates346_7.length then search edge 3 [2351, 6959] innerCandidates346_7 9310 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan346_8

def innerTail346_6 : List Nat := 6761 :: innerTail346_7
def innerCandidates346_6 : List Nat := [7229]
set_option maxRecDepth 4096 in
theorem innerCandidates346_6_checked : candidates edge 4 2351 6761 innerTail346_7 best.bound = innerCandidates346_6 := by
  unfold candidates
  have row : edge 6761 = SearchData.row6761.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan346_6 : scan edge (search edge 3) 4 [2351] 2351 innerTail346_6 best = best := by
  rw [innerTail346_6, scan, if_neg (by decide), innerCandidates346_6_checked]
  change scan edge (search edge 3) 4 [2351] 2351 innerTail346_7
    (if 3 ≤ innerCandidates346_6.length then search edge 3 [2351, 6761] innerCandidates346_6 9112 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan346_7

def innerTail346_5 : List Nat := 5519 :: innerTail346_6
def innerCandidates346_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates346_5_checked : candidates edge 4 2351 5519 innerTail346_6 best.bound = innerCandidates346_5 := by
  unfold candidates
  have row : edge 5519 = SearchData.row5519.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan346_5 : scan edge (search edge 3) 4 [2351] 2351 innerTail346_5 best = best := by
  rw [innerTail346_5, scan, if_neg (by decide), innerCandidates346_5_checked]
  change scan edge (search edge 3) 4 [2351] 2351 innerTail346_6
    (if 3 ≤ innerCandidates346_5.length then search edge 3 [2351, 5519] innerCandidates346_5 7870 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan346_6

def innerTail346_4 : List Nat := 5507 :: innerTail346_5
def innerCandidates346_4 : List Nat := [7643, 7853]
set_option maxRecDepth 4096 in
theorem innerCandidates346_4_checked : candidates edge 4 2351 5507 innerTail346_5 best.bound = innerCandidates346_4 := by
  unfold candidates
  have row : edge 5507 = SearchData.row5507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan346_4 : scan edge (search edge 3) 4 [2351] 2351 innerTail346_4 best = best := by
  rw [innerTail346_4, scan, if_neg (by decide), innerCandidates346_4_checked]
  change scan edge (search edge 3) 4 [2351] 2351 innerTail346_5
    (if 3 ≤ innerCandidates346_4.length then search edge 3 [2351, 5507] innerCandidates346_4 7858 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan346_5

def innerTail346_3 : List Nat := 4973 :: innerTail346_4
def innerCandidates346_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates346_3_checked : candidates edge 4 2351 4973 innerTail346_4 best.bound = innerCandidates346_3 := by
  unfold candidates
  have row : edge 4973 = SearchData.row4973.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan346_3 : scan edge (search edge 3) 4 [2351] 2351 innerTail346_3 best = best := by
  rw [innerTail346_3, scan, if_neg (by decide), innerCandidates346_3_checked]
  change scan edge (search edge 3) 4 [2351] 2351 innerTail346_4
    (if 3 ≤ innerCandidates346_3.length then search edge 3 [2351, 4973] innerCandidates346_3 7324 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan346_4

def innerTail346_2 : List Nat := 3881 :: innerTail346_3
def innerCandidates346_2 : List Nat := [5519]
set_option maxRecDepth 4096 in
theorem innerCandidates346_2_checked : candidates edge 4 2351 3881 innerTail346_3 best.bound = innerCandidates346_2 := by
  unfold candidates
  have row : edge 3881 = SearchData.row3881.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan346_2 : scan edge (search edge 3) 4 [2351] 2351 innerTail346_2 best = best := by
  rw [innerTail346_2, scan, if_neg (by decide), innerCandidates346_2_checked]
  change scan edge (search edge 3) 4 [2351] 2351 innerTail346_3
    (if 3 ≤ innerCandidates346_2.length then search edge 3 [2351, 3881] innerCandidates346_2 6232 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan346_3

def innerTail346_1 : List Nat := 3863 :: innerTail346_2
def innerCandidates346_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates346_1_checked : candidates edge 4 2351 3863 innerTail346_2 best.bound = innerCandidates346_1 := by
  unfold candidates
  have row : edge 3863 = SearchData.row3863.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan346_1 : scan edge (search edge 3) 4 [2351] 2351 innerTail346_1 best = best := by
  rw [innerTail346_1, scan, if_neg (by decide), innerCandidates346_1_checked]
  change scan edge (search edge 3) 4 [2351] 2351 innerTail346_2
    (if 3 ≤ innerCandidates346_1.length then search edge 3 [2351, 3863] innerCandidates346_1 6214 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan346_2

def innerTail346_0 : List Nat := 3617 :: innerTail346_1
def innerCandidates346_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates346_0_checked : candidates edge 4 2351 3617 innerTail346_1 best.bound = innerCandidates346_0 := by
  unfold candidates
  have row : edge 3617 = SearchData.row3617.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan346_0 : scan edge (search edge 3) 4 [2351] 2351 innerTail346_0 best = best := by
  rw [innerTail346_0, scan, if_neg (by decide), innerCandidates346_0_checked]
  change scan edge (search edge 3) 4 [2351] 2351 innerTail346_1
    (if 3 ≤ innerCandidates346_0.length then search edge 3 [2351, 3617] innerCandidates346_0 5968 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan346_1

theorem rootBranch346 : search edge 4 [2351] rootCandidates346 2351 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2351] 2351 innerTail346_0 best = best
  exact innerScan346_0

end ProjectEulerSolutions.PrimePairUpper.Search
