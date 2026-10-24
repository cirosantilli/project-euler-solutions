import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates22 : List Nat := [157, 241, 373, 379, 787, 829, 883, 919, 967, 1021, 1039, 1063, 1093, 1279, 1291, 1381, 1783, 1951, 2017, 2221, 2833, 3331, 3373, 3691, 3727, 3853, 4261, 4507, 4513, 4657, 4957, 4969, 5011, 5281, 5581, 6271, 6301, 6571, 6637, 6883, 6991, 7507, 7681, 7927, 8011, 8053, 8209, 8269, 8287]
set_option maxRecDepth 4096 in
theorem rootCandidates22_checked : candidates edge 5 0 97 tail23 best.bound = rootCandidates22 := by
  unfold candidates
  have row : edge 97 = PrimePairCache.smallRow97.contains := by funext q; rfl
  rw [row]
  decide

def innerTail22_49 : List Nat := []
theorem innerScan22_49 : scan edge (search edge 3) 4 [97] 97 innerTail22_49 best = best := rfl

def innerTail22_48 : List Nat := 8287 :: innerTail22_49
def innerCandidates22_48 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates22_48_checked : candidates edge 4 97 8287 innerTail22_49 best.bound = innerCandidates22_48 := by
  unfold candidates
  have row : edge 8287 = SearchData.row8287.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan22_48 : scan edge (search edge 3) 4 [97] 97 innerTail22_48 best = best := by
  rw [innerTail22_48, scan, if_neg (by decide), innerCandidates22_48_checked]
  change scan edge (search edge 3) 4 [97] 97 innerTail22_49
    (if 3 ≤ innerCandidates22_48.length then search edge 3 [97, 8287] innerCandidates22_48 8384 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan22_49

def innerTail22_47 : List Nat := 8269 :: innerTail22_48
def innerCandidates22_47 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates22_47_checked : candidates edge 4 97 8269 innerTail22_48 best.bound = innerCandidates22_47 := by
  unfold candidates
  have row : edge 8269 = SearchData.row8269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan22_47 : scan edge (search edge 3) 4 [97] 97 innerTail22_47 best = best := by
  rw [innerTail22_47, scan, if_neg (by decide), innerCandidates22_47_checked]
  change scan edge (search edge 3) 4 [97] 97 innerTail22_48
    (if 3 ≤ innerCandidates22_47.length then search edge 3 [97, 8269] innerCandidates22_47 8366 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan22_48

def innerTail22_46 : List Nat := 8209 :: innerTail22_47
def innerCandidates22_46 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates22_46_checked : candidates edge 4 97 8209 innerTail22_47 best.bound = innerCandidates22_46 := by
  unfold candidates
  have row : edge 8209 = SearchData.row8209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan22_46 : scan edge (search edge 3) 4 [97] 97 innerTail22_46 best = best := by
  rw [innerTail22_46, scan, if_neg (by decide), innerCandidates22_46_checked]
  change scan edge (search edge 3) 4 [97] 97 innerTail22_47
    (if 3 ≤ innerCandidates22_46.length then search edge 3 [97, 8209] innerCandidates22_46 8306 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan22_47

def innerTail22_45 : List Nat := 8053 :: innerTail22_46
def innerCandidates22_45 : List Nat := [8287]
set_option maxRecDepth 4096 in
theorem innerCandidates22_45_checked : candidates edge 4 97 8053 innerTail22_46 best.bound = innerCandidates22_45 := by
  unfold candidates
  have row : edge 8053 = SearchData.row8053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan22_45 : scan edge (search edge 3) 4 [97] 97 innerTail22_45 best = best := by
  rw [innerTail22_45, scan, if_neg (by decide), innerCandidates22_45_checked]
  change scan edge (search edge 3) 4 [97] 97 innerTail22_46
    (if 3 ≤ innerCandidates22_45.length then search edge 3 [97, 8053] innerCandidates22_45 8150 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan22_46

def innerTail22_44 : List Nat := 8011 :: innerTail22_45
def innerCandidates22_44 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates22_44_checked : candidates edge 4 97 8011 innerTail22_45 best.bound = innerCandidates22_44 := by
  unfold candidates
  have row : edge 8011 = SearchData.row8011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan22_44 : scan edge (search edge 3) 4 [97] 97 innerTail22_44 best = best := by
  rw [innerTail22_44, scan, if_neg (by decide), innerCandidates22_44_checked]
  change scan edge (search edge 3) 4 [97] 97 innerTail22_45
    (if 3 ≤ innerCandidates22_44.length then search edge 3 [97, 8011] innerCandidates22_44 8108 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan22_45

def innerTail22_43 : List Nat := 7927 :: innerTail22_44
def innerCandidates22_43 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates22_43_checked : candidates edge 4 97 7927 innerTail22_44 best.bound = innerCandidates22_43 := by
  unfold candidates
  have row : edge 7927 = SearchData.row7927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan22_43 : scan edge (search edge 3) 4 [97] 97 innerTail22_43 best = best := by
  rw [innerTail22_43, scan, if_neg (by decide), innerCandidates22_43_checked]
  change scan edge (search edge 3) 4 [97] 97 innerTail22_44
    (if 3 ≤ innerCandidates22_43.length then search edge 3 [97, 7927] innerCandidates22_43 8024 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan22_44

def innerTail22_42 : List Nat := 7681 :: innerTail22_43
def innerCandidates22_42 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates22_42_checked : candidates edge 4 97 7681 innerTail22_43 best.bound = innerCandidates22_42 := by
  unfold candidates
  have row : edge 7681 = SearchData.row7681.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan22_42 : scan edge (search edge 3) 4 [97] 97 innerTail22_42 best = best := by
  rw [innerTail22_42, scan, if_neg (by decide), innerCandidates22_42_checked]
  change scan edge (search edge 3) 4 [97] 97 innerTail22_43
    (if 3 ≤ innerCandidates22_42.length then search edge 3 [97, 7681] innerCandidates22_42 7778 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan22_43

def innerTail22_41 : List Nat := 7507 :: innerTail22_42
def innerCandidates22_41 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates22_41_checked : candidates edge 4 97 7507 innerTail22_42 best.bound = innerCandidates22_41 := by
  unfold candidates
  have row : edge 7507 = SearchData.row7507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan22_41 : scan edge (search edge 3) 4 [97] 97 innerTail22_41 best = best := by
  rw [innerTail22_41, scan, if_neg (by decide), innerCandidates22_41_checked]
  change scan edge (search edge 3) 4 [97] 97 innerTail22_42
    (if 3 ≤ innerCandidates22_41.length then search edge 3 [97, 7507] innerCandidates22_41 7604 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan22_42

def innerTail22_40 : List Nat := 6991 :: innerTail22_41
def innerCandidates22_40 : List Nat := [8011]
set_option maxRecDepth 4096 in
theorem innerCandidates22_40_checked : candidates edge 4 97 6991 innerTail22_41 best.bound = innerCandidates22_40 := by
  unfold candidates
  have row : edge 6991 = SearchData.row6991.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan22_40 : scan edge (search edge 3) 4 [97] 97 innerTail22_40 best = best := by
  rw [innerTail22_40, scan, if_neg (by decide), innerCandidates22_40_checked]
  change scan edge (search edge 3) 4 [97] 97 innerTail22_41
    (if 3 ≤ innerCandidates22_40.length then search edge 3 [97, 6991] innerCandidates22_40 7088 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan22_41

def innerTail22_39 : List Nat := 6883 :: innerTail22_40
def innerCandidates22_39 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates22_39_checked : candidates edge 4 97 6883 innerTail22_40 best.bound = innerCandidates22_39 := by
  unfold candidates
  have row : edge 6883 = SearchData.row6883.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan22_39 : scan edge (search edge 3) 4 [97] 97 innerTail22_39 best = best := by
  rw [innerTail22_39, scan, if_neg (by decide), innerCandidates22_39_checked]
  change scan edge (search edge 3) 4 [97] 97 innerTail22_40
    (if 3 ≤ innerCandidates22_39.length then search edge 3 [97, 6883] innerCandidates22_39 6980 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan22_40

def innerTail22_38 : List Nat := 6637 :: innerTail22_39
def innerCandidates22_38 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates22_38_checked : candidates edge 4 97 6637 innerTail22_39 best.bound = innerCandidates22_38 := by
  unfold candidates
  have row : edge 6637 = SearchData.row6637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan22_38 : scan edge (search edge 3) 4 [97] 97 innerTail22_38 best = best := by
  rw [innerTail22_38, scan, if_neg (by decide), innerCandidates22_38_checked]
  change scan edge (search edge 3) 4 [97] 97 innerTail22_39
    (if 3 ≤ innerCandidates22_38.length then search edge 3 [97, 6637] innerCandidates22_38 6734 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan22_39

def innerTail22_37 : List Nat := 6571 :: innerTail22_38
def innerCandidates22_37 : List Nat := [8269]
set_option maxRecDepth 4096 in
theorem innerCandidates22_37_checked : candidates edge 4 97 6571 innerTail22_38 best.bound = innerCandidates22_37 := by
  unfold candidates
  have row : edge 6571 = SearchData.row6571.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan22_37 : scan edge (search edge 3) 4 [97] 97 innerTail22_37 best = best := by
  rw [innerTail22_37, scan, if_neg (by decide), innerCandidates22_37_checked]
  change scan edge (search edge 3) 4 [97] 97 innerTail22_38
    (if 3 ≤ innerCandidates22_37.length then search edge 3 [97, 6571] innerCandidates22_37 6668 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan22_38

def innerTail22_36 : List Nat := 6301 :: innerTail22_37
def innerCandidates22_36 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates22_36_checked : candidates edge 4 97 6301 innerTail22_37 best.bound = innerCandidates22_36 := by
  unfold candidates
  have row : edge 6301 = SearchData.row6301.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan22_36 : scan edge (search edge 3) 4 [97] 97 innerTail22_36 best = best := by
  rw [innerTail22_36, scan, if_neg (by decide), innerCandidates22_36_checked]
  change scan edge (search edge 3) 4 [97] 97 innerTail22_37
    (if 3 ≤ innerCandidates22_36.length then search edge 3 [97, 6301] innerCandidates22_36 6398 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan22_37

def innerTail22_35 : List Nat := 6271 :: innerTail22_36
def innerCandidates22_35 : List Nat := [8011]
set_option maxRecDepth 4096 in
theorem innerCandidates22_35_checked : candidates edge 4 97 6271 innerTail22_36 best.bound = innerCandidates22_35 := by
  unfold candidates
  have row : edge 6271 = SearchData.row6271.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan22_35 : scan edge (search edge 3) 4 [97] 97 innerTail22_35 best = best := by
  rw [innerTail22_35, scan, if_neg (by decide), innerCandidates22_35_checked]
  change scan edge (search edge 3) 4 [97] 97 innerTail22_36
    (if 3 ≤ innerCandidates22_35.length then search edge 3 [97, 6271] innerCandidates22_35 6368 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan22_36

def innerTail22_34 : List Nat := 5581 :: innerTail22_35
def innerCandidates22_34 : List Nat := [6991]
set_option maxRecDepth 4096 in
theorem innerCandidates22_34_checked : candidates edge 4 97 5581 innerTail22_35 best.bound = innerCandidates22_34 := by
  unfold candidates
  have row : edge 5581 = SearchData.row5581.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan22_34 : scan edge (search edge 3) 4 [97] 97 innerTail22_34 best = best := by
  rw [innerTail22_34, scan, if_neg (by decide), innerCandidates22_34_checked]
  change scan edge (search edge 3) 4 [97] 97 innerTail22_35
    (if 3 ≤ innerCandidates22_34.length then search edge 3 [97, 5581] innerCandidates22_34 5678 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan22_35

def innerTail22_33 : List Nat := 5281 :: innerTail22_34
def innerCandidates22_33 : List Nat := [8287]
set_option maxRecDepth 4096 in
theorem innerCandidates22_33_checked : candidates edge 4 97 5281 innerTail22_34 best.bound = innerCandidates22_33 := by
  unfold candidates
  have row : edge 5281 = SearchData.row5281.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan22_33 : scan edge (search edge 3) 4 [97] 97 innerTail22_33 best = best := by
  rw [innerTail22_33, scan, if_neg (by decide), innerCandidates22_33_checked]
  change scan edge (search edge 3) 4 [97] 97 innerTail22_34
    (if 3 ≤ innerCandidates22_33.length then search edge 3 [97, 5281] innerCandidates22_33 5378 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan22_34

def innerTail22_32 : List Nat := 5011 :: innerTail22_33
def innerCandidates22_32 : List Nat := [6271, 8053]
set_option maxRecDepth 4096 in
theorem innerCandidates22_32_checked : candidates edge 4 97 5011 innerTail22_33 best.bound = innerCandidates22_32 := by
  unfold candidates
  have row : edge 5011 = SearchData.row5011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan22_32 : scan edge (search edge 3) 4 [97] 97 innerTail22_32 best = best := by
  rw [innerTail22_32, scan, if_neg (by decide), innerCandidates22_32_checked]
  change scan edge (search edge 3) 4 [97] 97 innerTail22_33
    (if 3 ≤ innerCandidates22_32.length then search edge 3 [97, 5011] innerCandidates22_32 5108 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan22_33

def innerTail22_31 : List Nat := 4969 :: innerTail22_32
def innerCandidates22_31 : List Nat := [7507]
set_option maxRecDepth 4096 in
theorem innerCandidates22_31_checked : candidates edge 4 97 4969 innerTail22_32 best.bound = innerCandidates22_31 := by
  unfold candidates
  have row : edge 4969 = SearchData.row4969.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan22_31 : scan edge (search edge 3) 4 [97] 97 innerTail22_31 best = best := by
  rw [innerTail22_31, scan, if_neg (by decide), innerCandidates22_31_checked]
  change scan edge (search edge 3) 4 [97] 97 innerTail22_32
    (if 3 ≤ innerCandidates22_31.length then search edge 3 [97, 4969] innerCandidates22_31 5066 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan22_32

def innerTail22_30 : List Nat := 4957 :: innerTail22_31
def innerCandidates22_30 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates22_30_checked : candidates edge 4 97 4957 innerTail22_31 best.bound = innerCandidates22_30 := by
  unfold candidates
  have row : edge 4957 = SearchData.row4957.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan22_30 : scan edge (search edge 3) 4 [97] 97 innerTail22_30 best = best := by
  rw [innerTail22_30, scan, if_neg (by decide), innerCandidates22_30_checked]
  change scan edge (search edge 3) 4 [97] 97 innerTail22_31
    (if 3 ≤ innerCandidates22_30.length then search edge 3 [97, 4957] innerCandidates22_30 5054 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan22_31

def innerTail22_29 : List Nat := 4657 :: innerTail22_30
def innerCandidates22_29 : List Nat := [8209]
set_option maxRecDepth 4096 in
theorem innerCandidates22_29_checked : candidates edge 4 97 4657 innerTail22_30 best.bound = innerCandidates22_29 := by
  unfold candidates
  have row : edge 4657 = SearchData.row4657.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan22_29 : scan edge (search edge 3) 4 [97] 97 innerTail22_29 best = best := by
  rw [innerTail22_29, scan, if_neg (by decide), innerCandidates22_29_checked]
  change scan edge (search edge 3) 4 [97] 97 innerTail22_30
    (if 3 ≤ innerCandidates22_29.length then search edge 3 [97, 4657] innerCandidates22_29 4754 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan22_30

def innerTail22_28 : List Nat := 4513 :: innerTail22_29
def innerCandidates22_28 : List Nat := [5581, 8287]
set_option maxRecDepth 4096 in
theorem innerCandidates22_28_checked : candidates edge 4 97 4513 innerTail22_29 best.bound = innerCandidates22_28 := by
  unfold candidates
  have row : edge 4513 = SearchData.row4513.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan22_28 : scan edge (search edge 3) 4 [97] 97 innerTail22_28 best = best := by
  rw [innerTail22_28, scan, if_neg (by decide), innerCandidates22_28_checked]
  change scan edge (search edge 3) 4 [97] 97 innerTail22_29
    (if 3 ≤ innerCandidates22_28.length then search edge 3 [97, 4513] innerCandidates22_28 4610 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan22_29

def innerTail22_27 : List Nat := 4507 :: innerTail22_28
def innerCandidates22_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates22_27_checked : candidates edge 4 97 4507 innerTail22_28 best.bound = innerCandidates22_27 := by
  unfold candidates
  have row : edge 4507 = SearchData.row4507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan22_27 : scan edge (search edge 3) 4 [97] 97 innerTail22_27 best = best := by
  rw [innerTail22_27, scan, if_neg (by decide), innerCandidates22_27_checked]
  change scan edge (search edge 3) 4 [97] 97 innerTail22_28
    (if 3 ≤ innerCandidates22_27.length then search edge 3 [97, 4507] innerCandidates22_27 4604 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan22_28

def innerTail22_26 : List Nat := 4261 :: innerTail22_27
def innerCandidates22_26 : List Nat := [5011, 5281]
set_option maxRecDepth 4096 in
theorem innerCandidates22_26_checked : candidates edge 4 97 4261 innerTail22_27 best.bound = innerCandidates22_26 := by
  unfold candidates
  have row : edge 4261 = SearchData.row4261.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan22_26 : scan edge (search edge 3) 4 [97] 97 innerTail22_26 best = best := by
  rw [innerTail22_26, scan, if_neg (by decide), innerCandidates22_26_checked]
  change scan edge (search edge 3) 4 [97] 97 innerTail22_27
    (if 3 ≤ innerCandidates22_26.length then search edge 3 [97, 4261] innerCandidates22_26 4358 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan22_27

def innerTail22_25 : List Nat := 3853 :: innerTail22_26
def innerCandidates22_25 : List Nat := [6991]
set_option maxRecDepth 4096 in
theorem innerCandidates22_25_checked : candidates edge 4 97 3853 innerTail22_26 best.bound = innerCandidates22_25 := by
  unfold candidates
  have row : edge 3853 = SearchData.row3853.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan22_25 : scan edge (search edge 3) 4 [97] 97 innerTail22_25 best = best := by
  rw [innerTail22_25, scan, if_neg (by decide), innerCandidates22_25_checked]
  change scan edge (search edge 3) 4 [97] 97 innerTail22_26
    (if 3 ≤ innerCandidates22_25.length then search edge 3 [97, 3853] innerCandidates22_25 3950 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan22_26

def innerTail22_24 : List Nat := 3727 :: innerTail22_25
def innerCandidates22_24 : List Nat := [5011]
set_option maxRecDepth 4096 in
theorem innerCandidates22_24_checked : candidates edge 4 97 3727 innerTail22_25 best.bound = innerCandidates22_24 := by
  unfold candidates
  have row : edge 3727 = SearchData.row3727.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan22_24 : scan edge (search edge 3) 4 [97] 97 innerTail22_24 best = best := by
  rw [innerTail22_24, scan, if_neg (by decide), innerCandidates22_24_checked]
  change scan edge (search edge 3) 4 [97] 97 innerTail22_25
    (if 3 ≤ innerCandidates22_24.length then search edge 3 [97, 3727] innerCandidates22_24 3824 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan22_25

def innerTail22_23 : List Nat := 3691 :: innerTail22_24
def innerCandidates22_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates22_23_checked : candidates edge 4 97 3691 innerTail22_24 best.bound = innerCandidates22_23 := by
  unfold candidates
  have row : edge 3691 = SearchData.row3691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan22_23 : scan edge (search edge 3) 4 [97] 97 innerTail22_23 best = best := by
  rw [innerTail22_23, scan, if_neg (by decide), innerCandidates22_23_checked]
  change scan edge (search edge 3) 4 [97] 97 innerTail22_24
    (if 3 ≤ innerCandidates22_23.length then search edge 3 [97, 3691] innerCandidates22_23 3788 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan22_24

def innerTail22_22 : List Nat := 3373 :: innerTail22_23
def innerCandidates22_22 : List Nat := [4957]
set_option maxRecDepth 4096 in
theorem innerCandidates22_22_checked : candidates edge 4 97 3373 innerTail22_23 best.bound = innerCandidates22_22 := by
  unfold candidates
  have row : edge 3373 = SearchData.row3373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan22_22 : scan edge (search edge 3) 4 [97] 97 innerTail22_22 best = best := by
  rw [innerTail22_22, scan, if_neg (by decide), innerCandidates22_22_checked]
  change scan edge (search edge 3) 4 [97] 97 innerTail22_23
    (if 3 ≤ innerCandidates22_22.length then search edge 3 [97, 3373] innerCandidates22_22 3470 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan22_23

def innerTail22_21 : List Nat := 3331 :: innerTail22_22
def innerCandidates22_21 : List Nat := [4261, 7927]
set_option maxRecDepth 4096 in
theorem innerCandidates22_21_checked : candidates edge 4 97 3331 innerTail22_22 best.bound = innerCandidates22_21 := by
  unfold candidates
  have row : edge 3331 = SearchData.row3331.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan22_21 : scan edge (search edge 3) 4 [97] 97 innerTail22_21 best = best := by
  rw [innerTail22_21, scan, if_neg (by decide), innerCandidates22_21_checked]
  change scan edge (search edge 3) 4 [97] 97 innerTail22_22
    (if 3 ≤ innerCandidates22_21.length then search edge 3 [97, 3331] innerCandidates22_21 3428 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan22_22

def innerTail22_20 : List Nat := 2833 :: innerTail22_21
def innerCandidates22_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates22_20_checked : candidates edge 4 97 2833 innerTail22_21 best.bound = innerCandidates22_20 := by
  unfold candidates
  have row : edge 2833 = SearchData.row2833.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan22_20 : scan edge (search edge 3) 4 [97] 97 innerTail22_20 best = best := by
  rw [innerTail22_20, scan, if_neg (by decide), innerCandidates22_20_checked]
  change scan edge (search edge 3) 4 [97] 97 innerTail22_21
    (if 3 ≤ innerCandidates22_20.length then search edge 3 [97, 2833] innerCandidates22_20 2930 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan22_21

def innerTail22_19 : List Nat := 2221 :: innerTail22_20
def innerCandidates22_19 : List Nat := [4513, 5581, 7681]
set_option maxRecDepth 4096 in
theorem innerCandidates22_19_checked : candidates edge 4 97 2221 innerTail22_20 best.bound = innerCandidates22_19 := by
  unfold candidates
  have row : edge 2221 = PrimePairCache.smallRow2221.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild22_19 : search edge 3 [97, 2221] innerCandidates22_19 2318 best = best := by decide

theorem innerScan22_19 : scan edge (search edge 3) 4 [97] 97 innerTail22_19 best = best := by
  rw [innerTail22_19, scan, if_neg (by decide), innerCandidates22_19_checked]
  change scan edge (search edge 3) 4 [97] 97 innerTail22_20
    (if 3 ≤ innerCandidates22_19.length then search edge 3 [97, 2221] innerCandidates22_19 2318 best else best) = best
  rw [if_pos (by decide), innerChild22_19]
  exact innerScan22_20

def innerTail22_18 : List Nat := 2017 :: innerTail22_19
def innerCandidates22_18 : List Nat := [2221]
set_option maxRecDepth 4096 in
theorem innerCandidates22_18_checked : candidates edge 4 97 2017 innerTail22_19 best.bound = innerCandidates22_18 := by
  unfold candidates
  have row : edge 2017 = PrimePairCache.smallRow2017.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan22_18 : scan edge (search edge 3) 4 [97] 97 innerTail22_18 best = best := by
  rw [innerTail22_18, scan, if_neg (by decide), innerCandidates22_18_checked]
  change scan edge (search edge 3) 4 [97] 97 innerTail22_19
    (if 3 ≤ innerCandidates22_18.length then search edge 3 [97, 2017] innerCandidates22_18 2114 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan22_19

def innerTail22_17 : List Nat := 1951 :: innerTail22_18
def innerCandidates22_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates22_17_checked : candidates edge 4 97 1951 innerTail22_18 best.bound = innerCandidates22_17 := by
  unfold candidates
  have row : edge 1951 = PrimePairCache.smallRow1951.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan22_17 : scan edge (search edge 3) 4 [97] 97 innerTail22_17 best = best := by
  rw [innerTail22_17, scan, if_neg (by decide), innerCandidates22_17_checked]
  change scan edge (search edge 3) 4 [97] 97 innerTail22_18
    (if 3 ≤ innerCandidates22_17.length then search edge 3 [97, 1951] innerCandidates22_17 2048 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan22_18

def innerTail22_16 : List Nat := 1783 :: innerTail22_17
def innerCandidates22_16 : List Nat := [5011, 8209]
set_option maxRecDepth 4096 in
theorem innerCandidates22_16_checked : candidates edge 4 97 1783 innerTail22_17 best.bound = innerCandidates22_16 := by
  unfold candidates
  have row : edge 1783 = PrimePairCache.smallRow1783.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan22_16 : scan edge (search edge 3) 4 [97] 97 innerTail22_16 best = best := by
  rw [innerTail22_16, scan, if_neg (by decide), innerCandidates22_16_checked]
  change scan edge (search edge 3) 4 [97] 97 innerTail22_17
    (if 3 ≤ innerCandidates22_16.length then search edge 3 [97, 1783] innerCandidates22_16 1880 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan22_17

def innerTail22_15 : List Nat := 1381 :: innerTail22_16
def innerCandidates22_15 : List Nat := [6571]
set_option maxRecDepth 4096 in
theorem innerCandidates22_15_checked : candidates edge 4 97 1381 innerTail22_16 best.bound = innerCandidates22_15 := by
  unfold candidates
  have row : edge 1381 = PrimePairCache.smallRow1381.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan22_15 : scan edge (search edge 3) 4 [97] 97 innerTail22_15 best = best := by
  rw [innerTail22_15, scan, if_neg (by decide), innerCandidates22_15_checked]
  change scan edge (search edge 3) 4 [97] 97 innerTail22_16
    (if 3 ≤ innerCandidates22_15.length then search edge 3 [97, 1381] innerCandidates22_15 1478 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan22_16

def innerTail22_14 : List Nat := 1291 :: innerTail22_15
def innerCandidates22_14 : List Nat := [4513, 8209]
set_option maxRecDepth 4096 in
theorem innerCandidates22_14_checked : candidates edge 4 97 1291 innerTail22_15 best.bound = innerCandidates22_14 := by
  unfold candidates
  have row : edge 1291 = PrimePairCache.smallRow1291.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan22_14 : scan edge (search edge 3) 4 [97] 97 innerTail22_14 best = best := by
  rw [innerTail22_14, scan, if_neg (by decide), innerCandidates22_14_checked]
  change scan edge (search edge 3) 4 [97] 97 innerTail22_15
    (if 3 ≤ innerCandidates22_14.length then search edge 3 [97, 1291] innerCandidates22_14 1388 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan22_15

def innerTail22_13 : List Nat := 1279 :: innerTail22_14
def innerCandidates22_13 : List Nat := [3691, 6271]
set_option maxRecDepth 4096 in
theorem innerCandidates22_13_checked : candidates edge 4 97 1279 innerTail22_14 best.bound = innerCandidates22_13 := by
  unfold candidates
  have row : edge 1279 = PrimePairCache.smallRow1279.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan22_13 : scan edge (search edge 3) 4 [97] 97 innerTail22_13 best = best := by
  rw [innerTail22_13, scan, if_neg (by decide), innerCandidates22_13_checked]
  change scan edge (search edge 3) 4 [97] 97 innerTail22_14
    (if 3 ≤ innerCandidates22_13.length then search edge 3 [97, 1279] innerCandidates22_13 1376 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan22_14

def innerTail22_12 : List Nat := 1093 :: innerTail22_13
def innerCandidates22_12 : List Nat := [3331]
set_option maxRecDepth 4096 in
theorem innerCandidates22_12_checked : candidates edge 4 97 1093 innerTail22_13 best.bound = innerCandidates22_12 := by
  unfold candidates
  have row : edge 1093 = PrimePairCache.smallRow1093.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan22_12 : scan edge (search edge 3) 4 [97] 97 innerTail22_12 best = best := by
  rw [innerTail22_12, scan, if_neg (by decide), innerCandidates22_12_checked]
  change scan edge (search edge 3) 4 [97] 97 innerTail22_13
    (if 3 ≤ innerCandidates22_12.length then search edge 3 [97, 1093] innerCandidates22_12 1190 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan22_13

def innerTail22_11 : List Nat := 1063 :: innerTail22_12
def innerCandidates22_11 : List Nat := [7507]
set_option maxRecDepth 4096 in
theorem innerCandidates22_11_checked : candidates edge 4 97 1063 innerTail22_12 best.bound = innerCandidates22_11 := by
  unfold candidates
  have row : edge 1063 = PrimePairCache.smallRow1063.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan22_11 : scan edge (search edge 3) 4 [97] 97 innerTail22_11 best = best := by
  rw [innerTail22_11, scan, if_neg (by decide), innerCandidates22_11_checked]
  change scan edge (search edge 3) 4 [97] 97 innerTail22_12
    (if 3 ≤ innerCandidates22_11.length then search edge 3 [97, 1063] innerCandidates22_11 1160 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan22_12

def innerTail22_10 : List Nat := 1039 :: innerTail22_11
def innerCandidates22_10 : List Nat := [1279]
set_option maxRecDepth 4096 in
theorem innerCandidates22_10_checked : candidates edge 4 97 1039 innerTail22_11 best.bound = innerCandidates22_10 := by
  unfold candidates
  have row : edge 1039 = PrimePairCache.smallRow1039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan22_10 : scan edge (search edge 3) 4 [97] 97 innerTail22_10 best = best := by
  rw [innerTail22_10, scan, if_neg (by decide), innerCandidates22_10_checked]
  change scan edge (search edge 3) 4 [97] 97 innerTail22_11
    (if 3 ≤ innerCandidates22_10.length then search edge 3 [97, 1039] innerCandidates22_10 1136 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan22_11

def innerTail22_9 : List Nat := 1021 :: innerTail22_10
def innerCandidates22_9 : List Nat := [5281]
set_option maxRecDepth 4096 in
theorem innerCandidates22_9_checked : candidates edge 4 97 1021 innerTail22_10 best.bound = innerCandidates22_9 := by
  unfold candidates
  have row : edge 1021 = PrimePairCache.smallRow1021.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan22_9 : scan edge (search edge 3) 4 [97] 97 innerTail22_9 best = best := by
  rw [innerTail22_9, scan, if_neg (by decide), innerCandidates22_9_checked]
  change scan edge (search edge 3) 4 [97] 97 innerTail22_10
    (if 3 ≤ innerCandidates22_9.length then search edge 3 [97, 1021] innerCandidates22_9 1118 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan22_10

def innerTail22_8 : List Nat := 967 :: innerTail22_9
def innerCandidates22_8 : List Nat := [1381, 1951, 4507, 5011, 8269]
set_option maxRecDepth 4096 in
theorem innerCandidates22_8_checked : candidates edge 4 97 967 innerTail22_9 best.bound = innerCandidates22_8 := by
  unfold candidates
  have row : edge 967 = PrimePairCache.smallRow967.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild22_8 : search edge 3 [97, 967] innerCandidates22_8 1064 best = best := by decide

theorem innerScan22_8 : scan edge (search edge 3) 4 [97] 97 innerTail22_8 best = best := by
  rw [innerTail22_8, scan, if_neg (by decide), innerCandidates22_8_checked]
  change scan edge (search edge 3) 4 [97] 97 innerTail22_9
    (if 3 ≤ innerCandidates22_8.length then search edge 3 [97, 967] innerCandidates22_8 1064 best else best) = best
  rw [if_pos (by decide), innerChild22_8]
  exact innerScan22_9

def innerTail22_7 : List Nat := 919 :: innerTail22_8
def innerCandidates22_7 : List Nat := [1279, 2017]
set_option maxRecDepth 4096 in
theorem innerCandidates22_7_checked : candidates edge 4 97 919 innerTail22_8 best.bound = innerCandidates22_7 := by
  unfold candidates
  have row : edge 919 = PrimePairCache.smallRow919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan22_7 : scan edge (search edge 3) 4 [97] 97 innerTail22_7 best = best := by
  rw [innerTail22_7, scan, if_neg (by decide), innerCandidates22_7_checked]
  change scan edge (search edge 3) 4 [97] 97 innerTail22_8
    (if 3 ≤ innerCandidates22_7.length then search edge 3 [97, 919] innerCandidates22_7 1016 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan22_8

def innerTail22_6 : List Nat := 883 :: innerTail22_7
def innerCandidates22_6 : List Nat := [1783, 6571]
set_option maxRecDepth 4096 in
theorem innerCandidates22_6_checked : candidates edge 4 97 883 innerTail22_7 best.bound = innerCandidates22_6 := by
  unfold candidates
  have row : edge 883 = PrimePairCache.smallRow883.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan22_6 : scan edge (search edge 3) 4 [97] 97 innerTail22_6 best = best := by
  rw [innerTail22_6, scan, if_neg (by decide), innerCandidates22_6_checked]
  change scan edge (search edge 3) 4 [97] 97 innerTail22_7
    (if 3 ≤ innerCandidates22_6.length then search edge 3 [97, 883] innerCandidates22_6 980 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan22_7

def innerTail22_5 : List Nat := 829 :: innerTail22_6
def innerCandidates22_5 : List Nat := [8209]
set_option maxRecDepth 4096 in
theorem innerCandidates22_5_checked : candidates edge 4 97 829 innerTail22_6 best.bound = innerCandidates22_5 := by
  unfold candidates
  have row : edge 829 = PrimePairCache.smallRow829.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan22_5 : scan edge (search edge 3) 4 [97] 97 innerTail22_5 best = best := by
  rw [innerTail22_5, scan, if_neg (by decide), innerCandidates22_5_checked]
  change scan edge (search edge 3) 4 [97] 97 innerTail22_6
    (if 3 ≤ innerCandidates22_5.length then search edge 3 [97, 829] innerCandidates22_5 926 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan22_6

def innerTail22_4 : List Nat := 787 :: innerTail22_5
def innerCandidates22_4 : List Nat := [2017, 4507]
set_option maxRecDepth 4096 in
theorem innerCandidates22_4_checked : candidates edge 4 97 787 innerTail22_5 best.bound = innerCandidates22_4 := by
  unfold candidates
  have row : edge 787 = PrimePairCache.smallRow787.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan22_4 : scan edge (search edge 3) 4 [97] 97 innerTail22_4 best = best := by
  rw [innerTail22_4, scan, if_neg (by decide), innerCandidates22_4_checked]
  change scan edge (search edge 3) 4 [97] 97 innerTail22_5
    (if 3 ≤ innerCandidates22_4.length then search edge 3 [97, 787] innerCandidates22_4 884 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan22_5

def innerTail22_3 : List Nat := 379 :: innerTail22_4
def innerCandidates22_3 : List Nat := [1063, 2221, 5581]
set_option maxRecDepth 4096 in
theorem innerCandidates22_3_checked : candidates edge 4 97 379 innerTail22_4 best.bound = innerCandidates22_3 := by
  unfold candidates
  have row : edge 379 = PrimePairCache.smallRow379.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild22_3 : search edge 3 [97, 379] innerCandidates22_3 476 best = best := by decide

theorem innerScan22_3 : scan edge (search edge 3) 4 [97] 97 innerTail22_3 best = best := by
  rw [innerTail22_3, scan, if_neg (by decide), innerCandidates22_3_checked]
  change scan edge (search edge 3) 4 [97] 97 innerTail22_4
    (if 3 ≤ innerCandidates22_3.length then search edge 3 [97, 379] innerCandidates22_3 476 best else best) = best
  rw [if_pos (by decide), innerChild22_3]
  exact innerScan22_4

def innerTail22_2 : List Nat := 373 :: innerTail22_3
def innerCandidates22_2 : List Nat := [4957]
set_option maxRecDepth 4096 in
theorem innerCandidates22_2_checked : candidates edge 4 97 373 innerTail22_3 best.bound = innerCandidates22_2 := by
  unfold candidates
  have row : edge 373 = PrimePairCache.smallRow373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan22_2 : scan edge (search edge 3) 4 [97] 97 innerTail22_2 best = best := by
  rw [innerTail22_2, scan, if_neg (by decide), innerCandidates22_2_checked]
  change scan edge (search edge 3) 4 [97] 97 innerTail22_3
    (if 3 ≤ innerCandidates22_2.length then search edge 3 [97, 373] innerCandidates22_2 470 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan22_3

def innerTail22_1 : List Nat := 241 :: innerTail22_2
def innerCandidates22_1 : List Nat := [883, 2833]
set_option maxRecDepth 4096 in
theorem innerCandidates22_1_checked : candidates edge 4 97 241 innerTail22_2 best.bound = innerCandidates22_1 := by
  unfold candidates
  have row : edge 241 = PrimePairCache.smallRow241.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan22_1 : scan edge (search edge 3) 4 [97] 97 innerTail22_1 best = best := by
  rw [innerTail22_1, scan, if_neg (by decide), innerCandidates22_1_checked]
  change scan edge (search edge 3) 4 [97] 97 innerTail22_2
    (if 3 ≤ innerCandidates22_1.length then search edge 3 [97, 241] innerCandidates22_1 338 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan22_2

def innerTail22_0 : List Nat := 157 :: innerTail22_1
def innerCandidates22_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates22_0_checked : candidates edge 4 97 157 innerTail22_1 best.bound = innerCandidates22_0 := by
  unfold candidates
  have row : edge 157 = PrimePairCache.smallRow157.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan22_0 : scan edge (search edge 3) 4 [97] 97 innerTail22_0 best = best := by
  rw [innerTail22_0, scan, if_neg (by decide), innerCandidates22_0_checked]
  change scan edge (search edge 3) 4 [97] 97 innerTail22_1
    (if 3 ≤ innerCandidates22_0.length then search edge 3 [97, 157] innerCandidates22_0 254 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan22_1

theorem rootBranch22 : search edge 4 [97] rootCandidates22 97 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [97] 97 innerTail22_0 best = best
  exact innerScan22_0

end ProjectEulerSolutions.PrimePairUpper.Search
