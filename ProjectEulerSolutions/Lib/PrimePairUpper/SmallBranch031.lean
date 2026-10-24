import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates799 : List Nat := [6571, 7549, 7933, 8221, 8311, 8329]
set_option maxRecDepth 4096 in
theorem rootCandidates799_checked : candidates edge 5 0 6151 tail800 best.bound = rootCandidates799 := by
  unfold candidates
  have row : edge 6151 = SearchData.row6151.contains := by funext q; rfl
  rw [row]
  decide

def innerTail799_6 : List Nat := []
theorem innerScan799_6 : scan edge (search edge 3) 4 [6151] 6151 innerTail799_6 best = best := rfl

def innerTail799_5 : List Nat := 8329 :: innerTail799_6
def innerCandidates799_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates799_5_checked : candidates edge 4 6151 8329 innerTail799_6 best.bound = innerCandidates799_5 := by
  unfold candidates
  have row : edge 8329 = SearchData.row8329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan799_5 : scan edge (search edge 3) 4 [6151] 6151 innerTail799_5 best = best := by
  rw [innerTail799_5, scan, if_neg (by decide), innerCandidates799_5_checked]
  change scan edge (search edge 3) 4 [6151] 6151 innerTail799_6
    (if 3 ≤ innerCandidates799_5.length then search edge 3 [6151, 8329] innerCandidates799_5 14480 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan799_6

def innerTail799_4 : List Nat := 8311 :: innerTail799_5
def innerCandidates799_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates799_4_checked : candidates edge 4 6151 8311 innerTail799_5 best.bound = innerCandidates799_4 := by
  unfold candidates
  have row : edge 8311 = SearchData.row8311.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan799_4 : scan edge (search edge 3) 4 [6151] 6151 innerTail799_4 best = best := by
  rw [innerTail799_4, scan, if_neg (by decide), innerCandidates799_4_checked]
  change scan edge (search edge 3) 4 [6151] 6151 innerTail799_5
    (if 3 ≤ innerCandidates799_4.length then search edge 3 [6151, 8311] innerCandidates799_4 14462 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan799_5

def innerTail799_3 : List Nat := 8221 :: innerTail799_4
def innerCandidates799_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates799_3_checked : candidates edge 4 6151 8221 innerTail799_4 best.bound = innerCandidates799_3 := by
  unfold candidates
  have row : edge 8221 = SearchData.row8221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan799_3 : scan edge (search edge 3) 4 [6151] 6151 innerTail799_3 best = best := by
  rw [innerTail799_3, scan, if_neg (by decide), innerCandidates799_3_checked]
  change scan edge (search edge 3) 4 [6151] 6151 innerTail799_4
    (if 3 ≤ innerCandidates799_3.length then search edge 3 [6151, 8221] innerCandidates799_3 14372 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan799_4

def innerTail799_2 : List Nat := 7933 :: innerTail799_3
def innerCandidates799_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates799_2_checked : candidates edge 4 6151 7933 innerTail799_3 best.bound = innerCandidates799_2 := by
  unfold candidates
  have row : edge 7933 = SearchData.row7933.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan799_2 : scan edge (search edge 3) 4 [6151] 6151 innerTail799_2 best = best := by
  rw [innerTail799_2, scan, if_neg (by decide), innerCandidates799_2_checked]
  change scan edge (search edge 3) 4 [6151] 6151 innerTail799_3
    (if 3 ≤ innerCandidates799_2.length then search edge 3 [6151, 7933] innerCandidates799_2 14084 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan799_3

def innerTail799_1 : List Nat := 7549 :: innerTail799_2
def innerCandidates799_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates799_1_checked : candidates edge 4 6151 7549 innerTail799_2 best.bound = innerCandidates799_1 := by
  unfold candidates
  have row : edge 7549 = SearchData.row7549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan799_1 : scan edge (search edge 3) 4 [6151] 6151 innerTail799_1 best = best := by
  rw [innerTail799_1, scan, if_neg (by decide), innerCandidates799_1_checked]
  change scan edge (search edge 3) 4 [6151] 6151 innerTail799_2
    (if 3 ≤ innerCandidates799_1.length then search edge 3 [6151, 7549] innerCandidates799_1 13700 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan799_2

def innerTail799_0 : List Nat := 6571 :: innerTail799_1
def innerCandidates799_0 : List Nat := [7549]
set_option maxRecDepth 4096 in
theorem innerCandidates799_0_checked : candidates edge 4 6151 6571 innerTail799_1 best.bound = innerCandidates799_0 := by
  unfold candidates
  have row : edge 6571 = SearchData.row6571.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan799_0 : scan edge (search edge 3) 4 [6151] 6151 innerTail799_0 best = best := by
  rw [innerTail799_0, scan, if_neg (by decide), innerCandidates799_0_checked]
  change scan edge (search edge 3) 4 [6151] 6151 innerTail799_1
    (if 3 ≤ innerCandidates799_0.length then search edge 3 [6151, 6571] innerCandidates799_0 12722 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan799_1

theorem rootBranch799 : search edge 4 [6151] rootCandidates799 6151 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6151] 6151 innerTail799_0 best = best
  exact innerScan799_0

def rootCandidates798 : List Nat := [6311, 6689, 6833, 7151, 8171, 8219]
set_option maxRecDepth 4096 in
theorem rootCandidates798_checked : candidates edge 5 0 6143 tail799 best.bound = rootCandidates798 := by
  unfold candidates
  have row : edge 6143 = SearchData.row6143.contains := by funext q; rfl
  rw [row]
  decide

def innerTail798_6 : List Nat := []
theorem innerScan798_6 : scan edge (search edge 3) 4 [6143] 6143 innerTail798_6 best = best := rfl

def innerTail798_5 : List Nat := 8219 :: innerTail798_6
def innerCandidates798_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates798_5_checked : candidates edge 4 6143 8219 innerTail798_6 best.bound = innerCandidates798_5 := by
  unfold candidates
  have row : edge 8219 = SearchData.row8219.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan798_5 : scan edge (search edge 3) 4 [6143] 6143 innerTail798_5 best = best := by
  rw [innerTail798_5, scan, if_neg (by decide), innerCandidates798_5_checked]
  change scan edge (search edge 3) 4 [6143] 6143 innerTail798_6
    (if 3 ≤ innerCandidates798_5.length then search edge 3 [6143, 8219] innerCandidates798_5 14362 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan798_6

def innerTail798_4 : List Nat := 8171 :: innerTail798_5
def innerCandidates798_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates798_4_checked : candidates edge 4 6143 8171 innerTail798_5 best.bound = innerCandidates798_4 := by
  unfold candidates
  have row : edge 8171 = SearchData.row8171.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan798_4 : scan edge (search edge 3) 4 [6143] 6143 innerTail798_4 best = best := by
  rw [innerTail798_4, scan, if_neg (by decide), innerCandidates798_4_checked]
  change scan edge (search edge 3) 4 [6143] 6143 innerTail798_5
    (if 3 ≤ innerCandidates798_4.length then search edge 3 [6143, 8171] innerCandidates798_4 14314 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan798_5

def innerTail798_3 : List Nat := 7151 :: innerTail798_4
def innerCandidates798_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates798_3_checked : candidates edge 4 6143 7151 innerTail798_4 best.bound = innerCandidates798_3 := by
  unfold candidates
  have row : edge 7151 = SearchData.row7151.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan798_3 : scan edge (search edge 3) 4 [6143] 6143 innerTail798_3 best = best := by
  rw [innerTail798_3, scan, if_neg (by decide), innerCandidates798_3_checked]
  change scan edge (search edge 3) 4 [6143] 6143 innerTail798_4
    (if 3 ≤ innerCandidates798_3.length then search edge 3 [6143, 7151] innerCandidates798_3 13294 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan798_4

def innerTail798_2 : List Nat := 6833 :: innerTail798_3
def innerCandidates798_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates798_2_checked : candidates edge 4 6143 6833 innerTail798_3 best.bound = innerCandidates798_2 := by
  unfold candidates
  have row : edge 6833 = SearchData.row6833.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan798_2 : scan edge (search edge 3) 4 [6143] 6143 innerTail798_2 best = best := by
  rw [innerTail798_2, scan, if_neg (by decide), innerCandidates798_2_checked]
  change scan edge (search edge 3) 4 [6143] 6143 innerTail798_3
    (if 3 ≤ innerCandidates798_2.length then search edge 3 [6143, 6833] innerCandidates798_2 12976 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan798_3

def innerTail798_1 : List Nat := 6689 :: innerTail798_2
def innerCandidates798_1 : List Nat := [7151]
set_option maxRecDepth 4096 in
theorem innerCandidates798_1_checked : candidates edge 4 6143 6689 innerTail798_2 best.bound = innerCandidates798_1 := by
  unfold candidates
  have row : edge 6689 = SearchData.row6689.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan798_1 : scan edge (search edge 3) 4 [6143] 6143 innerTail798_1 best = best := by
  rw [innerTail798_1, scan, if_neg (by decide), innerCandidates798_1_checked]
  change scan edge (search edge 3) 4 [6143] 6143 innerTail798_2
    (if 3 ≤ innerCandidates798_1.length then search edge 3 [6143, 6689] innerCandidates798_1 12832 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan798_2

def innerTail798_0 : List Nat := 6311 :: innerTail798_1
def innerCandidates798_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates798_0_checked : candidates edge 4 6143 6311 innerTail798_1 best.bound = innerCandidates798_0 := by
  unfold candidates
  have row : edge 6311 = SearchData.row6311.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan798_0 : scan edge (search edge 3) 4 [6143] 6143 innerTail798_0 best = best := by
  rw [innerTail798_0, scan, if_neg (by decide), innerCandidates798_0_checked]
  change scan edge (search edge 3) 4 [6143] 6143 innerTail798_1
    (if 3 ≤ innerCandidates798_0.length then search edge 3 [6143, 6311] innerCandidates798_0 12454 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan798_1

theorem rootBranch798 : search edge 4 [6143] rootCandidates798 6143 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6143] 6143 innerTail798_0 best = best
  exact innerScan798_0

def rootCandidates797 : List Nat := [6871, 7741, 8353]
set_option maxRecDepth 4096 in
theorem rootCandidates797_checked : candidates edge 5 0 6133 tail798 best.bound = rootCandidates797 := by
  unfold candidates
  have row : edge 6133 = SearchData.row6133.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates796 : List Nat := [6863, 6947, 7019, 7577, 8123]
set_option maxRecDepth 4096 in
theorem rootCandidates796_checked : candidates edge 5 0 6131 tail797 best.bound = rootCandidates796 := by
  unfold candidates
  have row : edge 6131 = SearchData.row6131.contains := by funext q; rfl
  rw [row]
  decide

def innerTail796_5 : List Nat := []
theorem innerScan796_5 : scan edge (search edge 3) 4 [6131] 6131 innerTail796_5 best = best := rfl

def innerTail796_4 : List Nat := 8123 :: innerTail796_5
def innerCandidates796_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates796_4_checked : candidates edge 4 6131 8123 innerTail796_5 best.bound = innerCandidates796_4 := by
  unfold candidates
  have row : edge 8123 = SearchData.row8123.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan796_4 : scan edge (search edge 3) 4 [6131] 6131 innerTail796_4 best = best := by
  rw [innerTail796_4, scan, if_neg (by decide), innerCandidates796_4_checked]
  change scan edge (search edge 3) 4 [6131] 6131 innerTail796_5
    (if 3 ≤ innerCandidates796_4.length then search edge 3 [6131, 8123] innerCandidates796_4 14254 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan796_5

def innerTail796_3 : List Nat := 7577 :: innerTail796_4
def innerCandidates796_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates796_3_checked : candidates edge 4 6131 7577 innerTail796_4 best.bound = innerCandidates796_3 := by
  unfold candidates
  have row : edge 7577 = SearchData.row7577.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan796_3 : scan edge (search edge 3) 4 [6131] 6131 innerTail796_3 best = best := by
  rw [innerTail796_3, scan, if_neg (by decide), innerCandidates796_3_checked]
  change scan edge (search edge 3) 4 [6131] 6131 innerTail796_4
    (if 3 ≤ innerCandidates796_3.length then search edge 3 [6131, 7577] innerCandidates796_3 13708 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan796_4

def innerTail796_2 : List Nat := 7019 :: innerTail796_3
def innerCandidates796_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates796_2_checked : candidates edge 4 6131 7019 innerTail796_3 best.bound = innerCandidates796_2 := by
  unfold candidates
  have row : edge 7019 = SearchData.row7019.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan796_2 : scan edge (search edge 3) 4 [6131] 6131 innerTail796_2 best = best := by
  rw [innerTail796_2, scan, if_neg (by decide), innerCandidates796_2_checked]
  change scan edge (search edge 3) 4 [6131] 6131 innerTail796_3
    (if 3 ≤ innerCandidates796_2.length then search edge 3 [6131, 7019] innerCandidates796_2 13150 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan796_3

def innerTail796_1 : List Nat := 6947 :: innerTail796_2
def innerCandidates796_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates796_1_checked : candidates edge 4 6131 6947 innerTail796_2 best.bound = innerCandidates796_1 := by
  unfold candidates
  have row : edge 6947 = SearchData.row6947.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan796_1 : scan edge (search edge 3) 4 [6131] 6131 innerTail796_1 best = best := by
  rw [innerTail796_1, scan, if_neg (by decide), innerCandidates796_1_checked]
  change scan edge (search edge 3) 4 [6131] 6131 innerTail796_2
    (if 3 ≤ innerCandidates796_1.length then search edge 3 [6131, 6947] innerCandidates796_1 13078 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan796_2

def innerTail796_0 : List Nat := 6863 :: innerTail796_1
def innerCandidates796_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates796_0_checked : candidates edge 4 6131 6863 innerTail796_1 best.bound = innerCandidates796_0 := by
  unfold candidates
  have row : edge 6863 = SearchData.row6863.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan796_0 : scan edge (search edge 3) 4 [6131] 6131 innerTail796_0 best = best := by
  rw [innerTail796_0, scan, if_neg (by decide), innerCandidates796_0_checked]
  change scan edge (search edge 3) 4 [6131] 6131 innerTail796_1
    (if 3 ≤ innerCandidates796_0.length then search edge 3 [6131, 6863] innerCandidates796_0 12994 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan796_1

theorem rootBranch796 : search edge 4 [6131] rootCandidates796 6131 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6131] 6131 innerTail796_0 best = best
  exact innerScan796_0

def rootCandidates795 : List Nat := [6619, 6871, 7489, 7993]
set_option maxRecDepth 4096 in
theorem rootCandidates795_checked : candidates edge 5 0 6121 tail796 best.bound = rootCandidates795 := by
  unfold candidates
  have row : edge 6121 = SearchData.row6121.contains := by funext q; rfl
  rw [row]
  decide

def innerTail795_4 : List Nat := []
theorem innerScan795_4 : scan edge (search edge 3) 4 [6121] 6121 innerTail795_4 best = best := rfl

def innerTail795_3 : List Nat := 7993 :: innerTail795_4
def innerCandidates795_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates795_3_checked : candidates edge 4 6121 7993 innerTail795_4 best.bound = innerCandidates795_3 := by
  unfold candidates
  have row : edge 7993 = SearchData.row7993.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan795_3 : scan edge (search edge 3) 4 [6121] 6121 innerTail795_3 best = best := by
  rw [innerTail795_3, scan, if_neg (by decide), innerCandidates795_3_checked]
  change scan edge (search edge 3) 4 [6121] 6121 innerTail795_4
    (if 3 ≤ innerCandidates795_3.length then search edge 3 [6121, 7993] innerCandidates795_3 14114 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan795_4

def innerTail795_2 : List Nat := 7489 :: innerTail795_3
def innerCandidates795_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates795_2_checked : candidates edge 4 6121 7489 innerTail795_3 best.bound = innerCandidates795_2 := by
  unfold candidates
  have row : edge 7489 = SearchData.row7489.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan795_2 : scan edge (search edge 3) 4 [6121] 6121 innerTail795_2 best = best := by
  rw [innerTail795_2, scan, if_neg (by decide), innerCandidates795_2_checked]
  change scan edge (search edge 3) 4 [6121] 6121 innerTail795_3
    (if 3 ≤ innerCandidates795_2.length then search edge 3 [6121, 7489] innerCandidates795_2 13610 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan795_3

def innerTail795_1 : List Nat := 6871 :: innerTail795_2
def innerCandidates795_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates795_1_checked : candidates edge 4 6121 6871 innerTail795_2 best.bound = innerCandidates795_1 := by
  unfold candidates
  have row : edge 6871 = SearchData.row6871.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan795_1 : scan edge (search edge 3) 4 [6121] 6121 innerTail795_1 best = best := by
  rw [innerTail795_1, scan, if_neg (by decide), innerCandidates795_1_checked]
  change scan edge (search edge 3) 4 [6121] 6121 innerTail795_2
    (if 3 ≤ innerCandidates795_1.length then search edge 3 [6121, 6871] innerCandidates795_1 12992 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan795_2

def innerTail795_0 : List Nat := 6619 :: innerTail795_1
def innerCandidates795_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates795_0_checked : candidates edge 4 6121 6619 innerTail795_1 best.bound = innerCandidates795_0 := by
  unfold candidates
  have row : edge 6619 = SearchData.row6619.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan795_0 : scan edge (search edge 3) 4 [6121] 6121 innerTail795_0 best = best := by
  rw [innerTail795_0, scan, if_neg (by decide), innerCandidates795_0_checked]
  change scan edge (search edge 3) 4 [6121] 6121 innerTail795_1
    (if 3 ≤ innerCandidates795_0.length then search edge 3 [6121, 6619] innerCandidates795_0 12740 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan795_1

theorem rootBranch795 : search edge 4 [6121] rootCandidates795 6121 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6121] 6121 innerTail795_0 best = best
  exact innerScan795_0

def rootCandidates794 : List Nat := [6311, 6353, 6659, 6833, 7001, 7691, 7793]
set_option maxRecDepth 4096 in
theorem rootCandidates794_checked : candidates edge 5 0 6113 tail795 best.bound = rootCandidates794 := by
  unfold candidates
  have row : edge 6113 = SearchData.row6113.contains := by funext q; rfl
  rw [row]
  decide

def innerTail794_7 : List Nat := []
theorem innerScan794_7 : scan edge (search edge 3) 4 [6113] 6113 innerTail794_7 best = best := rfl

def innerTail794_6 : List Nat := 7793 :: innerTail794_7
def innerCandidates794_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates794_6_checked : candidates edge 4 6113 7793 innerTail794_7 best.bound = innerCandidates794_6 := by
  unfold candidates
  have row : edge 7793 = SearchData.row7793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan794_6 : scan edge (search edge 3) 4 [6113] 6113 innerTail794_6 best = best := by
  rw [innerTail794_6, scan, if_neg (by decide), innerCandidates794_6_checked]
  change scan edge (search edge 3) 4 [6113] 6113 innerTail794_7
    (if 3 ≤ innerCandidates794_6.length then search edge 3 [6113, 7793] innerCandidates794_6 13906 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan794_7

def innerTail794_5 : List Nat := 7691 :: innerTail794_6
def innerCandidates794_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates794_5_checked : candidates edge 4 6113 7691 innerTail794_6 best.bound = innerCandidates794_5 := by
  unfold candidates
  have row : edge 7691 = SearchData.row7691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan794_5 : scan edge (search edge 3) 4 [6113] 6113 innerTail794_5 best = best := by
  rw [innerTail794_5, scan, if_neg (by decide), innerCandidates794_5_checked]
  change scan edge (search edge 3) 4 [6113] 6113 innerTail794_6
    (if 3 ≤ innerCandidates794_5.length then search edge 3 [6113, 7691] innerCandidates794_5 13804 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan794_6

def innerTail794_4 : List Nat := 7001 :: innerTail794_5
def innerCandidates794_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates794_4_checked : candidates edge 4 6113 7001 innerTail794_5 best.bound = innerCandidates794_4 := by
  unfold candidates
  have row : edge 7001 = SearchData.row7001.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan794_4 : scan edge (search edge 3) 4 [6113] 6113 innerTail794_4 best = best := by
  rw [innerTail794_4, scan, if_neg (by decide), innerCandidates794_4_checked]
  change scan edge (search edge 3) 4 [6113] 6113 innerTail794_5
    (if 3 ≤ innerCandidates794_4.length then search edge 3 [6113, 7001] innerCandidates794_4 13114 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan794_5

def innerTail794_3 : List Nat := 6833 :: innerTail794_4
def innerCandidates794_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates794_3_checked : candidates edge 4 6113 6833 innerTail794_4 best.bound = innerCandidates794_3 := by
  unfold candidates
  have row : edge 6833 = SearchData.row6833.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan794_3 : scan edge (search edge 3) 4 [6113] 6113 innerTail794_3 best = best := by
  rw [innerTail794_3, scan, if_neg (by decide), innerCandidates794_3_checked]
  change scan edge (search edge 3) 4 [6113] 6113 innerTail794_4
    (if 3 ≤ innerCandidates794_3.length then search edge 3 [6113, 6833] innerCandidates794_3 12946 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan794_4

def innerTail794_2 : List Nat := 6659 :: innerTail794_3
def innerCandidates794_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates794_2_checked : candidates edge 4 6113 6659 innerTail794_3 best.bound = innerCandidates794_2 := by
  unfold candidates
  have row : edge 6659 = SearchData.row6659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan794_2 : scan edge (search edge 3) 4 [6113] 6113 innerTail794_2 best = best := by
  rw [innerTail794_2, scan, if_neg (by decide), innerCandidates794_2_checked]
  change scan edge (search edge 3) 4 [6113] 6113 innerTail794_3
    (if 3 ≤ innerCandidates794_2.length then search edge 3 [6113, 6659] innerCandidates794_2 12772 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan794_3

def innerTail794_1 : List Nat := 6353 :: innerTail794_2
def innerCandidates794_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates794_1_checked : candidates edge 4 6113 6353 innerTail794_2 best.bound = innerCandidates794_1 := by
  unfold candidates
  have row : edge 6353 = SearchData.row6353.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan794_1 : scan edge (search edge 3) 4 [6113] 6113 innerTail794_1 best = best := by
  rw [innerTail794_1, scan, if_neg (by decide), innerCandidates794_1_checked]
  change scan edge (search edge 3) 4 [6113] 6113 innerTail794_2
    (if 3 ≤ innerCandidates794_1.length then search edge 3 [6113, 6353] innerCandidates794_1 12466 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan794_2

def innerTail794_0 : List Nat := 6311 :: innerTail794_1
def innerCandidates794_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates794_0_checked : candidates edge 4 6113 6311 innerTail794_1 best.bound = innerCandidates794_0 := by
  unfold candidates
  have row : edge 6311 = SearchData.row6311.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan794_0 : scan edge (search edge 3) 4 [6113] 6113 innerTail794_0 best = best := by
  rw [innerTail794_0, scan, if_neg (by decide), innerCandidates794_0_checked]
  change scan edge (search edge 3) 4 [6113] 6113 innerTail794_1
    (if 3 ≤ innerCandidates794_0.length then search edge 3 [6113, 6311] innerCandidates794_0 12424 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan794_1

theorem rootBranch794 : search edge 4 [6113] rootCandidates794 6113 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6113] 6113 innerTail794_0 best = best
  exact innerScan794_0

def rootCandidates793 : List Nat := [6947, 7211, 7487, 7589, 7877, 7883, 8093, 8123]
set_option maxRecDepth 4096 in
theorem rootCandidates793_checked : candidates edge 5 0 6101 tail794 best.bound = rootCandidates793 := by
  unfold candidates
  have row : edge 6101 = SearchData.row6101.contains := by funext q; rfl
  rw [row]
  decide

def innerTail793_8 : List Nat := []
theorem innerScan793_8 : scan edge (search edge 3) 4 [6101] 6101 innerTail793_8 best = best := rfl

def innerTail793_7 : List Nat := 8123 :: innerTail793_8
def innerCandidates793_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates793_7_checked : candidates edge 4 6101 8123 innerTail793_8 best.bound = innerCandidates793_7 := by
  unfold candidates
  have row : edge 8123 = SearchData.row8123.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan793_7 : scan edge (search edge 3) 4 [6101] 6101 innerTail793_7 best = best := by
  rw [innerTail793_7, scan, if_neg (by decide), innerCandidates793_7_checked]
  change scan edge (search edge 3) 4 [6101] 6101 innerTail793_8
    (if 3 ≤ innerCandidates793_7.length then search edge 3 [6101, 8123] innerCandidates793_7 14224 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan793_8

def innerTail793_6 : List Nat := 8093 :: innerTail793_7
def innerCandidates793_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates793_6_checked : candidates edge 4 6101 8093 innerTail793_7 best.bound = innerCandidates793_6 := by
  unfold candidates
  have row : edge 8093 = SearchData.row8093.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan793_6 : scan edge (search edge 3) 4 [6101] 6101 innerTail793_6 best = best := by
  rw [innerTail793_6, scan, if_neg (by decide), innerCandidates793_6_checked]
  change scan edge (search edge 3) 4 [6101] 6101 innerTail793_7
    (if 3 ≤ innerCandidates793_6.length then search edge 3 [6101, 8093] innerCandidates793_6 14194 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan793_7

def innerTail793_5 : List Nat := 7883 :: innerTail793_6
def innerCandidates793_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates793_5_checked : candidates edge 4 6101 7883 innerTail793_6 best.bound = innerCandidates793_5 := by
  unfold candidates
  have row : edge 7883 = SearchData.row7883.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan793_5 : scan edge (search edge 3) 4 [6101] 6101 innerTail793_5 best = best := by
  rw [innerTail793_5, scan, if_neg (by decide), innerCandidates793_5_checked]
  change scan edge (search edge 3) 4 [6101] 6101 innerTail793_6
    (if 3 ≤ innerCandidates793_5.length then search edge 3 [6101, 7883] innerCandidates793_5 13984 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan793_6

def innerTail793_4 : List Nat := 7877 :: innerTail793_5
def innerCandidates793_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates793_4_checked : candidates edge 4 6101 7877 innerTail793_5 best.bound = innerCandidates793_4 := by
  unfold candidates
  have row : edge 7877 = SearchData.row7877.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan793_4 : scan edge (search edge 3) 4 [6101] 6101 innerTail793_4 best = best := by
  rw [innerTail793_4, scan, if_neg (by decide), innerCandidates793_4_checked]
  change scan edge (search edge 3) 4 [6101] 6101 innerTail793_5
    (if 3 ≤ innerCandidates793_4.length then search edge 3 [6101, 7877] innerCandidates793_4 13978 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan793_5

def innerTail793_3 : List Nat := 7589 :: innerTail793_4
def innerCandidates793_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates793_3_checked : candidates edge 4 6101 7589 innerTail793_4 best.bound = innerCandidates793_3 := by
  unfold candidates
  have row : edge 7589 = SearchData.row7589.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan793_3 : scan edge (search edge 3) 4 [6101] 6101 innerTail793_3 best = best := by
  rw [innerTail793_3, scan, if_neg (by decide), innerCandidates793_3_checked]
  change scan edge (search edge 3) 4 [6101] 6101 innerTail793_4
    (if 3 ≤ innerCandidates793_3.length then search edge 3 [6101, 7589] innerCandidates793_3 13690 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan793_4

def innerTail793_2 : List Nat := 7487 :: innerTail793_3
def innerCandidates793_2 : List Nat := [7883]
set_option maxRecDepth 4096 in
theorem innerCandidates793_2_checked : candidates edge 4 6101 7487 innerTail793_3 best.bound = innerCandidates793_2 := by
  unfold candidates
  have row : edge 7487 = SearchData.row7487.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan793_2 : scan edge (search edge 3) 4 [6101] 6101 innerTail793_2 best = best := by
  rw [innerTail793_2, scan, if_neg (by decide), innerCandidates793_2_checked]
  change scan edge (search edge 3) 4 [6101] 6101 innerTail793_3
    (if 3 ≤ innerCandidates793_2.length then search edge 3 [6101, 7487] innerCandidates793_2 13588 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan793_3

def innerTail793_1 : List Nat := 7211 :: innerTail793_2
def innerCandidates793_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates793_1_checked : candidates edge 4 6101 7211 innerTail793_2 best.bound = innerCandidates793_1 := by
  unfold candidates
  have row : edge 7211 = SearchData.row7211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan793_1 : scan edge (search edge 3) 4 [6101] 6101 innerTail793_1 best = best := by
  rw [innerTail793_1, scan, if_neg (by decide), innerCandidates793_1_checked]
  change scan edge (search edge 3) 4 [6101] 6101 innerTail793_2
    (if 3 ≤ innerCandidates793_1.length then search edge 3 [6101, 7211] innerCandidates793_1 13312 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan793_2

def innerTail793_0 : List Nat := 6947 :: innerTail793_1
def innerCandidates793_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates793_0_checked : candidates edge 4 6101 6947 innerTail793_1 best.bound = innerCandidates793_0 := by
  unfold candidates
  have row : edge 6947 = SearchData.row6947.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan793_0 : scan edge (search edge 3) 4 [6101] 6101 innerTail793_0 best = best := by
  rw [innerTail793_0, scan, if_neg (by decide), innerCandidates793_0_checked]
  change scan edge (search edge 3) 4 [6101] 6101 innerTail793_1
    (if 3 ≤ innerCandidates793_0.length then search edge 3 [6101, 6947] innerCandidates793_0 13048 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan793_1

theorem rootBranch793 : search edge 4 [6101] rootCandidates793 6101 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6101] 6101 innerTail793_0 best = best
  exact innerScan793_0

def rootCandidates792 : List Nat := [8311]
set_option maxRecDepth 4096 in
theorem rootCandidates792_checked : candidates edge 5 0 6091 tail793 best.bound = rootCandidates792 := by
  unfold candidates
  have row : edge 6091 = SearchData.row6091.contains := by funext q; rfl
  rw [row]
  decide

end ProjectEulerSolutions.PrimePairUpper.Search
