import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates815 : List Nat := [6911, 7673, 8009, 8147, 8219, 8273]
set_option maxRecDepth 4096 in
theorem rootCandidates815_checked : candidates edge 5 0 6287 tail816 best.bound = rootCandidates815 := by
  unfold candidates
  have row : edge 6287 = SearchData.row6287.contains := by funext q; rfl
  rw [row]
  decide

def innerTail815_6 : List Nat := []
theorem innerScan815_6 : scan edge (search edge 3) 4 [6287] 6287 innerTail815_6 best = best := rfl

def innerTail815_5 : List Nat := 8273 :: innerTail815_6
def innerCandidates815_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates815_5_checked : candidates edge 4 6287 8273 innerTail815_6 best.bound = innerCandidates815_5 := by
  unfold candidates
  have row : edge 8273 = SearchData.row8273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan815_5 : scan edge (search edge 3) 4 [6287] 6287 innerTail815_5 best = best := by
  rw [innerTail815_5, scan, if_neg (by decide), innerCandidates815_5_checked]
  change scan edge (search edge 3) 4 [6287] 6287 innerTail815_6
    (if 3 ≤ innerCandidates815_5.length then search edge 3 [6287, 8273] innerCandidates815_5 14560 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan815_6

def innerTail815_4 : List Nat := 8219 :: innerTail815_5
def innerCandidates815_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates815_4_checked : candidates edge 4 6287 8219 innerTail815_5 best.bound = innerCandidates815_4 := by
  unfold candidates
  have row : edge 8219 = SearchData.row8219.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan815_4 : scan edge (search edge 3) 4 [6287] 6287 innerTail815_4 best = best := by
  rw [innerTail815_4, scan, if_neg (by decide), innerCandidates815_4_checked]
  change scan edge (search edge 3) 4 [6287] 6287 innerTail815_5
    (if 3 ≤ innerCandidates815_4.length then search edge 3 [6287, 8219] innerCandidates815_4 14506 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan815_5

def innerTail815_3 : List Nat := 8147 :: innerTail815_4
def innerCandidates815_3 : List Nat := [8219]
set_option maxRecDepth 4096 in
theorem innerCandidates815_3_checked : candidates edge 4 6287 8147 innerTail815_4 best.bound = innerCandidates815_3 := by
  unfold candidates
  have row : edge 8147 = SearchData.row8147.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan815_3 : scan edge (search edge 3) 4 [6287] 6287 innerTail815_3 best = best := by
  rw [innerTail815_3, scan, if_neg (by decide), innerCandidates815_3_checked]
  change scan edge (search edge 3) 4 [6287] 6287 innerTail815_4
    (if 3 ≤ innerCandidates815_3.length then search edge 3 [6287, 8147] innerCandidates815_3 14434 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan815_4

def innerTail815_2 : List Nat := 8009 :: innerTail815_3
def innerCandidates815_2 : List Nat := [8147]
set_option maxRecDepth 4096 in
theorem innerCandidates815_2_checked : candidates edge 4 6287 8009 innerTail815_3 best.bound = innerCandidates815_2 := by
  unfold candidates
  have row : edge 8009 = SearchData.row8009.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan815_2 : scan edge (search edge 3) 4 [6287] 6287 innerTail815_2 best = best := by
  rw [innerTail815_2, scan, if_neg (by decide), innerCandidates815_2_checked]
  change scan edge (search edge 3) 4 [6287] 6287 innerTail815_3
    (if 3 ≤ innerCandidates815_2.length then search edge 3 [6287, 8009] innerCandidates815_2 14296 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan815_3

def innerTail815_1 : List Nat := 7673 :: innerTail815_2
def innerCandidates815_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates815_1_checked : candidates edge 4 6287 7673 innerTail815_2 best.bound = innerCandidates815_1 := by
  unfold candidates
  have row : edge 7673 = SearchData.row7673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan815_1 : scan edge (search edge 3) 4 [6287] 6287 innerTail815_1 best = best := by
  rw [innerTail815_1, scan, if_neg (by decide), innerCandidates815_1_checked]
  change scan edge (search edge 3) 4 [6287] 6287 innerTail815_2
    (if 3 ≤ innerCandidates815_1.length then search edge 3 [6287, 7673] innerCandidates815_1 13960 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan815_2

def innerTail815_0 : List Nat := 6911 :: innerTail815_1
def innerCandidates815_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates815_0_checked : candidates edge 4 6287 6911 innerTail815_1 best.bound = innerCandidates815_0 := by
  unfold candidates
  have row : edge 6911 = SearchData.row6911.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan815_0 : scan edge (search edge 3) 4 [6287] 6287 innerTail815_0 best = best := by
  rw [innerTail815_0, scan, if_neg (by decide), innerCandidates815_0_checked]
  change scan edge (search edge 3) 4 [6287] 6287 innerTail815_1
    (if 3 ≤ innerCandidates815_0.length then search edge 3 [6287, 6911] innerCandidates815_0 13198 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan815_1

theorem rootBranch815 : search edge 4 [6287] rootCandidates815 6287 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6287] 6287 innerTail815_0 best = best
  exact innerScan815_0

def rootCandidates814 : List Nat := [7321, 7681, 8269]
set_option maxRecDepth 4096 in
theorem rootCandidates814_checked : candidates edge 5 0 6277 tail815 best.bound = rootCandidates814 := by
  unfold candidates
  have row : edge 6277 = SearchData.row6277.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates813 : List Nat := [6373, 8011]
set_option maxRecDepth 4096 in
theorem rootCandidates813_checked : candidates edge 5 0 6271 tail814 best.bound = rootCandidates813 := by
  unfold candidates
  have row : edge 6271 = SearchData.row6271.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates812 : List Nat := [6701, 6779, 6899, 7457, 7883, 8291, 8369]
set_option maxRecDepth 4096 in
theorem rootCandidates812_checked : candidates edge 5 0 6269 tail813 best.bound = rootCandidates812 := by
  unfold candidates
  have row : edge 6269 = SearchData.row6269.contains := by funext q; rfl
  rw [row]
  decide

def innerTail812_7 : List Nat := []
theorem innerScan812_7 : scan edge (search edge 3) 4 [6269] 6269 innerTail812_7 best = best := rfl

def innerTail812_6 : List Nat := 8369 :: innerTail812_7
def innerCandidates812_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates812_6_checked : candidates edge 4 6269 8369 innerTail812_7 best.bound = innerCandidates812_6 := by
  unfold candidates
  have row : edge 8369 = SearchData.row8369.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan812_6 : scan edge (search edge 3) 4 [6269] 6269 innerTail812_6 best = best := by
  rw [innerTail812_6, scan, if_neg (by decide), innerCandidates812_6_checked]
  change scan edge (search edge 3) 4 [6269] 6269 innerTail812_7
    (if 3 ≤ innerCandidates812_6.length then search edge 3 [6269, 8369] innerCandidates812_6 14638 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan812_7

def innerTail812_5 : List Nat := 8291 :: innerTail812_6
def innerCandidates812_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates812_5_checked : candidates edge 4 6269 8291 innerTail812_6 best.bound = innerCandidates812_5 := by
  unfold candidates
  have row : edge 8291 = SearchData.row8291.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan812_5 : scan edge (search edge 3) 4 [6269] 6269 innerTail812_5 best = best := by
  rw [innerTail812_5, scan, if_neg (by decide), innerCandidates812_5_checked]
  change scan edge (search edge 3) 4 [6269] 6269 innerTail812_6
    (if 3 ≤ innerCandidates812_5.length then search edge 3 [6269, 8291] innerCandidates812_5 14560 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan812_6

def innerTail812_4 : List Nat := 7883 :: innerTail812_5
def innerCandidates812_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates812_4_checked : candidates edge 4 6269 7883 innerTail812_5 best.bound = innerCandidates812_4 := by
  unfold candidates
  have row : edge 7883 = SearchData.row7883.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan812_4 : scan edge (search edge 3) 4 [6269] 6269 innerTail812_4 best = best := by
  rw [innerTail812_4, scan, if_neg (by decide), innerCandidates812_4_checked]
  change scan edge (search edge 3) 4 [6269] 6269 innerTail812_5
    (if 3 ≤ innerCandidates812_4.length then search edge 3 [6269, 7883] innerCandidates812_4 14152 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan812_5

def innerTail812_3 : List Nat := 7457 :: innerTail812_4
def innerCandidates812_3 : List Nat := [8369]
set_option maxRecDepth 4096 in
theorem innerCandidates812_3_checked : candidates edge 4 6269 7457 innerTail812_4 best.bound = innerCandidates812_3 := by
  unfold candidates
  have row : edge 7457 = SearchData.row7457.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan812_3 : scan edge (search edge 3) 4 [6269] 6269 innerTail812_3 best = best := by
  rw [innerTail812_3, scan, if_neg (by decide), innerCandidates812_3_checked]
  change scan edge (search edge 3) 4 [6269] 6269 innerTail812_4
    (if 3 ≤ innerCandidates812_3.length then search edge 3 [6269, 7457] innerCandidates812_3 13726 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan812_4

def innerTail812_2 : List Nat := 6899 :: innerTail812_3
def innerCandidates812_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates812_2_checked : candidates edge 4 6269 6899 innerTail812_3 best.bound = innerCandidates812_2 := by
  unfold candidates
  have row : edge 6899 = SearchData.row6899.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan812_2 : scan edge (search edge 3) 4 [6269] 6269 innerTail812_2 best = best := by
  rw [innerTail812_2, scan, if_neg (by decide), innerCandidates812_2_checked]
  change scan edge (search edge 3) 4 [6269] 6269 innerTail812_3
    (if 3 ≤ innerCandidates812_2.length then search edge 3 [6269, 6899] innerCandidates812_2 13168 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan812_3

def innerTail812_1 : List Nat := 6779 :: innerTail812_2
def innerCandidates812_1 : List Nat := [8369]
set_option maxRecDepth 4096 in
theorem innerCandidates812_1_checked : candidates edge 4 6269 6779 innerTail812_2 best.bound = innerCandidates812_1 := by
  unfold candidates
  have row : edge 6779 = SearchData.row6779.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan812_1 : scan edge (search edge 3) 4 [6269] 6269 innerTail812_1 best = best := by
  rw [innerTail812_1, scan, if_neg (by decide), innerCandidates812_1_checked]
  change scan edge (search edge 3) 4 [6269] 6269 innerTail812_2
    (if 3 ≤ innerCandidates812_1.length then search edge 3 [6269, 6779] innerCandidates812_1 13048 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan812_2

def innerTail812_0 : List Nat := 6701 :: innerTail812_1
def innerCandidates812_0 : List Nat := [6899]
set_option maxRecDepth 4096 in
theorem innerCandidates812_0_checked : candidates edge 4 6269 6701 innerTail812_1 best.bound = innerCandidates812_0 := by
  unfold candidates
  have row : edge 6701 = SearchData.row6701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan812_0 : scan edge (search edge 3) 4 [6269] 6269 innerTail812_0 best = best := by
  rw [innerTail812_0, scan, if_neg (by decide), innerCandidates812_0_checked]
  change scan edge (search edge 3) 4 [6269] 6269 innerTail812_1
    (if 3 ≤ innerCandidates812_0.length then search edge 3 [6269, 6701] innerCandidates812_0 12970 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan812_1

theorem rootBranch812 : search edge 4 [6269] rootCandidates812 6269 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6269] 6269 innerTail812_0 best = best
  exact innerScan812_0

def rootCandidates811 : List Nat := [6359, 6389, 7121, 7499, 7919, 8297]
set_option maxRecDepth 4096 in
theorem rootCandidates811_checked : candidates edge 5 0 6263 tail812 best.bound = rootCandidates811 := by
  unfold candidates
  have row : edge 6263 = SearchData.row6263.contains := by funext q; rfl
  rw [row]
  decide

def innerTail811_6 : List Nat := []
theorem innerScan811_6 : scan edge (search edge 3) 4 [6263] 6263 innerTail811_6 best = best := rfl

def innerTail811_5 : List Nat := 8297 :: innerTail811_6
def innerCandidates811_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates811_5_checked : candidates edge 4 6263 8297 innerTail811_6 best.bound = innerCandidates811_5 := by
  unfold candidates
  have row : edge 8297 = SearchData.row8297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan811_5 : scan edge (search edge 3) 4 [6263] 6263 innerTail811_5 best = best := by
  rw [innerTail811_5, scan, if_neg (by decide), innerCandidates811_5_checked]
  change scan edge (search edge 3) 4 [6263] 6263 innerTail811_6
    (if 3 ≤ innerCandidates811_5.length then search edge 3 [6263, 8297] innerCandidates811_5 14560 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan811_6

def innerTail811_4 : List Nat := 7919 :: innerTail811_5
def innerCandidates811_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates811_4_checked : candidates edge 4 6263 7919 innerTail811_5 best.bound = innerCandidates811_4 := by
  unfold candidates
  have row : edge 7919 = SearchData.row7919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan811_4 : scan edge (search edge 3) 4 [6263] 6263 innerTail811_4 best = best := by
  rw [innerTail811_4, scan, if_neg (by decide), innerCandidates811_4_checked]
  change scan edge (search edge 3) 4 [6263] 6263 innerTail811_5
    (if 3 ≤ innerCandidates811_4.length then search edge 3 [6263, 7919] innerCandidates811_4 14182 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan811_5

def innerTail811_3 : List Nat := 7499 :: innerTail811_4
def innerCandidates811_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates811_3_checked : candidates edge 4 6263 7499 innerTail811_4 best.bound = innerCandidates811_3 := by
  unfold candidates
  have row : edge 7499 = SearchData.row7499.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan811_3 : scan edge (search edge 3) 4 [6263] 6263 innerTail811_3 best = best := by
  rw [innerTail811_3, scan, if_neg (by decide), innerCandidates811_3_checked]
  change scan edge (search edge 3) 4 [6263] 6263 innerTail811_4
    (if 3 ≤ innerCandidates811_3.length then search edge 3 [6263, 7499] innerCandidates811_3 13762 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan811_4

def innerTail811_2 : List Nat := 7121 :: innerTail811_3
def innerCandidates811_2 : List Nat := [8297]
set_option maxRecDepth 4096 in
theorem innerCandidates811_2_checked : candidates edge 4 6263 7121 innerTail811_3 best.bound = innerCandidates811_2 := by
  unfold candidates
  have row : edge 7121 = SearchData.row7121.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan811_2 : scan edge (search edge 3) 4 [6263] 6263 innerTail811_2 best = best := by
  rw [innerTail811_2, scan, if_neg (by decide), innerCandidates811_2_checked]
  change scan edge (search edge 3) 4 [6263] 6263 innerTail811_3
    (if 3 ≤ innerCandidates811_2.length then search edge 3 [6263, 7121] innerCandidates811_2 13384 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan811_3

def innerTail811_1 : List Nat := 6389 :: innerTail811_2
def innerCandidates811_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates811_1_checked : candidates edge 4 6263 6389 innerTail811_2 best.bound = innerCandidates811_1 := by
  unfold candidates
  have row : edge 6389 = SearchData.row6389.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan811_1 : scan edge (search edge 3) 4 [6263] 6263 innerTail811_1 best = best := by
  rw [innerTail811_1, scan, if_neg (by decide), innerCandidates811_1_checked]
  change scan edge (search edge 3) 4 [6263] 6263 innerTail811_2
    (if 3 ≤ innerCandidates811_1.length then search edge 3 [6263, 6389] innerCandidates811_1 12652 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan811_2

def innerTail811_0 : List Nat := 6359 :: innerTail811_1
def innerCandidates811_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates811_0_checked : candidates edge 4 6263 6359 innerTail811_1 best.bound = innerCandidates811_0 := by
  unfold candidates
  have row : edge 6359 = SearchData.row6359.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan811_0 : scan edge (search edge 3) 4 [6263] 6263 innerTail811_0 best = best := by
  rw [innerTail811_0, scan, if_neg (by decide), innerCandidates811_0_checked]
  change scan edge (search edge 3) 4 [6263] 6263 innerTail811_1
    (if 3 ≤ innerCandidates811_0.length then search edge 3 [6263, 6359] innerCandidates811_0 12622 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan811_1

theorem rootBranch811 : search edge 4 [6263] rootCandidates811 6263 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6263] 6263 innerTail811_0 best = best
  exact innerScan811_0

def rootCandidates810 : List Nat := [7607, 8387]
set_option maxRecDepth 4096 in
theorem rootCandidates810_checked : candidates edge 5 0 6257 tail811 best.bound = rootCandidates810 := by
  unfold candidates
  have row : edge 6257 = SearchData.row6257.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates809 : List Nat := [6691, 6709, 7549, 7879, 8161]
set_option maxRecDepth 4096 in
theorem rootCandidates809_checked : candidates edge 5 0 6247 tail810 best.bound = rootCandidates809 := by
  unfold candidates
  have row : edge 6247 = SearchData.row6247.contains := by funext q; rfl
  rw [row]
  decide

def innerTail809_5 : List Nat := []
theorem innerScan809_5 : scan edge (search edge 3) 4 [6247] 6247 innerTail809_5 best = best := rfl

def innerTail809_4 : List Nat := 8161 :: innerTail809_5
def innerCandidates809_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates809_4_checked : candidates edge 4 6247 8161 innerTail809_5 best.bound = innerCandidates809_4 := by
  unfold candidates
  have row : edge 8161 = SearchData.row8161.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan809_4 : scan edge (search edge 3) 4 [6247] 6247 innerTail809_4 best = best := by
  rw [innerTail809_4, scan, if_neg (by decide), innerCandidates809_4_checked]
  change scan edge (search edge 3) 4 [6247] 6247 innerTail809_5
    (if 3 ≤ innerCandidates809_4.length then search edge 3 [6247, 8161] innerCandidates809_4 14408 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan809_5

def innerTail809_3 : List Nat := 7879 :: innerTail809_4
def innerCandidates809_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates809_3_checked : candidates edge 4 6247 7879 innerTail809_4 best.bound = innerCandidates809_3 := by
  unfold candidates
  have row : edge 7879 = SearchData.row7879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan809_3 : scan edge (search edge 3) 4 [6247] 6247 innerTail809_3 best = best := by
  rw [innerTail809_3, scan, if_neg (by decide), innerCandidates809_3_checked]
  change scan edge (search edge 3) 4 [6247] 6247 innerTail809_4
    (if 3 ≤ innerCandidates809_3.length then search edge 3 [6247, 7879] innerCandidates809_3 14126 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan809_4

def innerTail809_2 : List Nat := 7549 :: innerTail809_3
def innerCandidates809_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates809_2_checked : candidates edge 4 6247 7549 innerTail809_3 best.bound = innerCandidates809_2 := by
  unfold candidates
  have row : edge 7549 = SearchData.row7549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan809_2 : scan edge (search edge 3) 4 [6247] 6247 innerTail809_2 best = best := by
  rw [innerTail809_2, scan, if_neg (by decide), innerCandidates809_2_checked]
  change scan edge (search edge 3) 4 [6247] 6247 innerTail809_3
    (if 3 ≤ innerCandidates809_2.length then search edge 3 [6247, 7549] innerCandidates809_2 13796 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan809_3

def innerTail809_1 : List Nat := 6709 :: innerTail809_2
def innerCandidates809_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates809_1_checked : candidates edge 4 6247 6709 innerTail809_2 best.bound = innerCandidates809_1 := by
  unfold candidates
  have row : edge 6709 = SearchData.row6709.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan809_1 : scan edge (search edge 3) 4 [6247] 6247 innerTail809_1 best = best := by
  rw [innerTail809_1, scan, if_neg (by decide), innerCandidates809_1_checked]
  change scan edge (search edge 3) 4 [6247] 6247 innerTail809_2
    (if 3 ≤ innerCandidates809_1.length then search edge 3 [6247, 6709] innerCandidates809_1 12956 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan809_2

def innerTail809_0 : List Nat := 6691 :: innerTail809_1
def innerCandidates809_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates809_0_checked : candidates edge 4 6247 6691 innerTail809_1 best.bound = innerCandidates809_0 := by
  unfold candidates
  have row : edge 6691 = SearchData.row6691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan809_0 : scan edge (search edge 3) 4 [6247] 6247 innerTail809_0 best = best := by
  rw [innerTail809_0, scan, if_neg (by decide), innerCandidates809_0_checked]
  change scan edge (search edge 3) 4 [6247] 6247 innerTail809_1
    (if 3 ≤ innerCandidates809_0.length then search edge 3 [6247, 6691] innerCandidates809_0 12938 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan809_1

theorem rootBranch809 : search edge 4 [6247] rootCandidates809 6247 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6247] 6247 innerTail809_0 best = best
  exact innerScan809_0

def rootCandidates808 : List Nat := [6277, 7297, 8167, 8233]
set_option maxRecDepth 4096 in
theorem rootCandidates808_checked : candidates edge 5 0 6229 tail809 best.bound = rootCandidates808 := by
  unfold candidates
  have row : edge 6229 = SearchData.row6229.contains := by funext q; rfl
  rw [row]
  decide

def innerTail808_4 : List Nat := []
theorem innerScan808_4 : scan edge (search edge 3) 4 [6229] 6229 innerTail808_4 best = best := rfl

def innerTail808_3 : List Nat := 8233 :: innerTail808_4
def innerCandidates808_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates808_3_checked : candidates edge 4 6229 8233 innerTail808_4 best.bound = innerCandidates808_3 := by
  unfold candidates
  have row : edge 8233 = SearchData.row8233.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan808_3 : scan edge (search edge 3) 4 [6229] 6229 innerTail808_3 best = best := by
  rw [innerTail808_3, scan, if_neg (by decide), innerCandidates808_3_checked]
  change scan edge (search edge 3) 4 [6229] 6229 innerTail808_4
    (if 3 ≤ innerCandidates808_3.length then search edge 3 [6229, 8233] innerCandidates808_3 14462 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan808_4

def innerTail808_2 : List Nat := 8167 :: innerTail808_3
def innerCandidates808_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates808_2_checked : candidates edge 4 6229 8167 innerTail808_3 best.bound = innerCandidates808_2 := by
  unfold candidates
  have row : edge 8167 = SearchData.row8167.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan808_2 : scan edge (search edge 3) 4 [6229] 6229 innerTail808_2 best = best := by
  rw [innerTail808_2, scan, if_neg (by decide), innerCandidates808_2_checked]
  change scan edge (search edge 3) 4 [6229] 6229 innerTail808_3
    (if 3 ≤ innerCandidates808_2.length then search edge 3 [6229, 8167] innerCandidates808_2 14396 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan808_3

def innerTail808_1 : List Nat := 7297 :: innerTail808_2
def innerCandidates808_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates808_1_checked : candidates edge 4 6229 7297 innerTail808_2 best.bound = innerCandidates808_1 := by
  unfold candidates
  have row : edge 7297 = SearchData.row7297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan808_1 : scan edge (search edge 3) 4 [6229] 6229 innerTail808_1 best = best := by
  rw [innerTail808_1, scan, if_neg (by decide), innerCandidates808_1_checked]
  change scan edge (search edge 3) 4 [6229] 6229 innerTail808_2
    (if 3 ≤ innerCandidates808_1.length then search edge 3 [6229, 7297] innerCandidates808_1 13526 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan808_2

def innerTail808_0 : List Nat := 6277 :: innerTail808_1
def innerCandidates808_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates808_0_checked : candidates edge 4 6229 6277 innerTail808_1 best.bound = innerCandidates808_0 := by
  unfold candidates
  have row : edge 6277 = SearchData.row6277.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan808_0 : scan edge (search edge 3) 4 [6229] 6229 innerTail808_0 best = best := by
  rw [innerTail808_0, scan, if_neg (by decide), innerCandidates808_0_checked]
  change scan edge (search edge 3) 4 [6229] 6229 innerTail808_1
    (if 3 ≤ innerCandidates808_0.length then search edge 3 [6229, 6277] innerCandidates808_0 12506 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan808_1

theorem rootBranch808 : search edge 4 [6229] rootCandidates808 6229 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6229] 6229 innerTail808_0 best = best
  exact innerScan808_0

end ProjectEulerSolutions.PrimePairUpper.Search
