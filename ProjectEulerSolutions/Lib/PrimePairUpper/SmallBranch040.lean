import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates728 : List Nat := [6343, 6577, 6619, 7129, 7549, 7753]
set_option maxRecDepth 4096 in
theorem rootCandidates728_checked : candidates edge 5 0 5527 tail729 best.bound = rootCandidates728 := by
  unfold candidates
  have row : edge 5527 = SearchData.row5527.contains := by funext q; rfl
  rw [row]
  decide

def innerTail728_6 : List Nat := []
theorem innerScan728_6 : scan edge (search edge 3) 4 [5527] 5527 innerTail728_6 best = best := rfl

def innerTail728_5 : List Nat := 7753 :: innerTail728_6
def innerCandidates728_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates728_5_checked : candidates edge 4 5527 7753 innerTail728_6 best.bound = innerCandidates728_5 := by
  unfold candidates
  have row : edge 7753 = SearchData.row7753.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan728_5 : scan edge (search edge 3) 4 [5527] 5527 innerTail728_5 best = best := by
  rw [innerTail728_5, scan, if_neg (by decide), innerCandidates728_5_checked]
  change scan edge (search edge 3) 4 [5527] 5527 innerTail728_6
    (if 3 ≤ innerCandidates728_5.length then search edge 3 [5527, 7753] innerCandidates728_5 13280 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan728_6

def innerTail728_4 : List Nat := 7549 :: innerTail728_5
def innerCandidates728_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates728_4_checked : candidates edge 4 5527 7549 innerTail728_5 best.bound = innerCandidates728_4 := by
  unfold candidates
  have row : edge 7549 = SearchData.row7549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan728_4 : scan edge (search edge 3) 4 [5527] 5527 innerTail728_4 best = best := by
  rw [innerTail728_4, scan, if_neg (by decide), innerCandidates728_4_checked]
  change scan edge (search edge 3) 4 [5527] 5527 innerTail728_5
    (if 3 ≤ innerCandidates728_4.length then search edge 3 [5527, 7549] innerCandidates728_4 13076 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan728_5

def innerTail728_3 : List Nat := 7129 :: innerTail728_4
def innerCandidates728_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates728_3_checked : candidates edge 4 5527 7129 innerTail728_4 best.bound = innerCandidates728_3 := by
  unfold candidates
  have row : edge 7129 = SearchData.row7129.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan728_3 : scan edge (search edge 3) 4 [5527] 5527 innerTail728_3 best = best := by
  rw [innerTail728_3, scan, if_neg (by decide), innerCandidates728_3_checked]
  change scan edge (search edge 3) 4 [5527] 5527 innerTail728_4
    (if 3 ≤ innerCandidates728_3.length then search edge 3 [5527, 7129] innerCandidates728_3 12656 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan728_4

def innerTail728_2 : List Nat := 6619 :: innerTail728_3
def innerCandidates728_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates728_2_checked : candidates edge 4 5527 6619 innerTail728_3 best.bound = innerCandidates728_2 := by
  unfold candidates
  have row : edge 6619 = SearchData.row6619.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan728_2 : scan edge (search edge 3) 4 [5527] 5527 innerTail728_2 best = best := by
  rw [innerTail728_2, scan, if_neg (by decide), innerCandidates728_2_checked]
  change scan edge (search edge 3) 4 [5527] 5527 innerTail728_3
    (if 3 ≤ innerCandidates728_2.length then search edge 3 [5527, 6619] innerCandidates728_2 12146 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan728_3

def innerTail728_1 : List Nat := 6577 :: innerTail728_2
def innerCandidates728_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates728_1_checked : candidates edge 4 5527 6577 innerTail728_2 best.bound = innerCandidates728_1 := by
  unfold candidates
  have row : edge 6577 = SearchData.row6577.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan728_1 : scan edge (search edge 3) 4 [5527] 5527 innerTail728_1 best = best := by
  rw [innerTail728_1, scan, if_neg (by decide), innerCandidates728_1_checked]
  change scan edge (search edge 3) 4 [5527] 5527 innerTail728_2
    (if 3 ≤ innerCandidates728_1.length then search edge 3 [5527, 6577] innerCandidates728_1 12104 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan728_2

def innerTail728_0 : List Nat := 6343 :: innerTail728_1
def innerCandidates728_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates728_0_checked : candidates edge 4 5527 6343 innerTail728_1 best.bound = innerCandidates728_0 := by
  unfold candidates
  have row : edge 6343 = SearchData.row6343.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan728_0 : scan edge (search edge 3) 4 [5527] 5527 innerTail728_0 best = best := by
  rw [innerTail728_0, scan, if_neg (by decide), innerCandidates728_0_checked]
  change scan edge (search edge 3) 4 [5527] 5527 innerTail728_1
    (if 3 ≤ innerCandidates728_0.length then search edge 3 [5527, 6343] innerCandidates728_0 11870 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan728_1

theorem rootBranch728 : search edge 4 [5527] rootCandidates728 5527 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5527] 5527 innerTail728_0 best = best
  exact innerScan728_0

def rootCandidates727 : List Nat := [5701, 5851, 7159, 7951]
set_option maxRecDepth 4096 in
theorem rootCandidates727_checked : candidates edge 5 0 5521 tail728 best.bound = rootCandidates727 := by
  unfold candidates
  have row : edge 5521 = SearchData.row5521.contains := by funext q; rfl
  rw [row]
  decide

def innerTail727_4 : List Nat := []
theorem innerScan727_4 : scan edge (search edge 3) 4 [5521] 5521 innerTail727_4 best = best := rfl

def innerTail727_3 : List Nat := 7951 :: innerTail727_4
def innerCandidates727_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates727_3_checked : candidates edge 4 5521 7951 innerTail727_4 best.bound = innerCandidates727_3 := by
  unfold candidates
  have row : edge 7951 = SearchData.row7951.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan727_3 : scan edge (search edge 3) 4 [5521] 5521 innerTail727_3 best = best := by
  rw [innerTail727_3, scan, if_neg (by decide), innerCandidates727_3_checked]
  change scan edge (search edge 3) 4 [5521] 5521 innerTail727_4
    (if 3 ≤ innerCandidates727_3.length then search edge 3 [5521, 7951] innerCandidates727_3 13472 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan727_4

def innerTail727_2 : List Nat := 7159 :: innerTail727_3
def innerCandidates727_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates727_2_checked : candidates edge 4 5521 7159 innerTail727_3 best.bound = innerCandidates727_2 := by
  unfold candidates
  have row : edge 7159 = SearchData.row7159.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan727_2 : scan edge (search edge 3) 4 [5521] 5521 innerTail727_2 best = best := by
  rw [innerTail727_2, scan, if_neg (by decide), innerCandidates727_2_checked]
  change scan edge (search edge 3) 4 [5521] 5521 innerTail727_3
    (if 3 ≤ innerCandidates727_2.length then search edge 3 [5521, 7159] innerCandidates727_2 12680 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan727_3

def innerTail727_1 : List Nat := 5851 :: innerTail727_2
def innerCandidates727_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates727_1_checked : candidates edge 4 5521 5851 innerTail727_2 best.bound = innerCandidates727_1 := by
  unfold candidates
  have row : edge 5851 = SearchData.row5851.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan727_1 : scan edge (search edge 3) 4 [5521] 5521 innerTail727_1 best = best := by
  rw [innerTail727_1, scan, if_neg (by decide), innerCandidates727_1_checked]
  change scan edge (search edge 3) 4 [5521] 5521 innerTail727_2
    (if 3 ≤ innerCandidates727_1.length then search edge 3 [5521, 5851] innerCandidates727_1 11372 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan727_2

def innerTail727_0 : List Nat := 5701 :: innerTail727_1
def innerCandidates727_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates727_0_checked : candidates edge 4 5521 5701 innerTail727_1 best.bound = innerCandidates727_0 := by
  unfold candidates
  have row : edge 5701 = SearchData.row5701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan727_0 : scan edge (search edge 3) 4 [5521] 5521 innerTail727_0 best = best := by
  rw [innerTail727_0, scan, if_neg (by decide), innerCandidates727_0_checked]
  change scan edge (search edge 3) 4 [5521] 5521 innerTail727_1
    (if 3 ≤ innerCandidates727_0.length then search edge 3 [5521, 5701] innerCandidates727_0 11222 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan727_1

theorem rootBranch727 : search edge 4 [5521] rootCandidates727 5521 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5521] 5521 innerTail727_0 best = best
  exact innerScan727_0

def rootCandidates726 : List Nat := [5711, 6299, 7517, 8081]
set_option maxRecDepth 4096 in
theorem rootCandidates726_checked : candidates edge 5 0 5519 tail727 best.bound = rootCandidates726 := by
  unfold candidates
  have row : edge 5519 = SearchData.row5519.contains := by funext q; rfl
  rw [row]
  decide

def innerTail726_4 : List Nat := []
theorem innerScan726_4 : scan edge (search edge 3) 4 [5519] 5519 innerTail726_4 best = best := rfl

def innerTail726_3 : List Nat := 8081 :: innerTail726_4
def innerCandidates726_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates726_3_checked : candidates edge 4 5519 8081 innerTail726_4 best.bound = innerCandidates726_3 := by
  unfold candidates
  have row : edge 8081 = SearchData.row8081.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan726_3 : scan edge (search edge 3) 4 [5519] 5519 innerTail726_3 best = best := by
  rw [innerTail726_3, scan, if_neg (by decide), innerCandidates726_3_checked]
  change scan edge (search edge 3) 4 [5519] 5519 innerTail726_4
    (if 3 ≤ innerCandidates726_3.length then search edge 3 [5519, 8081] innerCandidates726_3 13600 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan726_4

def innerTail726_2 : List Nat := 7517 :: innerTail726_3
def innerCandidates726_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates726_2_checked : candidates edge 4 5519 7517 innerTail726_3 best.bound = innerCandidates726_2 := by
  unfold candidates
  have row : edge 7517 = SearchData.row7517.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan726_2 : scan edge (search edge 3) 4 [5519] 5519 innerTail726_2 best = best := by
  rw [innerTail726_2, scan, if_neg (by decide), innerCandidates726_2_checked]
  change scan edge (search edge 3) 4 [5519] 5519 innerTail726_3
    (if 3 ≤ innerCandidates726_2.length then search edge 3 [5519, 7517] innerCandidates726_2 13036 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan726_3

def innerTail726_1 : List Nat := 6299 :: innerTail726_2
def innerCandidates726_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates726_1_checked : candidates edge 4 5519 6299 innerTail726_2 best.bound = innerCandidates726_1 := by
  unfold candidates
  have row : edge 6299 = SearchData.row6299.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan726_1 : scan edge (search edge 3) 4 [5519] 5519 innerTail726_1 best = best := by
  rw [innerTail726_1, scan, if_neg (by decide), innerCandidates726_1_checked]
  change scan edge (search edge 3) 4 [5519] 5519 innerTail726_2
    (if 3 ≤ innerCandidates726_1.length then search edge 3 [5519, 6299] innerCandidates726_1 11818 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan726_2

def innerTail726_0 : List Nat := 5711 :: innerTail726_1
def innerCandidates726_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates726_0_checked : candidates edge 4 5519 5711 innerTail726_1 best.bound = innerCandidates726_0 := by
  unfold candidates
  have row : edge 5711 = SearchData.row5711.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan726_0 : scan edge (search edge 3) 4 [5519] 5519 innerTail726_0 best = best := by
  rw [innerTail726_0, scan, if_neg (by decide), innerCandidates726_0_checked]
  change scan edge (search edge 3) 4 [5519] 5519 innerTail726_1
    (if 3 ≤ innerCandidates726_0.length then search edge 3 [5519, 5711] innerCandidates726_0 11230 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan726_1

theorem rootBranch726 : search edge 4 [5519] rootCandidates726 5519 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5519] 5519 innerTail726_0 best = best
  exact innerScan726_0

def rootCandidates725 : List Nat := [5939, 6173, 6221, 6257, 6803, 7643, 7829, 7853, 7877, 8291]
set_option maxRecDepth 4096 in
theorem rootCandidates725_checked : candidates edge 5 0 5507 tail726 best.bound = rootCandidates725 := by
  unfold candidates
  have row : edge 5507 = SearchData.row5507.contains := by funext q; rfl
  rw [row]
  decide

def innerTail725_10 : List Nat := []
theorem innerScan725_10 : scan edge (search edge 3) 4 [5507] 5507 innerTail725_10 best = best := rfl

def innerTail725_9 : List Nat := 8291 :: innerTail725_10
def innerCandidates725_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates725_9_checked : candidates edge 4 5507 8291 innerTail725_10 best.bound = innerCandidates725_9 := by
  unfold candidates
  have row : edge 8291 = SearchData.row8291.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan725_9 : scan edge (search edge 3) 4 [5507] 5507 innerTail725_9 best = best := by
  rw [innerTail725_9, scan, if_neg (by decide), innerCandidates725_9_checked]
  change scan edge (search edge 3) 4 [5507] 5507 innerTail725_10
    (if 3 ≤ innerCandidates725_9.length then search edge 3 [5507, 8291] innerCandidates725_9 13798 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan725_10

def innerTail725_8 : List Nat := 7877 :: innerTail725_9
def innerCandidates725_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates725_8_checked : candidates edge 4 5507 7877 innerTail725_9 best.bound = innerCandidates725_8 := by
  unfold candidates
  have row : edge 7877 = SearchData.row7877.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan725_8 : scan edge (search edge 3) 4 [5507] 5507 innerTail725_8 best = best := by
  rw [innerTail725_8, scan, if_neg (by decide), innerCandidates725_8_checked]
  change scan edge (search edge 3) 4 [5507] 5507 innerTail725_9
    (if 3 ≤ innerCandidates725_8.length then search edge 3 [5507, 7877] innerCandidates725_8 13384 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan725_9

def innerTail725_7 : List Nat := 7853 :: innerTail725_8
def innerCandidates725_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates725_7_checked : candidates edge 4 5507 7853 innerTail725_8 best.bound = innerCandidates725_7 := by
  unfold candidates
  have row : edge 7853 = SearchData.row7853.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan725_7 : scan edge (search edge 3) 4 [5507] 5507 innerTail725_7 best = best := by
  rw [innerTail725_7, scan, if_neg (by decide), innerCandidates725_7_checked]
  change scan edge (search edge 3) 4 [5507] 5507 innerTail725_8
    (if 3 ≤ innerCandidates725_7.length then search edge 3 [5507, 7853] innerCandidates725_7 13360 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan725_8

def innerTail725_6 : List Nat := 7829 :: innerTail725_7
def innerCandidates725_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates725_6_checked : candidates edge 4 5507 7829 innerTail725_7 best.bound = innerCandidates725_6 := by
  unfold candidates
  have row : edge 7829 = SearchData.row7829.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan725_6 : scan edge (search edge 3) 4 [5507] 5507 innerTail725_6 best = best := by
  rw [innerTail725_6, scan, if_neg (by decide), innerCandidates725_6_checked]
  change scan edge (search edge 3) 4 [5507] 5507 innerTail725_7
    (if 3 ≤ innerCandidates725_6.length then search edge 3 [5507, 7829] innerCandidates725_6 13336 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan725_7

def innerTail725_5 : List Nat := 7643 :: innerTail725_6
def innerCandidates725_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates725_5_checked : candidates edge 4 5507 7643 innerTail725_6 best.bound = innerCandidates725_5 := by
  unfold candidates
  have row : edge 7643 = SearchData.row7643.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan725_5 : scan edge (search edge 3) 4 [5507] 5507 innerTail725_5 best = best := by
  rw [innerTail725_5, scan, if_neg (by decide), innerCandidates725_5_checked]
  change scan edge (search edge 3) 4 [5507] 5507 innerTail725_6
    (if 3 ≤ innerCandidates725_5.length then search edge 3 [5507, 7643] innerCandidates725_5 13150 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan725_6

def innerTail725_4 : List Nat := 6803 :: innerTail725_5
def innerCandidates725_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates725_4_checked : candidates edge 4 5507 6803 innerTail725_5 best.bound = innerCandidates725_4 := by
  unfold candidates
  have row : edge 6803 = SearchData.row6803.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan725_4 : scan edge (search edge 3) 4 [5507] 5507 innerTail725_4 best = best := by
  rw [innerTail725_4, scan, if_neg (by decide), innerCandidates725_4_checked]
  change scan edge (search edge 3) 4 [5507] 5507 innerTail725_5
    (if 3 ≤ innerCandidates725_4.length then search edge 3 [5507, 6803] innerCandidates725_4 12310 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan725_5

def innerTail725_3 : List Nat := 6257 :: innerTail725_4
def innerCandidates725_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates725_3_checked : candidates edge 4 5507 6257 innerTail725_4 best.bound = innerCandidates725_3 := by
  unfold candidates
  have row : edge 6257 = SearchData.row6257.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan725_3 : scan edge (search edge 3) 4 [5507] 5507 innerTail725_3 best = best := by
  rw [innerTail725_3, scan, if_neg (by decide), innerCandidates725_3_checked]
  change scan edge (search edge 3) 4 [5507] 5507 innerTail725_4
    (if 3 ≤ innerCandidates725_3.length then search edge 3 [5507, 6257] innerCandidates725_3 11764 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan725_4

def innerTail725_2 : List Nat := 6221 :: innerTail725_3
def innerCandidates725_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates725_2_checked : candidates edge 4 5507 6221 innerTail725_3 best.bound = innerCandidates725_2 := by
  unfold candidates
  have row : edge 6221 = SearchData.row6221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan725_2 : scan edge (search edge 3) 4 [5507] 5507 innerTail725_2 best = best := by
  rw [innerTail725_2, scan, if_neg (by decide), innerCandidates725_2_checked]
  change scan edge (search edge 3) 4 [5507] 5507 innerTail725_3
    (if 3 ≤ innerCandidates725_2.length then search edge 3 [5507, 6221] innerCandidates725_2 11728 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan725_3

def innerTail725_1 : List Nat := 6173 :: innerTail725_2
def innerCandidates725_1 : List Nat := [7853]
set_option maxRecDepth 4096 in
theorem innerCandidates725_1_checked : candidates edge 4 5507 6173 innerTail725_2 best.bound = innerCandidates725_1 := by
  unfold candidates
  have row : edge 6173 = SearchData.row6173.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan725_1 : scan edge (search edge 3) 4 [5507] 5507 innerTail725_1 best = best := by
  rw [innerTail725_1, scan, if_neg (by decide), innerCandidates725_1_checked]
  change scan edge (search edge 3) 4 [5507] 5507 innerTail725_2
    (if 3 ≤ innerCandidates725_1.length then search edge 3 [5507, 6173] innerCandidates725_1 11680 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan725_2

def innerTail725_0 : List Nat := 5939 :: innerTail725_1
def innerCandidates725_0 : List Nat := [6173]
set_option maxRecDepth 4096 in
theorem innerCandidates725_0_checked : candidates edge 4 5507 5939 innerTail725_1 best.bound = innerCandidates725_0 := by
  unfold candidates
  have row : edge 5939 = SearchData.row5939.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan725_0 : scan edge (search edge 3) 4 [5507] 5507 innerTail725_0 best = best := by
  rw [innerTail725_0, scan, if_neg (by decide), innerCandidates725_0_checked]
  change scan edge (search edge 3) 4 [5507] 5507 innerTail725_1
    (if 3 ≤ innerCandidates725_0.length then search edge 3 [5507, 5939] innerCandidates725_0 11446 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan725_1

theorem rootBranch725 : search edge 4 [5507] rootCandidates725 5507 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5507] 5507 innerTail725_0 best = best
  exact innerScan725_0

def rootCandidates724 : List Nat := [5821, 6469, 7027, 7393, 7603, 8311, 8353]
set_option maxRecDepth 4096 in
theorem rootCandidates724_checked : candidates edge 5 0 5503 tail725 best.bound = rootCandidates724 := by
  unfold candidates
  have row : edge 5503 = SearchData.row5503.contains := by funext q; rfl
  rw [row]
  decide

def innerTail724_7 : List Nat := []
theorem innerScan724_7 : scan edge (search edge 3) 4 [5503] 5503 innerTail724_7 best = best := rfl

def innerTail724_6 : List Nat := 8353 :: innerTail724_7
def innerCandidates724_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates724_6_checked : candidates edge 4 5503 8353 innerTail724_7 best.bound = innerCandidates724_6 := by
  unfold candidates
  have row : edge 8353 = SearchData.row8353.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan724_6 : scan edge (search edge 3) 4 [5503] 5503 innerTail724_6 best = best := by
  rw [innerTail724_6, scan, if_neg (by decide), innerCandidates724_6_checked]
  change scan edge (search edge 3) 4 [5503] 5503 innerTail724_7
    (if 3 ≤ innerCandidates724_6.length then search edge 3 [5503, 8353] innerCandidates724_6 13856 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan724_7

def innerTail724_5 : List Nat := 8311 :: innerTail724_6
def innerCandidates724_5 : List Nat := [8353]
set_option maxRecDepth 4096 in
theorem innerCandidates724_5_checked : candidates edge 4 5503 8311 innerTail724_6 best.bound = innerCandidates724_5 := by
  unfold candidates
  have row : edge 8311 = SearchData.row8311.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan724_5 : scan edge (search edge 3) 4 [5503] 5503 innerTail724_5 best = best := by
  rw [innerTail724_5, scan, if_neg (by decide), innerCandidates724_5_checked]
  change scan edge (search edge 3) 4 [5503] 5503 innerTail724_6
    (if 3 ≤ innerCandidates724_5.length then search edge 3 [5503, 8311] innerCandidates724_5 13814 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan724_6

def innerTail724_4 : List Nat := 7603 :: innerTail724_5
def innerCandidates724_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates724_4_checked : candidates edge 4 5503 7603 innerTail724_5 best.bound = innerCandidates724_4 := by
  unfold candidates
  have row : edge 7603 = SearchData.row7603.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan724_4 : scan edge (search edge 3) 4 [5503] 5503 innerTail724_4 best = best := by
  rw [innerTail724_4, scan, if_neg (by decide), innerCandidates724_4_checked]
  change scan edge (search edge 3) 4 [5503] 5503 innerTail724_5
    (if 3 ≤ innerCandidates724_4.length then search edge 3 [5503, 7603] innerCandidates724_4 13106 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan724_5

def innerTail724_3 : List Nat := 7393 :: innerTail724_4
def innerCandidates724_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates724_3_checked : candidates edge 4 5503 7393 innerTail724_4 best.bound = innerCandidates724_3 := by
  unfold candidates
  have row : edge 7393 = SearchData.row7393.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan724_3 : scan edge (search edge 3) 4 [5503] 5503 innerTail724_3 best = best := by
  rw [innerTail724_3, scan, if_neg (by decide), innerCandidates724_3_checked]
  change scan edge (search edge 3) 4 [5503] 5503 innerTail724_4
    (if 3 ≤ innerCandidates724_3.length then search edge 3 [5503, 7393] innerCandidates724_3 12896 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan724_4

def innerTail724_2 : List Nat := 7027 :: innerTail724_3
def innerCandidates724_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates724_2_checked : candidates edge 4 5503 7027 innerTail724_3 best.bound = innerCandidates724_2 := by
  unfold candidates
  have row : edge 7027 = SearchData.row7027.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan724_2 : scan edge (search edge 3) 4 [5503] 5503 innerTail724_2 best = best := by
  rw [innerTail724_2, scan, if_neg (by decide), innerCandidates724_2_checked]
  change scan edge (search edge 3) 4 [5503] 5503 innerTail724_3
    (if 3 ≤ innerCandidates724_2.length then search edge 3 [5503, 7027] innerCandidates724_2 12530 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan724_3

def innerTail724_1 : List Nat := 6469 :: innerTail724_2
def innerCandidates724_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates724_1_checked : candidates edge 4 5503 6469 innerTail724_2 best.bound = innerCandidates724_1 := by
  unfold candidates
  have row : edge 6469 = SearchData.row6469.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan724_1 : scan edge (search edge 3) 4 [5503] 5503 innerTail724_1 best = best := by
  rw [innerTail724_1, scan, if_neg (by decide), innerCandidates724_1_checked]
  change scan edge (search edge 3) 4 [5503] 5503 innerTail724_2
    (if 3 ≤ innerCandidates724_1.length then search edge 3 [5503, 6469] innerCandidates724_1 11972 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan724_2

def innerTail724_0 : List Nat := 5821 :: innerTail724_1
def innerCandidates724_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates724_0_checked : candidates edge 4 5503 5821 innerTail724_1 best.bound = innerCandidates724_0 := by
  unfold candidates
  have row : edge 5821 = SearchData.row5821.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan724_0 : scan edge (search edge 3) 4 [5503] 5503 innerTail724_0 best = best := by
  rw [innerTail724_0, scan, if_neg (by decide), innerCandidates724_0_checked]
  change scan edge (search edge 3) 4 [5503] 5503 innerTail724_1
    (if 3 ≤ innerCandidates724_0.length then search edge 3 [5503, 5821] innerCandidates724_0 11324 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan724_1

theorem rootBranch724 : search edge 4 [5503] rootCandidates724 5503 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5503] 5503 innerTail724_0 best = best
  exact innerScan724_0

def rootCandidates723 : List Nat := [5591, 5867, 5879, 6359, 7211, 7349, 7481]
set_option maxRecDepth 4096 in
theorem rootCandidates723_checked : candidates edge 5 0 5501 tail724 best.bound = rootCandidates723 := by
  unfold candidates
  have row : edge 5501 = SearchData.row5501.contains := by funext q; rfl
  rw [row]
  decide

def innerTail723_7 : List Nat := []
theorem innerScan723_7 : scan edge (search edge 3) 4 [5501] 5501 innerTail723_7 best = best := rfl

def innerTail723_6 : List Nat := 7481 :: innerTail723_7
def innerCandidates723_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates723_6_checked : candidates edge 4 5501 7481 innerTail723_7 best.bound = innerCandidates723_6 := by
  unfold candidates
  have row : edge 7481 = SearchData.row7481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan723_6 : scan edge (search edge 3) 4 [5501] 5501 innerTail723_6 best = best := by
  rw [innerTail723_6, scan, if_neg (by decide), innerCandidates723_6_checked]
  change scan edge (search edge 3) 4 [5501] 5501 innerTail723_7
    (if 3 ≤ innerCandidates723_6.length then search edge 3 [5501, 7481] innerCandidates723_6 12982 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan723_7

def innerTail723_5 : List Nat := 7349 :: innerTail723_6
def innerCandidates723_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates723_5_checked : candidates edge 4 5501 7349 innerTail723_6 best.bound = innerCandidates723_5 := by
  unfold candidates
  have row : edge 7349 = SearchData.row7349.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan723_5 : scan edge (search edge 3) 4 [5501] 5501 innerTail723_5 best = best := by
  rw [innerTail723_5, scan, if_neg (by decide), innerCandidates723_5_checked]
  change scan edge (search edge 3) 4 [5501] 5501 innerTail723_6
    (if 3 ≤ innerCandidates723_5.length then search edge 3 [5501, 7349] innerCandidates723_5 12850 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan723_6

def innerTail723_4 : List Nat := 7211 :: innerTail723_5
def innerCandidates723_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates723_4_checked : candidates edge 4 5501 7211 innerTail723_5 best.bound = innerCandidates723_4 := by
  unfold candidates
  have row : edge 7211 = SearchData.row7211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan723_4 : scan edge (search edge 3) 4 [5501] 5501 innerTail723_4 best = best := by
  rw [innerTail723_4, scan, if_neg (by decide), innerCandidates723_4_checked]
  change scan edge (search edge 3) 4 [5501] 5501 innerTail723_5
    (if 3 ≤ innerCandidates723_4.length then search edge 3 [5501, 7211] innerCandidates723_4 12712 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan723_5

def innerTail723_3 : List Nat := 6359 :: innerTail723_4
def innerCandidates723_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates723_3_checked : candidates edge 4 5501 6359 innerTail723_4 best.bound = innerCandidates723_3 := by
  unfold candidates
  have row : edge 6359 = SearchData.row6359.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan723_3 : scan edge (search edge 3) 4 [5501] 5501 innerTail723_3 best = best := by
  rw [innerTail723_3, scan, if_neg (by decide), innerCandidates723_3_checked]
  change scan edge (search edge 3) 4 [5501] 5501 innerTail723_4
    (if 3 ≤ innerCandidates723_3.length then search edge 3 [5501, 6359] innerCandidates723_3 11860 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan723_4

def innerTail723_2 : List Nat := 5879 :: innerTail723_3
def innerCandidates723_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates723_2_checked : candidates edge 4 5501 5879 innerTail723_3 best.bound = innerCandidates723_2 := by
  unfold candidates
  have row : edge 5879 = SearchData.row5879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan723_2 : scan edge (search edge 3) 4 [5501] 5501 innerTail723_2 best = best := by
  rw [innerTail723_2, scan, if_neg (by decide), innerCandidates723_2_checked]
  change scan edge (search edge 3) 4 [5501] 5501 innerTail723_3
    (if 3 ≤ innerCandidates723_2.length then search edge 3 [5501, 5879] innerCandidates723_2 11380 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan723_3

def innerTail723_1 : List Nat := 5867 :: innerTail723_2
def innerCandidates723_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates723_1_checked : candidates edge 4 5501 5867 innerTail723_2 best.bound = innerCandidates723_1 := by
  unfold candidates
  have row : edge 5867 = SearchData.row5867.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan723_1 : scan edge (search edge 3) 4 [5501] 5501 innerTail723_1 best = best := by
  rw [innerTail723_1, scan, if_neg (by decide), innerCandidates723_1_checked]
  change scan edge (search edge 3) 4 [5501] 5501 innerTail723_2
    (if 3 ≤ innerCandidates723_1.length then search edge 3 [5501, 5867] innerCandidates723_1 11368 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan723_2

def innerTail723_0 : List Nat := 5591 :: innerTail723_1
def innerCandidates723_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates723_0_checked : candidates edge 4 5501 5591 innerTail723_1 best.bound = innerCandidates723_0 := by
  unfold candidates
  have row : edge 5591 = SearchData.row5591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan723_0 : scan edge (search edge 3) 4 [5501] 5501 innerTail723_0 best = best := by
  rw [innerTail723_0, scan, if_neg (by decide), innerCandidates723_0_checked]
  change scan edge (search edge 3) 4 [5501] 5501 innerTail723_1
    (if 3 ≤ innerCandidates723_0.length then search edge 3 [5501, 5591] innerCandidates723_0 11092 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan723_1

theorem rootBranch723 : search edge 4 [5501] rootCandidates723 5501 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5501] 5501 innerTail723_0 best = best
  exact innerScan723_0

def rootCandidates722 : List Nat := [5507, 5987, 6029, 6491, 7103, 7247, 7487, 8297, 8369]
set_option maxRecDepth 4096 in
theorem rootCandidates722_checked : candidates edge 5 0 5483 tail723 best.bound = rootCandidates722 := by
  unfold candidates
  have row : edge 5483 = SearchData.row5483.contains := by funext q; rfl
  rw [row]
  decide

def innerTail722_9 : List Nat := []
theorem innerScan722_9 : scan edge (search edge 3) 4 [5483] 5483 innerTail722_9 best = best := rfl

def innerTail722_8 : List Nat := 8369 :: innerTail722_9
def innerCandidates722_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates722_8_checked : candidates edge 4 5483 8369 innerTail722_9 best.bound = innerCandidates722_8 := by
  unfold candidates
  have row : edge 8369 = SearchData.row8369.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan722_8 : scan edge (search edge 3) 4 [5483] 5483 innerTail722_8 best = best := by
  rw [innerTail722_8, scan, if_neg (by decide), innerCandidates722_8_checked]
  change scan edge (search edge 3) 4 [5483] 5483 innerTail722_9
    (if 3 ≤ innerCandidates722_8.length then search edge 3 [5483, 8369] innerCandidates722_8 13852 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan722_9

def innerTail722_7 : List Nat := 8297 :: innerTail722_8
def innerCandidates722_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates722_7_checked : candidates edge 4 5483 8297 innerTail722_8 best.bound = innerCandidates722_7 := by
  unfold candidates
  have row : edge 8297 = SearchData.row8297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan722_7 : scan edge (search edge 3) 4 [5483] 5483 innerTail722_7 best = best := by
  rw [innerTail722_7, scan, if_neg (by decide), innerCandidates722_7_checked]
  change scan edge (search edge 3) 4 [5483] 5483 innerTail722_8
    (if 3 ≤ innerCandidates722_7.length then search edge 3 [5483, 8297] innerCandidates722_7 13780 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan722_8

def innerTail722_6 : List Nat := 7487 :: innerTail722_7
def innerCandidates722_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates722_6_checked : candidates edge 4 5483 7487 innerTail722_7 best.bound = innerCandidates722_6 := by
  unfold candidates
  have row : edge 7487 = SearchData.row7487.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan722_6 : scan edge (search edge 3) 4 [5483] 5483 innerTail722_6 best = best := by
  rw [innerTail722_6, scan, if_neg (by decide), innerCandidates722_6_checked]
  change scan edge (search edge 3) 4 [5483] 5483 innerTail722_7
    (if 3 ≤ innerCandidates722_6.length then search edge 3 [5483, 7487] innerCandidates722_6 12970 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan722_7

def innerTail722_5 : List Nat := 7247 :: innerTail722_6
def innerCandidates722_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates722_5_checked : candidates edge 4 5483 7247 innerTail722_6 best.bound = innerCandidates722_5 := by
  unfold candidates
  have row : edge 7247 = SearchData.row7247.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan722_5 : scan edge (search edge 3) 4 [5483] 5483 innerTail722_5 best = best := by
  rw [innerTail722_5, scan, if_neg (by decide), innerCandidates722_5_checked]
  change scan edge (search edge 3) 4 [5483] 5483 innerTail722_6
    (if 3 ≤ innerCandidates722_5.length then search edge 3 [5483, 7247] innerCandidates722_5 12730 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan722_6

def innerTail722_4 : List Nat := 7103 :: innerTail722_5
def innerCandidates722_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates722_4_checked : candidates edge 4 5483 7103 innerTail722_5 best.bound = innerCandidates722_4 := by
  unfold candidates
  have row : edge 7103 = SearchData.row7103.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan722_4 : scan edge (search edge 3) 4 [5483] 5483 innerTail722_4 best = best := by
  rw [innerTail722_4, scan, if_neg (by decide), innerCandidates722_4_checked]
  change scan edge (search edge 3) 4 [5483] 5483 innerTail722_5
    (if 3 ≤ innerCandidates722_4.length then search edge 3 [5483, 7103] innerCandidates722_4 12586 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan722_5

def innerTail722_3 : List Nat := 6491 :: innerTail722_4
def innerCandidates722_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates722_3_checked : candidates edge 4 5483 6491 innerTail722_4 best.bound = innerCandidates722_3 := by
  unfold candidates
  have row : edge 6491 = SearchData.row6491.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan722_3 : scan edge (search edge 3) 4 [5483] 5483 innerTail722_3 best = best := by
  rw [innerTail722_3, scan, if_neg (by decide), innerCandidates722_3_checked]
  change scan edge (search edge 3) 4 [5483] 5483 innerTail722_4
    (if 3 ≤ innerCandidates722_3.length then search edge 3 [5483, 6491] innerCandidates722_3 11974 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan722_4

def innerTail722_2 : List Nat := 6029 :: innerTail722_3
def innerCandidates722_2 : List Nat := [6491]
set_option maxRecDepth 4096 in
theorem innerCandidates722_2_checked : candidates edge 4 5483 6029 innerTail722_3 best.bound = innerCandidates722_2 := by
  unfold candidates
  have row : edge 6029 = SearchData.row6029.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan722_2 : scan edge (search edge 3) 4 [5483] 5483 innerTail722_2 best = best := by
  rw [innerTail722_2, scan, if_neg (by decide), innerCandidates722_2_checked]
  change scan edge (search edge 3) 4 [5483] 5483 innerTail722_3
    (if 3 ≤ innerCandidates722_2.length then search edge 3 [5483, 6029] innerCandidates722_2 11512 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan722_3

def innerTail722_1 : List Nat := 5987 :: innerTail722_2
def innerCandidates722_1 : List Nat := [6491]
set_option maxRecDepth 4096 in
theorem innerCandidates722_1_checked : candidates edge 4 5483 5987 innerTail722_2 best.bound = innerCandidates722_1 := by
  unfold candidates
  have row : edge 5987 = SearchData.row5987.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan722_1 : scan edge (search edge 3) 4 [5483] 5483 innerTail722_1 best = best := by
  rw [innerTail722_1, scan, if_neg (by decide), innerCandidates722_1_checked]
  change scan edge (search edge 3) 4 [5483] 5483 innerTail722_2
    (if 3 ≤ innerCandidates722_1.length then search edge 3 [5483, 5987] innerCandidates722_1 11470 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan722_2

def innerTail722_0 : List Nat := 5507 :: innerTail722_1
def innerCandidates722_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates722_0_checked : candidates edge 4 5483 5507 innerTail722_1 best.bound = innerCandidates722_0 := by
  unfold candidates
  have row : edge 5507 = SearchData.row5507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan722_0 : scan edge (search edge 3) 4 [5483] 5483 innerTail722_0 best = best := by
  rw [innerTail722_0, scan, if_neg (by decide), innerCandidates722_0_checked]
  change scan edge (search edge 3) 4 [5483] 5483 innerTail722_1
    (if 3 ≤ innerCandidates722_0.length then search edge 3 [5483, 5507] innerCandidates722_0 10990 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan722_1

theorem rootBranch722 : search edge 4 [5483] rootCandidates722 5483 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5483] 5483 innerTail722_0 best = best
  exact innerScan722_0

def rootCandidates721 : List Nat := [5581, 5827, 5857, 5869, 6271, 6607, 7321, 7927, 8209]
set_option maxRecDepth 4096 in
theorem rootCandidates721_checked : candidates edge 5 0 5479 tail722 best.bound = rootCandidates721 := by
  unfold candidates
  have row : edge 5479 = SearchData.row5479.contains := by funext q; rfl
  rw [row]
  decide

def innerTail721_9 : List Nat := []
theorem innerScan721_9 : scan edge (search edge 3) 4 [5479] 5479 innerTail721_9 best = best := rfl

def innerTail721_8 : List Nat := 8209 :: innerTail721_9
def innerCandidates721_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates721_8_checked : candidates edge 4 5479 8209 innerTail721_9 best.bound = innerCandidates721_8 := by
  unfold candidates
  have row : edge 8209 = SearchData.row8209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan721_8 : scan edge (search edge 3) 4 [5479] 5479 innerTail721_8 best = best := by
  rw [innerTail721_8, scan, if_neg (by decide), innerCandidates721_8_checked]
  change scan edge (search edge 3) 4 [5479] 5479 innerTail721_9
    (if 3 ≤ innerCandidates721_8.length then search edge 3 [5479, 8209] innerCandidates721_8 13688 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan721_9

def innerTail721_7 : List Nat := 7927 :: innerTail721_8
def innerCandidates721_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates721_7_checked : candidates edge 4 5479 7927 innerTail721_8 best.bound = innerCandidates721_7 := by
  unfold candidates
  have row : edge 7927 = SearchData.row7927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan721_7 : scan edge (search edge 3) 4 [5479] 5479 innerTail721_7 best = best := by
  rw [innerTail721_7, scan, if_neg (by decide), innerCandidates721_7_checked]
  change scan edge (search edge 3) 4 [5479] 5479 innerTail721_8
    (if 3 ≤ innerCandidates721_7.length then search edge 3 [5479, 7927] innerCandidates721_7 13406 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan721_8

def innerTail721_6 : List Nat := 7321 :: innerTail721_7
def innerCandidates721_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates721_6_checked : candidates edge 4 5479 7321 innerTail721_7 best.bound = innerCandidates721_6 := by
  unfold candidates
  have row : edge 7321 = SearchData.row7321.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan721_6 : scan edge (search edge 3) 4 [5479] 5479 innerTail721_6 best = best := by
  rw [innerTail721_6, scan, if_neg (by decide), innerCandidates721_6_checked]
  change scan edge (search edge 3) 4 [5479] 5479 innerTail721_7
    (if 3 ≤ innerCandidates721_6.length then search edge 3 [5479, 7321] innerCandidates721_6 12800 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan721_7

def innerTail721_5 : List Nat := 6607 :: innerTail721_6
def innerCandidates721_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates721_5_checked : candidates edge 4 5479 6607 innerTail721_6 best.bound = innerCandidates721_5 := by
  unfold candidates
  have row : edge 6607 = SearchData.row6607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan721_5 : scan edge (search edge 3) 4 [5479] 5479 innerTail721_5 best = best := by
  rw [innerTail721_5, scan, if_neg (by decide), innerCandidates721_5_checked]
  change scan edge (search edge 3) 4 [5479] 5479 innerTail721_6
    (if 3 ≤ innerCandidates721_5.length then search edge 3 [5479, 6607] innerCandidates721_5 12086 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan721_6

def innerTail721_4 : List Nat := 6271 :: innerTail721_5
def innerCandidates721_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates721_4_checked : candidates edge 4 5479 6271 innerTail721_5 best.bound = innerCandidates721_4 := by
  unfold candidates
  have row : edge 6271 = SearchData.row6271.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan721_4 : scan edge (search edge 3) 4 [5479] 5479 innerTail721_4 best = best := by
  rw [innerTail721_4, scan, if_neg (by decide), innerCandidates721_4_checked]
  change scan edge (search edge 3) 4 [5479] 5479 innerTail721_5
    (if 3 ≤ innerCandidates721_4.length then search edge 3 [5479, 6271] innerCandidates721_4 11750 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan721_5

def innerTail721_3 : List Nat := 5869 :: innerTail721_4
def innerCandidates721_3 : List Nat := [8209]
set_option maxRecDepth 4096 in
theorem innerCandidates721_3_checked : candidates edge 4 5479 5869 innerTail721_4 best.bound = innerCandidates721_3 := by
  unfold candidates
  have row : edge 5869 = SearchData.row5869.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan721_3 : scan edge (search edge 3) 4 [5479] 5479 innerTail721_3 best = best := by
  rw [innerTail721_3, scan, if_neg (by decide), innerCandidates721_3_checked]
  change scan edge (search edge 3) 4 [5479] 5479 innerTail721_4
    (if 3 ≤ innerCandidates721_3.length then search edge 3 [5479, 5869] innerCandidates721_3 11348 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan721_4

def innerTail721_2 : List Nat := 5857 :: innerTail721_3
def innerCandidates721_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates721_2_checked : candidates edge 4 5479 5857 innerTail721_3 best.bound = innerCandidates721_2 := by
  unfold candidates
  have row : edge 5857 = SearchData.row5857.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan721_2 : scan edge (search edge 3) 4 [5479] 5479 innerTail721_2 best = best := by
  rw [innerTail721_2, scan, if_neg (by decide), innerCandidates721_2_checked]
  change scan edge (search edge 3) 4 [5479] 5479 innerTail721_3
    (if 3 ≤ innerCandidates721_2.length then search edge 3 [5479, 5857] innerCandidates721_2 11336 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan721_3

def innerTail721_1 : List Nat := 5827 :: innerTail721_2
def innerCandidates721_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates721_1_checked : candidates edge 4 5479 5827 innerTail721_2 best.bound = innerCandidates721_1 := by
  unfold candidates
  have row : edge 5827 = SearchData.row5827.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan721_1 : scan edge (search edge 3) 4 [5479] 5479 innerTail721_1 best = best := by
  rw [innerTail721_1, scan, if_neg (by decide), innerCandidates721_1_checked]
  change scan edge (search edge 3) 4 [5479] 5479 innerTail721_2
    (if 3 ≤ innerCandidates721_1.length then search edge 3 [5479, 5827] innerCandidates721_1 11306 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan721_2

def innerTail721_0 : List Nat := 5581 :: innerTail721_1
def innerCandidates721_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates721_0_checked : candidates edge 4 5479 5581 innerTail721_1 best.bound = innerCandidates721_0 := by
  unfold candidates
  have row : edge 5581 = SearchData.row5581.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan721_0 : scan edge (search edge 3) 4 [5479] 5479 innerTail721_0 best = best := by
  rw [innerTail721_0, scan, if_neg (by decide), innerCandidates721_0_checked]
  change scan edge (search edge 3) 4 [5479] 5479 innerTail721_1
    (if 3 ≤ innerCandidates721_0.length then search edge 3 [5479, 5581] innerCandidates721_0 11060 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan721_1

theorem rootBranch721 : search edge 4 [5479] rootCandidates721 5479 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5479] 5479 innerTail721_0 best = best
  exact innerScan721_0

end ProjectEulerSolutions.PrimePairUpper.Search
