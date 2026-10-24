import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates624 : List Nat := [4877, 5153, 5231, 5237, 5471, 5693, 5717, 5849, 6263, 6473, 6827, 7757]
set_option maxRecDepth 4096 in
theorem rootCandidates624_checked : candidates edge 5 0 4643 tail625 best.bound = rootCandidates624 := by
  unfold candidates
  have row : edge 4643 = SearchData.row4643.contains := by funext q; rfl
  rw [row]
  decide

def innerTail624_12 : List Nat := []
theorem innerScan624_12 : scan edge (search edge 3) 4 [4643] 4643 innerTail624_12 best = best := rfl

def innerTail624_11 : List Nat := 7757 :: innerTail624_12
def innerCandidates624_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates624_11_checked : candidates edge 4 4643 7757 innerTail624_12 best.bound = innerCandidates624_11 := by
  unfold candidates
  have row : edge 7757 = SearchData.row7757.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan624_11 : scan edge (search edge 3) 4 [4643] 4643 innerTail624_11 best = best := by
  rw [innerTail624_11, scan, if_neg (by decide), innerCandidates624_11_checked]
  change scan edge (search edge 3) 4 [4643] 4643 innerTail624_12
    (if 3 ≤ innerCandidates624_11.length then search edge 3 [4643, 7757] innerCandidates624_11 12400 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan624_12

def innerTail624_10 : List Nat := 6827 :: innerTail624_11
def innerCandidates624_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates624_10_checked : candidates edge 4 4643 6827 innerTail624_11 best.bound = innerCandidates624_10 := by
  unfold candidates
  have row : edge 6827 = SearchData.row6827.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan624_10 : scan edge (search edge 3) 4 [4643] 4643 innerTail624_10 best = best := by
  rw [innerTail624_10, scan, if_neg (by decide), innerCandidates624_10_checked]
  change scan edge (search edge 3) 4 [4643] 4643 innerTail624_11
    (if 3 ≤ innerCandidates624_10.length then search edge 3 [4643, 6827] innerCandidates624_10 11470 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan624_11

def innerTail624_9 : List Nat := 6473 :: innerTail624_10
def innerCandidates624_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates624_9_checked : candidates edge 4 4643 6473 innerTail624_10 best.bound = innerCandidates624_9 := by
  unfold candidates
  have row : edge 6473 = SearchData.row6473.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan624_9 : scan edge (search edge 3) 4 [4643] 4643 innerTail624_9 best = best := by
  rw [innerTail624_9, scan, if_neg (by decide), innerCandidates624_9_checked]
  change scan edge (search edge 3) 4 [4643] 4643 innerTail624_10
    (if 3 ≤ innerCandidates624_9.length then search edge 3 [4643, 6473] innerCandidates624_9 11116 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan624_10

def innerTail624_8 : List Nat := 6263 :: innerTail624_9
def innerCandidates624_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates624_8_checked : candidates edge 4 4643 6263 innerTail624_9 best.bound = innerCandidates624_8 := by
  unfold candidates
  have row : edge 6263 = SearchData.row6263.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan624_8 : scan edge (search edge 3) 4 [4643] 4643 innerTail624_8 best = best := by
  rw [innerTail624_8, scan, if_neg (by decide), innerCandidates624_8_checked]
  change scan edge (search edge 3) 4 [4643] 4643 innerTail624_9
    (if 3 ≤ innerCandidates624_8.length then search edge 3 [4643, 6263] innerCandidates624_8 10906 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan624_9

def innerTail624_7 : List Nat := 5849 :: innerTail624_8
def innerCandidates624_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates624_7_checked : candidates edge 4 4643 5849 innerTail624_8 best.bound = innerCandidates624_7 := by
  unfold candidates
  have row : edge 5849 = SearchData.row5849.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan624_7 : scan edge (search edge 3) 4 [4643] 4643 innerTail624_7 best = best := by
  rw [innerTail624_7, scan, if_neg (by decide), innerCandidates624_7_checked]
  change scan edge (search edge 3) 4 [4643] 4643 innerTail624_8
    (if 3 ≤ innerCandidates624_7.length then search edge 3 [4643, 5849] innerCandidates624_7 10492 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan624_8

def innerTail624_6 : List Nat := 5717 :: innerTail624_7
def innerCandidates624_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates624_6_checked : candidates edge 4 4643 5717 innerTail624_7 best.bound = innerCandidates624_6 := by
  unfold candidates
  have row : edge 5717 = SearchData.row5717.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan624_6 : scan edge (search edge 3) 4 [4643] 4643 innerTail624_6 best = best := by
  rw [innerTail624_6, scan, if_neg (by decide), innerCandidates624_6_checked]
  change scan edge (search edge 3) 4 [4643] 4643 innerTail624_7
    (if 3 ≤ innerCandidates624_6.length then search edge 3 [4643, 5717] innerCandidates624_6 10360 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan624_7

def innerTail624_5 : List Nat := 5693 :: innerTail624_6
def innerCandidates624_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates624_5_checked : candidates edge 4 4643 5693 innerTail624_6 best.bound = innerCandidates624_5 := by
  unfold candidates
  have row : edge 5693 = SearchData.row5693.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan624_5 : scan edge (search edge 3) 4 [4643] 4643 innerTail624_5 best = best := by
  rw [innerTail624_5, scan, if_neg (by decide), innerCandidates624_5_checked]
  change scan edge (search edge 3) 4 [4643] 4643 innerTail624_6
    (if 3 ≤ innerCandidates624_5.length then search edge 3 [4643, 5693] innerCandidates624_5 10336 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan624_6

def innerTail624_4 : List Nat := 5471 :: innerTail624_5
def innerCandidates624_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates624_4_checked : candidates edge 4 4643 5471 innerTail624_5 best.bound = innerCandidates624_4 := by
  unfold candidates
  have row : edge 5471 = SearchData.row5471.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan624_4 : scan edge (search edge 3) 4 [4643] 4643 innerTail624_4 best = best := by
  rw [innerTail624_4, scan, if_neg (by decide), innerCandidates624_4_checked]
  change scan edge (search edge 3) 4 [4643] 4643 innerTail624_5
    (if 3 ≤ innerCandidates624_4.length then search edge 3 [4643, 5471] innerCandidates624_4 10114 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan624_5

def innerTail624_3 : List Nat := 5237 :: innerTail624_4
def innerCandidates624_3 : List Nat := [5471, 5693]
set_option maxRecDepth 4096 in
theorem innerCandidates624_3_checked : candidates edge 4 4643 5237 innerTail624_4 best.bound = innerCandidates624_3 := by
  unfold candidates
  have row : edge 5237 = SearchData.row5237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan624_3 : scan edge (search edge 3) 4 [4643] 4643 innerTail624_3 best = best := by
  rw [innerTail624_3, scan, if_neg (by decide), innerCandidates624_3_checked]
  change scan edge (search edge 3) 4 [4643] 4643 innerTail624_4
    (if 3 ≤ innerCandidates624_3.length then search edge 3 [4643, 5237] innerCandidates624_3 9880 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan624_4

def innerTail624_2 : List Nat := 5231 :: innerTail624_3
def innerCandidates624_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates624_2_checked : candidates edge 4 4643 5231 innerTail624_3 best.bound = innerCandidates624_2 := by
  unfold candidates
  have row : edge 5231 = SearchData.row5231.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan624_2 : scan edge (search edge 3) 4 [4643] 4643 innerTail624_2 best = best := by
  rw [innerTail624_2, scan, if_neg (by decide), innerCandidates624_2_checked]
  change scan edge (search edge 3) 4 [4643] 4643 innerTail624_3
    (if 3 ≤ innerCandidates624_2.length then search edge 3 [4643, 5231] innerCandidates624_2 9874 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan624_3

def innerTail624_1 : List Nat := 5153 :: innerTail624_2
def innerCandidates624_1 : List Nat := [5849]
set_option maxRecDepth 4096 in
theorem innerCandidates624_1_checked : candidates edge 4 4643 5153 innerTail624_2 best.bound = innerCandidates624_1 := by
  unfold candidates
  have row : edge 5153 = SearchData.row5153.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan624_1 : scan edge (search edge 3) 4 [4643] 4643 innerTail624_1 best = best := by
  rw [innerTail624_1, scan, if_neg (by decide), innerCandidates624_1_checked]
  change scan edge (search edge 3) 4 [4643] 4643 innerTail624_2
    (if 3 ≤ innerCandidates624_1.length then search edge 3 [4643, 5153] innerCandidates624_1 9796 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan624_2

def innerTail624_0 : List Nat := 4877 :: innerTail624_1
def innerCandidates624_0 : List Nat := [5693]
set_option maxRecDepth 4096 in
theorem innerCandidates624_0_checked : candidates edge 4 4643 4877 innerTail624_1 best.bound = innerCandidates624_0 := by
  unfold candidates
  have row : edge 4877 = SearchData.row4877.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan624_0 : scan edge (search edge 3) 4 [4643] 4643 innerTail624_0 best = best := by
  rw [innerTail624_0, scan, if_neg (by decide), innerCandidates624_0_checked]
  change scan edge (search edge 3) 4 [4643] 4643 innerTail624_1
    (if 3 ≤ innerCandidates624_0.length then search edge 3 [4643, 4877] innerCandidates624_0 9520 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan624_1

theorem rootBranch624 : search edge 4 [4643] rootCandidates624 4643 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4643] 4643 innerTail624_0 best = best
  exact innerScan624_0

def rootCandidates623 : List Nat := [4951, 5749, 5827, 6121, 6211, 6373, 6709, 7177, 7237]
set_option maxRecDepth 4096 in
theorem rootCandidates623_checked : candidates edge 5 0 4639 tail624 best.bound = rootCandidates623 := by
  unfold candidates
  have row : edge 4639 = SearchData.row4639.contains := by funext q; rfl
  rw [row]
  decide

def innerTail623_9 : List Nat := []
theorem innerScan623_9 : scan edge (search edge 3) 4 [4639] 4639 innerTail623_9 best = best := rfl

def innerTail623_8 : List Nat := 7237 :: innerTail623_9
def innerCandidates623_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates623_8_checked : candidates edge 4 4639 7237 innerTail623_9 best.bound = innerCandidates623_8 := by
  unfold candidates
  have row : edge 7237 = SearchData.row7237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan623_8 : scan edge (search edge 3) 4 [4639] 4639 innerTail623_8 best = best := by
  rw [innerTail623_8, scan, if_neg (by decide), innerCandidates623_8_checked]
  change scan edge (search edge 3) 4 [4639] 4639 innerTail623_9
    (if 3 ≤ innerCandidates623_8.length then search edge 3 [4639, 7237] innerCandidates623_8 11876 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan623_9

def innerTail623_7 : List Nat := 7177 :: innerTail623_8
def innerCandidates623_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates623_7_checked : candidates edge 4 4639 7177 innerTail623_8 best.bound = innerCandidates623_7 := by
  unfold candidates
  have row : edge 7177 = SearchData.row7177.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan623_7 : scan edge (search edge 3) 4 [4639] 4639 innerTail623_7 best = best := by
  rw [innerTail623_7, scan, if_neg (by decide), innerCandidates623_7_checked]
  change scan edge (search edge 3) 4 [4639] 4639 innerTail623_8
    (if 3 ≤ innerCandidates623_7.length then search edge 3 [4639, 7177] innerCandidates623_7 11816 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan623_8

def innerTail623_6 : List Nat := 6709 :: innerTail623_7
def innerCandidates623_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates623_6_checked : candidates edge 4 4639 6709 innerTail623_7 best.bound = innerCandidates623_6 := by
  unfold candidates
  have row : edge 6709 = SearchData.row6709.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan623_6 : scan edge (search edge 3) 4 [4639] 4639 innerTail623_6 best = best := by
  rw [innerTail623_6, scan, if_neg (by decide), innerCandidates623_6_checked]
  change scan edge (search edge 3) 4 [4639] 4639 innerTail623_7
    (if 3 ≤ innerCandidates623_6.length then search edge 3 [4639, 6709] innerCandidates623_6 11348 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan623_7

def innerTail623_5 : List Nat := 6373 :: innerTail623_6
def innerCandidates623_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates623_5_checked : candidates edge 4 4639 6373 innerTail623_6 best.bound = innerCandidates623_5 := by
  unfold candidates
  have row : edge 6373 = SearchData.row6373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan623_5 : scan edge (search edge 3) 4 [4639] 4639 innerTail623_5 best = best := by
  rw [innerTail623_5, scan, if_neg (by decide), innerCandidates623_5_checked]
  change scan edge (search edge 3) 4 [4639] 4639 innerTail623_6
    (if 3 ≤ innerCandidates623_5.length then search edge 3 [4639, 6373] innerCandidates623_5 11012 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan623_6

def innerTail623_4 : List Nat := 6211 :: innerTail623_5
def innerCandidates623_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates623_4_checked : candidates edge 4 4639 6211 innerTail623_5 best.bound = innerCandidates623_4 := by
  unfold candidates
  have row : edge 6211 = SearchData.row6211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan623_4 : scan edge (search edge 3) 4 [4639] 4639 innerTail623_4 best = best := by
  rw [innerTail623_4, scan, if_neg (by decide), innerCandidates623_4_checked]
  change scan edge (search edge 3) 4 [4639] 4639 innerTail623_5
    (if 3 ≤ innerCandidates623_4.length then search edge 3 [4639, 6211] innerCandidates623_4 10850 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan623_5

def innerTail623_3 : List Nat := 6121 :: innerTail623_4
def innerCandidates623_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates623_3_checked : candidates edge 4 4639 6121 innerTail623_4 best.bound = innerCandidates623_3 := by
  unfold candidates
  have row : edge 6121 = SearchData.row6121.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan623_3 : scan edge (search edge 3) 4 [4639] 4639 innerTail623_3 best = best := by
  rw [innerTail623_3, scan, if_neg (by decide), innerCandidates623_3_checked]
  change scan edge (search edge 3) 4 [4639] 4639 innerTail623_4
    (if 3 ≤ innerCandidates623_3.length then search edge 3 [4639, 6121] innerCandidates623_3 10760 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan623_4

def innerTail623_2 : List Nat := 5827 :: innerTail623_3
def innerCandidates623_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates623_2_checked : candidates edge 4 4639 5827 innerTail623_3 best.bound = innerCandidates623_2 := by
  unfold candidates
  have row : edge 5827 = SearchData.row5827.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan623_2 : scan edge (search edge 3) 4 [4639] 4639 innerTail623_2 best = best := by
  rw [innerTail623_2, scan, if_neg (by decide), innerCandidates623_2_checked]
  change scan edge (search edge 3) 4 [4639] 4639 innerTail623_3
    (if 3 ≤ innerCandidates623_2.length then search edge 3 [4639, 5827] innerCandidates623_2 10466 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan623_3

def innerTail623_1 : List Nat := 5749 :: innerTail623_2
def innerCandidates623_1 : List Nat := [6211]
set_option maxRecDepth 4096 in
theorem innerCandidates623_1_checked : candidates edge 4 4639 5749 innerTail623_2 best.bound = innerCandidates623_1 := by
  unfold candidates
  have row : edge 5749 = SearchData.row5749.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan623_1 : scan edge (search edge 3) 4 [4639] 4639 innerTail623_1 best = best := by
  rw [innerTail623_1, scan, if_neg (by decide), innerCandidates623_1_checked]
  change scan edge (search edge 3) 4 [4639] 4639 innerTail623_2
    (if 3 ≤ innerCandidates623_1.length then search edge 3 [4639, 5749] innerCandidates623_1 10388 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan623_2

def innerTail623_0 : List Nat := 4951 :: innerTail623_1
def innerCandidates623_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates623_0_checked : candidates edge 4 4639 4951 innerTail623_1 best.bound = innerCandidates623_0 := by
  unfold candidates
  have row : edge 4951 = SearchData.row4951.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan623_0 : scan edge (search edge 3) 4 [4639] 4639 innerTail623_0 best = best := by
  rw [innerTail623_0, scan, if_neg (by decide), innerCandidates623_0_checked]
  change scan edge (search edge 3) 4 [4639] 4639 innerTail623_1
    (if 3 ≤ innerCandidates623_0.length then search edge 3 [4639, 4951] innerCandidates623_0 9590 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan623_1

theorem rootBranch623 : search edge 4 [4639] rootCandidates623 4639 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4639] 4639 innerTail623_0 best = best
  exact innerScan623_0

def rootCandidates622 : List Nat := [4919, 5009, 5081, 5471, 5939, 6173, 6569, 7517, 7949]
set_option maxRecDepth 4096 in
theorem rootCandidates622_checked : candidates edge 5 0 4637 tail623 best.bound = rootCandidates622 := by
  unfold candidates
  have row : edge 4637 = SearchData.row4637.contains := by funext q; rfl
  rw [row]
  decide

def innerTail622_9 : List Nat := []
theorem innerScan622_9 : scan edge (search edge 3) 4 [4637] 4637 innerTail622_9 best = best := rfl

def innerTail622_8 : List Nat := 7949 :: innerTail622_9
def innerCandidates622_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates622_8_checked : candidates edge 4 4637 7949 innerTail622_9 best.bound = innerCandidates622_8 := by
  unfold candidates
  have row : edge 7949 = SearchData.row7949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan622_8 : scan edge (search edge 3) 4 [4637] 4637 innerTail622_8 best = best := by
  rw [innerTail622_8, scan, if_neg (by decide), innerCandidates622_8_checked]
  change scan edge (search edge 3) 4 [4637] 4637 innerTail622_9
    (if 3 ≤ innerCandidates622_8.length then search edge 3 [4637, 7949] innerCandidates622_8 12586 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan622_9

def innerTail622_7 : List Nat := 7517 :: innerTail622_8
def innerCandidates622_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates622_7_checked : candidates edge 4 4637 7517 innerTail622_8 best.bound = innerCandidates622_7 := by
  unfold candidates
  have row : edge 7517 = SearchData.row7517.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan622_7 : scan edge (search edge 3) 4 [4637] 4637 innerTail622_7 best = best := by
  rw [innerTail622_7, scan, if_neg (by decide), innerCandidates622_7_checked]
  change scan edge (search edge 3) 4 [4637] 4637 innerTail622_8
    (if 3 ≤ innerCandidates622_7.length then search edge 3 [4637, 7517] innerCandidates622_7 12154 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan622_8

def innerTail622_6 : List Nat := 6569 :: innerTail622_7
def innerCandidates622_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates622_6_checked : candidates edge 4 4637 6569 innerTail622_7 best.bound = innerCandidates622_6 := by
  unfold candidates
  have row : edge 6569 = SearchData.row6569.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan622_6 : scan edge (search edge 3) 4 [4637] 4637 innerTail622_6 best = best := by
  rw [innerTail622_6, scan, if_neg (by decide), innerCandidates622_6_checked]
  change scan edge (search edge 3) 4 [4637] 4637 innerTail622_7
    (if 3 ≤ innerCandidates622_6.length then search edge 3 [4637, 6569] innerCandidates622_6 11206 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan622_7

def innerTail622_5 : List Nat := 6173 :: innerTail622_6
def innerCandidates622_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates622_5_checked : candidates edge 4 4637 6173 innerTail622_6 best.bound = innerCandidates622_5 := by
  unfold candidates
  have row : edge 6173 = SearchData.row6173.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan622_5 : scan edge (search edge 3) 4 [4637] 4637 innerTail622_5 best = best := by
  rw [innerTail622_5, scan, if_neg (by decide), innerCandidates622_5_checked]
  change scan edge (search edge 3) 4 [4637] 4637 innerTail622_6
    (if 3 ≤ innerCandidates622_5.length then search edge 3 [4637, 6173] innerCandidates622_5 10810 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan622_6

def innerTail622_4 : List Nat := 5939 :: innerTail622_5
def innerCandidates622_4 : List Nat := [6173]
set_option maxRecDepth 4096 in
theorem innerCandidates622_4_checked : candidates edge 4 4637 5939 innerTail622_5 best.bound = innerCandidates622_4 := by
  unfold candidates
  have row : edge 5939 = SearchData.row5939.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan622_4 : scan edge (search edge 3) 4 [4637] 4637 innerTail622_4 best = best := by
  rw [innerTail622_4, scan, if_neg (by decide), innerCandidates622_4_checked]
  change scan edge (search edge 3) 4 [4637] 4637 innerTail622_5
    (if 3 ≤ innerCandidates622_4.length then search edge 3 [4637, 5939] innerCandidates622_4 10576 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan622_5

def innerTail622_3 : List Nat := 5471 :: innerTail622_4
def innerCandidates622_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates622_3_checked : candidates edge 4 4637 5471 innerTail622_4 best.bound = innerCandidates622_3 := by
  unfold candidates
  have row : edge 5471 = SearchData.row5471.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan622_3 : scan edge (search edge 3) 4 [4637] 4637 innerTail622_3 best = best := by
  rw [innerTail622_3, scan, if_neg (by decide), innerCandidates622_3_checked]
  change scan edge (search edge 3) 4 [4637] 4637 innerTail622_4
    (if 3 ≤ innerCandidates622_3.length then search edge 3 [4637, 5471] innerCandidates622_3 10108 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan622_4

def innerTail622_2 : List Nat := 5081 :: innerTail622_3
def innerCandidates622_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates622_2_checked : candidates edge 4 4637 5081 innerTail622_3 best.bound = innerCandidates622_2 := by
  unfold candidates
  have row : edge 5081 = SearchData.row5081.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan622_2 : scan edge (search edge 3) 4 [4637] 4637 innerTail622_2 best = best := by
  rw [innerTail622_2, scan, if_neg (by decide), innerCandidates622_2_checked]
  change scan edge (search edge 3) 4 [4637] 4637 innerTail622_3
    (if 3 ≤ innerCandidates622_2.length then search edge 3 [4637, 5081] innerCandidates622_2 9718 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan622_3

def innerTail622_1 : List Nat := 5009 :: innerTail622_2
def innerCandidates622_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates622_1_checked : candidates edge 4 4637 5009 innerTail622_2 best.bound = innerCandidates622_1 := by
  unfold candidates
  have row : edge 5009 = SearchData.row5009.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan622_1 : scan edge (search edge 3) 4 [4637] 4637 innerTail622_1 best = best := by
  rw [innerTail622_1, scan, if_neg (by decide), innerCandidates622_1_checked]
  change scan edge (search edge 3) 4 [4637] 4637 innerTail622_2
    (if 3 ≤ innerCandidates622_1.length then search edge 3 [4637, 5009] innerCandidates622_1 9646 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan622_2

def innerTail622_0 : List Nat := 4919 :: innerTail622_1
def innerCandidates622_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates622_0_checked : candidates edge 4 4637 4919 innerTail622_1 best.bound = innerCandidates622_0 := by
  unfold candidates
  have row : edge 4919 = SearchData.row4919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan622_0 : scan edge (search edge 3) 4 [4637] 4637 innerTail622_0 best = best := by
  rw [innerTail622_0, scan, if_neg (by decide), innerCandidates622_0_checked]
  change scan edge (search edge 3) 4 [4637] 4637 innerTail622_1
    (if 3 ≤ innerCandidates622_0.length then search edge 3 [4637, 4919] innerCandidates622_0 9556 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan622_1

theorem rootBranch622 : search edge 4 [4637] rootCandidates622 4637 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4637] 4637 innerTail622_0 best = best
  exact innerScan622_0

def rootCandidates621 : List Nat := [5347, 5641, 5779, 6673, 6679, 7489, 7573, 8287]
set_option maxRecDepth 4096 in
theorem rootCandidates621_checked : candidates edge 5 0 4621 tail622 best.bound = rootCandidates621 := by
  unfold candidates
  have row : edge 4621 = SearchData.row4621.contains := by funext q; rfl
  rw [row]
  decide

def innerTail621_8 : List Nat := []
theorem innerScan621_8 : scan edge (search edge 3) 4 [4621] 4621 innerTail621_8 best = best := rfl

def innerTail621_7 : List Nat := 8287 :: innerTail621_8
def innerCandidates621_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates621_7_checked : candidates edge 4 4621 8287 innerTail621_8 best.bound = innerCandidates621_7 := by
  unfold candidates
  have row : edge 8287 = SearchData.row8287.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan621_7 : scan edge (search edge 3) 4 [4621] 4621 innerTail621_7 best = best := by
  rw [innerTail621_7, scan, if_neg (by decide), innerCandidates621_7_checked]
  change scan edge (search edge 3) 4 [4621] 4621 innerTail621_8
    (if 3 ≤ innerCandidates621_7.length then search edge 3 [4621, 8287] innerCandidates621_7 12908 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan621_8

def innerTail621_6 : List Nat := 7573 :: innerTail621_7
def innerCandidates621_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates621_6_checked : candidates edge 4 4621 7573 innerTail621_7 best.bound = innerCandidates621_6 := by
  unfold candidates
  have row : edge 7573 = SearchData.row7573.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan621_6 : scan edge (search edge 3) 4 [4621] 4621 innerTail621_6 best = best := by
  rw [innerTail621_6, scan, if_neg (by decide), innerCandidates621_6_checked]
  change scan edge (search edge 3) 4 [4621] 4621 innerTail621_7
    (if 3 ≤ innerCandidates621_6.length then search edge 3 [4621, 7573] innerCandidates621_6 12194 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan621_7

def innerTail621_5 : List Nat := 7489 :: innerTail621_6
def innerCandidates621_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates621_5_checked : candidates edge 4 4621 7489 innerTail621_6 best.bound = innerCandidates621_5 := by
  unfold candidates
  have row : edge 7489 = SearchData.row7489.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan621_5 : scan edge (search edge 3) 4 [4621] 4621 innerTail621_5 best = best := by
  rw [innerTail621_5, scan, if_neg (by decide), innerCandidates621_5_checked]
  change scan edge (search edge 3) 4 [4621] 4621 innerTail621_6
    (if 3 ≤ innerCandidates621_5.length then search edge 3 [4621, 7489] innerCandidates621_5 12110 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan621_6

def innerTail621_4 : List Nat := 6679 :: innerTail621_5
def innerCandidates621_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates621_4_checked : candidates edge 4 4621 6679 innerTail621_5 best.bound = innerCandidates621_4 := by
  unfold candidates
  have row : edge 6679 = SearchData.row6679.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan621_4 : scan edge (search edge 3) 4 [4621] 4621 innerTail621_4 best = best := by
  rw [innerTail621_4, scan, if_neg (by decide), innerCandidates621_4_checked]
  change scan edge (search edge 3) 4 [4621] 4621 innerTail621_5
    (if 3 ≤ innerCandidates621_4.length then search edge 3 [4621, 6679] innerCandidates621_4 11300 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan621_5

def innerTail621_3 : List Nat := 6673 :: innerTail621_4
def innerCandidates621_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates621_3_checked : candidates edge 4 4621 6673 innerTail621_4 best.bound = innerCandidates621_3 := by
  unfold candidates
  have row : edge 6673 = SearchData.row6673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan621_3 : scan edge (search edge 3) 4 [4621] 4621 innerTail621_3 best = best := by
  rw [innerTail621_3, scan, if_neg (by decide), innerCandidates621_3_checked]
  change scan edge (search edge 3) 4 [4621] 4621 innerTail621_4
    (if 3 ≤ innerCandidates621_3.length then search edge 3 [4621, 6673] innerCandidates621_3 11294 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan621_4

def innerTail621_2 : List Nat := 5779 :: innerTail621_3
def innerCandidates621_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates621_2_checked : candidates edge 4 4621 5779 innerTail621_3 best.bound = innerCandidates621_2 := by
  unfold candidates
  have row : edge 5779 = SearchData.row5779.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan621_2 : scan edge (search edge 3) 4 [4621] 4621 innerTail621_2 best = best := by
  rw [innerTail621_2, scan, if_neg (by decide), innerCandidates621_2_checked]
  change scan edge (search edge 3) 4 [4621] 4621 innerTail621_3
    (if 3 ≤ innerCandidates621_2.length then search edge 3 [4621, 5779] innerCandidates621_2 10400 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan621_3

def innerTail621_1 : List Nat := 5641 :: innerTail621_2
def innerCandidates621_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates621_1_checked : candidates edge 4 4621 5641 innerTail621_2 best.bound = innerCandidates621_1 := by
  unfold candidates
  have row : edge 5641 = SearchData.row5641.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan621_1 : scan edge (search edge 3) 4 [4621] 4621 innerTail621_1 best = best := by
  rw [innerTail621_1, scan, if_neg (by decide), innerCandidates621_1_checked]
  change scan edge (search edge 3) 4 [4621] 4621 innerTail621_2
    (if 3 ≤ innerCandidates621_1.length then search edge 3 [4621, 5641] innerCandidates621_1 10262 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan621_2

def innerTail621_0 : List Nat := 5347 :: innerTail621_1
def innerCandidates621_0 : List Nat := [8287]
set_option maxRecDepth 4096 in
theorem innerCandidates621_0_checked : candidates edge 4 4621 5347 innerTail621_1 best.bound = innerCandidates621_0 := by
  unfold candidates
  have row : edge 5347 = SearchData.row5347.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan621_0 : scan edge (search edge 3) 4 [4621] 4621 innerTail621_0 best = best := by
  rw [innerTail621_0, scan, if_neg (by decide), innerCandidates621_0_checked]
  change scan edge (search edge 3) 4 [4621] 4621 innerTail621_1
    (if 3 ≤ innerCandidates621_0.length then search edge 3 [4621, 5347] innerCandidates621_0 9968 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan621_1

theorem rootBranch621 : search edge 4 [4621] rootCandidates621 4621 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4621] 4621 innerTail621_0 best = best
  exact innerScan621_0

def rootCandidates620 : List Nat := [5569, 6073, 6379, 6421, 6577, 7603]
set_option maxRecDepth 4096 in
theorem rootCandidates620_checked : candidates edge 5 0 4603 tail621 best.bound = rootCandidates620 := by
  unfold candidates
  have row : edge 4603 = SearchData.row4603.contains := by funext q; rfl
  rw [row]
  decide

def innerTail620_6 : List Nat := []
theorem innerScan620_6 : scan edge (search edge 3) 4 [4603] 4603 innerTail620_6 best = best := rfl

def innerTail620_5 : List Nat := 7603 :: innerTail620_6
def innerCandidates620_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates620_5_checked : candidates edge 4 4603 7603 innerTail620_6 best.bound = innerCandidates620_5 := by
  unfold candidates
  have row : edge 7603 = SearchData.row7603.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan620_5 : scan edge (search edge 3) 4 [4603] 4603 innerTail620_5 best = best := by
  rw [innerTail620_5, scan, if_neg (by decide), innerCandidates620_5_checked]
  change scan edge (search edge 3) 4 [4603] 4603 innerTail620_6
    (if 3 ≤ innerCandidates620_5.length then search edge 3 [4603, 7603] innerCandidates620_5 12206 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan620_6

def innerTail620_4 : List Nat := 6577 :: innerTail620_5
def innerCandidates620_4 : List Nat := [7603]
set_option maxRecDepth 4096 in
theorem innerCandidates620_4_checked : candidates edge 4 4603 6577 innerTail620_5 best.bound = innerCandidates620_4 := by
  unfold candidates
  have row : edge 6577 = SearchData.row6577.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan620_4 : scan edge (search edge 3) 4 [4603] 4603 innerTail620_4 best = best := by
  rw [innerTail620_4, scan, if_neg (by decide), innerCandidates620_4_checked]
  change scan edge (search edge 3) 4 [4603] 4603 innerTail620_5
    (if 3 ≤ innerCandidates620_4.length then search edge 3 [4603, 6577] innerCandidates620_4 11180 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan620_5

def innerTail620_3 : List Nat := 6421 :: innerTail620_4
def innerCandidates620_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates620_3_checked : candidates edge 4 4603 6421 innerTail620_4 best.bound = innerCandidates620_3 := by
  unfold candidates
  have row : edge 6421 = SearchData.row6421.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan620_3 : scan edge (search edge 3) 4 [4603] 4603 innerTail620_3 best = best := by
  rw [innerTail620_3, scan, if_neg (by decide), innerCandidates620_3_checked]
  change scan edge (search edge 3) 4 [4603] 4603 innerTail620_4
    (if 3 ≤ innerCandidates620_3.length then search edge 3 [4603, 6421] innerCandidates620_3 11024 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan620_4

def innerTail620_2 : List Nat := 6379 :: innerTail620_3
def innerCandidates620_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates620_2_checked : candidates edge 4 4603 6379 innerTail620_3 best.bound = innerCandidates620_2 := by
  unfold candidates
  have row : edge 6379 = SearchData.row6379.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan620_2 : scan edge (search edge 3) 4 [4603] 4603 innerTail620_2 best = best := by
  rw [innerTail620_2, scan, if_neg (by decide), innerCandidates620_2_checked]
  change scan edge (search edge 3) 4 [4603] 4603 innerTail620_3
    (if 3 ≤ innerCandidates620_2.length then search edge 3 [4603, 6379] innerCandidates620_2 10982 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan620_3

def innerTail620_1 : List Nat := 6073 :: innerTail620_2
def innerCandidates620_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates620_1_checked : candidates edge 4 4603 6073 innerTail620_2 best.bound = innerCandidates620_1 := by
  unfold candidates
  have row : edge 6073 = SearchData.row6073.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan620_1 : scan edge (search edge 3) 4 [4603] 4603 innerTail620_1 best = best := by
  rw [innerTail620_1, scan, if_neg (by decide), innerCandidates620_1_checked]
  change scan edge (search edge 3) 4 [4603] 4603 innerTail620_2
    (if 3 ≤ innerCandidates620_1.length then search edge 3 [4603, 6073] innerCandidates620_1 10676 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan620_2

def innerTail620_0 : List Nat := 5569 :: innerTail620_1
def innerCandidates620_0 : List Nat := [6577]
set_option maxRecDepth 4096 in
theorem innerCandidates620_0_checked : candidates edge 4 4603 5569 innerTail620_1 best.bound = innerCandidates620_0 := by
  unfold candidates
  have row : edge 5569 = SearchData.row5569.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan620_0 : scan edge (search edge 3) 4 [4603] 4603 innerTail620_0 best = best := by
  rw [innerTail620_0, scan, if_neg (by decide), innerCandidates620_0_checked]
  change scan edge (search edge 3) 4 [4603] 4603 innerTail620_1
    (if 3 ≤ innerCandidates620_0.length then search edge 3 [4603, 5569] innerCandidates620_0 10172 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan620_1

theorem rootBranch620 : search edge 4 [4603] rootCandidates620 4603 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4603] 4603 innerTail620_0 best = best
  exact innerScan620_0

def rootCandidates619 : List Nat := [4969, 5113, 5563, 5581, 5749, 6043, 6079, 7213, 7993, 8011, 8263, 8293]
set_option maxRecDepth 4096 in
theorem rootCandidates619_checked : candidates edge 5 0 4597 tail620 best.bound = rootCandidates619 := by
  unfold candidates
  have row : edge 4597 = SearchData.row4597.contains := by funext q; rfl
  rw [row]
  decide

def innerTail619_12 : List Nat := []
theorem innerScan619_12 : scan edge (search edge 3) 4 [4597] 4597 innerTail619_12 best = best := rfl

def innerTail619_11 : List Nat := 8293 :: innerTail619_12
def innerCandidates619_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates619_11_checked : candidates edge 4 4597 8293 innerTail619_12 best.bound = innerCandidates619_11 := by
  unfold candidates
  have row : edge 8293 = SearchData.row8293.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan619_11 : scan edge (search edge 3) 4 [4597] 4597 innerTail619_11 best = best := by
  rw [innerTail619_11, scan, if_neg (by decide), innerCandidates619_11_checked]
  change scan edge (search edge 3) 4 [4597] 4597 innerTail619_12
    (if 3 ≤ innerCandidates619_11.length then search edge 3 [4597, 8293] innerCandidates619_11 12890 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan619_12

def innerTail619_10 : List Nat := 8263 :: innerTail619_11
def innerCandidates619_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates619_10_checked : candidates edge 4 4597 8263 innerTail619_11 best.bound = innerCandidates619_10 := by
  unfold candidates
  have row : edge 8263 = SearchData.row8263.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan619_10 : scan edge (search edge 3) 4 [4597] 4597 innerTail619_10 best = best := by
  rw [innerTail619_10, scan, if_neg (by decide), innerCandidates619_10_checked]
  change scan edge (search edge 3) 4 [4597] 4597 innerTail619_11
    (if 3 ≤ innerCandidates619_10.length then search edge 3 [4597, 8263] innerCandidates619_10 12860 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan619_11

def innerTail619_9 : List Nat := 8011 :: innerTail619_10
def innerCandidates619_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates619_9_checked : candidates edge 4 4597 8011 innerTail619_10 best.bound = innerCandidates619_9 := by
  unfold candidates
  have row : edge 8011 = SearchData.row8011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan619_9 : scan edge (search edge 3) 4 [4597] 4597 innerTail619_9 best = best := by
  rw [innerTail619_9, scan, if_neg (by decide), innerCandidates619_9_checked]
  change scan edge (search edge 3) 4 [4597] 4597 innerTail619_10
    (if 3 ≤ innerCandidates619_9.length then search edge 3 [4597, 8011] innerCandidates619_9 12608 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan619_10

def innerTail619_8 : List Nat := 7993 :: innerTail619_9
def innerCandidates619_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates619_8_checked : candidates edge 4 4597 7993 innerTail619_9 best.bound = innerCandidates619_8 := by
  unfold candidates
  have row : edge 7993 = SearchData.row7993.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan619_8 : scan edge (search edge 3) 4 [4597] 4597 innerTail619_8 best = best := by
  rw [innerTail619_8, scan, if_neg (by decide), innerCandidates619_8_checked]
  change scan edge (search edge 3) 4 [4597] 4597 innerTail619_9
    (if 3 ≤ innerCandidates619_8.length then search edge 3 [4597, 7993] innerCandidates619_8 12590 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan619_9

def innerTail619_7 : List Nat := 7213 :: innerTail619_8
def innerCandidates619_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates619_7_checked : candidates edge 4 4597 7213 innerTail619_8 best.bound = innerCandidates619_7 := by
  unfold candidates
  have row : edge 7213 = SearchData.row7213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan619_7 : scan edge (search edge 3) 4 [4597] 4597 innerTail619_7 best = best := by
  rw [innerTail619_7, scan, if_neg (by decide), innerCandidates619_7_checked]
  change scan edge (search edge 3) 4 [4597] 4597 innerTail619_8
    (if 3 ≤ innerCandidates619_7.length then search edge 3 [4597, 7213] innerCandidates619_7 11810 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan619_8

def innerTail619_6 : List Nat := 6079 :: innerTail619_7
def innerCandidates619_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates619_6_checked : candidates edge 4 4597 6079 innerTail619_7 best.bound = innerCandidates619_6 := by
  unfold candidates
  have row : edge 6079 = SearchData.row6079.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan619_6 : scan edge (search edge 3) 4 [4597] 4597 innerTail619_6 best = best := by
  rw [innerTail619_6, scan, if_neg (by decide), innerCandidates619_6_checked]
  change scan edge (search edge 3) 4 [4597] 4597 innerTail619_7
    (if 3 ≤ innerCandidates619_6.length then search edge 3 [4597, 6079] innerCandidates619_6 10676 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan619_7

def innerTail619_5 : List Nat := 6043 :: innerTail619_6
def innerCandidates619_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates619_5_checked : candidates edge 4 4597 6043 innerTail619_6 best.bound = innerCandidates619_5 := by
  unfold candidates
  have row : edge 6043 = SearchData.row6043.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan619_5 : scan edge (search edge 3) 4 [4597] 4597 innerTail619_5 best = best := by
  rw [innerTail619_5, scan, if_neg (by decide), innerCandidates619_5_checked]
  change scan edge (search edge 3) 4 [4597] 4597 innerTail619_6
    (if 3 ≤ innerCandidates619_5.length then search edge 3 [4597, 6043] innerCandidates619_5 10640 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan619_6

def innerTail619_4 : List Nat := 5749 :: innerTail619_5
def innerCandidates619_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates619_4_checked : candidates edge 4 4597 5749 innerTail619_5 best.bound = innerCandidates619_4 := by
  unfold candidates
  have row : edge 5749 = SearchData.row5749.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan619_4 : scan edge (search edge 3) 4 [4597] 4597 innerTail619_4 best = best := by
  rw [innerTail619_4, scan, if_neg (by decide), innerCandidates619_4_checked]
  change scan edge (search edge 3) 4 [4597] 4597 innerTail619_5
    (if 3 ≤ innerCandidates619_4.length then search edge 3 [4597, 5749] innerCandidates619_4 10346 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan619_5

def innerTail619_3 : List Nat := 5581 :: innerTail619_4
def innerCandidates619_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates619_3_checked : candidates edge 4 4597 5581 innerTail619_4 best.bound = innerCandidates619_3 := by
  unfold candidates
  have row : edge 5581 = SearchData.row5581.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan619_3 : scan edge (search edge 3) 4 [4597] 4597 innerTail619_3 best = best := by
  rw [innerTail619_3, scan, if_neg (by decide), innerCandidates619_3_checked]
  change scan edge (search edge 3) 4 [4597] 4597 innerTail619_4
    (if 3 ≤ innerCandidates619_3.length then search edge 3 [4597, 5581] innerCandidates619_3 10178 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan619_4

def innerTail619_2 : List Nat := 5563 :: innerTail619_3
def innerCandidates619_2 : List Nat := [6079]
set_option maxRecDepth 4096 in
theorem innerCandidates619_2_checked : candidates edge 4 4597 5563 innerTail619_3 best.bound = innerCandidates619_2 := by
  unfold candidates
  have row : edge 5563 = SearchData.row5563.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan619_2 : scan edge (search edge 3) 4 [4597] 4597 innerTail619_2 best = best := by
  rw [innerTail619_2, scan, if_neg (by decide), innerCandidates619_2_checked]
  change scan edge (search edge 3) 4 [4597] 4597 innerTail619_3
    (if 3 ≤ innerCandidates619_2.length then search edge 3 [4597, 5563] innerCandidates619_2 10160 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan619_3

def innerTail619_1 : List Nat := 5113 :: innerTail619_2
def innerCandidates619_1 : List Nat := [8293]
set_option maxRecDepth 4096 in
theorem innerCandidates619_1_checked : candidates edge 4 4597 5113 innerTail619_2 best.bound = innerCandidates619_1 := by
  unfold candidates
  have row : edge 5113 = SearchData.row5113.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan619_1 : scan edge (search edge 3) 4 [4597] 4597 innerTail619_1 best = best := by
  rw [innerTail619_1, scan, if_neg (by decide), innerCandidates619_1_checked]
  change scan edge (search edge 3) 4 [4597] 4597 innerTail619_2
    (if 3 ≤ innerCandidates619_1.length then search edge 3 [4597, 5113] innerCandidates619_1 9710 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan619_2

def innerTail619_0 : List Nat := 4969 :: innerTail619_1
def innerCandidates619_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates619_0_checked : candidates edge 4 4597 4969 innerTail619_1 best.bound = innerCandidates619_0 := by
  unfold candidates
  have row : edge 4969 = SearchData.row4969.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan619_0 : scan edge (search edge 3) 4 [4597] 4597 innerTail619_0 best = best := by
  rw [innerTail619_0, scan, if_neg (by decide), innerCandidates619_0_checked]
  change scan edge (search edge 3) 4 [4597] 4597 innerTail619_1
    (if 3 ≤ innerCandidates619_0.length then search edge 3 [4597, 4969] innerCandidates619_0 9566 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan619_1

theorem rootBranch619 : search edge 4 [4597] rootCandidates619 4597 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4597] 4597 innerTail619_0 best = best
  exact innerScan619_0

end ProjectEulerSolutions.PrimePairUpper.Search
