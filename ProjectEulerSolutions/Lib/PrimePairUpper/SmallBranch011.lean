import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates959 : List Nat := [8069]
set_option maxRecDepth 4096 in
theorem rootCandidates959_checked : candidates edge 5 0 7577 tail960 best.bound = rootCandidates959 := by
  unfold candidates
  have row : edge 7577 = SearchData.row7577.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates958 : List Nat := [7603, 7621, 8263, 8329]
set_option maxRecDepth 4096 in
theorem rootCandidates958_checked : candidates edge 5 0 7573 tail959 best.bound = rootCandidates958 := by
  unfold candidates
  have row : edge 7573 = SearchData.row7573.contains := by funext q; rfl
  rw [row]
  decide

def innerTail958_4 : List Nat := []
theorem innerScan958_4 : scan edge (search edge 3) 4 [7573] 7573 innerTail958_4 best = best := rfl

def innerTail958_3 : List Nat := 8329 :: innerTail958_4
def innerCandidates958_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates958_3_checked : candidates edge 4 7573 8329 innerTail958_4 best.bound = innerCandidates958_3 := by
  unfold candidates
  have row : edge 8329 = SearchData.row8329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan958_3 : scan edge (search edge 3) 4 [7573] 7573 innerTail958_3 best = best := by
  rw [innerTail958_3, scan, if_neg (by decide), innerCandidates958_3_checked]
  change scan edge (search edge 3) 4 [7573] 7573 innerTail958_4
    (if 3 ≤ innerCandidates958_3.length then search edge 3 [7573, 8329] innerCandidates958_3 15902 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan958_4

def innerTail958_2 : List Nat := 8263 :: innerTail958_3
def innerCandidates958_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates958_2_checked : candidates edge 4 7573 8263 innerTail958_3 best.bound = innerCandidates958_2 := by
  unfold candidates
  have row : edge 8263 = SearchData.row8263.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan958_2 : scan edge (search edge 3) 4 [7573] 7573 innerTail958_2 best = best := by
  rw [innerTail958_2, scan, if_neg (by decide), innerCandidates958_2_checked]
  change scan edge (search edge 3) 4 [7573] 7573 innerTail958_3
    (if 3 ≤ innerCandidates958_2.length then search edge 3 [7573, 8263] innerCandidates958_2 15836 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan958_3

def innerTail958_1 : List Nat := 7621 :: innerTail958_2
def innerCandidates958_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates958_1_checked : candidates edge 4 7573 7621 innerTail958_2 best.bound = innerCandidates958_1 := by
  unfold candidates
  have row : edge 7621 = SearchData.row7621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan958_1 : scan edge (search edge 3) 4 [7573] 7573 innerTail958_1 best = best := by
  rw [innerTail958_1, scan, if_neg (by decide), innerCandidates958_1_checked]
  change scan edge (search edge 3) 4 [7573] 7573 innerTail958_2
    (if 3 ≤ innerCandidates958_1.length then search edge 3 [7573, 7621] innerCandidates958_1 15194 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan958_2

def innerTail958_0 : List Nat := 7603 :: innerTail958_1
def innerCandidates958_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates958_0_checked : candidates edge 4 7573 7603 innerTail958_1 best.bound = innerCandidates958_0 := by
  unfold candidates
  have row : edge 7603 = SearchData.row7603.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan958_0 : scan edge (search edge 3) 4 [7573] 7573 innerTail958_0 best = best := by
  rw [innerTail958_0, scan, if_neg (by decide), innerCandidates958_0_checked]
  change scan edge (search edge 3) 4 [7573] 7573 innerTail958_1
    (if 3 ≤ innerCandidates958_0.length then search edge 3 [7573, 7603] innerCandidates958_0 15176 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan958_1

theorem rootBranch958 : search edge 4 [7573] rootCandidates958 7573 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [7573] 7573 innerTail958_0 best = best
  exact innerScan958_0

def rootCandidates957 : List Nat := [7723, 7963]
set_option maxRecDepth 4096 in
theorem rootCandidates957_checked : candidates edge 5 0 7561 tail958 best.bound = rootCandidates957 := by
  unfold candidates
  have row : edge 7561 = SearchData.row7561.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates956 : List Nat := [8111]
set_option maxRecDepth 4096 in
theorem rootCandidates956_checked : candidates edge 5 0 7559 tail957 best.bound = rootCandidates956 := by
  unfold candidates
  have row : edge 7559 = SearchData.row7559.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates955 : List Nat := [8089, 8209]
set_option maxRecDepth 4096 in
theorem rootCandidates955_checked : candidates edge 5 0 7549 tail956 best.bound = rootCandidates955 := by
  unfold candidates
  have row : edge 7549 = SearchData.row7549.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates954 : List Nat := [8093]
set_option maxRecDepth 4096 in
theorem rootCandidates954_checked : candidates edge 5 0 7547 tail955 best.bound = rootCandidates954 := by
  unfold candidates
  have row : edge 7547 = SearchData.row7547.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates953 : List Nat := [7883, 8111]
set_option maxRecDepth 4096 in
theorem rootCandidates953_checked : candidates edge 5 0 7541 tail954 best.bound = rootCandidates953 := by
  unfold candidates
  have row : edge 7541 = SearchData.row7541.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates952 : List Nat := [8101, 8179]
set_option maxRecDepth 4096 in
theorem rootCandidates952_checked : candidates edge 5 0 7537 tail953 best.bound = rootCandidates952 := by
  unfold candidates
  have row : edge 7537 = SearchData.row7537.contains := by funext q; rfl
  rw [row]
  decide

end ProjectEulerSolutions.PrimePairUpper.Search
