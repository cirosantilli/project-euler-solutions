import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates999 : List Nat := []
set_option maxRecDepth 4096 in
theorem rootCandidates999_checked : candidates edge 5 0 7933 tail1000 best.bound = rootCandidates999 := by
  unfold candidates
  have row : edge 7933 = SearchData.row7933.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates998 : List Nat := []
set_option maxRecDepth 4096 in
theorem rootCandidates998_checked : candidates edge 5 0 7927 tail999 best.bound = rootCandidates998 := by
  unfold candidates
  have row : edge 7927 = SearchData.row7927.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates997 : List Nat := [8117]
set_option maxRecDepth 4096 in
theorem rootCandidates997_checked : candidates edge 5 0 7919 tail998 best.bound = rootCandidates997 := by
  unfold candidates
  have row : edge 7919 = SearchData.row7919.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates996 : List Nat := [7919, 8039, 8081, 8297]
set_option maxRecDepth 4096 in
theorem rootCandidates996_checked : candidates edge 5 0 7907 tail997 best.bound = rootCandidates996 := by
  unfold candidates
  have row : edge 7907 = SearchData.row7907.contains := by funext q; rfl
  rw [row]
  decide

def innerTail996_4 : List Nat := []
theorem innerScan996_4 : scan edge (search edge 3) 4 [7907] 7907 innerTail996_4 best = best := rfl

def innerTail996_3 : List Nat := 8297 :: innerTail996_4
def innerCandidates996_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates996_3_checked : candidates edge 4 7907 8297 innerTail996_4 best.bound = innerCandidates996_3 := by
  unfold candidates
  have row : edge 8297 = SearchData.row8297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan996_3 : scan edge (search edge 3) 4 [7907] 7907 innerTail996_3 best = best := by
  rw [innerTail996_3, scan, if_neg (by decide), innerCandidates996_3_checked]
  change scan edge (search edge 3) 4 [7907] 7907 innerTail996_4
    (if 3 ≤ innerCandidates996_3.length then search edge 3 [7907, 8297] innerCandidates996_3 16204 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan996_4

def innerTail996_2 : List Nat := 8081 :: innerTail996_3
def innerCandidates996_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates996_2_checked : candidates edge 4 7907 8081 innerTail996_3 best.bound = innerCandidates996_2 := by
  unfold candidates
  have row : edge 8081 = SearchData.row8081.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan996_2 : scan edge (search edge 3) 4 [7907] 7907 innerTail996_2 best = best := by
  rw [innerTail996_2, scan, if_neg (by decide), innerCandidates996_2_checked]
  change scan edge (search edge 3) 4 [7907] 7907 innerTail996_3
    (if 3 ≤ innerCandidates996_2.length then search edge 3 [7907, 8081] innerCandidates996_2 15988 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan996_3

def innerTail996_1 : List Nat := 8039 :: innerTail996_2
def innerCandidates996_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates996_1_checked : candidates edge 4 7907 8039 innerTail996_2 best.bound = innerCandidates996_1 := by
  unfold candidates
  have row : edge 8039 = SearchData.row8039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan996_1 : scan edge (search edge 3) 4 [7907] 7907 innerTail996_1 best = best := by
  rw [innerTail996_1, scan, if_neg (by decide), innerCandidates996_1_checked]
  change scan edge (search edge 3) 4 [7907] 7907 innerTail996_2
    (if 3 ≤ innerCandidates996_1.length then search edge 3 [7907, 8039] innerCandidates996_1 15946 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan996_2

def innerTail996_0 : List Nat := 7919 :: innerTail996_1
def innerCandidates996_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates996_0_checked : candidates edge 4 7907 7919 innerTail996_1 best.bound = innerCandidates996_0 := by
  unfold candidates
  have row : edge 7919 = SearchData.row7919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan996_0 : scan edge (search edge 3) 4 [7907] 7907 innerTail996_0 best = best := by
  rw [innerTail996_0, scan, if_neg (by decide), innerCandidates996_0_checked]
  change scan edge (search edge 3) 4 [7907] 7907 innerTail996_1
    (if 3 ≤ innerCandidates996_0.length then search edge 3 [7907, 7919] innerCandidates996_0 15826 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan996_1

theorem rootBranch996 : search edge 4 [7907] rootCandidates996 7907 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [7907] 7907 innerTail996_0 best = best
  exact innerScan996_0

def rootCandidates995 : List Nat := [8081]
set_option maxRecDepth 4096 in
theorem rootCandidates995_checked : candidates edge 5 0 7901 tail996 best.bound = rootCandidates995 := by
  unfold candidates
  have row : edge 7901 = SearchData.row7901.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates994 : List Nat := []
set_option maxRecDepth 4096 in
theorem rootCandidates994_checked : candidates edge 5 0 7883 tail995 best.bound = rootCandidates994 := by
  unfold candidates
  have row : edge 7883 = SearchData.row7883.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates993 : List Nat := [8353]
set_option maxRecDepth 4096 in
theorem rootCandidates993_checked : candidates edge 5 0 7879 tail994 best.bound = rootCandidates993 := by
  unfold candidates
  have row : edge 7879 = SearchData.row7879.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates992 : List Nat := []
set_option maxRecDepth 4096 in
theorem rootCandidates992_checked : candidates edge 5 0 7877 tail993 best.bound = rootCandidates992 := by
  unfold candidates
  have row : edge 7877 = SearchData.row7877.contains := by funext q; rfl
  rw [row]
  decide

end ProjectEulerSolutions.PrimePairUpper.Search
