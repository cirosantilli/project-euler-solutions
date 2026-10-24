import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates1047 : List Nat := []
set_option maxRecDepth 4096 in
theorem rootCandidates1047_checked : candidates edge 5 0 8387 tail1048 best.bound = rootCandidates1047 := by
  unfold candidates
  have row : edge 8387 = SearchData.row8387.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates1046 : List Nat := []
set_option maxRecDepth 4096 in
theorem rootCandidates1046_checked : candidates edge 5 0 8377 tail1047 best.bound = rootCandidates1046 := by
  unfold candidates
  have row : edge 8377 = SearchData.row8377.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates1045 : List Nat := []
set_option maxRecDepth 4096 in
theorem rootCandidates1045_checked : candidates edge 5 0 8369 tail1046 best.bound = rootCandidates1045 := by
  unfold candidates
  have row : edge 8369 = SearchData.row8369.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates1044 : List Nat := []
set_option maxRecDepth 4096 in
theorem rootCandidates1044_checked : candidates edge 5 0 8363 tail1045 best.bound = rootCandidates1044 := by
  unfold candidates
  have row : edge 8363 = SearchData.row8363.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates1043 : List Nat := []
set_option maxRecDepth 4096 in
theorem rootCandidates1043_checked : candidates edge 5 0 8353 tail1044 best.bound = rootCandidates1043 := by
  unfold candidates
  have row : edge 8353 = SearchData.row8353.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates1042 : List Nat := []
set_option maxRecDepth 4096 in
theorem rootCandidates1042_checked : candidates edge 5 0 8329 tail1043 best.bound = rootCandidates1042 := by
  unfold candidates
  have row : edge 8329 = SearchData.row8329.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates1041 : List Nat := []
set_option maxRecDepth 4096 in
theorem rootCandidates1041_checked : candidates edge 5 0 8317 tail1042 best.bound = rootCandidates1041 := by
  unfold candidates
  have row : edge 8317 = SearchData.row8317.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates1040 : List Nat := [8353, 8377]
set_option maxRecDepth 4096 in
theorem rootCandidates1040_checked : candidates edge 5 0 8311 tail1041 best.bound = rootCandidates1040 := by
  unfold candidates
  have row : edge 8311 = SearchData.row8311.contains := by funext q; rfl
  rw [row]
  decide

end ProjectEulerSolutions.PrimePairUpper.Search
