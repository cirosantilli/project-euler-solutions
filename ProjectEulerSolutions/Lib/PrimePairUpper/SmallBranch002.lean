import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates1031 : List Nat := []
set_option maxRecDepth 4096 in
theorem rootCandidates1031_checked : candidates edge 5 0 8237 tail1032 best.bound = rootCandidates1031 := by
  unfold candidates
  have row : edge 8237 = SearchData.row8237.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates1030 : List Nat := [8329]
set_option maxRecDepth 4096 in
theorem rootCandidates1030_checked : candidates edge 5 0 8233 tail1031 best.bound = rootCandidates1030 := by
  unfold candidates
  have row : edge 8233 = SearchData.row8233.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates1029 : List Nat := [8273]
set_option maxRecDepth 4096 in
theorem rootCandidates1029_checked : candidates edge 5 0 8231 tail1030 best.bound = rootCandidates1029 := by
  unfold candidates
  have row : edge 8231 = SearchData.row8231.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates1028 : List Nat := []
set_option maxRecDepth 4096 in
theorem rootCandidates1028_checked : candidates edge 5 0 8221 tail1029 best.bound = rootCandidates1028 := by
  unfold candidates
  have row : edge 8221 = SearchData.row8221.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates1027 : List Nat := []
set_option maxRecDepth 4096 in
theorem rootCandidates1027_checked : candidates edge 5 0 8219 tail1028 best.bound = rootCandidates1027 := by
  unfold candidates
  have row : edge 8219 = SearchData.row8219.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates1026 : List Nat := []
set_option maxRecDepth 4096 in
theorem rootCandidates1026_checked : candidates edge 5 0 8209 tail1027 best.bound = rootCandidates1026 := by
  unfold candidates
  have row : edge 8209 = SearchData.row8209.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates1025 : List Nat := []
set_option maxRecDepth 4096 in
theorem rootCandidates1025_checked : candidates edge 5 0 8191 tail1026 best.bound = rootCandidates1025 := by
  unfold candidates
  have row : edge 8191 = SearchData.row8191.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates1024 : List Nat := [8293]
set_option maxRecDepth 4096 in
theorem rootCandidates1024_checked : candidates edge 5 0 8179 tail1025 best.bound = rootCandidates1024 := by
  unfold candidates
  have row : edge 8179 = SearchData.row8179.contains := by funext q; rfl
  rw [row]
  decide

end ProjectEulerSolutions.PrimePairUpper.Search
