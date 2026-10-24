import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates1015 : List Nat := []
set_option maxRecDepth 4096 in
theorem rootCandidates1015_checked : candidates edge 5 0 8093 tail1016 best.bound = rootCandidates1015 := by
  unfold candidates
  have row : edge 8093 = SearchData.row8093.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates1014 : List Nat := []
set_option maxRecDepth 4096 in
theorem rootCandidates1014_checked : candidates edge 5 0 8089 tail1015 best.bound = rootCandidates1014 := by
  unfold candidates
  have row : edge 8089 = SearchData.row8089.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates1013 : List Nat := [8219]
set_option maxRecDepth 4096 in
theorem rootCandidates1013_checked : candidates edge 5 0 8087 tail1014 best.bound = rootCandidates1013 := by
  unfold candidates
  have row : edge 8087 = SearchData.row8087.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates1012 : List Nat := []
set_option maxRecDepth 4096 in
theorem rootCandidates1012_checked : candidates edge 5 0 8081 tail1013 best.bound = rootCandidates1012 := by
  unfold candidates
  have row : edge 8081 = SearchData.row8081.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates1011 : List Nat := []
set_option maxRecDepth 4096 in
theorem rootCandidates1011_checked : candidates edge 5 0 8069 tail1012 best.bound = rootCandidates1011 := by
  unfold candidates
  have row : edge 8069 = SearchData.row8069.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates1010 : List Nat := [8269]
set_option maxRecDepth 4096 in
theorem rootCandidates1010_checked : candidates edge 5 0 8059 tail1011 best.bound = rootCandidates1010 := by
  unfold candidates
  have row : edge 8059 = SearchData.row8059.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates1009 : List Nat := [8287]
set_option maxRecDepth 4096 in
theorem rootCandidates1009_checked : candidates edge 5 0 8053 tail1010 best.bound = rootCandidates1009 := by
  unfold candidates
  have row : edge 8053 = SearchData.row8053.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates1008 : List Nat := []
set_option maxRecDepth 4096 in
theorem rootCandidates1008_checked : candidates edge 5 0 8039 tail1009 best.bound = rootCandidates1008 := by
  unfold candidates
  have row : edge 8039 = SearchData.row8039.contains := by funext q; rfl
  rw [row]
  decide

end ProjectEulerSolutions.PrimePairUpper.Search
