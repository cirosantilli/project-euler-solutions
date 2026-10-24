import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates1039 : List Nat := []
set_option maxRecDepth 4096 in
theorem rootCandidates1039_checked : candidates edge 5 0 8297 tail1040 best.bound = rootCandidates1039 := by
  unfold candidates
  have row : edge 8297 = SearchData.row8297.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates1038 : List Nat := []
set_option maxRecDepth 4096 in
theorem rootCandidates1038_checked : candidates edge 5 0 8293 tail1039 best.bound = rootCandidates1038 := by
  unfold candidates
  have row : edge 8293 = SearchData.row8293.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates1037 : List Nat := []
set_option maxRecDepth 4096 in
theorem rootCandidates1037_checked : candidates edge 5 0 8291 tail1038 best.bound = rootCandidates1037 := by
  unfold candidates
  have row : edge 8291 = SearchData.row8291.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates1036 : List Nat := []
set_option maxRecDepth 4096 in
theorem rootCandidates1036_checked : candidates edge 5 0 8287 tail1037 best.bound = rootCandidates1036 := by
  unfold candidates
  have row : edge 8287 = SearchData.row8287.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates1035 : List Nat := []
set_option maxRecDepth 4096 in
theorem rootCandidates1035_checked : candidates edge 5 0 8273 tail1036 best.bound = rootCandidates1035 := by
  unfold candidates
  have row : edge 8273 = SearchData.row8273.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates1034 : List Nat := []
set_option maxRecDepth 4096 in
theorem rootCandidates1034_checked : candidates edge 5 0 8269 tail1035 best.bound = rootCandidates1034 := by
  unfold candidates
  have row : edge 8269 = SearchData.row8269.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates1033 : List Nat := []
set_option maxRecDepth 4096 in
theorem rootCandidates1033_checked : candidates edge 5 0 8263 tail1034 best.bound = rootCandidates1033 := by
  unfold candidates
  have row : edge 8263 = SearchData.row8263.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates1032 : List Nat := []
set_option maxRecDepth 4096 in
theorem rootCandidates1032_checked : candidates edge 5 0 8243 tail1033 best.bound = rootCandidates1032 := by
  unfold candidates
  have row : edge 8243 = SearchData.row8243.contains := by funext q; rfl
  rw [row]
  decide

end ProjectEulerSolutions.PrimePairUpper.Search
