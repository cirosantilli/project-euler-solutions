import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates975 : List Nat := [7823, 7853]
set_option maxRecDepth 4096 in
theorem rootCandidates975_checked : candidates edge 5 0 7703 tail976 best.bound = rootCandidates975 := by
  unfold candidates
  have row : edge 7703 = SearchData.row7703.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates974 : List Nat := [7873]
set_option maxRecDepth 4096 in
theorem rootCandidates974_checked : candidates edge 5 0 7699 tail975 best.bound = rootCandidates974 := by
  unfold candidates
  have row : edge 7699 = SearchData.row7699.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates973 : List Nat := [8147]
set_option maxRecDepth 4096 in
theorem rootCandidates973_checked : candidates edge 5 0 7691 tail974 best.bound = rootCandidates973 := by
  unfold candidates
  have row : edge 7691 = SearchData.row7691.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates972 : List Nat := [8311]
set_option maxRecDepth 4096 in
theorem rootCandidates972_checked : candidates edge 5 0 7687 tail973 best.bound = rootCandidates972 := by
  unfold candidates
  have row : edge 7687 = SearchData.row7687.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates971 : List Nat := []
set_option maxRecDepth 4096 in
theorem rootCandidates971_checked : candidates edge 5 0 7681 tail972 best.bound = rootCandidates971 := by
  unfold candidates
  have row : edge 7681 = SearchData.row7681.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates970 : List Nat := [7823]
set_option maxRecDepth 4096 in
theorem rootCandidates970_checked : candidates edge 5 0 7673 tail971 best.bound = rootCandidates970 := by
  unfold candidates
  have row : edge 7673 = SearchData.row7673.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates969 : List Nat := [8311]
set_option maxRecDepth 4096 in
theorem rootCandidates969_checked : candidates edge 5 0 7669 tail970 best.bound = rootCandidates969 := by
  unfold candidates
  have row : edge 7669 = SearchData.row7669.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates968 : List Nat := []
set_option maxRecDepth 4096 in
theorem rootCandidates968_checked : candidates edge 5 0 7649 tail969 best.bound = rootCandidates968 := by
  unfold candidates
  have row : edge 7649 = SearchData.row7649.contains := by funext q; rfl
  rw [row]
  decide

end ProjectEulerSolutions.PrimePairUpper.Search
