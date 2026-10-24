import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates1023 : List Nat := []
set_option maxRecDepth 4096 in
theorem rootCandidates1023_checked : candidates edge 5 0 8171 tail1024 best.bound = rootCandidates1023 := by
  unfold candidates
  have row : edge 8171 = SearchData.row8171.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates1022 : List Nat := [8209, 8263]
set_option maxRecDepth 4096 in
theorem rootCandidates1022_checked : candidates edge 5 0 8167 tail1023 best.bound = rootCandidates1022 := by
  unfold candidates
  have row : edge 8167 = SearchData.row8167.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates1021 : List Nat := [8233]
set_option maxRecDepth 4096 in
theorem rootCandidates1021_checked : candidates edge 5 0 8161 tail1022 best.bound = rootCandidates1021 := by
  unfold candidates
  have row : edge 8161 = SearchData.row8161.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates1020 : List Nat := [8219]
set_option maxRecDepth 4096 in
theorem rootCandidates1020_checked : candidates edge 5 0 8147 tail1021 best.bound = rootCandidates1020 := by
  unfold candidates
  have row : edge 8147 = SearchData.row8147.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates1019 : List Nat := []
set_option maxRecDepth 4096 in
theorem rootCandidates1019_checked : candidates edge 5 0 8123 tail1020 best.bound = rootCandidates1019 := by
  unfold candidates
  have row : edge 8123 = SearchData.row8123.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates1018 : List Nat := []
set_option maxRecDepth 4096 in
theorem rootCandidates1018_checked : candidates edge 5 0 8117 tail1019 best.bound = rootCandidates1018 := by
  unfold candidates
  have row : edge 8117 = SearchData.row8117.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates1017 : List Nat := [8123, 8291]
set_option maxRecDepth 4096 in
theorem rootCandidates1017_checked : candidates edge 5 0 8111 tail1018 best.bound = rootCandidates1017 := by
  unfold candidates
  have row : edge 8111 = SearchData.row8111.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates1016 : List Nat := [8233, 8317]
set_option maxRecDepth 4096 in
theorem rootCandidates1016_checked : candidates edge 5 0 8101 tail1017 best.bound = rootCandidates1016 := by
  unfold candidates
  have row : edge 8101 = SearchData.row8101.contains := by funext q; rfl
  rw [row]
  decide

end ProjectEulerSolutions.PrimePairUpper.Search
