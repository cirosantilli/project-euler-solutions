import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates943 : List Nat := [8191, 8317]
set_option maxRecDepth 4096 in
theorem rootCandidates943_checked : candidates edge 5 0 7477 tail944 best.bound = rootCandidates943 := by
  unfold candidates
  have row : edge 7477 = SearchData.row7477.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates942 : List Nat := [7927]
set_option maxRecDepth 4096 in
theorem rootCandidates942_checked : candidates edge 5 0 7459 tail943 best.bound = rootCandidates942 := by
  unfold candidates
  have row : edge 7459 = SearchData.row7459.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates941 : List Nat := [7757, 8369]
set_option maxRecDepth 4096 in
theorem rootCandidates941_checked : candidates edge 5 0 7457 tail942 best.bound = rootCandidates941 := by
  unfold candidates
  have row : edge 7457 = SearchData.row7457.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates940 : List Nat := [7853]
set_option maxRecDepth 4096 in
theorem rootCandidates940_checked : candidates edge 5 0 7451 tail941 best.bound = rootCandidates940 := by
  unfold candidates
  have row : edge 7451 = SearchData.row7451.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates939 : List Nat := [7559, 8219]
set_option maxRecDepth 4096 in
theorem rootCandidates939_checked : candidates edge 5 0 7433 tail940 best.bound = rootCandidates939 := by
  unfold candidates
  have row : edge 7433 = SearchData.row7433.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates938 : List Nat := [7639, 8017, 8263]
set_option maxRecDepth 4096 in
theorem rootCandidates938_checked : candidates edge 5 0 7417 tail939 best.bound = rootCandidates938 := by
  unfold candidates
  have row : edge 7417 = SearchData.row7417.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates937 : List Nat := [8329]
set_option maxRecDepth 4096 in
theorem rootCandidates937_checked : candidates edge 5 0 7411 tail938 best.bound = rootCandidates937 := by
  unfold candidates
  have row : edge 7411 = SearchData.row7411.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates936 : List Nat := [7687, 7867]
set_option maxRecDepth 4096 in
theorem rootCandidates936_checked : candidates edge 5 0 7393 tail937 best.bound = rootCandidates936 := by
  unfold candidates
  have row : edge 7393 = SearchData.row7393.contains := by funext q; rfl
  rw [row]
  decide

end ProjectEulerSolutions.PrimePairUpper.Search
