import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates991 : List Nat := [8293, 8353]
set_option maxRecDepth 4096 in
theorem rootCandidates991_checked : candidates edge 5 0 7873 tail992 best.bound = rootCandidates991 := by
  unfold candidates
  have row : edge 7873 = SearchData.row7873.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates990 : List Nat := []
set_option maxRecDepth 4096 in
theorem rootCandidates990_checked : candidates edge 5 0 7867 tail991 best.bound = rootCandidates990 := by
  unfold candidates
  have row : edge 7867 = SearchData.row7867.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates989 : List Nat := [7883]
set_option maxRecDepth 4096 in
theorem rootCandidates989_checked : candidates edge 5 0 7853 tail990 best.bound = rootCandidates989 := by
  unfold candidates
  have row : edge 7853 = SearchData.row7853.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates988 : List Nat := []
set_option maxRecDepth 4096 in
theorem rootCandidates988_checked : candidates edge 5 0 7841 tail989 best.bound = rootCandidates988 := by
  unfold candidates
  have row : edge 7841 = SearchData.row7841.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates987 : List Nat := []
set_option maxRecDepth 4096 in
theorem rootCandidates987_checked : candidates edge 5 0 7829 tail988 best.bound = rootCandidates987 := by
  unfold candidates
  have row : edge 7829 = SearchData.row7829.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates986 : List Nat := [8093]
set_option maxRecDepth 4096 in
theorem rootCandidates986_checked : candidates edge 5 0 7823 tail987 best.bound = rootCandidates986 := by
  unfold candidates
  have row : edge 7823 = SearchData.row7823.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates985 : List Nat := [8147]
set_option maxRecDepth 4096 in
theorem rootCandidates985_checked : candidates edge 5 0 7817 tail986 best.bound = rootCandidates985 := by
  unfold candidates
  have row : edge 7817 = SearchData.row7817.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates984 : List Nat := [7919, 8009, 8363]
set_option maxRecDepth 4096 in
theorem rootCandidates984_checked : candidates edge 5 0 7793 tail985 best.bound = rootCandidates984 := by
  unfold candidates
  have row : edge 7793 = SearchData.row7793.contains := by funext q; rfl
  rw [row]
  decide

end ProjectEulerSolutions.PrimePairUpper.Search
